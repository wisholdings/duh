import json
import pandas as pd
from sqlalchemy import create_engine, exc as sqlalchemy_exc, text, inspect
import os
import sys
import traceback
import re

# --- Configuration Loading ---
try:
    # Define BASE_OUTPUT_DIR directly
    BASE_OUTPUT_DIR = r"D:\EIA_STACK"  # Adjust to your actual base directory
    print("Using BASE_OUTPUT_DIR:", BASE_OUTPUT_DIR)
except Exception as e:
    print(f"Warning: Error setting BASE_OUTPUT_DIR ({e}). Using current directory as base.")
    BASE_OUTPUT_DIR = "."

try:
    # Load region configuration from config/regions.json
    # Changed to point to config at the same level as scripts
    script_dir = os.path.dirname(__file__)  # Directory of the script (scripts)
    project_root = os.path.abspath(os.path.join(script_dir, '..'))  # Parent directory (BestVersion)
    regions_config_path = os.path.join(project_root, 'config', 'regions.json')  # BestVersion/config/regions.json
    with open(regions_config_path, 'r') as f:
        regions = json.load(f)
    print(f"Successfully loaded region configuration from {regions_config_path}")
except FileNotFoundError:
    print(f"Error: Region configuration file not found at {regions_config_path}.")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: Could not decode {regions_config_path}. Please check its JSON format.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred loading region configuration: {e}")
    print(traceback.format_exc())
    sys.exit(1)

# --- CapabilityDataFetcher Class Definition ---
class CapabilityDataFetcher:
    def __init__(self, region, energy_type, base_dir=BASE_OUTPUT_DIR, start_date='2019-01-01', db_connection_string=None):
        """
        Initialize the CapabilityDataFetcher. Saves output as Parquet.

        :param region: Region name (e.g., 'CALIFORNIA', 'CAROLINA').
        :param energy_type: Type of energy capability (e.g., 'Solar', 'NG', 'Wind').
        :param base_dir: Base directory for output files (default: BASE_OUTPUT_DIR).
        :param start_date: Start date for filtering data (default: '2019-01-01').
        :param db_connection_string: SQL Server connection string (default provided if None).
        """
        self.region = region.upper()  # Keep region name upper for DB table matching
        self.energy_type = energy_type.upper()  # Normalize to uppercase for consistency
        self.base_dir = base_dir
        self.start_date = start_date
        self.db_connection_string = db_connection_string or (
            "mssql+pymssql://powergasmaster:Aberlour2023@"
            "gaspowstosbxarmservus2001.database.windows.net:1433/gasstosbxarmservus2001"
        )

        # Define output directory structure
        self.region_output_dir = os.path.join(self.base_dir, self.region.lower(), "Capacity")
        self.output_file = os.path.join(self.region_output_dir, f"{self.region.lower()}_{self.energy_type.lower()}.parquet")

    def discover_capability_tables(self, engine):
        """
        Dynamically discover all available capability tables in the database.
        Returns a dictionary mapping energy types to table names.
        """
        try:
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            
            # Filter for capability tables
            capability_tables = [table for table in tables if 'CAPABILITY' in table.upper() and 'PRODUCTION' in table.upper()]
            
            print(f"  Found {len(capability_tables)} capability tables:")
            for table in capability_tables:
                print(f"    {table}")
            
            # Extract energy types from table names
            energy_table_mapping = {}
            for table in capability_tables:
                # Pattern: PRODUCTION_{ENERGY_TYPE}_CAPABILITY
                match = re.search(r'PRODUCTION_(.+)_CAPABILITY', table.upper())
                if match:
                    energy_type = match.group(1)
                    # Normalize energy type names to match your new classification
                    energy_type = self.normalize_energy_type(energy_type)
                    energy_table_mapping[energy_type] = table
                    print(f"    Mapped: {energy_type} -> {table}")
            
            return energy_table_mapping
            
        except Exception as e:
            print(f"[ERROR] Failed to discover capability tables: {e}")
            print(traceback.format_exc())
            return {}

    def normalize_energy_type(self, energy_type):
        """
        Normalize energy type names to consistent format matching new technology categories.
        """
        # Updated mappings to match your new technology classification
        normalize_map = {
            # Natural Gas variations
            'NATURALGAS': 'NATURAL_GAS',
            'NATURAL_GAS': 'NATURAL_GAS',
            'NG': 'NATURAL_GAS',
            
            # Battery variations
            'BATTERIES': 'BATTERIES',
            'BATTERY': 'BATTERIES',
            'BAT': 'BATTERIES',
            
            # Solar variations
            'SOLAR': 'SOLAR',
            'SOLAR_BATTERIES': 'SOLAR_BATTERIES',
            
            # Storage variations
            'PUMPED_STORAGE': 'PUMPED_STORAGE',
            'PUMPED': 'PUMPED_STORAGE',
            
            # Wind variations
            'WIND': 'WIND',
            
            # Nuclear variations
            'NUCLEAR': 'NUCLEAR',
            'NUC': 'NUCLEAR',
            
            # Hydroelectric variations
            'HYDROELECTRIC': 'HYDROELECTRIC',
            'HYDRO': 'HYDROELECTRIC',
            'WAT': 'HYDROELECTRIC',
            
            # Coal variations
            'COAL': 'COAL',
            
            # Oil variations
            'OIL': 'OIL',
            
            # Geothermal variations
            'GEOTHERMAL': 'GEOTHERMAL',
            'GEO': 'GEOTHERMAL',
            
            # Biomass variations
            'BIOMASS': 'BIOMASS',
            
            # Other variations
            'OTHER': 'OTHER'
        }
        
        return normalize_map.get(energy_type, energy_type)

    def get_region_columns_for_table(self, engine, table_name, region):
        """
        Get all columns for a specific region in a capability table.
        Returns list of column names that match the region pattern.
        """
        try:
            inspector = inspect(engine)
            columns = inspector.get_columns(table_name)
            column_names = [col['name'] for col in columns]
            
            # Find columns that match the region pattern
            region_columns = []
            
            # Handle region name mapping for column matching
            region_column_mapping = {
                'CALIFORNIA': 'California',
                'CAROLINA': 'Carolina', 
                'CAROLINAS': 'Carolina',
                'CENTRAL': 'Central',
                'FLORIDA': 'Florida',
                'MIDATLANTIC': 'Midatlantic',
                'MID-ATLANTIC': 'Midatlantic',
                'MIDWEST': 'Midwest',
                'NEWENGLAND': 'Newengland',
                'NEW ENGLAND': 'Newengland',
                'NEWYORK': 'Newyork',
                'NEW YORK': 'Newyork',
                'NORTHWEST': 'Northwest',
                'SOUTHEAST': 'Southeast',
                'SOUTHWEST': 'Southwest',
                'TENNESSEE': 'Tennessee',
                'TEXAS': 'Texas'
            }
            
            # Get the actual column prefix used in the database
            db_region_name = region_column_mapping.get(region, region.title())
            
            for col in column_names:
                # Look for columns that start with the mapped region name and contain capacity info
                if (col.startswith(f"{db_region_name}_") and 
                    ('Total_Capacity' in col or 'CAPACITY' in col.upper())):
                    region_columns.append(col)
            
            print(f"    Found columns for {region} (mapped to {db_region_name}) in {table_name}: {region_columns}")
            return region_columns
            
        except Exception as e:
            print(f"[ERROR] Failed to get columns for {table_name}: {e}")
            return []

    def should_fetch_energy(self):
        """
        Check if the energy type should be fetched for the region.
        Updated exclusion logic for new technology categories.
        """
        # Updated exclusion logic for new technology categories
        exclusions = [
            # Wind exclusions (regions with minimal wind capacity)
            (["CAROLINA", "TENNESSEE", "SOUTHEAST", "FLORIDA"], "WIND"),
            
            # Solar exclusions (regions with minimal solar capacity)
            (["NEWYORK"], "SOLAR"),
            
            # Solar_Batteries exclusions (very limited deployment)
            (["NEWYORK", "CAROLINA", "TENNESSEE", "SOUTHEAST"], "SOLAR_BATTERIES"),
            
            # Geothermal exclusions (only viable in certain regions)
            (["CAROLINA", "TENNESSEE", "SOUTHEAST", "FLORIDA", "NEWYORK", "NEWENGLAND", "MIDWEST", "MIDATLANTIC"], "GEOTHERMAL"),
            
            # Pumped Storage exclusions (requires specific geography)
            (["FLORIDA", "TEXAS"], "PUMPED_STORAGE"),
        ]
        
        for excluded_regions, excluded_energy in exclusions:
            if self.region in excluded_regions and self.energy_type == excluded_energy:
                print(f"Skipping {self.region} {self.energy_type} data ({excluded_energy} excluded for this region).")
                return False
        
        return True

    def fetch_capability_data(self):
        """Fetch capability data from SQL database."""
        if not self.should_fetch_energy():
            return None

        print(f"Connecting to database for {self.region} {self.energy_type}...")
        try:
            engine = create_engine(self.db_connection_string)
            
            # Discover available capability tables
            energy_table_mapping = self.discover_capability_tables(engine)
            
            if not energy_table_mapping:
                print(f"[ERROR] No capability tables found in database")
                return None
            
            # Find the table for this energy type
            table_name = energy_table_mapping.get(self.energy_type)
            if not table_name:
                print(f"Warning: No table found for energy type: {self.energy_type}")
                print(f"Available energy types: {list(energy_table_mapping.keys())}")
                return None

            # Get region-specific columns for this table
            region_columns = self.get_region_columns_for_table(engine, table_name, self.region)
            
            if not region_columns:
                print(f"Warning: No columns found for region {self.region} in table {table_name}")
                return None

            # Build the query with all available region columns
            columns_str = ', '.join([f'[{col}]' for col in region_columns])
            
            query = f"""
            SELECT
                [Datetime],
                {columns_str}
            FROM [dbo].[{table_name}]
            WHERE [Datetime] >= %s
            ORDER BY [Datetime]
            """
            
            print(f"Executing query: {query}")
            print(f"Fetching {self.region} {self.energy_type} capability data from SQL...")
            
            df_capability = pd.read_sql(query, engine, params=(self.start_date,))
            print(f"  Fetched {len(df_capability)} records.")
            print(f"  Columns: {list(df_capability.columns)}")
            
            return df_capability

        except sqlalchemy_exc.SQLAlchemyError as e:
            print(f"[ERROR] Database connection or query failed for {self.region} {self.energy_type}: {e}")
            print(traceback.format_exc())
            return None
        except ImportError:
            print("[ERROR] Required database driver (e.g., pymssql) not installed.")
            return None
        except Exception as e:
            print(f"[ERROR] Error during SQL fetch for {self.region} {self.energy_type}: {e}")
            print(traceback.format_exc())
            return None

    def process_data(self, df_capability):
        """Process the fetched capability data."""
        if df_capability is None or df_capability.empty:
            print("Skipping processing for None or empty DataFrame.")
            return None

        print(f"Processing {self.region} {self.energy_type} data...")
        try:
            df_processed = df_capability.copy()
            if "Datetime" in df_processed.columns:
                df_processed.rename(columns={"Datetime": "date"}, inplace=True)
            elif "date" not in df_processed.columns:
                print("Error: Neither 'Datetime' nor 'date' column found. Cannot process.")
                return None

            if not pd.api.types.is_datetime64_any_dtype(df_processed['date']):
                print("  Converting 'date' column to datetime...")
                df_processed["date"] = pd.to_datetime(df_processed["date"], errors='coerce')
                rows_before = len(df_processed)
                df_processed.dropna(subset=['date'], inplace=True)
                if len(df_processed) < rows_before:
                    print(f"  Removed {rows_before - len(df_processed)} rows with invalid dates.")

            # Check for null columns
            null_columns = df_processed.drop(columns=['date']).isnull().all()
            if null_columns.any():
                null_col_names = null_columns[null_columns].index.tolist()
                print(f"  Warning: The following columns contain only null values: {null_col_names}")

            print(f"Processed {self.region} {self.energy_type} data to {len(df_processed)} rows starting from {self.start_date}")
            print(f"  Date range: {df_processed['date'].min()} to {df_processed['date'].max()}")
            print(f"  Final columns: {list(df_processed.columns)}")
            
            return df_processed
        except Exception as e:
            print(f"Error during data processing for {self.region} {self.energy_type}: {e}")
            print(traceback.format_exc())
            return None

    def save_data(self, df_capability):
        """Save the processed DataFrame to a Parquet file if data exists."""
        if df_capability is None or df_capability.empty:
            print(f"No {self.energy_type} data processed or available to save for {self.region}.")
            return

        print(f"Saving {self.region} {self.energy_type} capability data to Parquet: '{self.output_file}'")
        try:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            float_cols = df_capability.select_dtypes(include=['float64', 'float32']).columns
            if not float_cols.empty:
                print(f"  Rounding float columns: {list(float_cols)}")
                df_capability.loc[:, float_cols] = df_capability[float_cols].round(2)

            df_capability.to_parquet(self.output_file, index=False, engine='pyarrow')
            print(f"  {self.energy_type} capability data successfully saved as Parquet.")
            
            # Verify saved file
            try:
                saved_df = pd.read_parquet(self.output_file, engine='pyarrow')
                print(f"  Verified: Saved file has {len(saved_df)} rows and {len(saved_df.columns)} columns")
                del saved_df
            except Exception as e:
                print(f"  Warning: Could not verify saved file: {e}")
                
        except Exception as e:
            print(f"[ERROR] Failed to save {self.energy_type} capability Parquet file for {self.region}: {e}")
            print(traceback.format_exc())

    def run(self):
        """Execute the full capability data fetching, processing, and saving process if output file doesn't exist."""
        print(f"\n--- Running Capability Data Fetcher for: {self.region} - {self.energy_type} ---")

        if os.path.exists(self.output_file):
            print(f"Skipping fetch/process/save: Output file already exists at '{self.output_file}'")
            print(f"--- Finished Capability Data Fetcher for: {self.region} - {self.energy_type} (Skipped) ---")
            return

        df_capability = None
        processed_df = None
        try:
            df_capability = self.fetch_capability_data()
            processed_df = self.process_data(df_capability)
            self.save_data(processed_df)
        except Exception as e:
            print(f"[CRITICAL ERROR] Unhandled exception in run for {self.region} {self.energy_type}: {e}")
            print(traceback.format_exc())
        finally:
            if df_capability is not None:
                del df_capability
            if processed_df is not None:
                del processed_df
            print(f"--- Finished Capability Data Fetcher for: {self.region} - {self.energy_type} ---")

# --- Main Execution Function ---
def discover_all_energy_types():
    """
    Discover all available energy types from the database capability tables.
    """
    try:
        db_connection_string = (
        )
        engine = create_engine(db_connection_string)
        
        # Create a temporary fetcher to use its discovery methods
        temp_fetcher = CapabilityDataFetcher("TEMP", "TEMP")
        energy_table_mapping = temp_fetcher.discover_capability_tables(engine)
        
        return list(energy_table_mapping.keys())
        
    except Exception as e:
        print(f"[ERROR] Failed to discover energy types: {e}")
        print("Falling back to hardcoded energy types...")
        # Updated fallback list to match new technology categories
        return [
            "NATURAL_GAS", "SOLAR", "SOLAR_BATTERIES", "BATTERIES", "PUMPED_STORAGE", 
            "COAL", "GEOTHERMAL", "BIOMASS", "WIND", "NUCLEAR", "HYDROELECTRIC", 
            "OIL", "OTHER"
        ]

# --- Main Execution Block ---
if __name__ == "__main__":
    print("===== Starting Dynamic Capability Data Fetching Process =====")
    if not regions:
        print("No regions found in config/regions.json. Nothing to process.")
        sys.exit(0)

    # Discover all available energy types dynamically
    print("\nDiscovering available energy types from database...")
    energy_types = discover_all_energy_types()
    print(f"Energy types to process: {energy_types}")

    success_count = 0
    total_count = len(regions) * len(energy_types)

    # Process each region and energy type
    for region in regions.keys():
        print(f"\nProcessing region: {region}")
        for energy_type in energy_types:
            try:
                fetcher = CapabilityDataFetcher(region, energy_type)
                fetcher.run()
                success_count += 1
            except Exception as e:
                print(f"[ERROR] Failed to process {region} - {energy_type}: {e}")
                print(traceback.format_exc())

    print(f"\n===== Capability Data Fetching Process Finished =====")
    print(f"Success rate: {success_count}/{total_count} combinations processed successfully")