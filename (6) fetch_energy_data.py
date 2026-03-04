import json
import pandas as pd
from sqlalchemy import create_engine, exc as sqlalchemy_exc, text, inspect
import os
import sys
import traceback
from datetime import datetime, timedelta

# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

# --- Configuration Loading ---
try:
    from config.settings import BASE_OUTPUT_DIR
    print("Successfully imported BASE_OUTPUT_DIR from config.settings.")
except ImportError:
    print("Warning: Could not import BASE_OUTPUT_DIR from config.settings. Using current directory as base.")
    BASE_OUTPUT_DIR = "."
except Exception as e:
    print(f"Warning: Error importing config ({e}). Using current directory as base.")
    BASE_OUTPUT_DIR = "."

try:
    regions_config_path = os.path.join(project_root, 'config', 'regions.json')
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

# --- EnergyDataFetcher Class Definition ---
class EnergyDataFetcher:
    def __init__(self, region, output_dir=BASE_OUTPUT_DIR, start_date='2019-01-01', db_connection_string=None):
        """
        Initialize the EnergyDataFetcher. Saves output as Parquet.

        :param region: Region name for the energy data (e.g., 'CALIFORNIA').
        :param output_dir: Base directory for output files (default: BASE_OUTPUT_DIR from settings).
        :param start_date: Start date for filtering data (default: '2019-01-01').
        :param db_connection_string: SQL Server connection string (default provided if None).
        """
        self.region = region.upper()
        self.base_output_dir = output_dir
        self.start_date = start_date
        self.db_connection_string = db_connection_string 
        self.region_output_dir = os.path.join(self.base_output_dir, self.region.lower(), "EIA")
        self.output_file = os.path.join(self.region_output_dir, f"{self.region.lower()}_EIA.parquet")
        self.table_name = f"PRODUCTION_{self.region}_ENERGY_EIA_HOURLY"

    def get_available_columns(self, engine):
        """
        Dynamically discover all available energy columns for the region.
        Returns a list of column names excluding DATETIME.
        """
        try:
            inspector = inspect(engine)
            
            # Check if table exists
            if not inspector.has_table(self.table_name):
                print(f"[ERROR] Table {self.table_name} does not exist.")
                return []
            
            # Get all columns from the table
            columns = inspector.get_columns(self.table_name)
            column_names = [col['name'] for col in columns]
            
            print(f"  All columns found in {self.table_name}: {column_names}")
            
            # Filter out DATETIME column and get energy columns
            energy_columns = [col for col in column_names if col.upper() != 'DATETIME']
            
            print(f"  Energy columns to fetch: {energy_columns}")
            return energy_columns
            
        except Exception as e:
            print(f"[ERROR] Failed to get column information for {self.table_name}: {e}")
            print(traceback.format_exc())
            return []

    def validate_table_exists(self, engine):
        """Check if the table exists and has data."""
        try:
            # Check if table exists
            inspector = inspect(engine)
            if not inspector.has_table(self.table_name):
                print(f"[ERROR] Table {self.table_name} does not exist.")
                return False
            
            # Check if table has data
            with engine.connect() as conn:
                count_query = text(f"SELECT COUNT(*) as row_count FROM [{self.table_name}]")
                result = conn.execute(count_query)
                row_count = result.fetchone()[0]
                
                if row_count == 0:
                    print(f"[WARNING] Table {self.table_name} exists but contains no data.")
                    return False
                
                print(f"  Table {self.table_name} found with {row_count} rows.")
                return True
                
        except Exception as e:
            print(f"[ERROR] Failed to validate table {self.table_name}: {e}")
            print(traceback.format_exc())
            return False

    def fetch_energy_data(self):
        """Fetch energy data from SQL database, dynamically getting all available energy columns."""
        print(f"Connecting to database...")
        try:
            engine = create_engine(self.db_connection_string)
            
            # Validate table exists
            if not self.validate_table_exists(engine):
                return pd.DataFrame()
            
            # Get all available energy columns dynamically
            energy_columns = self.get_available_columns(engine)
            
            if not energy_columns:
                print(f"[ERROR] No energy columns found for {self.region}")
                return pd.DataFrame()
            
            # Build the SELECT query with all available columns
            all_columns = ['DATETIME'] + energy_columns
            columns_str = ', '.join([f'[{col}]' for col in all_columns])
            
            query = f"SELECT {columns_str} FROM [dbo].[{self.table_name}] ORDER BY [DATETIME]"
            
            print(f"Executing query: {query}")
            
            df_energy = pd.read_sql(query, engine)
            print(f"  Fetched {len(df_energy)} records for {self.region}.")
            print("  Columns fetched:", df_energy.columns.tolist())
            print("  Sample data:\n", df_energy.head())
            print("  Null counts:\n", df_energy.isnull().sum())
            
            return df_energy
            
        except sqlalchemy_exc.SQLAlchemyError as e:
            print(f"[ERROR] Database connection or query failed for {self.region}: {e}")
            print(f"Query attempted: {query if 'query' in locals() else 'Query not constructed'}")
            print(traceback.format_exc())
            return pd.DataFrame()
        except ImportError:
            print("[ERROR] Required database driver (e.g., pymssql) not installed.")
            return pd.DataFrame()
        except Exception as e:
            print(f"[ERROR] Unexpected error fetching energy data for {self.region}: {e}")
            print(traceback.format_exc())
            return pd.DataFrame()

    def process_data(self, df_energy):
        """Rename columns, convert to datetime, and filter by start date."""
        if df_energy.empty:
            print("Skipping processing for empty DataFrame.")
            return df_energy
        
        print("Processing fetched data...")
        try:
            # Rename DATETIME to date
            if "DATETIME" in df_energy.columns:
                df_energy.rename(columns={"DATETIME": "date"}, inplace=True)
            elif "date" not in df_energy.columns:
                print("Error: Neither 'DATETIME' nor 'date' column found. Cannot process.")
                return pd.DataFrame()
            
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df_energy['date']):
                print("  Converting 'date' column to datetime...")
                df_energy["date"] = pd.to_datetime(df_energy["date"], errors='coerce')
                df_energy.dropna(subset=['date'], inplace=True)
            
            # Check for null columns
            null_columns = df_energy.drop(columns=['date']).isnull().all()
            if null_columns.any():
                null_col_names = null_columns[null_columns].index.tolist()
                print(f"  Warning: The following columns contain only null values: {null_col_names}")
                # Optionally drop null columns
                # df_energy = df_energy.drop(columns=null_col_names)
            
            # Filter by start date
            print(f"  Filtering data from {self.start_date}...")
            start_dt = pd.to_datetime(self.start_date)
            filtered_df = df_energy[df_energy['date'] >= start_dt].copy()
            
            print(f"  Filtered {self.region} energy data to {len(filtered_df)} rows.")
            print("  Processed columns:", filtered_df.columns.tolist())
            print("  Date range:", f"{filtered_df['date'].min()} to {filtered_df['date'].max()}")
            print("  Processed sample data:\n", filtered_df.head())
            
            return filtered_df
            
        except Exception as e:
            print(f"Error during data processing for {self.region}: {e}")
            print(traceback.format_exc())
            return pd.DataFrame()

    def save_data(self, df_energy):
        """Save the processed DataFrame to Parquet and CSV files."""
        if df_energy.empty:
            print("No data to save.")
            return
        
        print(f"Saving {self.region} energy data to Parquet: '{self.output_file}'")
        try:
            # Debug: Print DataFrame details before saving
            print("  DataFrame before saving:")
            print("  Columns:", df_energy.columns.tolist())
            print("  Rows:", len(df_energy))
            print("  Sample:\n", df_energy.head())

            # Create output directory
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

            # Round float columns
            float_cols = df_energy.select_dtypes(include=['float64', 'float32']).columns
            if not float_cols.empty:
                print(f"  Rounding float columns: {list(float_cols)}")
                df_energy[float_cols] = df_energy[float_cols].round(2)

            # Save to Parquet
            df_energy.to_parquet(self.output_file, index=False, engine='pyarrow')
            print(f"  Energy data successfully saved to Parquet.")

            # Save to CSV for verification
            csv_file = self.output_file.replace('.parquet', '.csv')
            df_energy.to_csv(csv_file, index=False)
            print(f"  Energy data also saved to CSV: '{csv_file}'")

            # Verify Parquet file with a delay to ensure file is flushed
            import time
            time.sleep(1)  # Wait 1 second to ensure file is fully written
            saved_df = pd.read_parquet(self.output_file, engine='pyarrow')
            print("  Verified saved Parquet file:")
            print("  Columns:", saved_df.columns.tolist())
            print("  Rows:", len(saved_df))
            print("  Sample:\n", saved_df.head())

            # Verify CSV file
            csv_df = pd.read_csv(csv_file)
            print("  Verified saved CSV file:")
            print("  Columns:", csv_df.columns.tolist())
            print("  Rows:", len(csv_df))
            print("  Sample:\n", csv_df.head())

            # Check file size
            file_size = os.path.getsize(self.output_file) / (1024 * 1024)  # Size in MB
            print(f"  Parquet file size: {file_size:.2f} MB")
            
        except Exception as e:
            print(f"[ERROR] Failed to save or verify energy Parquet/CSV file for {self.region}: {e}")
            print(traceback.format_exc())

    def run(self):
        """Execute the full energy data fetching, processing, and saving process."""
        print(f"\n--- Running Energy Data Fetcher for: {self.region} ---")
        df_energy = None
        processed_df = None
        try:
            df_energy = self.fetch_energy_data()
            processed_df = self.process_data(df_energy)
            self.save_data(processed_df)
        except Exception as e:
            print(f"[CRITICAL ERROR] Unhandled exception in run for {self.region}: {e}")
            print(traceback.format_exc())
        finally:
            if df_energy is not None:
                del df_energy
            if processed_df is not None:
                del processed_df
            print(f"--- Finished Energy Data Fetcher for: {self.region} ---")

# --- Main Execution Block ---
if __name__ == "__main__":
    print("===== Starting Energy Data Fetching Process =====")
    if not regions:
        print("No regions found in config/regions.json. Nothing to process.")
        sys.exit(0)
    
    for region in regions.keys():
        print(f"\nFetching energy data for region: {region}")
        try:
            fetcher = EnergyDataFetcher(region)
            fetcher.run()
        except Exception as e:
            print(f"[ERROR] Failed to process region {region}: {e}")
            print(traceback.format_exc())
    
    print("\n===== Energy Data Fetching Process Finished =====")