# Combined Load and Weather Data Merger Script

import json
import os
import sys
import pandas as pd
from sqlalchemy import create_engine, exc as sqlalchemy_exc
import gc
import traceback
from typing import Optional, Dict # Import Dict

# Ensure pyarrow and pymssql are installed:
# pip install pyarrow pymssql SQLAlchemy

# --- Path Setup ---
# Get the absolute path of the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Assuming the script is located at project_root/scripts/your_script.py
# and config/src are located at project_root/config/ or project_root/src/
# The project root is the parent directory of the script directory.
# os.path.join(script_dir, '..') goes up one level from 'scripts/' to the project root.
project_root = os.path.abspath(os.path.join(script_dir, '..'))

# Add the project root directory to sys.path
# This makes 'config', 'src', etc. importable as packages
# Check if it's already in sys.path to avoid duplicates if this script is imported elsewhere
if project_root not in sys.path:
    sys.path.append(project_root)

print(f"Script directory: {script_dir}")
print(f"Calculated project root (parent of script dir): {project_root}")
print(f"sys.path updated to include: {project_root}")


# --- Configuration Loading ---
# Initialize variables to None or empty in case of import/load errors
BASE_OUTPUT_DIR = None
regions = {}
load_column_mapping: Dict[str, str] = {} # Add type hint for clarity

try:
    # Now attempt to import config.settings - it should be found because project_root is in sys.path
    from config.settings import BASE_OUTPUT_DIR
    print("Successfully imported BASE_OUTPUT_DIR from config.settings.")

    # Construct the paths to config files relative to the project root
    config_dir = os.path.join(project_root, 'config')
    regions_config_path = os.path.join(config_dir, 'regions.json')

    if not os.path.exists(regions_config_path):
         raise FileNotFoundError(f"Regions config not found at: {regions_config_path}")

    with open(regions_config_path, 'r') as f:
        regions = json.load(f)
    print(f"Successfully loaded region configuration from {regions_config_path}")

    # --- Load Column Mapping (defined directly in the original runner) ---
    load_column_mapping = {
        "california": "CALIFORNIA_LOAD",
        "carolina": "CAROLINA_LOAD",
        "central": "CENTRAL_LOAD",
        "florida": "FLORIDA_LOAD",
        "midwest": "MIDWEST_LOAD",
        "midatlantic": "MIDATLANTIC_LOAD",
        "northwest": "NORTHWEST_LOAD",
        "southeast": "SOUTHEAST_LOAD",
        "southwest": "SOUTHWEST_LOAD",
        "tennessee": "TENNESSEE_LOAD",
        "texas": "TEXAS_LOAD",
        "newyork": "NEWYORK_LOAD",
        "newengland": "NEWENGLAND_LOAD",
    }
    print(f"Defined load column mapping for {len(load_column_mapping)} regions.")


except ImportError as e:
    print(f"Error: Could not import BASE_OUTPUT_DIR from config.settings.")
    print(f"Details: {e}")
    print(f"Please ensure config/settings.py exists at '{os.path.join(project_root, 'config', 'settings.py')}'")
    print("and that the directory '{os.path.join(project_root, 'config')}' contains an '__init__.py' file.")
    # Exit if essential config cannot be loaded
    sys.exit(1)
except FileNotFoundError as e:
    print(f"Error loading configuration file: {e}")
    print(f"Please ensure regions.json exists in '{os.path.join(project_root, 'config')}/'.")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"Error decoding JSON configuration file: {e}")
    print("Please check the format of regions.json.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during configuration loading: {e}")
    print(traceback.format_exc())
    sys.exit(1)

# Ensure BASE_OUTPUT_DIR is valid after loading
if not BASE_OUTPUT_DIR or BASE_OUTPUT_DIR == ".":
     print("Error: BASE_OUTPUT_DIR is not properly configured (is None, empty, or '.'). Cannot proceed.")
     sys.exit(1)
print(f"Using Base Output Directory for data: {BASE_OUTPUT_DIR}")

# --- Constants ---
DATE_COL = 'date' # Consistent column name for datetime


# --- LoadWeatherMerger Class Definition ---
class LoadWeatherMerger:
    def __init__(self,
                 region: str,
                 load_column: str,
                 start_date: str = '2019-01-01 00:00:00',
                 base_dir: str = BASE_OUTPUT_DIR, # Default uses the loaded global
                 db_connection_string: Optional[str] = None):
        """
        Initialize the LoadWeatherMerger. Merges weather Parquet data with SQL load data.

        Args:
            region: Region name (e.g., 'ss', 'carolina').
            load_column: Name of the load column in the SQL database.
            start_date: Start date string for SQL query.
            base_dir: Base directory for file paths.
            db_connection_string: SQL Server connection string. Uses default if None.
        """
        if not isinstance(region, str) or not region: raise ValueError("region must be a non-empty string.")
        if not isinstance(load_column, str) or not load_column: raise ValueError("load_column must be a non-empty string.")
        if not isinstance(start_date, str) or not start_date: raise ValueError("start_date must be a non-empty string.")
        if not isinstance(base_dir, str) or not base_dir: raise ValueError("base_dir must be a non-empty string.")

        self.region = region.lower()
        self.load_column = load_column
        self.start_date = start_date
        self.base_dir = base_dir

        # --- Database Connection String ---
        # Keeping the hardcoded connection string here as it was in the original src file.
        # Consider moving this to config.settings for better practice.
        self.db_connection_string = db_connection_string or (
        )
        if not self.db_connection_string:
             raise ValueError("Database connection string is not provided and default is empty.")

        # Define file paths
        self.region_dir = os.path.join(self.base_dir, self.region)

        # Input weather file path: This is the output of the trimming step
        self.weather_filepath = os.path.join(self.region_dir, f"{self.region}_meteo_final_forecast.parquet")

        # Output path
        self.output_dir = os.path.join(self.region_dir, "final")
        self.output_filepath = os.path.join(self.output_dir, f"{self.region}_meteo_historical_with_load.parquet")

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)


    def fetch_load_data(self) -> Optional[pd.DataFrame]:
        """
        Fetches load data from SQL database, sets 'date' index.
        Returns an indexed DataFrame with 'date' as index and load column, or None on failure or empty data.
        """
        print(f"Connecting to database...")
        # Use %s placeholder for pymssql compatibility with parameters
        # Select only 'datetime' and the specific load column to minimize data transfer
        query = f"""
        SELECT
            datetime,
            "{self.load_column}"
        FROM
            [dbo].[PRODUCTION_REGION_LOAD_HOURLY]
        WHERE
            datetime >= %s
        ORDER BY
            datetime
        """
        print(f"Fetching '{self.load_column}' data from SQL (from {self.start_date})...")
        engine = None # Initialize engine variable
        df_load = None # Initialize DataFrame variable
        try:
            # Check if pymssql driver is importable before creating engine
            try:
                import pymssql
            except ImportError:
                 raise ImportError("Required database driver 'pymssql' not installed.")

            engine = create_engine(self.db_connection_string)
            # Use context manager for connection and read_sql to ensure resources are closed
            with engine.connect() as connection:
                 df_load = pd.read_sql(query, connection, params=(self.start_date,))

            if df_load.empty:
                print(f"Warning: No load data found for '{self.load_column}' from {self.start_date}.")
                return None # Return None if no load data

            print(f"  Fetched {len(df_load)} load records.")

            # --- Data Cleaning and Preparation ---
            print("  Preparing load data...")
            # Rename datetime column to the consistent DATE_COL
            df_load.rename(columns={"datetime": DATE_COL}, inplace=True)

            # Convert date column to datetime, coercing errors to NaT
            initial_rows = len(df_load)
            df_load[DATE_COL] = pd.to_datetime(df_load[DATE_COL], errors='coerce')
            df_load.dropna(subset=[DATE_COL], inplace=True) # Drop rows where conversion failed
            dropped_rows = initial_rows - len(df_load)
            if dropped_rows > 0:
                 print(f"  Dropped {dropped_rows} load rows with invalid date values.")

            if df_load.empty:
                 print("Warning: Load data is empty after date cleaning.")
                 return None

            # Handle duplicate timestamps
            if df_load[DATE_COL].duplicated().any():
                print(f"  Warning: Duplicate timestamps found in load data. Keeping last entry for each date.")
                df_load = df_load.drop_duplicates(subset=[DATE_COL], keep='last')

            # Handle timezone if present (make naive for consistency with weather data loading)
            if df_load[DATE_COL].dt.tz is not None:
                 print("  Converting load date column to timezone-naive.")
                 df_load[DATE_COL] = df_load[DATE_COL].dt.tz_localize(None)

            # Set the date column as index
            df_load.set_index(DATE_COL, inplace=True)
            print("  Load data prepared and indexed.")
            return df_load

        except ImportError as e:
             print(f"[ERROR] Database driver error: {e}")
             print("Please ensure 'pymssql' is installed (`pip install pymssql`).")
             return None
        except sqlalchemy_exc.SQLAlchemyError as e:
             print(f"[ERROR] Database connection or query failed: {e}")
             # print(f"Query attempted (debug): {query % repr(self.start_date)}")
             print(traceback.format_exc())
             return None
        except Exception as e:
            print(f"[ERROR] Unexpected error fetching or preparing load data: {e}")
            print(traceback.format_exc())
            return None
        finally:
            # Dispose of the engine connection pool
            if engine:
                 engine.dispose()


    def process_and_merge(self, load_df_indexed: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """
        Reads the entire weather Parquet file and merges with the indexed load data.

        Args:
            load_df_indexed: DataFrame with load data, indexed by 'date', or None.

        Returns:
            A DataFrame containing the merged data, indexed by 'date', or None if merging fails or no data exists.
        """
        print(f"Processing weather file: {self.weather_filepath}")
        weather_df: Optional[pd.DataFrame] = None # Initialize weather DataFrame variable
        weather_exists = False # Initialize the flag here to ensure it's always defined

        # --- Load Weather Data ---
        if os.path.exists(self.weather_filepath):
            try:
                print(f"Reading weather Parquet file: {self.weather_filepath}...")
                # Use use_nullable_dtypes=True for compatibility with pyarrow
                weather_df = pd.read_parquet(self.weather_filepath, engine='pyarrow', use_nullable_dtypes=True)
                # FutureWarning fix: Replace use_nullable_dtypes=True with dtype_backend='numpy_nullable'
                # weather_df = pd.read_parquet(self.weather_filepath, engine='pyarrow', dtype_backend='numpy_nullable') # Use this in newer pandas
                print(f"  Read {len(weather_df)} weather records.")

                if weather_df.empty:
                    print("Warning: Weather Parquet file is empty.")
                    # weather_exists remains False
                else:
                    # --- Data Cleaning and Preparation ---
                    print("  Preparing weather data...")
                    if DATE_COL not in weather_df.columns:
                         print(f"Error: '{DATE_COL}' column not found in weather file. Cannot use weather data.")
                         # weather_exists remains False
                         weather_df = pd.DataFrame() # Ensure it's an empty DF or None

                    elif not pd.api.types.is_datetime64_any_dtype(weather_df[DATE_COL]):
                         print(f"  Warning: Converting '{DATE_COL}' in weather data to datetime.")
                         initial_rows = len(weather_df)
                         weather_df[DATE_COL] = pd.to_datetime(weather_df[DATE_COL], errors='coerce')
                         weather_df.dropna(subset=[DATE_COL], inplace=True)
                         dropped_rows = initial_rows - len(weather_df)
                         if dropped_rows > 0:
                             print(f"  Dropped {dropped_rows} weather rows with invalid date values.")

                         if weather_df.empty:
                              print("Warning: Weather data is empty after date cleaning.")
                              # weather_exists remains False

                    # If preparation was successful and df is still not empty:
                    if weather_df is not None and not weather_df.empty: # Add check for None
                         # Handle duplicate timestamps
                         if weather_df[DATE_COL].duplicated().any():
                              print("  Warning: Duplicate timestamps found in weather data. Keeping last entry for each date.")
                              weather_df = weather_df.drop_duplicates(subset=[DATE_COL], keep='last')

                         # Ensure timezone consistency (make naive)
                         if weather_df[DATE_COL].dt.tz is not None:
                              print(f"  Converting weather data timezone to timezone-naive.")
                              weather_df[DATE_COL] = weather_df[DATE_COL].dt.tz_localize(None)

                         # Set the date column as index
                         weather_df.set_index(DATE_COL, inplace=True)
                         print("  Weather data prepared and indexed.")
                         weather_exists = True # <--- Set to True here

            except Exception as e:
                print(f"[ERROR] Failed reading or preparing weather Parquet file: {e}")
                print(traceback.format_exc())
                weather_exists = False # Indicate failure
                weather_df = None # Ensure DF is None after error

        else: # File does not exist
            print(f"Warning: Weather Parquet file not found at '{self.weather_filepath}'.")
            weather_exists = False # Explicitly set to False


        load_exists = load_df_indexed is not None and not load_df_indexed.empty

        # --- Perform the merge ---
        final_df = None
        try:
            print("Joining weather data with load data...")

            if weather_exists and load_exists:
                # Perform an outer join on the date index
                final_df = weather_df.join(load_df_indexed, how='outer')
                print(f"  Performed outer join between weather ({len(weather_df)} indexed) and load ({len(load_df_indexed)} indexed).")

            elif weather_exists:
                # If only weather data exists, use it as the base
                final_df = weather_df.copy()
                # Add the load column with missing values so the schema is consistent
                if self.load_column not in final_df.columns:
                    print(f"  Only weather data available. Adding empty '{self.load_column}' column.")
                    final_df[self.load_column] = pd.NA # Use nullable type

            elif load_exists:
                # If only load data exists, use it as the base
                final_df = load_df_indexed.copy()
                print("  Only load data available. Weather columns will be missing.")
                 # Weather columns from the expected schema won't be present here unless added explicitly

            else:
                print("Error: No valid weather or load data available for merging.")
                return None # Indicate nothing could be merged

            # Explicit cleanup of intermediate DataFrames
            if 'weather_df' in locals() and weather_df is not None: del weather_df
            if load_df_indexed is not None: del load_df_indexed
            gc.collect()

            if final_df is None or final_df.empty:
                 print("Warning: Merging resulted in an empty DataFrame.")
                 return None # Indicate empty result

            # Sort final result by date index
            if isinstance(final_df.index, pd.DatetimeIndex):
                print("Sorting final merged data by date index...")
                final_df.sort_index(inplace=True)
            else:
                 print("Warning: Index is not DatetimeIndex after join, cannot sort by index.")
                 # Attempt to sort by the 'date' column if it exists and is still datetime
                 if DATE_COL in final_df.columns and pd.api.types.is_datetime64_any_dtype(final_df[DATE_COL]):
                      print(f"  Attempting to sort by '{DATE_COL}' column instead.")
                      final_df.sort_values(by=DATE_COL, inplace=True)
                 else:
                      print("  Cannot sort final DataFrame.")


            # Reset index before returning
            return final_df.reset_index()

        except Exception as e:
            print(f"[ERROR] Failed merging weather and load data: {e}")
            print(traceback.format_exc())
            # Explicit cleanup in case of error during merge
            if 'weather_df' in locals() and weather_df is not None: del weather_df
            if load_df_indexed is not None: del load_df_indexed
            if 'final_df' in locals() and final_df is not None: del final_df
            gc.collect()
            return None


    def save_merged_data(self, merged_df: pd.DataFrame) -> bool:
        """Saves the merged DataFrame to the final output Parquet file. Returns True on success."""
        if merged_df is None or merged_df.empty:
             print("Warning: No data available to save.")
             return False

        print(f"Saving merged data ({len(merged_df)} rows) to Parquet: '{self.output_filepath}'")
        try:
            # Output directory created in __init__
            # os.makedirs(self.output_dir, exist_ok=True)

            print("  Rounding float columns to 2 decimal places...")
            # Select float columns specifically for rounding.
            float_cols = merged_df.select_dtypes(include=['float64', 'float32']).columns
            if not float_cols.empty:
                 # Use .loc to avoid SettingWithCopyWarning
                 merged_df.loc[:, float_cols] = merged_df[float_cols].round(2)
            else:
                print("   No float columns found to round.")

            merged_df.to_parquet(self.output_filepath, index=False, engine='pyarrow')
            # FutureWarning fix: Replace engine='pyarrow' with engine='pyarrow', index=False
            # merged_df.to_parquet(self.output_filepath, engine='pyarrow', index=False)
            print(f"  Successfully saved merged Parquet data.")
            return True # Indicate success
        except Exception as e:
            print(f"[ERROR] Failed to save merged Parquet data: {e}")
            print(traceback.format_exc())
            return False # Indicate failure
        finally:
             # Explicit cleanup of the large DataFrame
             if 'merged_df' in locals() and merged_df is not None: del merged_df
             gc.collect()


    def run(self):
        """Executes the full load-weather merging process."""
        print(f"\n--- Running Load/Weather Merger for: {self.region} (Parquet) ---")
        final_merged_df = None
        load_df_indexed = None
        save_successful = False # Track if save was successful
        try:
            # 1. Fetch Load Data
            load_df_indexed = self.fetch_load_data()
            # load_df_indexed can be None or empty DataFrame here

            # 2. Process Weather Data and Merge
            final_merged_df = self.process_and_merge(load_df_indexed)
            # final_merged_df can be None or empty DataFrame here

            # 3. Save the Merged Data
            if final_merged_df is not None and not final_merged_df.empty:
                save_successful = self.save_merged_data(final_merged_df)
                if save_successful:
                    print(f"--- Merge process complete for region: {self.region} ---")
                else:
                     print(f"--- Merge process for region {self.region} failed during save. ---")
            else:
                print(f"--- Merge process for region {self.region} resulted in no data after merging. No file saved. ---")

        except Exception as e:
            # This catches unhandled errors in the run method itself
            print(f"[CRITICAL ERROR] Unhandled exception in merger run for region {self.region}: {e}")
            print(traceback.format_exc())
        finally:
             # Explicit memory cleanup regardless of success or failure
             # Check if variables are defined and not None before deleting
             if 'load_df_indexed' in locals() and load_df_indexed is not None: del load_df_indexed
             if 'final_merged_df' in locals() and final_merged_df is not None: del final_merged_df
             gc.collect() # Run garbage collection
             print(f"\nLoad/Weather merge process finished for region: {self.region}.")


# --- Main Execution Block ---
if __name__ == "__main__":
    # This block runs only when the script is executed directly

    print("===== Starting Load and Weather Data Merge Process =====")

    # Configuration loading is handled at the top of the script outside this block,
    # so BASE_OUTPUT_DIR, regions, and load_column_mapping are already available here.

    if not regions:
        print("No regions loaded from config/regions.json. Nothing to process.")
        sys.exit(0)

    if not load_column_mapping:
         print("Load column mapping is empty. Nothing to process.")
         sys.exit(0)

    # BASE_OUTPUT_DIR validity is checked after loading config at the top

    regions_with_errors = []
    # Also check if regions is actually a dict with keys as expected
    if not isinstance(regions, dict) or not all(isinstance(k, str) for k in regions.keys()):
        print(f"Error: Expected regions config to be a dictionary with string keys, but got {type(regions)}. Exiting.")
        sys.exit(1)


    print(f"Found {len(regions)} regions in config: {list(regions.keys())}")
    print(f"Using Base Output Directory for data: {BASE_OUTPUT_DIR}")

    # --- Date Range for SQL Fetch (Hardcoded as in original runner) ---
    sql_start_date = '2019-01-01 00:00:00'
    print(f"Fetching load data from SQL starting from: {sql_start_date}")


    for region_name in regions.keys():
        print(f"\n{'='*20} Processing Region: {region_name.upper()} {'='*20}")
        try:
            # Get the load column name for the region, with a fallback message
            load_column = load_column_mapping.get(region_name)
            if not load_column:
                 print(f"Warning: No load column mapping found for region '{region_name}'. Skipping this region.")
                 regions_with_errors.append(f"{region_name} (No Load Column Mapping)")
                 continue # Skip to the next region

            print(f"Using load column: {load_column}")

            # Pass loaded variables to the constructor
            merger = LoadWeatherMerger(
                region=region_name,
                load_column=load_column,
                start_date=sql_start_date,
                base_dir=BASE_OUTPUT_DIR
                # db_connection_string could be passed here if needed from settings
            )
            merger.run()

        except ValueError as e:
             # Handle specific initialization errors from __init__ (e.g., bad input args)
             print(f"\n---!!! Configuration/Initialization Error processing {region_name.upper()}: {e} !!!---")
             traceback.print_exc()
             regions_with_errors.append(f"{region_name} (Config/Init Error)")
        except Exception as e:
             # Catch any other critical unhandled errors per region
             print(f"\n---!!! CRITICAL UNHANDLED ERROR processing {region_name.upper()}: {e} !!!---")
             traceback.print_exc()
             regions_with_errors.append(f"{region_name} (Critical Error)")

    print("\n" + "="*60)
    print("--- Load and Weather Data Merge Process Complete ---")
    if regions_with_errors:
        print(f"Regions encountering errors or skipped ({len(regions_with_errors)}):")
        for region_err in regions_with_errors: print(f"  - {region_err}")
    else:
        print("All configured regions processed successfully.")
    print("="*60)