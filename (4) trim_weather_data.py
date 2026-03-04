# Combined Weather Data Trimmer Script

import json
import os
import sys
import pandas as pd
import gc
import traceback
from typing import List

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
sys.path.append(project_root)

print(f"Script directory: {script_dir}")
print(f"Calculated project root (parent of script dir): {project_root}")
print(f"sys.path updated to include: {project_root}")


# --- Configuration Loading ---
# Initialize variables to None or empty in case of import/load errors
BASE_OUTPUT_DIR = None
regions = {}

try:
    # Now attempt to import config.settings - it should be found because project_root is in sys.path
    from config.settings import BASE_OUTPUT_DIR
    print("Successfully imported BASE_OUTPUT_DIR from config.settings.")

    # Construct the path to regions.json relative to the project root
    config_dir = os.path.join(project_root, 'config')
    regions_config_path = os.path.join(config_dir, 'regions.json')

    if not os.path.exists(regions_config_path):
         raise FileNotFoundError(f"Regions config not found at: {regions_config_path}")

    with open(regions_config_path, 'r') as f:
        regions = json.load(f)
    print(f"Successfully loaded region configuration from {regions_config_path}")

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

# --- Date Range (Hardcoded as in original script) ---
start_date_str = '2019-01-01 00:00:00'
end_date_str = '2027-12-31 23:00:00'
print(f"Using trimming date range: {start_date_str} to {end_date_str}")


# --- WeatherDataTrimmer Class Definition ---
class WeatherDataTrimmer:
    def __init__(self, region: str, start_date_str: str, end_date_str: str, base_dir: str):
        """
        Initialize the WeatherDataTrimmer for Parquet files.

        Args:
            region: Region name (e.g., 'ss', 'carolina') to construct file paths.
            start_date_str: Start of the date range (inclusive, 'YYYY-MM-DD HH:MM:SS').
            end_date_str: End of the date range (inclusive, 'YYYY-MM-DD HH:MM:SS').
            base_dir: Base directory for data output (e.g., 'data').
        """
        if not isinstance(base_dir, str) or not base_dir:
            raise ValueError("base_dir must be a non-empty string.")
        if not isinstance(start_date_str, str) or not start_date_str:
            raise ValueError("start_date_str must be a non-empty string.")
        if not isinstance(end_date_str, str) or not end_date_str:
            raise ValueError("end_date_str must be a non-empty string.")

        self.region = region.lower()
        self.base_dir = base_dir
        self.start_date_str = start_date_str
        self.end_date_str = end_date_str
        self.region_dir = os.path.join(self.base_dir, self.region)

        # --- File paths for processing and deletion ---
        # This is the file produced by the Analogous Forecast step, which should contain
        # combined historical, current forecast, and analogous forecast data.
        self.file_to_filter = os.path.join(self.region_dir, f"{self.region}_meteo_final_forecast.parquet")

        # These are the intermediate files to clean up after the final file is trimmed.
        # Include the original fetcher outputs, the merger outputs.
        self.files_to_delete = [
            os.path.join(self.region_dir, f"{self.region}_meteo2_weather.parquet"),    # From fetcher (historical)
            # os.path.join(self.region_dir, f"{self.region}_meteo_forecast.parquet"),  # From fetcher (forecast)
            # os.path.join(self.region_dir, f"{self.region}_meteo_combined.parquet"), # From merger (hist + initial forecast)
            # os.path.join(self.region_dir, f"{self.region}_meteo_weather.sqlite"),    # From fetcher (if used)
            # Add other potential intermediate files if necessary, e.g., logs
            # os.path.join(self.region_dir, f"{self.region}_failed_historical_log.csv"),
            # os.path.join(self.region_dir, f"{self.region}_failed_forecast_log.csv")
        ]

        # Convert date strings to datetime objects *once* during initialization
        # Use errors='raise' here to catch bad date formats early
        try:
            self.start_dt = pd.to_datetime(self.start_date_str, errors='raise')
            self.end_dt = pd.to_datetime(self.end_date_str, errors='raise')
        except ValueError as e:
             raise ValueError(f"Invalid date format in start_date_str ('{start_date_str}') or end_date_str ('{end_date_str}'): {e}")

        # Ensure region directory exists just in case (redundant if BASE_OUTPUT_DIR is handled, but harmless)
        os.makedirs(self.region_dir, exist_ok=True)


    def trim_file(self):
        """Trim the specified Parquet file based on the date range and overwrite it."""
        print(f"--- Trimming Parquet file: {self.file_to_filter} ---")

        if not os.path.exists(self.file_to_filter):
            print(f"Error: File to filter not found at '{self.file_to_filter}'. Skipping trimming step.")
            return False # Indicate failure/skip

        try:
            print(f"Reading Parquet file: {self.file_to_filter}...")
            # Use use_nullable_dtypes=True for compatibility with pyarrow
            df = pd.read_parquet(self.file_to_filter, engine='pyarrow', use_nullable_dtypes=True)
            rows_before = len(df)
            print(f"  Read {rows_before} rows.")

            # Ensure 'date' column exists and is datetime
            if 'date' not in df.columns:
                print(f"Error: 'date' column not found in {self.file_to_filter}. Cannot trim.")
                return False # Indicate failure
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                 print(f"  Warning: Attempting to convert 'date' column to datetime.")
                 # Use errors='coerce' during conversion from other types in the DataFrame
                 df['date'] = pd.to_datetime(df['date'], errors='coerce')
                 initial_rows = len(df)
                 df.dropna(subset=['date'], inplace=True) # Drop rows where conversion failed
                 dropped_rows = initial_rows - len(df)
                 if dropped_rows > 0:
                     print(f"  Dropped {dropped_rows} rows with invalid date values during conversion.")

            if df.empty:
                 print("  DataFrame is empty after date conversion/dropna. No data to trim.")
                 # Decide behavior for empty output file: save empty or delete original?
                 # Saving empty is safer, it indicates the filtering result.
                 filtered_df = pd.DataFrame(columns=df.columns) # Create empty df with original columns
                 print("  Filtered down to 0 rows.")
                 # Continue to save the empty file
            else:
                # Check and handle timezone (Parquet can store timezone info)
                if df['date'].dt.tz is not None:
                    print("  Converting date column to timezone-naive.")
                    df['date'] = df['date'].dt.tz_localize(None)

                print(f"Filtering dates between {self.start_date_str} and {self.end_date_str} (inclusive)...")
                filtered_df = df[(df['date'] >= self.start_dt) & (df['date'] <= self.end_dt)].copy() # Use pre-converted datetime objects
                rows_after = len(filtered_df)
                print(f"  Filtered down to {rows_after} rows ({rows_before - rows_after} rows removed).")


            if not filtered_df.empty:
                print("  Rounding all numerical columns to 2 decimal places...")
                # Select numeric columns (int, float). Be careful if is_analogous_forecast is int.
                # Select specific numeric types or exclude known non-numeric
                numeric_cols = filtered_df.select_dtypes(include='number').columns
                 # Exclude boolean/flag columns that might be stored as numbers
                numeric_cols = numeric_cols.drop('is_analogous_forecast', errors='ignore') # Use errors='ignore' in case the column doesn't exist

                if not numeric_cols.empty:
                    # Use .loc to avoid SettingWithCopyWarning
                    filtered_df.loc[:, numeric_cols] = filtered_df[numeric_cols].round(2)
                else:
                    print("   No numeric columns found to round.")


            print(f"Saving trimmed and formatted data back to Parquet file {self.file_to_filter}...")
            # os.makedirs handled in __init__ now, but harmless here too
            # os.makedirs(os.path.dirname(self.file_to_filter), exist_ok=True)
            filtered_df.to_parquet(self.file_to_filter, index=False, engine='pyarrow')
            print("  Successfully saved trimmed and formatted Parquet file.")
            save_successful = True

            del df, filtered_df # Explicitly clear memory
            gc.collect()
            return save_successful # Indicate success

        except Exception as e:
            print(f"An error occurred during file trimming: {e}")
            print(traceback.format_exc()) # Print full traceback
            # Explicitly clear memory in case of error
            if 'df' in locals(): del df
            if 'filtered_df' in locals(): del filtered_df
            gc.collect()
            return False # Indicate failure


    def delete_files(self):
        """Delete the specified intermediate files (including Parquet and SQLite)."""
        print("\n--- Attempting to delete intermediate files ---")
        for file_path in self.files_to_delete:
            print(f"  Deleting file: {file_path}")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"    Successfully deleted.")
                except OSError as e:
                    print(f"    Error deleting: {e}")
                except Exception as e:
                    print(f"    An unexpected error occurred while deleting: {e}")
                    print(traceback.format_exc())
            else:
                print(f"    File not found, cannot delete.")
        print("--- Intermediate file deletion process complete ---")


    def run(self):
        """Execute the full trimming and deletion process."""
        print(f"Starting data trimming and cleanup for region: {self.region}")
        trim_successful = False
        try:
            trim_successful = self.trim_file() # Run trimming
            # Decide whether to delete based on trim success.
            # It's safer to delete intermediate files only if the final trimmed file was successfully saved.
            if trim_successful:
                 self.delete_files() # Run deletion only if trimming/saving worked
            else:
                 print("\nSkipping deletion of intermediate files because trimming and saving failed.")

        except Exception as e:
            # This catches errors during the run method itself, not just trim_file/delete_files
            print(f"\n[CRITICAL ERROR] Unhandled exception during run for region {self.region}: {e}")
            print(traceback.format_exc())
        finally:
            gc.collect() # Final garbage collection
            print(f"\nTrimming and cleanup process finished for region: {self.region}.")


# --- Main Execution Block ---
if __name__ == "__main__":
    # This block runs only when the script is executed directly

    print("===== Starting Weather Data Trimming and Cleanup Process =====")

    # Configuration and Date Range loading are handled at the top of the script
    # outside this block, so BASE_OUTPUT_DIR, regions, start_date_str,
    # and end_date_str are already available here.

    if not regions:
        print("No regions loaded from config/regions.json. Nothing to process.")
        sys.exit(0)

    # BASE_OUTPUT_DIR validity is checked after loading config at the top

    regions_with_errors = []
    # Also check if regions is actually a dict with keys as expected
    if not isinstance(regions, dict) or not all(isinstance(k, str) for k in regions.keys()):
        print(f"Error: Expected regions config to be a dictionary with string keys, but got {type(regions)}. Exiting.")
        sys.exit(1)

    print(f"Found {len(regions)} regions in config: {list(regions.keys())}")
    print(f"Trimming data to range: {start_date_str} to {end_date_str}")


    for region_name in regions.keys():
        print(f"\n{'='*20} Processing Region: {region_name.upper()} {'='*20}")
        try:
            # Pass loaded variables to the constructor
            trimmer = WeatherDataTrimmer(
                region=region_name,
                start_date_str=start_date_str,
                end_date_str=end_date_str,
                base_dir=BASE_OUTPUT_DIR
            )
            trimmer.run()
        except ValueError as e:
             # Handle specific initialization errors from __init__ (e.g., bad date format)
             print(f"\n---!!! Configuration/Initialization Error processing {region_name.upper()}: {e} !!!---")
             traceback.print_exc()
             regions_with_errors.append(f"{region_name} (Config/Init Error)")
        except Exception as e:
             print(f"\n---!!! CRITICAL UNHANDLED ERROR processing {region_name.upper()}: {e} !!!---")
             traceback.print_exc()
             regions_with_errors.append(f"{region_name} (Critical Error)")

    print("\n" + "="*60)
    print("--- Weather Data Trimming and Cleanup Complete ---")
    if regions_with_errors:
        print(f"Regions encountering errors ({len(regions_with_errors)}):")
        for region_err in regions_with_errors: print(f"  - {region_err}")
    else:
        print("All configured regions processed successfully.")
    print("="*60)