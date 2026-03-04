import json
import os
import sys
import pandas as pd
import gc
import traceback

# --- Path Setup ---
# Get the absolute path of the directory containing the script
script_dir = os.path.dirname(__file__)

# Assuming the script is located at project_root/some_directory/scripts/your_script.py
# and config is located at project_root/config/
# The project root is typically two directories up from the script location.
project_root = os.path.abspath(os.path.join(script_dir, '..'))

# Add the project root directory to sys.path
sys.path.append(project_root)

print(f"Script directory: {script_dir}")
print(f"Assuming project root: {project_root}")
print(f"sys.path updated to include: {project_root}")


# --- Configuration Loading ---
try:
    # Now attempt to import config.settings - it should be found because project_root is in sys.path
    from config.settings import BASE_OUTPUT_DIR
    print("Successfully imported BASE_OUTPUT_DIR from config.settings.")
except ImportError as e:
    print(f"Error: Could not import BASE_OUTPUT_DIR from config.settings.")
    print(f"Details: {e}")
    print(f"Please ensure config/settings.py exists at the project root '{project_root}'")
    print("and that 'config' is a valid Python package (usually needs an __init__.py file inside the config directory).")
    # Exit if essential config cannot be loaded
    sys.exit(1)
except Exception as e:
    # Catch any other unexpected errors during settings import
    print(f"An unexpected error occurred during import from config.settings: {e}")
    print(traceback.format_exc())
    sys.exit(1)


# Load region configuration
try:
    # Construct the path to regions.json relative to the project root
    regions_config_path = os.path.join(project_root, 'config', 'regions.json')

    with open(regions_config_path, 'r') as f:
        regions = json.load(f)
    print(f"Successfully loaded region configuration from {regions_config_path}")
    # print(f"Regions found: {list(regions.keys())}") # Optional: print regions

except FileNotFoundError:
    print(f"Error: Region configuration file not found at {regions_config_path}.")
    print("Please ensure config/regions.json exists at the project root.")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: Could not decode {regions_config_path}. Please check its JSON format.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred loading region configuration: {e}")
    print(traceback.format_exc())
    sys.exit(1)


# --- Constants ---
DATETIME_COL = 'datetime'

# --- WeatherDataMerger Class Definition ---
class WeatherDataMerger:
    def __init__(self, region, base_dir=BASE_OUTPUT_DIR):
        """
        Initialize the WeatherDataMerger for Parquet files.

        :param region: Region name (e.g., 'ss', 'carolina').
        :param base_dir: Base directory containing region subdirectories.
        """
        self.region = region.lower()
        self.base_dir = base_dir # This comes from the imported config
        self.region_dir = os.path.join(self.base_dir, self.region)

        # File paths for Parquet files
        self.historical_parquet = os.path.join(self.region_dir, f"{self.region}_meteo_weather.parquet")
        self.forecast_parquet = os.path.join(self.region_dir, f"{self.region}_meteo_forecast.parquet")
        self.merged_parquet = os.path.join(self.region_dir, f"{self.region}_meteo_combined.parquet")

        # Ensure region directory exists
        os.makedirs(self.region_dir, exist_ok=True)


    def check_files_exist(self):
        """Check if input Parquet files exist."""
        historical_exists = os.path.exists(self.historical_parquet)
        forecast_exists = os.path.exists(self.forecast_parquet)

        if not historical_exists:
            print(f"Warning: Historical data file not found at '{self.historical_parquet}'. Merged file will only contain forecast data (if available).")
        if not forecast_exists:
            print(f"Warning: Forecast data file not found at '{self.forecast_parquet}'.")

        return historical_exists, forecast_exists


    def get_last_historical_datetime(self):
        """
        Efficiently finds the last datetime in the historical Parquet file
        by reading only the datetime column.
        Returns None if the file is empty, column doesn't exist, or on error.
        """
        print(f"Finding last timestamp in {self.historical_parquet}...")
        try:
            # Read only the datetime column.
            # Ensure 'pyarrow' engine is available (`pip install pyarrow`).
            # Use use_nullable_dtypes=True to avoid issues with pyarrow and missing data
            datetime_series = pd.read_parquet(
                self.historical_parquet,
                columns=[DATETIME_COL],
                engine='pyarrow',
                use_nullable_dtypes=True # Recommended for pyarrow
            )[DATETIME_COL]

            if not datetime_series.empty:
                # Convert to datetime if it's not already (parquet should handle this)
                if not pd.api.types.is_datetime64_any_dtype(datetime_series):
                     datetime_series = pd.to_datetime(datetime_series)

                last_datetime = datetime_series.max()

                if pd.notna(last_datetime):
                    print(f"  Last historical timestamp: {last_datetime}")
                    return last_datetime
                else:
                    print("  Historical file contains only null/NaT datetimes.")
                    return None
            else:
                print("  Historical Parquet file is empty.")
                return None
        except FileNotFoundError:
             print("  Historical file not found (inside get_last_historical_datetime).")
             return None
        except KeyError:
            print(f"  Error: Column '{DATETIME_COL}' not found in {self.historical_parquet}. Cannot determine last timestamp.")
            return None
        except Exception as e:
            print(f"  Error reading last datetime from {self.historical_parquet}: {e}. Assuming no historical data.")
            print(traceback.format_exc())
            return None

    def merge_data(self):
        """
        Merges historical and forecast Parquet data.
        Reads historical (if exists) and forecast, filters forecast data
        occurring after the last historical timestamp, concatenates,
        and writes a new combined Parquet file.
        """
        historical_exists, forecast_exists = self.check_files_exist()

        if not historical_exists and not forecast_exists:
            print("Neither historical nor forecast Parquet file found. Nothing to merge.")
            return False # Indicate merge couldn't happen

        hist_df = None
        fcst_df = None

        # 1. Read historical data (if it exists)
        if historical_exists:
            print(f"Reading historical data from: {self.historical_parquet}")
            try:
                hist_df = pd.read_parquet(self.historical_parquet, engine='pyarrow', use_nullable_dtypes=True)
                # Ensure datetime column exists and is datetime type
                if DATETIME_COL not in hist_df.columns:
                     print(f"  Error: Historical file missing '{DATETIME_COL}' column. Treating as no historical data.")
                     hist_df = None # Reset hist_df
                     historical_exists = False # Update flag
                elif not pd.api.types.is_datetime64_any_dtype(hist_df[DATETIME_COL]):
                     print(f"  Warning: Converting '{DATETIME_COL}' in historical data to datetime.")
                     hist_df[DATETIME_COL] = pd.to_datetime(hist_df[DATETIME_COL])
                print(f"  Read {len(hist_df)} historical records.")
                if hist_df.empty:
                     print("  Historical file is empty after reading.")

            except Exception as e:
                print(f"  Error reading historical file '{self.historical_parquet}': {e}. Proceeding without historical data.")
                print(traceback.format_exc())
                hist_df = None # Ensure it's None on error
                historical_exists = False # Update flag


        # 2. Read forecast data (if it exists)
        if forecast_exists:
            print(f"Reading forecast data from: {self.forecast_parquet}")
            try:
                fcst_df = pd.read_parquet(self.forecast_parquet, engine='pyarrow', use_nullable_dtypes=True)
                 # Ensure datetime column exists and is datetime type
                if DATETIME_COL not in fcst_df.columns:
                     print(f"  Error: Forecast file missing '{DATETIME_COL}' column. Cannot use forecast data.")
                     fcst_df = None # Reset fcst_df
                elif not pd.api.types.is_datetime64_any_dtype(fcst_df[DATETIME_COL]):
                     print(f"  Warning: Converting '{DATETIME_COL}' in forecast data to datetime.")
                     fcst_df[DATETIME_COL] = pd.to_datetime(fcst_df[DATETIME_COL])
                print(f"  Read {len(fcst_df)} forecast records.")
                if fcst_df.empty:
                    print("  Forecast file is empty after reading.")
            except Exception as e:
                print(f"  Error reading forecast file '{self.forecast_parquet}': {e}. Cannot use forecast data.")
                print(traceback.format_exc())
                fcst_df = None

        # If after reading, both are None/empty, exit
        if (hist_df is None or hist_df.empty) and (fcst_df is None or fcst_df.empty):
             print("Both historical and forecast data are unavailable or empty after reading. No merge possible.")
             return False

        # 3. Find last historical timestamp (if historical data was loaded)
        last_hist_dt = None
        if hist_df is not None and not hist_df.empty:
            # Get max directly from loaded dataframe
            try:
                last_hist_dt = hist_df[DATETIME_COL].max()
                if pd.notna(last_hist_dt):
                     print(f"  Last historical timestamp from loaded data: {last_hist_dt}")
                else:
                     print("  Max datetime in loaded historical data is NaT.")
                     last_hist_dt = None # Treat as no valid last date
            except Exception as e:
                # This shouldn't happen if the column exists and is datetime, but defensive coding
                print(f"  Error getting max date from loaded historical df: {e}")
                last_hist_dt = None # Fallback

        # 4. Filter forecast data
        new_fcst_df = None
        if fcst_df is not None and not fcst_df.empty:
            if last_hist_dt is not None:
                print(f"Filtering forecast data strictly after {last_hist_dt}...")
                # Ensure datetime column is correct type before comparison
                if not pd.api.types.is_datetime64_any_dtype(fcst_df[DATETIME_COL]):
                    # This should have been done during read, but double-check
                    fcst_df[DATETIME_COL] = pd.to_datetime(fcst_df[DATETIME_COL])

                # Perform the filtering
                new_fcst_df = fcst_df[fcst_df[DATETIME_COL] > last_hist_dt].copy()
                print(f"  Found {len(new_fcst_df)} new forecast records.")
                if new_fcst_df.empty:
                     print("  No forecast data found after the last historical timestamp.")

            else:
                # If no historical data, keep all forecast data
                print("No valid historical data found, using all forecast records.")
                new_fcst_df = fcst_df.copy()
        else:
            print("No forecast data available to filter or append.")


        # 5. Combine historical and new forecast data
        print("\nCombining dataframes...")
        combined_df = pd.DataFrame() # Start with an empty df

        if hist_df is not None and not hist_df.empty:
            combined_df = pd.concat([combined_df, hist_df], ignore_index=True, sort=False)
            print(f"  Added {len(hist_df)} historical records.")
            # Free up memory
            del hist_df
            gc.collect()

        if new_fcst_df is not None and not new_fcst_df.empty:
            combined_df = pd.concat([combined_df, new_fcst_df], ignore_index=True, sort=False)
            print(f"  Added {len(new_fcst_df)} new forecast records.")
            # Free up memory
            del new_fcst_df
            # del fcst_df # Keep fcst_df for potential debugging if needed, but it's likely not needed after filtering
            gc.collect()


        if combined_df.empty:
             print("  No data available to combine after filtering.")
             return False

        # 6. Sort final dataframe
        print(f"Sorting combined data by '{DATETIME_COL}'...")
        # Ensure the column is datetime type before sorting
        if not pd.api.types.is_datetime64_any_dtype(combined_df[DATETIME_COL]):
             print(f"  Warning: Converting '{DATETIME_COL}' in combined data to datetime for sorting.")
             combined_df[DATETIME_COL] = pd.to_datetime(combined_df[DATETIME_COL])

        combined_df.sort_values(by=DATETIME_COL, inplace=True)

        # Optional: Remove duplicates just in case (e.g., if forecast included the last historical point)
        initial_rows = len(combined_df)
        combined_df.drop_duplicates(subset=[DATETIME_COL], keep='last', inplace=True)
        if len(combined_df) < initial_rows:
             print(f"  Removed {initial_rows - len(combined_df)} duplicate datetime entries.")

        print(f"  Final combined dataframe shape: {combined_df.shape}")


        # 7. Write the combined data to the final Parquet file
        try:
            print(f"\nWriting merged data to Parquet file: {self.merged_parquet}")
            # Use specified engine, default compression (usually 'snappy' for pyarrow)
            combined_df.to_parquet(self.merged_parquet, index=False, engine='pyarrow')
            print("  Successfully saved merged Parquet file.")
            return True # Indicate success

        except Exception as e:
            print(f"  [ERROR] Failed to write merged Parquet file: {e}")
            print(traceback.format_exc())
            return False # Indicate failure
        finally:
             del combined_df # Final cleanup
             gc.collect()


    def run(self):
        """Execute the full Parquet merging process."""
        print(f"Starting Parquet merge for region: {self.region}")
        success = self.merge_data() # Call the refactored method
        if success:
            print("Parquet merge process complete.")
        else:
            print("Merge process finished, but encountered errors or no data was available to merge.")


# --- Main Execution Block ---
if __name__ == "__main__":
    # This block runs only when the script is executed directly

    print("===== Starting Weather Data Merge Process =====")

    if not regions:
        print("No regions found in config/regions.json. Nothing to process.")
        sys.exit(0)

    # Process each region defined in regions.json
    for region_key, region_info in regions.items():
        # Use region_key which is the string identifier (e.g., 'ss', 'carolina')
        region_name = region_key

        print("\n" + "="*60)
        print(f"Processing region: {region_name}")
        print("="*60)

        try:
            merger = WeatherDataMerger(region=region_name, base_dir=BASE_OUTPUT_DIR)
            merger.run()
        except Exception as e:
            print(f"\n!! UNEXPECTED ERROR DURING MERGE FOR REGION: {region_name} !!")
            print(f"Error: {e}")
            print(traceback.format_exc())
        finally:
            # Add a pause or separator between regions
            print("\n" + "-"*60 + "\n")

    print("===== Weather Data Merge Process Finished =====")