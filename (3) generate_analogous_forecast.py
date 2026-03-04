# Combined Analogous Forecast Generator Script

import json
import os
import sys
import pandas as pd
import gc
import traceback
from typing import Dict, List, Optional

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
month_mapping = {}

try:
    # Now attempt to import config.settings - it should be found because project_root is in sys.path
    from config.settings import BASE_OUTPUT_DIR
    print("Successfully imported BASE_OUTPUT_DIR from config.settings.")

    # Construct the paths to config files relative to the project root
    config_dir = os.path.join(project_root, 'config')
    regions_config_path = os.path.join(config_dir, 'regions.json')
    month_map_config_path = os.path.join(config_dir, 'month_mapping.json')

    if not os.path.exists(regions_config_path):
         raise FileNotFoundError(f"Regions config not found at: {regions_config_path}")
    if not os.path.exists(month_map_config_path):
         raise FileNotFoundError(f"Month map config not found at: {month_map_config_path}")

    with open(regions_config_path, 'r') as f:
        regions = json.load(f)
    print(f"Successfully loaded region configuration from {regions_config_path}")

    with open(month_map_config_path, 'r') as f:
        month_mapping = json.load(f)
    print(f"Successfully loaded month mapping from {month_map_config_path}")

except ImportError as e:
    print(f"Error: Could not import BASE_OUTPUT_DIR from config.settings.")
    print(f"Details: {e}")
    print(f"Please ensure config/settings.py exists at '{os.path.join(project_root, 'config', 'settings.py')}'")
    print("and that the directory '{os.path.join(project_root, 'config')}' contains an '__init__.py' file.")
    # Exit if essential config cannot be loaded
    sys.exit(1)
except FileNotFoundError as e:
    print(f"Error loading configuration file: {e}")
    print(f"Please ensure the necessary config files exist in '{os.path.join(project_root, 'config')}/'.")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"Error decoding JSON configuration file: {e}")
    print("Please check the format of regions.json and month_mapping.json.")
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


# --- AnalogousForecastGenerator Class Definition ---
class AnalogousForecastGenerator:
    """
    Generates a long-term forecast Parquet file by appending analogous
    historical data to an existing combined weather data Parquet file
    based on a month mapping. Optimized using vectorized operations.
    Deletes the input combined file after success.
    """
    def __init__(self,
                 region: str,
                 month_mapping: Dict[str, str],
                 base_dir: str): # Make base_dir a required string argument
        """
        Initialize the AnalogousForecastGenerator for Parquet files.

        Args:
            region: Region name (e.g., 'ss', 'carolina').
            month_mapping: Dictionary mapping future months ('YYYY-MM') to
                           historical months ('YYYY-MM').
            base_dir: Base directory for data output (e.g., 'data').
        """
        if not isinstance(month_mapping, dict) or not month_mapping:
             raise ValueError("month_mapping must be a non-empty dictionary.")
        if not isinstance(base_dir, str) or not base_dir:
            raise ValueError("base_dir must be a non-empty string.")


        self.region = region.lower()
        self.month_mapping = month_mapping
        self.base_dir = base_dir # This comes from the imported config, passed to init
        self.region_dir = os.path.join(self.base_dir, self.region)

        # File paths for Parquet files
        self.input_file = os.path.join(self.region_dir, f"{self.region}_meteo_combined.parquet")
        self.output_file = os.path.join(self.region_dir, f"{self.region}_meteo_final_forecast.parquet")

        # Ensure region directory exists for output
        os.makedirs(self.region_dir, exist_ok=True)

    def load_combined_data(self) -> Optional[pd.DataFrame]:
        """Load combined weather data from Parquet, rename column, ensure timezone-naive."""
        print(f"Loading combined data from Parquet: {self.input_file}")
        if not os.path.exists(self.input_file):
            print(f"Error: Input Parquet file not found at '{self.input_file}'")
            return None

        try:
            combined_data = pd.read_parquet(self.input_file, engine='pyarrow', use_nullable_dtypes=True)
            # Check if 'datetime' column exists, prefer 'date'
            if "datetime" in combined_data.columns:
                if "date" in combined_data.columns:
                     print("Warning: Both 'datetime' and 'date' columns found. Keeping 'date'.")
                     combined_data.drop(columns=['datetime'], inplace=True, errors='ignore') # Drop the duplicate
                else:
                     combined_data.rename(columns={"datetime": "date"}, inplace=True)

            if "date" not in combined_data.columns:
                 print("Error: Neither 'datetime' nor 'date' column found in input Parquet file.")
                 return None

            print(f"  Successfully loaded {len(combined_data)} records.")

            # Ensure 'date' column is datetime and timezone-naive
            if not pd.api.types.is_datetime64_any_dtype(combined_data['date']):
                 print(f"  Warning: Attempting to convert 'date' column to datetime.")
                 combined_data['date'] = pd.to_datetime(combined_data['date'], errors='coerce')
                 # Drop rows where conversion failed
                 initial_rows = len(combined_data)
                 combined_data.dropna(subset=['date'], inplace=True)
                 dropped_rows = initial_rows - len(combined_data)
                 if dropped_rows > 0:
                     print(f"  Dropped {dropped_rows} rows due to invalid date conversion.")

            if combined_data.empty:
                 print("  Combined data is empty after loading/cleaning.")
                 return None

            if combined_data["date"].dt.tz is not None:
                print("  Converting date column to timezone-naive.")
                combined_data["date"] = combined_data["date"].dt.tz_localize(None)

            # Sort data by date after loading for consistency (optional but good practice)
            combined_data.sort_values(by='date', inplace=True, ignore_index=True) # ignore_index resets index after sort


            return combined_data
        except FileNotFoundError: # Should be caught by os.path.exists, but defensive
            print(f"Error: Input Parquet file not found at '{self.input_file}'")
            return None
        except Exception as e:
            print(f"Error: Failed to read or process Parquet {self.input_file}: {e}")
            print(traceback.format_exc())
            return None

    def generate_forecast(self, combined_data: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Generate long-term analogous forecast based on month mapping using
        vectorized operations, preserving hourly profile.
        """
        analogous_forecast_dfs = []
        print("\nGenerating long-term analogous forecast...")

        required_hist_months = set(self.month_mapping.values())
        print(f"  Identifying source data for historical months: {required_hist_months}")

        if combined_data.empty:
             print("  Input combined data is empty. Cannot generate forecast.")
             return []

        combined_data_copy = combined_data.copy()
        try:
            combined_data_copy['_yyyymm'] = combined_data_copy["date"].dt.to_period('M').astype(str)
            source_data_df = combined_data_copy[combined_data_copy['_yyyymm'].isin(required_hist_months)].copy()
            source_data_df.drop(columns=['_yyyymm'], inplace=True, errors='ignore')
        except Exception as e:
             print(f"Error during pre-filtering source data: {e}")
             combined_data_copy.drop(columns=['_yyyymm'], inplace=True, errors='ignore')
             del combined_data_copy 
             gc.collect()
             return []
        finally:
             combined_data_copy.drop(columns=['_yyyymm'], inplace=True, errors='ignore')
             del combined_data_copy 
             gc.collect()


        if source_data_df.empty:
            print("  Warning: No source data found matching any required historical months within the combined data.")
            if 'source_data_df' in locals(): del source_data_df # Ensure it's deleted
            gc.collect()
            return []

        print(f"  Found {len(source_data_df)} relevant source rows for analogous forecast.")
        source_data_df['_hist_yyyymm'] = source_data_df["date"].dt.to_period('M').astype(str)

        for future_month_str, hist_month_str in self.month_mapping.items():
            print(f"  Processing: Forecast {future_month_str} from source {hist_month_str}")
            hist_chunk = source_data_df[source_data_df["_hist_yyyymm"] == hist_month_str].copy()

            if hist_chunk.empty:
                print(f"    Warning: No source data found for {hist_month_str}. Skipping forecast for {future_month_str}.")
                del hist_chunk 
                gc.collect()
                continue
            try:
                future_year, future_month = map(int, future_month_str.split("-"))
            except ValueError:
                print(f"    Error: Invalid future month format '{future_month_str}'. Expected YYYY-MM. Skipping.")
                del hist_chunk 
                gc.collect()
                continue
            except Exception as e:
                print(f"    Error processing future month string '{future_month_str}': {e}. Skipping.")
                print(traceback.format_exc())
                del hist_chunk 
                gc.collect()
                continue

            print(f"    Generating new dates for {len(hist_chunk)} rows (preserving time)...")
            try:
                day = hist_chunk['date'].dt.day; hour = hist_chunk['date'].dt.hour
                minute = hist_chunk['date'].dt.minute; second = hist_chunk['date'].dt.second
                temp_date_df = pd.DataFrame({
                    'year': future_year, 'month': future_month, 'day': day,
                    'hour': hour, 'minute': minute, 'second': second
                }, index=hist_chunk.index) 
                new_dates_series = pd.to_datetime(temp_date_df, errors='coerce')
                forecast_chunk = hist_chunk.copy()
                forecast_chunk["date"] = new_dates_series
                initial_rows = len(forecast_chunk)
                forecast_chunk.dropna(subset=['date'], inplace=True)
                dropped_rows = initial_rows - len(forecast_chunk)
                if dropped_rows > 0:
                    print(f"    Dropped {dropped_rows} rows for {future_month_str} due to invalid date construction (e.g., Feb 29 in non-leap year).")
            except Exception as e:
                 print(f"    Error during vectorized date generation for {future_month_str}: {e}")
                 print(traceback.format_exc())
                 if 'hist_chunk' in locals(): del hist_chunk
                 if 'forecast_chunk' in locals(): del forecast_chunk
                 if 'temp_date_df' in locals(): del temp_date_df
                 if 'new_dates_series' in locals(): del new_dates_series
                 gc.collect()
                 continue 

            if forecast_chunk.empty:
                print(f"    Warning: No valid forecast rows generated for {future_month_str} after date validation.")
                del hist_chunk, forecast_chunk 
                gc.collect()
                continue

            forecast_chunk.drop(columns=['_hist_yyyymm'], inplace=True, errors='ignore')
            forecast_chunk["is_analogous_forecast"] = True 
            analogous_forecast_dfs.append(forecast_chunk)
            print(f"    Generated {len(forecast_chunk)} valid forecast rows for {future_month_str}.")

            del hist_chunk, forecast_chunk 
            if 'temp_date_df' in locals(): del temp_date_df
            if 'new_dates_series' in locals(): del new_dates_series
            gc.collect()

        source_data_df.drop(columns=['_hist_yyyymm'], inplace=True, errors='ignore')
        del source_data_df 
        gc.collect()
        return analogous_forecast_dfs


    def combine_and_save(self, original_combined_data: pd.DataFrame, analogous_forecast_dfs: List[pd.DataFrame]) -> bool:
        """
        Combine original data with pre-filtered analogous forecast, deduplicate, 
        save to Parquet, and return status.
        Returns True if the final file was saved successfully, False otherwise.
        """
        output_df = None
        save_successful = False

        # Start with a copy of the original combined data.
        output_df = original_combined_data.copy()
        output_df['is_analogous_forecast'] = False # Flag for original data

        if analogous_forecast_dfs:
            print("\nFiltering and combining original data with analogous forecast...")

            # 1. Get the last date from the original combined data (which includes provider's forecast)
            last_original_date = None
            if not original_combined_data.empty and 'date' in original_combined_data.columns:
                # Ensure 'date' is datetime before taking max
                if not pd.api.types.is_datetime64_any_dtype(original_combined_data['date']):
                    print("  Warning: 'date' column in original_combined_data is not datetime. Attempting conversion for max_date.")
                    try:
                        original_combined_data['date'] = pd.to_datetime(original_combined_data['date'])
                    except Exception as e_conv:
                        print(f"  Error converting original_combined_data 'date' to datetime: {e_conv}. Cannot filter analogous data by date.")
                        # Proceed without date filtering if conversion fails
                
                if pd.api.types.is_datetime64_any_dtype(original_combined_data['date']): # Re-check after potential conversion
                    last_original_date = original_combined_data['date'].max()
                    if pd.notna(last_original_date):
                        print(f"  Last date in original combined data (provider's forecast end): {last_original_date}")
                    else:
                        print("  Warning: Last date in original combined data is NaT. No date-based filtering of analogous data will occur.")
                        last_original_date = None # Explicitly set to None if NaT
            else:
                print("  Warning: Original combined data is empty or 'date' column missing. Cannot determine last date for filtering analogous data.")

            # 2. Filter analogous forecast dataframes to remove rows with dates <= last_original_date
            filtered_analogous_forecast_dfs = []
            total_analogous_rows_before_filter = 0
            total_analogous_rows_after_filter = 0

            for i, forecast_df_chunk in enumerate(analogous_forecast_dfs):
                total_analogous_rows_before_filter += len(forecast_df_chunk)
                
                # Ensure 'date' column in chunk is datetime for comparison
                if not pd.api.types.is_datetime64_any_dtype(forecast_df_chunk['date']):
                    print(f"    Warning: Converting 'date' in analogous chunk {i} to datetime for filtering.")
                    try:
                        forecast_df_chunk.loc[:, 'date'] = pd.to_datetime(forecast_df_chunk['date'], errors='coerce')
                        forecast_df_chunk.dropna(subset=['date'], inplace=True) # Drop if conversion failed
                    except Exception as e_conv_chunk:
                        print(f"    Error converting analogous chunk {i} 'date' to datetime: {e_conv_chunk}. Skipping this chunk for filtering.")
                        continue # Skip this chunk if date conversion fails

                if forecast_df_chunk.empty:
                    print(f"    Analogous chunk {i} is empty after potential date conversion. Skipping.")
                    continue
                
                if last_original_date is not None:
                    # Keep only rows strictly after the last original date
                    # .copy() is important here to avoid SettingWithCopyWarning on slices
                    filtered_chunk = forecast_df_chunk[forecast_df_chunk['date'] > last_original_date].copy()
                    rows_removed = len(forecast_df_chunk) - len(filtered_chunk)
                    if rows_removed > 0:
                        print(f"    Analogous chunk {i}: Removed {rows_removed} rows with dates <= {last_original_date}.")
                else:
                    # If no last_original_date, keep the whole chunk (no date-based filtering)
                    filtered_chunk = forecast_df_chunk.copy()
                    print(f"    Analogous chunk {i}: No date-based filtering applied (no valid last_original_date).")

                if not filtered_chunk.empty:
                    # Ensure the flag is True for the new forecast data (should be set in generate_forecast)
                    if 'is_analogous_forecast' not in filtered_chunk.columns or not filtered_chunk['is_analogous_forecast'].all():
                         filtered_chunk.loc[:, 'is_analogous_forecast'] = True
                    filtered_analogous_forecast_dfs.append(filtered_chunk)
                    total_analogous_rows_after_filter += len(filtered_chunk)
            
            print(f"  Total analogous rows before date-based filtering: {total_analogous_rows_before_filter}")
            print(f"  Total analogous rows after date-based filtering: {total_analogous_rows_after_filter}")

            # 3. Concatenate if there's filtered analogous data
            if filtered_analogous_forecast_dfs:
                try:
                    analogous_forecast_df_combined = pd.concat(filtered_analogous_forecast_dfs, ignore_index=True)
                    print(f"  Total filtered analogous forecast rows to append: {len(analogous_forecast_df_combined)}")

                    if not analogous_forecast_df_combined.empty:
                        # Combine original data and new (filtered) forecast data
                        output_df = pd.concat([output_df, analogous_forecast_df_combined], ignore_index=True, sort=False)
                        print(f"  Combined original ({len(original_combined_data)}) and new filtered forecast ({len(analogous_forecast_df_combined)}) records.")
                        del analogous_forecast_df_combined # Free memory
                        gc.collect()
                    else:
                        print("  No analogous forecast data remained after filtering. Output contains original data only.")
                except Exception as e_concat:
                    print(f"  Error during concatenation of filtered analogous data: {e_concat}")
                    print(traceback.format_exc())
                    # output_df still holds the original data at this point
            else:
                print("  No analogous forecast data available or all was filtered out. Output contains original data only.")

            # Ensure the flag column in the combined DF has no NaNs
            if 'is_analogous_forecast' in output_df.columns:
                 output_df["is_analogous_forecast"].fillna(False, inplace=True) 

            print(f"  Total rows before final sorting/deduplication: {len(output_df)}")
            print("  Sorting and deduplicating based on 'date'...")

            if output_df.empty:
                 print("  DataFrame is empty before final processing.")
                 save_successful = False
                 if 'output_df' in locals(): del output_df 
                 gc.collect()
                 return save_successful

            if not pd.api.types.is_datetime64_any_dtype(output_df['date']):
                 print(f"  Warning: Converting 'date' in combined data to datetime for sorting/deduplication.")
                 output_df['date'] = pd.to_datetime(output_df['date'], errors='coerce')
                 output_df.dropna(subset=['date'], inplace=True)

            if output_df.empty:
                 print("  DataFrame is empty after date conversion/dropna.")
                 save_successful = False
                 if 'output_df' in locals(): del output_df
                 gc.collect()
                 return save_successful 

            output_df.sort_values(by="date", inplace=True, ignore_index=True)
            initial_row_count = len(output_df)
            output_df.drop_duplicates(subset=['date'], keep='first', inplace=True)
            final_row_count = len(output_df)
            duplicates_removed = initial_row_count - final_row_count
            if duplicates_removed > 0:
                 print(f"  Removed {duplicates_removed} duplicate date entries (kept first - primarily for internal consistency or unexpected overlaps).")
            print(f"  Final dataset size: {final_row_count} unique records.")

        else: # analogous_forecast_dfs was empty to begin with
            print("\nNo analogous forecast dataframes provided. Output contains original data only.")
        
        # --- Save the final DataFrame to Parquet ---
        if output_df is not None and not output_df.empty:
            print(f"\nSaving final data ({len(output_df)} rows) to Parquet: '{self.output_file}'")
            try:
                os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
                numeric_cols = output_df.select_dtypes(include='number').columns
                numeric_cols = numeric_cols.drop('is_analogous_forecast', errors='ignore')
                if not numeric_cols.empty:
                     print(f"  Rounding numeric columns: {list(numeric_cols)}")
                     output_df.loc[:, numeric_cols] = output_df[numeric_cols].round(2)
                output_df.to_parquet(self.output_file, index=False, engine='pyarrow')
                print("  Successfully saved final data.")
                save_successful = True
            except Exception as e:
                print(f"Error: Failed saving final Parquet file '{self.output_file}': {e}")
                print(traceback.format_exc())
                save_successful = False
        else:
             print("Warning: No valid data available to save after processing.")
             save_successful = False

        if 'output_df' in locals() and output_df is not None: del output_df 
        gc.collect()
        return save_successful

    def delete_input_file(self):
        """Deletes the input combined Parquet file."""
        print(f"\nAttempting to delete input Parquet file: {self.input_file}")
        if os.path.exists(self.input_file):
            try:
                if os.path.exists(self.output_file):
                    os.remove(self.input_file)
                    print(f"  Successfully deleted input file.")
                else:
                    print(f"  Skipping deletion of input file '{self.input_file}' because output file '{self.output_file}' was not found (save likely failed).")
            except OSError as e:
                print(f"  Error deleting input file '{self.input_file}': {e}")
            except Exception as e:
                 print(f"  Unexpected error deleting input file '{self.input_file}': {e}")
                 print(traceback.format_exc())
        else:
            print(f"  Input file not found at '{self.input_file}' (already deleted or never existed).")

    def run(self):
        """Execute the full process: Load, Generate, Combine, Save, Delete Input."""
        print(f"Starting analogous forecast generation for region: {self.region} (using Parquet)")
        combined_data = None 

        try:
            combined_data = self.load_combined_data()
            if combined_data is None or combined_data.empty:
                print("Aborting run for this region due to failure loading input data or input data is empty.")
                return 

            analogous_forecast_dfs = self.generate_forecast(combined_data)
            save_status = self.combine_and_save(combined_data, analogous_forecast_dfs)

            # if save_status:
            #     self.delete_input_file()
            # else:
            #     print("\nSkipping deletion of input file because the final output file was NOT successfully saved.")

        except Exception as e:
            print(f"\n[CRITICAL ERROR] Unhandled exception during run for region {self.region}: {e}")
            print(traceback.format_exc())
        finally:
            if 'combined_data' in locals() and combined_data is not None:
                 del combined_data
            if 'analogous_forecast_dfs' in locals() and analogous_forecast_dfs:
                 analogous_forecast_dfs.clear()
                 del analogous_forecast_dfs
            gc.collect() 
            print(f"\nAnalogous weather processing finished for region: {self.region}.")


# --- Main Execution Block ---
if __name__ == "__main__":
    print("===== Starting Analogous Forecast Generation Process =====")
    if not regions:
        print("No regions loaded from config/regions.json. Nothing to process.")
        sys.exit(0)
    if not month_mapping:
         print("No month mapping loaded from config/month_mapping.json. Nothing to process.")
         sys.exit(0)
    regions_with_errors = []
    if not isinstance(regions, dict) or not all(isinstance(k, str) for k in regions.keys()):
        print(f"Error: Expected regions config to be a dictionary with string keys, but got {type(regions)}. Exiting.")
        sys.exit(1)

    print(f"Found {len(regions)} regions in config: {list(regions.keys())}")
    print(f"Using Month Mapping (sample): {list(month_mapping.items())[:min(5, len(month_mapping))]}...")

    for region_name in regions.keys():
        print(f"\n{'='*20} Processing Region: {region_name.upper()} {'='*20}")
        try:
            generator = AnalogousForecastGenerator(region_name, month_mapping, base_dir=BASE_OUTPUT_DIR)
            generator.run()
        except ValueError as e:
             print(f"\n---!!! Configuration/Initialization Error processing {region_name.upper()}: {e} !!!---")
             traceback.print_exc()
             regions_with_errors.append(f"{region_name} (Config/Init Error)")
        except Exception as e:
             print(f"\n---!!! CRITICAL UNHANDLED ERROR processing {region_name.upper()}: {e} !!!---")
             traceback.print_exc()
             regions_with_errors.append(f"{region_name} (Critical Error)")

    print("\n" + "="*60)
    print("--- Analogous Forecast Generation Complete ---")
    if regions_with_errors:
        print(f"Regions encountering errors ({len(regions_with_errors)}):")
        for region_err in regions_with_errors: print(f"  - {region_err}")
    else:
        print("All configured regions processed successfully.")
    print("="*60)