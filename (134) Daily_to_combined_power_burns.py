import os
import json
import sys
import pandas as pd
from datetime import datetime
import traceback
import warnings

# --- Configuration Loading (BASE_OUTPUT_DIR) ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from config.settings import BASE_OUTPUT_DIR
    print(f"Loaded BASE_OUTPUT_DIR from config: {BASE_OUTPUT_DIR}")
except (ImportError, NameError, FileNotFoundError) as e:
    print(f"Could not import BASE_OUTPUT_DIR from config.settings ({e}). Using default.")
    BASE_OUTPUT_DIR = "D:\\EIA_STACK"  # FALLBACK - IMPORTANT: Update if needed
    print(f"Using default BASE_OUTPUT_DIR: {BASE_OUTPUT_DIR}")

# --- Load Regions Configuration ---
regions_list = []
try:
    script_location_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    project_root = os.path.abspath(os.path.join(script_location_dir, '..'))
    
    paths_to_check = [
        os.path.join(project_root, 'config', 'regions.json'),
        os.path.join(script_location_dir, 'config', 'regions.json'),
        os.path.join(os.getcwd(), 'config', 'regions.json')
    ]
    regions_config_path = None
    for path in paths_to_check:
        if os.path.exists(path):
            regions_config_path = path
            print(f"Found regions config at: {regions_config_path}")
            break
    
    if regions_config_path:
        with open(regions_config_path, 'r') as f:
            regions_config_data = json.load(f)
        
        if isinstance(regions_config_data, dict):
            regions_list = list(regions_config_data.keys())
        elif isinstance(regions_config_data, list):
            if all(isinstance(item, str) for item in regions_config_data):
                regions_list = regions_config_data
            elif all(isinstance(item, dict) and 'name' in item for item in regions_config_data):
                regions_list = [item['name'] for item in regions_config_data]
            else:
                raise ValueError("Regions config list is not in a recognized format.")
        else:
            raise ValueError("Regions config file is not a dictionary or a list.")
            
        if not regions_list:
            print("WARNING: Regions config loaded but resulted in an empty list of regions.")
        else:
            print(f"Successfully loaded regions: {regions_list}")
    else:
        print("ERROR: Core regions config file 'config/regions.json' not found.")
        sys.exit(1)

except FileNotFoundError as e:
    print(f"ERROR: Regions config file not found. {e}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Critical error loading regions configuration: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- File Configuration ---
HOURLY_COMBINED_INPUT_FILENAME = "final_power_burns_multi_features.csv"
DAILY_ESTIMATED_GAS_OUTPUT_FILENAME_PATTERN = "{region}_estimated_daily_gas_mmcf.csv"
COMBINED_DAILY_FOR_SQL_FILENAME = "power_burns_daily.csv"


# --- Column Name Definitions ---
HOURLY_DATE_COL = 'date'
HOURLY_ESTIMATE_SUFFIX = "_Hourly_MMCF_Multi"
DAILY_DATE_COL_OUTPUT = 'Date' 
REGIONAL_DATA_COL_NAME = "Estimated_Daily_Gas_MMcf_Total" 
L48_POWER_BURNS_COL_NAME = "L48_Power_Burns" 

# --- Pandas Options ---
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

def aggregate_hourly_to_daily_regional_and_combine():
    """
    Loads hourly combined gas estimates, aggregates them to daily sums
    for each region, saves regional CSVs. Then, calculates a total L48
    daily sum and saves it to a separate file for SQL upload.
    """
    print("\n=====================================================================")
    print("=== Aggregating Hourly to Daily (Regional & Combined L48) ===")
    print("=====================================================================")

    input_filepath = os.path.join(BASE_OUTPUT_DIR, HOURLY_COMBINED_INPUT_FILENAME)
    combined_sql_output_filepath = os.path.join(BASE_OUTPUT_DIR, COMBINED_DAILY_FOR_SQL_FILENAME)

    print(f"Input hourly file: {input_filepath}")

    if not regions_list:
        print("ERROR: No regions configured to process. Exiting.")
        sys.exit(1)

    # 1. Load Hourly Data
    if not os.path.exists(input_filepath):
        print(f"ERROR: Input file not found: {input_filepath}")
        sys.exit(1)

    try:
        print(f"Loading hourly data from '{os.path.basename(input_filepath)}'...")
        df_hourly = pd.read_csv(input_filepath)
        
        if HOURLY_DATE_COL not in df_hourly.columns:
            print(f"ERROR: Expected date column '{HOURLY_DATE_COL}' not found in '{input_filepath}'.")
            sys.exit(1)
        df_hourly[HOURLY_DATE_COL] = pd.to_datetime(df_hourly[HOURLY_DATE_COL], errors='coerce')
        
        initial_rows = len(df_hourly)
        df_hourly.dropna(subset=[HOURLY_DATE_COL], inplace=True)
        if len(df_hourly) < initial_rows:
            print(f"    Dropped {initial_rows - len(df_hourly)} rows with invalid date entries.")

        if df_hourly.empty:
            print("ERROR: No valid data after loading and date parsing. Exiting.")
            sys.exit(1)
        print(f"Successfully loaded {len(df_hourly)} hourly records.")

    except Exception as e:
        print(f"ERROR: Could not load or parse input file '{input_filepath}': {e}")
        traceback.print_exc()
        sys.exit(1)

    # 2. Create Daily Date Column for Grouping & Clean Regional Hourly Columns
    df_hourly[DAILY_DATE_COL_OUTPUT] = df_hourly[HOURLY_DATE_COL].dt.normalize()

    print("\n--- Pre-processing Regional Hourly Columns (Numeric Conversion & NaN Fill) ---")
    actual_regional_hourly_cols_found = []
    for region_name_iter in regions_list:
        col_to_check = f"{region_name_iter.upper()}{HOURLY_ESTIMATE_SUFFIX}"
        if col_to_check in df_hourly.columns:
            df_hourly[col_to_check] = pd.to_numeric(df_hourly[col_to_check], errors='coerce').fillna(0)
            actual_regional_hourly_cols_found.append(col_to_check)
            # print(f"    Processed column: {col_to_check}") # Optional: for verbose logging
        # else: # Optional: for verbose logging
            # print(f"    Column not found, skipping pre-processing: {col_to_check}")
    
    if not actual_regional_hourly_cols_found:
        print("WARNING: No regional hourly data columns (ending with '{HOURLY_ESTIMATE_SUFFIX}') were found and processed in the input file.")
    else:
        print(f"    Found and pre-processed {len(actual_regional_hourly_cols_found)} regional hourly columns.")


    successful_regions_count = 0
    failed_regions_details = []

    # 3. Process each region and save individual daily files
    print("\n--- Processing Regions and Saving Individual Daily Files ---")
    for region_name in regions_list:
        region_upper = region_name.upper()
        # print(f"  --- Processing Region: {region_upper} ---") # Already printed if verbose

        hourly_col_for_region = f"{region_upper}{HOURLY_ESTIMATE_SUFFIX}"

        if hourly_col_for_region not in actual_regional_hourly_cols_found: # Check against pre-processed list
            msg = f"    WARNING: Hourly column '{hourly_col_for_region}' not found or not processed for {region_upper}. Skipping for individual file."
            print(msg)
            failed_regions_details.append(f"{region_name} (Column not found/processed: {hourly_col_for_region})")
            continue

        try:
            # print(f"    Aggregating data for {hourly_col_for_region}...") # Verbose
            df_region_sum = df_hourly.groupby(DAILY_DATE_COL_OUTPUT, as_index=False).agg(
                {hourly_col_for_region: 'sum'}
            )
            
            df_region_sum_for_file = df_region_sum.rename(columns={hourly_col_for_region: REGIONAL_DATA_COL_NAME})
            df_region_sum_for_file = df_region_sum_for_file[[DAILY_DATE_COL_OUTPUT, REGIONAL_DATA_COL_NAME]]
            
            region_final_dir = os.path.join(BASE_OUTPUT_DIR, region_name, "final")
            output_filepath_region = os.path.join(region_final_dir, DAILY_ESTIMATED_GAS_OUTPUT_FILENAME_PATTERN.format(region=region_name))
            
            # print(f"    Saving daily summed gas for {region_upper} to: {output_filepath_region}") # Verbose
            os.makedirs(region_final_dir, exist_ok=True)
            
            df_region_sum_for_file[REGIONAL_DATA_COL_NAME] = df_region_sum_for_file[REGIONAL_DATA_COL_NAME].round(3)
            df_region_sum_for_file.to_csv(output_filepath_region, index=False, date_format='%Y-%m-%d')
            print(f"    Successfully saved individual daily file for {region_upper}.")
            successful_regions_count += 1

        except Exception as e:
            msg = f"    ERROR processing or saving individual file for region {region_upper}: {e}"
            print(msg)
            traceback.print_exc()
            failed_regions_details.append(f"{region_name} (Error saving individual file: {type(e).__name__})")
    
    # 4. Calculate and save combined L48 daily sum for SQL upload
    print("\n--- Calculating and Saving Combined L48 Daily Data for SQL Upload File ---")
    if df_hourly.empty: 
        print("ERROR: Hourly data is empty. Cannot create L48 combined file.")
    elif not actual_regional_hourly_cols_found:
        print("WARNING: No regional hourly columns found to sum for L48 total. Creating L48 file with zeros.")
        unique_dates = df_hourly[[DAILY_DATE_COL_OUTPUT]].drop_duplicates().sort_values(DAILY_DATE_COL_OUTPUT)
        df_for_sql = pd.DataFrame({
            'date': unique_dates[DAILY_DATE_COL_OUTPUT],
            L48_POWER_BURNS_COL_NAME: 0
        })
        df_for_sql[L48_POWER_BURNS_COL_NAME] = df_for_sql[L48_POWER_BURNS_COL_NAME].astype(int)
    else:
        print(f"Summing L48 total from hourly columns: {actual_regional_hourly_cols_found}")
        # Sum regional hourly columns row-wise to get total L48 hourly consumption
        df_hourly['L48_Hourly_Sum_Temp'] = df_hourly[actual_regional_hourly_cols_found].sum(axis=1)

        # Aggregate this hourly L48 sum to daily
        df_l48_daily_sum = df_hourly.groupby(DAILY_DATE_COL_OUTPUT, as_index=False)['L48_Hourly_Sum_Temp'].sum()

        df_for_sql = df_l48_daily_sum.rename(columns={
            DAILY_DATE_COL_OUTPUT: 'date', # Match SQL script's expected input col name
            'L48_Hourly_Sum_Temp': L48_POWER_BURNS_COL_NAME
        })
        
        df_for_sql[L48_POWER_BURNS_COL_NAME] = df_for_sql[L48_POWER_BURNS_COL_NAME].round(0).astype(int)
    
    # Save the df_for_sql regardless of whether it has data or is zeroed out (if applicable)
    try:
        print(f"Saving L48 daily data for SQL to: {combined_sql_output_filepath}")
        df_for_sql.to_csv(combined_sql_output_filepath, index=False, date_format='%Y-%m-%d')
        print(f"Successfully saved '{COMBINED_DAILY_FOR_SQL_FILENAME}'.")
    except Exception as e:
        print(f"ERROR saving L48 combined daily file '{combined_sql_output_filepath}': {e}")
        traceback.print_exc()


    # 5. Summary
    print("\n=====================================================================")
    print("=== Regional & Combined L48 Daily Aggregation Process Summary ===")
    print(f"Total regions configured: {len(regions_list)}")
    print(f"Successfully processed and saved individual regional files for: {successful_regions_count} regions")
    if failed_regions_details:
        print(f"Failed or skipped regions for individual files: {len(failed_regions_details)}")
        for detail in failed_regions_details:
            print(f"  - {detail}")
    
    if os.path.exists(combined_sql_output_filepath):
        print(f"L48 Combined file '{COMBINED_DAILY_FOR_SQL_FILENAME}' for SQL created successfully.")
    else: # Should not happen if df_for_sql is always created, but good to check
        print(f"WARNING: L48 Combined file '{COMBINED_DAILY_FOR_SQL_FILENAME}' was NOT created.")

    print("=====================================================================")

if __name__ == "__main__":
    aggregate_hourly_to_daily_regional_and_combine()