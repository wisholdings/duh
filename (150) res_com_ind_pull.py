import os
import json
import sys
import pandas as pd
import traceback
import time
from datetime import datetime
import requests
import warnings

# --- Configuration Loading ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from config.settings import BASE_OUTPUT_DIR
    print(f"Loaded BASE_OUTPUT_DIR from config: {BASE_OUTPUT_DIR}")
except (ImportError, NameError, FileNotFoundError) as e:
    print(f"Could not import BASE_OUTPUT_DIR from config.settings ({e}). Using default.")
    BASE_OUTPUT_DIR = "D:\\EIA_NATGAS_RCI_DATA" # EXAMPLE: Change this path
    print(f"Using default BASE_OUTPUT_DIR: {BASE_OUTPUT_DIR}")

# --- EIA API Configuration ---
API_KEY = ""

# --- File Configuration for the final combined file ---
FINAL_COMBINED_FILENAME = "ALL_REGIONS_res_com_ind_combined.csv"
FINAL_COMBINED_FILEPATH = os.path.join(BASE_OUTPUT_DIR, FINAL_COMBINED_FILENAME)


warnings.filterwarnings('ignore', message='Unverified HTTPS request')

def fetch_eia_data_by_process_type(api_key, region_codes, region_name_log, process_name_filter, data_type_label_log):
    api_url = "https://api.eia.gov/v2/natural-gas/cons/sum/data/"
    limit = 5000
    today_date_str = datetime.now().strftime("%Y-%m")
    print(f"    Fetching {data_type_label_log} data for region '{region_name_log}', codes: {region_codes}, process: '{process_name_filter}'")
    offset = 0
    current_data_type_df = pd.DataFrame()
    while True:
        params = {
            "api_key": api_key, "frequency": "monthly", "data[0]": "value",
            "start": "2019-01", "end": today_date_str,
            "sort[0][column]": "period", "sort[0][direction]": "desc",
            "offset": offset, "length": limit
        }
        for i, code in enumerate(region_codes):
            params[f"facets[duoarea][{i}]"] = code
        try:
            response = requests.get(api_url, params=params, verify=False, timeout=60)
            if response.status_code == 200:
                data = response.json()
                if 'response' in data and 'data' in data['response']:
                    page_data = data['response']['data']
                    if not page_data:
                        print(f"      No more {data_type_label_log} data from API for {region_name_log} (offset {offset}).")
                        break
                    df = pd.DataFrame(page_data)
                    print(f"      Fetched {len(df)} {data_type_label_log} records for {region_name_log} (offset {offset})...")
                    expected_cols = ['period', 'duoarea', 'process-name', 'value']
                    if not all(col in df.columns for col in expected_cols):
                        print(f"      Warning: Missing expected columns in {data_type_label_log} API response batch for {region_name_log}. Skipping batch.")
                        print(f"      Found columns: {df.columns.tolist()}")
                        offset += limit
                        continue
                    df = df[expected_cols]
                    df_filtered = df[df['process-name'] == process_name_filter].copy()
                    if not df_filtered.empty:
                        df_filtered['period'] = pd.to_datetime(df_filtered['period'], errors='coerce')
                        df_filtered['duoarea'] = df_filtered['duoarea'].astype(str)
                        df_filtered['value'] = pd.to_numeric(df_filtered['value'], errors='coerce')
                        df_filtered.dropna(subset=['period', 'value'], inplace=True)
                        current_data_type_df = pd.concat([current_data_type_df, df_filtered], ignore_index=True)
                    offset += limit
                else:
                    print(f"      API response format unexpected for {data_type_label_log} for {region_name_log}. Data: {str(data)[:200]}...")
                    return pd.DataFrame()
            elif response.status_code == 404:
                print(f"      Warning: {data_type_label_log} data not found (404) for {region_name_log}, codes {region_codes} at offset {offset}.")
                break
            elif response.status_code == 429:
                print(f"    ERROR: API rate limit hit (429) for {data_type_label_log} for {region_name_log}.")
                return pd.DataFrame()
            else:
                print(f"    ERROR: Failed to fetch {data_type_label_log} data for {region_name_log}. Status: {response.status_code}, Response: {response.text[:200]}")
                return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            print(f"    ERROR: Network or request error for {data_type_label_log} for {region_name_log}: {e}")
            return pd.DataFrame()
        except json.JSONDecodeError as e:
            print(f"    ERROR: Could not decode JSON response for {data_type_label_log} for {region_name_log}: {e}")
            if 'response' in locals() and response is not None:
                 print(f"    Response Text: {response.text[:500]}...")
            return pd.DataFrame()
        except Exception as e:
            print(f"    ERROR: An unexpected error occurred during fetch for {data_type_label_log} for {region_name_log}: {e}")
            traceback.print_exc()
            return pd.DataFrame()
    print(f"    Finished fetching {data_type_label_log} for {region_name_log}. Total relevant rows: {len(current_data_type_df)}")
    return current_data_type_df

def summarize_data_monthly(raw_df, data_type_label, output_col_name, region_name_log):
    if raw_df.empty:
        print(f"    No raw {data_type_label} data to process for {region_name_log}.")
        return pd.DataFrame(columns=['datetime', output_col_name])
    print(f"    Calculating monthly sums for {data_type_label} data for {region_name_log}...")
    if 'period' not in raw_df.columns or 'value' not in raw_df.columns:
        print(f"    Raw {data_type_label} data for {region_name_log} is missing 'period' or 'value' column.")
        return pd.DataFrame(columns=['datetime', output_col_name])
    if raw_df.empty:
        print(f"    No valid {data_type_label} data after internal cleaning for {region_name_log}.")
        return pd.DataFrame(columns=['datetime', output_col_name])
    try:
        monthly_sum = raw_df.groupby(raw_df['period'].dt.to_period('M'))['value'].sum()
    except Exception as e:
        print(f"    Error during groupby/sum for {data_type_label} data for {region_name_log}: {e}")
        return pd.DataFrame(columns=['datetime', output_col_name])
    if monthly_sum.empty:
        print(f"    Monthly summation resulted in empty {data_type_label} data for {region_name_log}.")
        return pd.DataFrame(columns=['datetime', output_col_name])
    df_processed = monthly_sum.reset_index()
    df_processed['period'] = df_processed['period'].dt.to_timestamp()
    df_processed.rename(columns={'period': 'datetime', 'value': output_col_name}, inplace=True)
    print(f"    Successfully summarized {data_type_label} data for {region_name_log}. Shape: {df_processed.shape}")
    return df_processed

regions = {
    "texas": ["STX"], "florida": ["SFL"], "carolina": ["SNC", "SSC"], "california": ["SCA"],
    "newengland": ["SME", "SCT", "SMA", "SNH", "SRI", "SVT"], "newyork": ["SNY"],
    "midatlantic": ["SMD", "SDE", "SPA", "SVA", "SWV"],
    "midwest": ["SIL", "SIN", "SMO", "SMI", "SMN", "SWI", "SIA", "SOH", "SKS", "SKY"],
    "southeast": ["SAL", "SAR", "SGA", "SMS", "SLA"],
    "southwest": ["SAZ", "SNM", "SNV", "SUT", "SCO"],
    "northwest": ["SID", "SOR", "SWA", "SWY", "SMT"],
    "central": ["SNE", "SNJ", "SND", "SSD", "SOK"], "tennessee": ["STN"]
}
data_types_to_process = {
    "Residential": {"filter": "Residential Consumption", "col_name": "Residential_MMcf"},
    "Commercial":  {"filter": "Commercial Consumption",  "col_name": "Commercial_MMcf"},
    "Industrial":  {"filter": "Industrial Consumption",  "col_name": "Industrial_MMcf"}
}
base_output_columns = [cfg["col_name"] for data_label, cfg in data_types_to_process.items()]
consump_sum_col_name = 'consump_sum'
# expected_output_columns_ordered will be dynamically built for each region

print("\n===================================================================")
print("=== Starting EIA Natural Gas Combined (Res/Com/Ind) Data Processing ===")
print("===================================================================")
start_total_time = time.time()
successful_regions_processing = [] # Renamed to avoid conflict
failed_regions_processing = []     # Renamed
all_regional_final_dfs = [] # List to store each region's final DataFrame for later combination

for region_name, region_codes in regions.items():
    region_upper = region_name.upper()
    print(f"\n\n################ Processing Region: {region_upper} ################")
    start_region_time = time.time()
    region_success_flag = False
    current_region_df_for_master_combine = pd.DataFrame() # To store the df for this region

    try:
        region_base_dir = os.path.join(BASE_OUTPUT_DIR, region_name)
        final_dir = os.path.join(region_base_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        print(f"  Output directory: {final_dir}")

        list_of_processed_dfs_for_region = []
        any_data_type_yielded_values = False
        
        # Dynamically create column names for this region
        regional_residential_col = f"{region_upper}_{data_types_to_process['Residential']['col_name']}"
        regional_commercial_col = f"{region_upper}_{data_types_to_process['Commercial']['col_name']}"
        regional_industrial_col = f"{region_upper}_{data_types_to_process['Industrial']['col_name']}"
        regional_sum_col = f"{region_upper}_{consump_sum_col_name}"

        regional_data_types_config = {
            "Residential": {"filter": "Residential Consumption", "col_name": regional_residential_col},
            "Commercial":  {"filter": "Commercial Consumption",  "col_name": regional_commercial_col},
            "Industrial":  {"filter": "Industrial Consumption",  "col_name": regional_industrial_col}
        }

        for data_label, config in regional_data_types_config.items(): # Use regional_data_types_config
            print(f"\n  --- Processing {data_label} Data for {region_upper} ---")
            raw_df = fetch_eia_data_by_process_type(
                API_KEY, region_codes, region_upper, config["filter"], data_label
            )
            summed_df = summarize_data_monthly(
                raw_df, data_label, config["col_name"], region_upper # Pass regional col name
            )
            if 'datetime' in summed_df.columns:
                list_of_processed_dfs_for_region.append(summed_df)
                if not summed_df.empty and config["col_name"] in summed_df.columns and not summed_df[config["col_name"]].isnull().all():
                    any_data_type_yielded_values = True
            else:
                print(f"    Warning: {data_label} data for {region_upper} resulted in a DataFrame without 'datetime'. Skipping for merge.")

        if not list_of_processed_dfs_for_region:
            print(f"  No DataFrames were suitable for merging for {region_upper}. Skipping save.")
            failed_regions_processing.append(f"{region_name} (No data suitable for merge)")
            continue

        all_datetimes_series = pd.Series(dtype='datetime64[ns]')
        for df_ in list_of_processed_dfs_for_region:
            if 'datetime' in df_.columns and not df_['datetime'].empty:
                all_datetimes_series = pd.concat([all_datetimes_series, df_['datetime']])
        if all_datetimes_series.empty:
            if not any_data_type_yielded_values:
                 print(f"  No actual data values found for any type for {region_upper}. Skipping save.")
                 failed_regions_processing.append(f"{region_name} (No values for any type)")
            else: 
                 print(f"  No 'datetime' values found across all processed data for {region_upper}. Skipping merge and save.")
                 failed_regions_processing.append(f"{region_name} (No datetime data for merge base)")
            continue
            
        merged_df = pd.DataFrame({'datetime': all_datetimes_series.drop_duplicates().sort_values()}).reset_index(drop=True)
        for df_to_merge in list_of_processed_dfs_for_region:
            data_cols_in_df_to_merge = [col for col in df_to_merge.columns if col != 'datetime']
            if not df_to_merge.empty and data_cols_in_df_to_merge:
                if not df_to_merge[data_cols_in_df_to_merge].isnull().all().all():
                    merged_df = pd.merge(merged_df, df_to_merge[['datetime'] + data_cols_in_df_to_merge], on='datetime', how='left')
        
        regional_columns_to_sum = [
            regional_residential_col, regional_commercial_col, regional_industrial_col
        ]
        actual_regional_cols_to_sum = [col for col in regional_columns_to_sum if col in merged_df.columns]
        if actual_regional_cols_to_sum:
            merged_df[regional_sum_col] = merged_df[actual_regional_cols_to_sum].sum(axis=1, min_count=1)
        else:
            merged_df[regional_sum_col] = pd.NA
            
        region_specific_output_columns = ['datetime'] + actual_regional_cols_to_sum + ([regional_sum_col] if regional_sum_col in merged_df else [])
        
        # Ensure all defined regional columns exist, even if empty from fetch/sum
        for col_name in [regional_residential_col, regional_commercial_col, regional_industrial_col, regional_sum_col]:
             if col_name not in merged_df.columns:
                merged_df[col_name] = pd.NA

        # Select only the relevant columns for this region's file
        final_region_df_cols = ['datetime', regional_residential_col, regional_commercial_col, regional_industrial_col, regional_sum_col]
        # Filter out any columns that might not have been created if all underlying data was missing
        final_region_df_cols = [col for col in final_region_df_cols if col in merged_df.columns]
        current_region_df_for_master_combine = merged_df[final_region_df_cols].copy()


        # For the individual region file, only Res, Com, Ind specific to that region.
        # The `expected_output_columns_ordered` was for the old single-column structure per region.
        # Now we want datetime, REGION_Residential_MMcf, REGION_Commercial_MMcf, REGION_Industrial_MMcf, REGION_consump_sum
        output_cols_for_region_file = ['datetime', 
                                       regional_data_types_config["Residential"]["col_name"],
                                       regional_data_types_config["Commercial"]["col_name"],
                                       regional_data_types_config["Industrial"]["col_name"],
                                       regional_sum_col]
        # Filter out columns that might not exist if all data for a type was missing
        output_cols_for_region_file = [col for col in output_cols_for_region_file if col in merged_df.columns]
        
        df_for_region_csv = merged_df[output_cols_for_region_file].copy()


        data_columns_in_region_csv = [col for col in df_for_region_csv.columns if col != 'datetime']
        if not data_columns_in_region_csv or df_for_region_csv[data_columns_in_region_csv].isnull().all().all():
            print(f"  All data columns in the final DataFrame for region {region_upper} are empty (NaN). Skipping save.")
            if not any(region_name in fr for fr in failed_regions_processing):
                 failed_regions_processing.append(f"{region_name} (Merged data columns all NaN)")
            continue

        output_filename = f"{region_name}_res_com_ind.csv" # This individual file will now have region-specific column names
        output_filepath = os.path.join(final_dir, output_filename)
        
        print(f"  Saving combined monthly data for {region_upper} to: {output_filepath}")
        df_for_region_csv.to_csv(output_filepath, index=False, float_format='%.3f')
        print(f"  Successfully saved combined data for {region_upper}. Shape: {df_for_region_csv.shape}")
        
        # Add the successfully processed regional DataFrame to the list for master combination
        if not current_region_df_for_master_combine.empty:
            all_regional_final_dfs.append(current_region_df_for_master_combine)
        
        region_success_flag = True

    except Exception as e:
        print(f"  CRITICAL ERROR processing region {region_upper}: {e}")
        traceback.print_exc()
        if not any(region_name in fr for fr in failed_regions_processing):
            failed_regions_processing.append(f"{region_name} (Critical Error: {type(e).__name__})")
    finally:
        if region_success_flag:
            successful_regions_processing.append(region_name)
        end_region_time = time.time()
        print(f"  Region {region_upper} processing time: {end_region_time - start_region_time:.2f} seconds.")
        print(f"################ End Processing Region: {region_upper} ################")

# --- Combine All Regional Processed DataFrames into a Single Master File ---
if all_regional_final_dfs:
    print("\n\n=======================================================")
    print("=== Combining All Regional Data into Master File ===")
    print("=======================================================")
    
    # Start with the first DataFrame in the list as the base for merging
    master_combined_df = all_regional_final_dfs[0].copy() 
    
    # Loop through the rest of the DataFrames and merge them
    for i in range(1, len(all_regional_final_dfs)):
        # Check if 'datetime' exists, if not, this df might be problematic
        if 'datetime' not in all_regional_final_dfs[i].columns:
            print(f"Warning: DataFrame for an intermediate region (index {i}) is missing 'datetime' column. Skipping its merge.")
            continue
        master_combined_df = pd.merge(master_combined_df, all_regional_final_dfs[i], on='datetime', how='outer')

    # Sort by datetime
    master_combined_df.sort_values(by='datetime', inplace=True)
    master_combined_df.reset_index(drop=True, inplace=True)
    
    # Add date_published column
    master_combined_df['date_published'] = datetime.now().strftime('%Y-%m-%d')
    
    # Save the master combined file
    try:
        os.makedirs(BASE_OUTPUT_DIR, exist_ok=True) # Ensure base output dir exists
        master_combined_df.to_csv(FINAL_COMBINED_FILEPATH, index=False, float_format='%.3f')
        print(f"Successfully saved master combined file to: {FINAL_COMBINED_FILEPATH}")
        print(f"Master combined file shape: {master_combined_df.shape}")
    except Exception as e:
        print(f"ERROR saving master combined file {FINAL_COMBINED_FILEPATH}: {e}")
        traceback.print_exc()
else:
    print("\nNo regional DataFrames were successfully processed to create a master combined file.")


# --- End of Processing Summary ---
end_total_time = time.time()
total_duration = end_total_time - start_total_time
print("\n\n=============================================")
print("========== Processing Summary ==========")
print(f"Total regions attempted: {len(regions.keys())}")
print(f"Successfully processed regions: {len(successful_regions_processing)} ({', '.join(successful_regions_processing) if successful_regions_processing else 'None'})")
print(f"Failed regions: {len(failed_regions_processing)}")

if failed_regions_processing:
    print("\n--- Failure Details (Regions) ---")
    for fr_info in failed_regions_processing:
        print(f"  - {fr_info}")
    print("\nPlease review the detailed logs above for specific errors within each region.")
elif len(successful_regions_processing) == len(regions.keys()):
     print("\nAll specified regions processed successfully.")
else:
     print("\nProcessing finished. Some regions might have been skipped or had an unclassified status. Check logs.")
     print(f"Successful count: {len(successful_regions_processing)}, Failed count: {len(failed_regions_processing)}")

if os.path.exists(FINAL_COMBINED_FILEPATH) and all_regional_final_dfs:
    print(f"\nMaster combined file '{FINAL_COMBINED_FILENAME}' created successfully.")
elif all_regional_final_dfs: # implies save failed
    print(f"\nMaster combined file '{FINAL_COMBINED_FILENAME}' FAILED to save (but data was processed).")
else: # implies no data to combine
    print(f"\nMaster combined file '{FINAL_COMBINED_FILENAME}' was NOT created as no regional data was available.")


print(f"\nTotal execution time: {total_duration:.2f} seconds.")
print("=============================================")

if failed_regions_processing or not all_regional_final_dfs: # If any region failed OR no data to combine
    sys.exit(1)
else:
    sys.exit(0)