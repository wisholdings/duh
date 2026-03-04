# -*- coding: utf-8 -*-
import os
import json
import sys
import pandas as pd
import numpy as np # Still needed for np.nan if coerce results in NaN during to_numeric
import traceback
import time
from datetime import datetime

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
    current_dir_for_default = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    project_root_for_default = os.path.abspath(os.path.join(current_dir_for_default, '..'))
    BASE_OUTPUT_DIR = os.path.join(project_root_for_default, "EIA_STACK") 
    if not os.path.exists(BASE_OUTPUT_DIR):
         BASE_OUTPUT_DIR = os.path.join(project_root_for_default, "EIA_STACK_DEFAULT")
    print(f"Using default/fallback BASE_OUTPUT_DIR: {BASE_OUTPUT_DIR}")

# --- Load Regions Configuration ---
try:
    script_dir_for_regions = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    project_root = os.path.abspath(os.path.join(script_dir_for_regions, '..'))
    paths_to_check = [
        os.path.join(project_root, 'config', 'regions.json'),
        os.path.join(script_dir_for_regions, 'config', 'regions.json'),
        os.path.join('config', 'regions.json')
    ]
    regions_config_path = None
    for path in paths_to_check:
        if os.path.exists(path):
            regions_config_path = path
            print(f"Found regions config at: {regions_config_path}")
            break
    if regions_config_path:
        with open(regions_config_path, 'r') as f:
            regions_data = json.load(f)
        print("Successfully loaded regions config.")
    else:
        raise FileNotFoundError("Core regions config file 'config/regions.json' not found.")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Exiting due to missing critical regions configuration.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading regions config: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- Main Processing Function ---
def extract_hourly_features():
    print("\n=====================================================")
    print("=== Starting Hourly NG MW & Feature Extraction (from Parquet) ===")
    print("=====================================================")
    start_total_time = time.time()
    successful_regions = []
    failed_regions = []

    if isinstance(regions_data, dict):
        regions_to_process = list(regions_data.keys())
    elif isinstance(regions_data, list):
        if all(isinstance(item, str) for item in regions_data):
            regions_to_process = regions_data
        elif all(isinstance(item, dict) and 'name' in item for item in regions_data):
            regions_to_process = [item['name'] for item in regions_data]
        else:
            print("Error: Regions list in config is not in a recognized format. Exiting.")
            sys.exit(1)
    else:
        print("Error: Regions configuration is not in an expected format. Exiting.")
        sys.exit(1)

    print(f"\nConfiguration loaded.")
    print(f"Regions to process: {regions_to_process}")

    if not regions_to_process:
        print("No regions found in the configuration file. Exiting.")
        return

    print(f"Total regions to process: {len(regions_to_process)}")

    for region_name in regions_to_process:
        region_upper = region_name.upper()
        print(f"\n\n################ Processing Region: {region_upper} ################")
        start_region_time = time.time()
        region_success = False

        try:
            region_base_dir = os.path.join(BASE_OUTPUT_DIR, region_name)
            final_dir = os.path.join(region_base_dir, "final")
            input_file = os.path.join(final_dir, "combined.parquet")
            output_file = os.path.join(final_dir, "hourly_ng_mw_features.csv")

            print(f"  Input file: {input_file}")
            print(f"  Output file: {output_file}")

            if not os.path.exists(input_file):
                print(f"  Error: Input file not found: {input_file}. Skipping region.")
                failed_regions.append(f"{region_name} (Input file missing: {os.path.basename(input_file)})")
                continue

            print(f"  Reading {input_file}...")
            date_col_name = 'date'
            ng_mw_col_name = f"{region_upper}_NG_MW"
            ng_ratio_col_name = f"{region_upper}_NG_RATIO"
            
            load_col_name = f"{region_upper}_LOAD"
            wat_mw_col_name = f"{region_upper}_WAT_MW"
            wnd_mw_col_name = f"{region_upper}_WND_MW"
            sun_mw_col_name = f"{region_upper}_SUN_MW"
            col_mw_col_name = f"{region_upper}_COL_MW"
            
            # Define the order of desired additional features (after NG_RATIO, before temp)
            ordered_additional_feature_templates = [
                load_col_name, wat_mw_col_name, wnd_mw_col_name, sun_mw_col_name, col_mw_col_name
            ]
            df_hourly = None

            try:
                df_hourly = pd.read_parquet(input_file)

                if date_col_name not in df_hourly.columns:
                    raise ValueError(f"Date column '{date_col_name}' not found.")
                if not pd.api.types.is_datetime64_any_dtype(df_hourly[date_col_name]):
                     print(f"    Warning: '{date_col_name}' not datetime. Attempting conversion.")
                     df_hourly[date_col_name] = pd.to_datetime(df_hourly[date_col_name], errors='coerce')
                
                initial_rows = len(df_hourly)
                df_hourly.dropna(subset=[date_col_name], inplace=True)
                if len(df_hourly) < initial_rows:
                    print(f"    Warning: Dropped {initial_rows - len(df_hourly)} rows with invalid '{date_col_name}'.")

                if df_hourly.empty:
                    print(f"  Input DataFrame empty after date processing. Skipping.")
                    failed_regions.append(f"{region_name} (Empty Input or all dates invalid)")
                    continue
                
                if ng_mw_col_name not in df_hourly.columns:
                    print(f"  Error: NG MW column '{ng_mw_col_name}' not found. Skipping region.")
                    failed_regions.append(f"{region_name} (Missing NG_MW column: {ng_mw_col_name})")
                    continue

            except (ValueError, KeyError) as e:
                print(f"  Error during initial processing of {os.path.basename(input_file)}: {e}")
                failed_regions.append(f"{region_name} (Parquet Read/Critical Column Error: {str(e)})")
                continue
            except Exception as e:
                print(f"  Unexpected Error reading/processing Parquet {input_file}: {e}")
                traceback.print_exc()
                failed_regions.append(f"{region_name} (Parquet Read/Processing Error)")
                continue
            
            print(f"  Successfully read and validated critical columns from {os.path.basename(input_file)}. Rows: {len(df_hourly)}.")

            # --- Dynamically build the list of columns to select based on existence ---
            selected_columns_for_output = [date_col_name, ng_mw_col_name] # Start with mandatory columns

            # Check for NG_RATIO
            if ng_ratio_col_name in df_hourly.columns:
                selected_columns_for_output.append(ng_ratio_col_name)
            else:
                print(f"  Info: Optional column '{ng_ratio_col_name}' not found. It will not be included.")

            # Check for other additional features in their defined order
            found_additional_features = []
            for col_template in ordered_additional_feature_templates:
                if col_template in df_hourly.columns:
                    selected_columns_for_output.append(col_template)
                    found_additional_features.append(col_template) # Keep track of found ones
                else:
                    print(f"  Info: Optional column '{col_template}' not found. It will not be included.")
            
            # Check for temperature columns
            temp_cols_found = sorted([col for col in df_hourly.columns if col.endswith('_temperature_2m')])
            if temp_cols_found:
                print(f"  Found temperature columns: {temp_cols_found}")
                selected_columns_for_output.extend(temp_cols_found)
            else:
                print(f"  Info: No columns ending with '_temperature_2m' found. They will not be included.")
            
            # Ensure uniqueness in selected_columns_for_output while preserving order
            selected_columns_for_output = list(dict.fromkeys(selected_columns_for_output))
            print(f"  Final list of columns to be included in output: {selected_columns_for_output}")

            # Create the processed DataFrame with only the selected, existing columns
            processed_df = df_hourly[selected_columns_for_output].copy()
            
            # --- Numeric Conversion for relevant existing columns ---
            # Define all column names that *should* be numeric if they exist (excluding date)
            potential_numeric_col_names = [ng_mw_col_name] # ng_mw is always there if we reach this point
            if ng_ratio_col_name in processed_df.columns: # Check processed_df as it's our source of truth now
                potential_numeric_col_names.append(ng_ratio_col_name)
            
            potential_numeric_col_names.extend(found_additional_features) # these are already confirmed to be in processed_df
            potential_numeric_col_names.extend(temp_cols_found)         # these are also confirmed to be in processed_df

            # Filter to unique names actually in processed_df and not 'date'
            numeric_cols_to_convert = sorted(list(set(
                col for col in potential_numeric_col_names if col in processed_df.columns and col != date_col_name
            )))

            print(f"  Converting to numeric (if not already): {numeric_cols_to_convert}")
            for col in numeric_cols_to_convert:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                nan_count = processed_df[col].isnull().sum()
                if nan_count > 0:
                    print(f"    Warning: Column '{col}' had {nan_count} values converted to NaN (due to non-numeric or missing data).")
            
            # The order of columns in processed_df is already set by selected_columns_for_output
            # which was built in the desired order.

            print(f"  Final columns in output DataFrame: {processed_df.columns.tolist()}")
            print(f"  Shape of DataFrame to save: {processed_df.shape}")

            if processed_df.empty and date_col_name in processed_df.columns and len(processed_df.columns) == 1 :
                 print(f"  Warning: Processed DataFrame for {region_name} only contains the date column.")
            elif processed_df.empty:
                 print(f"  Warning: Processed DataFrame for {region_name} is completely empty before saving.")

            processed_df.sort_values(by=date_col_name, inplace=True)

            print(f"  Saving hourly NG MW and features data to: {output_file}")
            os.makedirs(final_dir, exist_ok=True)
            processed_df.to_csv(output_file, index=False, float_format='%.4f', date_format='%Y-%m-%d %H:%M:%S')
            print(f"  Successfully saved hourly NG MW and features data for {region_upper}.")
            region_success = True

        except FileNotFoundError as e: # This might catch the input_file not found again, though handled earlier
            print(f"  Error: File not found during processing for {region_upper}: {e}")
            # Ensure it's added to failed_regions if not already
            if not any(region_name in fr for fr in failed_regions):
                 failed_regions.append(f"{region_name} (FileNotFound Error: {os.path.basename(str(e))})")
        except Exception as e:
            print(f"  ERROR: An unexpected error occurred processing region {region_upper}: {e}")
            traceback.print_exc()
            if not any(region_name in fr for fr in failed_regions):
                failed_regions.append(f"{region_name} (Unexpected Error: {type(e).__name__})")

        finally:
            if region_success:
                successful_regions.append(region_name)
            end_region_time = time.time()
            print(f"  Region processing time: {end_region_time - start_region_time:.2f} seconds.")
            print(f"################ End Processing Region: {region_upper} ################")

    end_total_time = time.time()
    total_duration = end_total_time - start_total_time

    print("\n\n=============================================")
    print("========== Hourly Feature Extraction Summary (from Parquet) ==========")
    print(f"Total regions attempted: {len(regions_to_process)}")
    print(f"Successfully processed regions: {len(successful_regions)}")
    print(f"Failed regions: {len(failed_regions)}")

    if failed_regions:
        print("\n--- Failure Details (Regions) ---")
        for fr_info in sorted(list(set(failed_regions))): # Use set to ensure unique error messages per region
            print(f"  - {fr_info}")
        print("\nPlease review the detailed logs.")
    elif len(successful_regions) == len(regions_to_process) and not failed_regions and len(regions_to_process) > 0:
        print("\nAll specified regions processed successfully.")
    elif not regions_to_process:
        print("\nNo regions were specified for processing.")
    else:
        print("\nProcessing finished. Review logs for details.")
        print(f"Successful: {successful_regions}")
        print(f"Failed: {failed_regions}")

    print(f"\nTotal execution time: {total_duration:.2f} seconds.")
    print("=============================================")

    if failed_regions:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    extract_hourly_features()