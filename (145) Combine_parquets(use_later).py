# -*- coding: utf-8 -*-
import os
import pandas as pd
import sys
import json
import traceback

# Reuse the BASE_OUTPUT_DIR loading logic
try:
    # Attempt to get the directory of the current script
    script_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
    parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from config.settings import BASE_OUTPUT_DIR
    print(f"Loaded BASE_OUTPUT_DIR from config: {BASE_OUTPUT_DIR}")
except ImportError:
    print("Could not import BASE_OUTPUT_DIR from config.settings. Using default.")
    # Fallback if __file__ is not defined or structure is different
    script_dir_fallback = os.getcwd()
    BASE_OUTPUT_DIR = os.path.abspath(os.path.join(script_dir_fallback, '..', 'EIA_STACK'))
    print(f"Using default BASE_OUTPUT_DIR: {BASE_OUTPUT_DIR}")
except NameError: # This can happen if __file__ is not defined (e.g. in some interactive environments)
    print("Could not determine script directory using __file__. Using current working directory for default BASE_OUTPUT_DIR.")
    BASE_OUTPUT_DIR = os.path.abspath(os.path.join(os.getcwd(), '..', 'EIA_STACK')) # Fallback to CWD based path
    print(f"Using default BASE_OUTPUT_DIR: {BASE_OUTPUT_DIR}")


# Load regions from config file
try:
    script_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
    project_root_guess1 = os.path.abspath(os.path.join(script_dir, '..'))
    regions_config_path = os.path.join(project_root_guess1, 'config', 'regions.json')

    if not os.path.exists(regions_config_path):
        # Fallback for cases where script might be in a sub-sub-directory or structure is different
        project_root_guess2 = os.path.abspath(os.path.join(script_dir, '..', '..'))
        regions_config_path = os.path.join(project_root_guess2, 'config', 'regions.json')
    
    if not os.path.exists(regions_config_path):
        # Final fallback to current working directory structure (if script is run from project root)
        regions_config_path = os.path.join('config', 'regions.json')

    if not os.path.exists(regions_config_path):
        print(f"Error: Core regions config file 'config/regions.json' not found at expected paths. Exiting.")
        sys.exit(1)

    with open(regions_config_path, 'r') as f:
        regions = json.load(f)
    print(f"Loaded regions from {regions_config_path}")
except FileNotFoundError:
    print(f"Error: Core regions config file 'config/regions.json' not found. Exiting.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading regions config: {e}. Exiting.")
    sys.exit(1)

# Process each region
for region_key in regions: # Iterate through keys if regions is a dict, or items if it's a list of strings
    region_name = region_key # Assuming region_key is the string name like "PJM", "MISO" etc.
    print(f"\n--- Processing region: {region_name.upper()} ---")
    try:
        # Define paths for train, predict, and combined Parquet files
        train_path = os.path.join(BASE_OUTPUT_DIR, region_name, 'final', 'train.parquet')
        predict_path = os.path.join(BASE_OUTPUT_DIR, region_name, 'final', 'predict.parquet')
        combined_path = os.path.join(BASE_OUTPUT_DIR, region_name, 'final', 'combined.parquet')

        # Check if both input files exist
        if not os.path.exists(train_path):
            print(f"Train Parquet file not found for region {region_name} at {train_path}. Skipping.")
            continue
        if not os.path.exists(predict_path):
            print(f"Predict Parquet file not found for region {region_name} at {predict_path}. Skipping.")
            continue

        # Load the Parquet files
        print(f"Loading train and predict Parquet files for {region_name}...")
        train_df = pd.read_parquet(train_path)
        predict_df = pd.read_parquet(predict_path)
        print(f"  Train data shape: {train_df.shape}")
        print(f"  Predict data shape: {predict_df.shape}")

        # Combine the DataFrames
        print("Combining DataFrames...")
        combined_df = pd.concat([train_df, predict_df], ignore_index=True)
        print(f"  Combined shape before fill: {combined_df.shape}")

        # Check for NaNs before filling
        nan_counts_before = combined_df.isnull().sum()
        nan_columns_before = nan_counts_before[nan_counts_before > 0]
        if not nan_columns_before.empty:
            print(f"  NaN counts before filling for region {region_name}:")
            print(nan_columns_before)
        else:
            print(f"  No NaNs found before filling for region {region_name}.")

        # Forward fill NaNs, then backward fill remaining NaNs
        print("Applying forward fill (ffill) and backward fill (bfill) to handle NaNs...")
        combined_df.ffill(inplace=True)
        combined_df.bfill(inplace=True)
        
        # Check for NaNs after filling
        nan_counts_after = combined_df.isnull().sum().sum() # Sum of all NaNs in the DataFrame
        if nan_counts_after > 0:
            print(f"  Warning: NaNs still present after ffill and bfill for region {region_name}! Count: {nan_counts_after}")
            print(combined_df.isnull().sum()[combined_df.isnull().sum() > 0]) # Show columns with remaining NaNs
        else:
            print(f"  Successfully filled all NaNs for region {region_name}.")


        # Save the combined DataFrame to a new Parquet file
        combined_df.to_parquet(combined_path, index=False)
        print(f"Successfully combined, filled, and saved data for {region_name} into {combined_path}")
        print(f"Combined file has {len(combined_df)} rows (train: {len(train_df)}, predict: {len(predict_df)})")

    except Exception as e:
        print(f"Error processing region {region_name}: {e}")
        print(traceback.format_exc())
        continue

print("\nCombining process complete for all regions.")