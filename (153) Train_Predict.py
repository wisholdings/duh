import pandas as pd
import os
import sys

# --- Configuration ---
# Base directory where all regional folders (california, central, etc.) are located
BASE_DATA_DIR = r"D:\EIA_STACK"

# List of all regions to process
REGIONS = [
    "california", "carolina", "central", "florida", "midatlantic", "midwest",
    "newengland", "newyork", "northwest", "southeast", "southwest", "tennessee", "texas"
]

# The different consumption types to process for each region
CONSUMPTION_TYPES = ["residential", "commercial", "industrial"]


def split_file_for_region(region_name, consumption_type, base_dir):
    """
    Loads a specific regional consumption file, splits it into train/predict sets,
    and saves the output files to the correct prediction folder.
    """
    region_upper = region_name.upper()
    consumption_cap = consumption_type.capitalize()

    # --- Dynamically define file paths and column names ---
    # Input comes from the 'final' subfolder where the previous script saved its output
    input_dir = os.path.join(base_dir, region_name, "final")
    input_file_name = f"{region_name}_{consumption_type}_mmcf_WEATHER_FEATURES.csv"
    input_filepath = os.path.join(input_dir, input_file_name)

    # Output goes to the new 'L48_Predictions' subfolder
    output_dir = os.path.join(base_dir, region_name, "L48_Predictions")
    train_output_file = os.path.join(output_dir, f"{region_name}_{consumption_type}_train.csv")
    predict_output_file = os.path.join(output_dir, f"{region_name}_{consumption_type}_predict.csv")

    # The target column to split on (e.g., 'FLORIDA_Residential_hourly_MMcf')
    target_col = f"{region_upper}_{consumption_cap}_hourly_MMcf"

    print(f"\n--- Processing: {region_upper} - {consumption_cap} ---")

    # 1. LOAD DATA
    try:
        df_full = pd.read_csv(input_filepath, parse_dates=['datetime'])
        df_full = df_full.sort_values("datetime").reset_index(drop=True)
        print(f"  [1] Loaded data from: {input_filepath}")
        
        if target_col not in df_full.columns:
            print(f"  -> FATAL ERROR: Target column '{target_col}' not found. Skipping file.")
            return

    except FileNotFoundError:
        print(f"  -> WARNING: Input file not found. Skipping.")
        return
    except Exception as e:
        print(f"  -> FATAL ERROR: Could not load the file. Reason: {e}. Skipping.")
        return

    # 2. FIND THE SPLIT POINT
    print(f"  [2] Finding the first empty value in '{target_col}'...")
    try:
        # idxmax() on a boolean series finds the index of the first 'True' value
        first_nan_index = df_full[target_col].isnull().idxmax()
        
        # A check to ensure NaNs were actually found. If no NaNs, idxmax returns 0.
        if first_nan_index == 0 and not df_full[target_col].isnull().iloc[0]:
            print("  -> ERROR: No empty values found in target column. Cannot split. Skipping file.")
            return
        
        print(f"  -> Split point found at index: {first_nan_index}")

    except Exception as e:
        print(f"  -> ERROR: Could not determine split point. Reason: {e}. Skipping file.")
        return

    # 3. SPLIT THE DATAFRAME
    print(f"  [3] Splitting the data...")
    df_train = df_full.iloc[:first_nan_index].copy()
    df_predict = df_full.iloc[first_nan_index:].copy()
    print(f"  -> Train set shape: {df_train.shape}")
    print(f"  -> Predict set shape: {df_predict.shape}")

    # 4. SAVE THE FILES
    print(f"  [4] Saving the new CSV files...")
    try:
        # Ensure the output directory exists before saving
        os.makedirs(output_dir, exist_ok=True)
        
        df_train.to_csv(train_output_file, index=False, date_format='%Y-%m-%d %H:%M:%S')
        print(f"  -> Training data saved to: {train_output_file}")
        
        df_predict.to_csv(predict_output_file, index=False, date_format='%Y-%m-%d %H:%M:%S')
        print(f"  -> Prediction data saved to: {predict_output_file}")

    except Exception as e:
        print(f"  -> ERROR: Could not save files. Reason: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    print("="*60)
    print("--- SCRIPT TO CREATE TRAIN/PREDICT FILES FOR ALL REGIONS ---")
    print(f"--- Base Directory: {BASE_DATA_DIR} ---")
    print("="*60)

    for region in REGIONS:
        for consumption in CONSUMPTION_TYPES:
            split_file_for_region(region, consumption, BASE_DATA_DIR)

    print("\n" + "="*60)
    print("--- All Regions and Types Processed ---")
    print("="*60)