import os
import sys
import pandas as pd
import numpy as np
import warnings

# --- Main Configuration ---
# This is the root directory where all the regional folders are.
BASE_DATA_DIR = r"D:\EIA_STACK"

# --- 2. DEFINE WEATHER BASELINES FOR DEMAND SHAPES ---
HEATING_BASELINE_TEMP_F = 65.0 # Demand increases when temp is BELOW this
COOLING_BASELINE_TEMP_F = 75.0 # Demand increases when temp is ABOVE this

# --- Pandas Options ---
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


def disaggregate_gas(df_group, monthly_gas_col, hourly_shape_col):
    """
    Disaggregates a monthly gas total into hourly values based on a shape column.
    """
    if df_group[monthly_gas_col].isnull().all():
        df_group['hourly_gas_burn'] = np.nan
        return df_group

    monthly_gas_total = df_group[monthly_gas_col].iloc[0]
    hourly_shape = df_group[hourly_shape_col]
    monthly_shape_sum = hourly_shape.sum()

    if monthly_shape_sum > 0 and monthly_gas_total > 0:
        # Distribute monthly total proportionally to the hourly shape
        df_group['hourly_gas_burn'] = monthly_gas_total * (hourly_shape / monthly_shape_sum)
    else: 
        # If there's no weather demand or no consumption, distribute evenly across hours
        num_hours_in_month = len(df_group)
        if num_hours_in_month > 0:
            df_group['hourly_gas_burn'] = monthly_gas_total / num_hours_in_month
        else:
            df_group['hourly_gas_burn'] = 0

    return df_group

def process_all_regions():
    """
    Main processing loop. Dynamically finds regions and weather columns,
    then disaggregates monthly gas consumption to hourly.
    """
    print("\n=========================================================")
    print("=== Processing Regional Hourly Gas Consumption Data ===")
    print("=========================================================")
    
    # Dynamically find all subdirectories in the base data directory
    try:
        region_names = [d for d in os.listdir(BASE_DATA_DIR) if os.path.isdir(os.path.join(BASE_DATA_DIR, d))]
        if not region_names:
            print(f"FATAL: No region subdirectories found in {BASE_DATA_DIR}. Exiting.")
            return
    except FileNotFoundError:
        print(f"FATAL: Base data directory not found at {BASE_DATA_DIR}. Exiting.")
        return

    # Loop through each discovered region
    for region_name in region_names:
        
        region_upper = region_name.upper()
        print(f"\n\n--- Processing Region: {region_upper} ---")

        # --- Define Paths for the Current Region ---
        DATA_DIRECTORY = os.path.join(BASE_DATA_DIR, region_name, "final")
        HOURLY_FEATURES_FILE = os.path.join(DATA_DIRECTORY, "hourly_mw_features.csv")
        MONTHLY_CONSUMPTION_FILE = os.path.join(DATA_DIRECTORY, f"{region_name}_res_com_ind.csv")
        OUTPUT_DIR = DATA_DIRECTORY

        # --- 1. Validate Input Files ---
        if not os.path.exists(MONTHLY_CONSUMPTION_FILE):
            print(f"  FATAL for {region_upper}: Monthly consumption file not found at: {MONTHLY_CONSUMPTION_FILE}"); continue
        if not os.path.exists(HOURLY_FEATURES_FILE):
            print(f"  FATAL for {region_upper}: Hourly features file not found at: {HOURLY_FEATURES_FILE}"); continue

        # --- 2. Dynamically Discover Temperature Columns ---
        try:
            print(f"  Discovering temperature columns from: {HOURLY_FEATURES_FILE}")
            header_df = pd.read_csv(HOURLY_FEATURES_FILE, nrows=0)
            date_col_name = next((col for col in header_df.columns if 'date' in col.lower() or 'time' in col.lower()), None)
            if not date_col_name:
                print(f"  FATAL for {region_upper}: No date/time column found in hourly features file."); continue
            
            weather_cols = [col for col in header_df.columns if '_temperature_2m' in col]
            if not weather_cols:
                print(f"  WARNING: No '_temperature_2m' columns found for {region_upper}. Skipping region.")
                continue
            print(f"  Found {len(weather_cols)} temperature columns.")
        except Exception as e:
            print(f"  FATAL for {region_upper}: Could not read header of hourly features file. Error: {e}"); continue

        # --- 3. Load and Prepare Monthly Consumption Data ---
        print(f"  Loading monthly consumption from: {MONTHLY_CONSUMPTION_FILE}")
        df_monthly_gas = pd.read_csv(MONTHLY_CONSUMPTION_FILE)
        monthly_date_col = next((col for col in df_monthly_gas.columns if 'date' in col.lower() or 'time' in col.lower()), None)
        if monthly_date_col:
            df_monthly_gas.rename(columns={monthly_date_col: 'datetime'}, inplace=True)
            df_monthly_gas['datetime'] = pd.to_datetime(df_monthly_gas['datetime'])
        else:
            print(f"  FATAL for {region_upper}: No date column found in {os.path.basename(MONTHLY_CONSUMPTION_FILE)}"); continue
        
        # --- 4. Robustly Interpolate Monthly Data ---
        consumption_cols_to_process = [c for c in df_monthly_gas.columns if c.startswith(region_upper) and c.endswith("_MMcf")]
        df_monthly_gas.set_index('datetime', inplace=True)
        
        # Ensure a complete monthly index to handle missing months
        full_date_range = pd.date_range(start=df_monthly_gas.index.min(), end=df_monthly_gas.index.max(), freq='MS')
        df_monthly_gas = df_monthly_gas.reindex(full_date_range)

        # Fill missing values using seasonal forward/backward fill
        for col in consumption_cols_to_process:
            if df_monthly_gas[col].isnull().any():
                df_monthly_gas[col] = df_monthly_gas.groupby(df_monthly_gas.index.month)[col].transform(lambda x: x.bfill().ffill())
        
        df_monthly_gas.reset_index(inplace=True); df_monthly_gas.rename(columns={'index': 'datetime'}, inplace=True)
        print("  Monthly data interpolation complete.")

        # --- 5. Load Hourly Weather Data ---
        try:
            print(f"  Loading hourly weather data...")
            df_hourly_weather = pd.read_csv(HOURLY_FEATURES_FILE, usecols=[date_col_name] + weather_cols)
            df_hourly_weather.rename(columns={date_col_name: 'datetime'}, inplace=True)
            df_hourly_weather['datetime'] = pd.to_datetime(df_hourly_weather['datetime'])
        except Exception as e:
            print(f"  FATAL for {region_upper}: Could not load hourly weather data. Error: {e}"); continue

        # --- 6. Create Weather Shapes and Disaggregate ---
        print("  Creating weather-based demand shapes...")
        df_hourly_weather['avg_temp'] = df_hourly_weather[weather_cols].mean(axis=1)
        df_hourly_weather['heating_shape'] = (HEATING_BASELINE_TEMP_F - df_hourly_weather['avg_temp']).clip(lower=0)
        df_hourly_weather['cooling_shape'] = (df_hourly_weather['avg_temp'] - COOLING_BASELINE_TEMP_F).clip(lower=0)
        df_hourly_weather['weather_shape'] = df_hourly_weather['heating_shape'] + df_hourly_weather['cooling_shape']
        df_hourly_weather['even_shape'] = 1 # For industrial load

        os.makedirs(OUTPUT_DIR, exist_ok=True) 

        for monthly_gas_col_name in consumption_cols_to_process:
            print(f"\n  --- Processing: {monthly_gas_col_name} ---")

            # Prepare data for merging by creating a common 'month' key
            df_hourly_weather['month'] = df_hourly_weather['datetime'].dt.to_period("M")
            df_monthly_gas['month'] = df_monthly_gas['datetime'].dt.to_period("M")

            # Merge monthly totals onto the hourly dataframe
            df_merged = pd.merge(df_hourly_weather, df_monthly_gas[['month', monthly_gas_col_name]], on="month", how="left")
            
            # Choose the correct shape based on consumption type
            if 'Industrial' in monthly_gas_col_name:
                shape_to_use = 'even_shape'
                print("    -> Using EVEN distribution for Industrial consumption.")
            else:
                shape_to_use = 'weather_shape'
                print("    -> Using combined HEATING+COOLING shape for Residential/Commercial.")

            # Apply the disaggregation function to each month's data
            df_processed = df_merged.groupby("month", group_keys=False).apply(
                disaggregate_gas, monthly_gas_col=monthly_gas_col_name, hourly_shape_col=shape_to_use
            )

            # --- 7. Save Output File ---
            output_col_name = monthly_gas_col_name.replace('_MMcf', '_hourly_MMcf')
            df_processed.rename(columns={'hourly_gas_burn': output_col_name}, inplace=True)

            final_cols_to_keep = ['datetime'] + weather_cols + [output_col_name]
            df_final = df_processed[final_cols_to_keep]

            output_filename = f"{monthly_gas_col_name.lower()}_weather_features.csv"
            output_filepath = os.path.join(OUTPUT_DIR, output_filename)

            df_final.to_csv(output_filepath, index=False, float_format='%.6f', date_format='%Y-%m-%d %H:%M:%S')
            print(f"    Successfully saved clean file: {output_filepath}")

    print("\n=========================================================")
    print("=== All Regions Processed ===")
    print("=========================================================")

if __name__ == "__main__":
    process_all_regions()