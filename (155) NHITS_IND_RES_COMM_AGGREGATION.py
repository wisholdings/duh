# -*- coding: utf-8 -*-
import pandas as pd
import os
import sys
from functools import reduce

# --- Configuration ---
# The directory where the HYBRID scripts saved all their results
RESULTS_DIR = r"D:\EIA_STACK\HYBRID_RESULTS"  # Changed from NHITS_RESULTS
# The final directory for the aggregated output files
FINAL_OUTPUT_DIR = r"D:\EIA_STACK"

# Lists to iterate through
REGIONS = [
    "california", "carolina", "central", "florida", "midatlantic", "midwest",
    "newengland", "newyork", "northwest", "southeast", "southwest", "tennessee", "texas"
]
CONSUMPTION_TYPES = ["residential", "commercial", "industrial"]

def find_forecast_file(region_name, consumption_type, results_dir):
    """
    Constructs the expected filename for a region's daily forecast CSV
    based on the simplified '{region}_{consumption}.csv' naming convention.
    """
    region_upper = region_name.upper()
    
    # The directory structure for a specific region's prediction files
    predictions_dir = os.path.join(results_dir, region_upper, "Predictions")

    # Construct the simple filename (matching hybrid model output)
    simple_filename = f"{region_name}_{consumption_type}.csv"
    
    expected_file = os.path.join(predictions_dir, simple_filename)
    
    if os.path.exists(expected_file):
        return expected_file
    
    # Fallback: Try with "_hybrid" suffix in case it was named that way
    hybrid_filename = f"{region_name}_{consumption_type}_final_hybrid_forecast.csv"
    hybrid_file = os.path.join(predictions_dir, hybrid_filename)
    
    if os.path.exists(hybrid_file):
        return hybrid_file
    
    # Return None if the file is not found
    return None


# --- Main Execution Block ---
if __name__ == "__main__":
    print("="*60)
    print("--- Consolidating All Regional Daily Forecasts ---")
    print("--- FROM HYBRID MODEL RESULTS ---")
    print(f"--- Input Directory: {RESULTS_DIR}")
    print(f"--- Output Directory: {FINAL_OUTPUT_DIR}")
    print("="*60)

    # Track success/failure
    successful_loads = []
    failed_loads = []

    # Loop through each consumption type to create a separate consolidated file for each
    for consumption in CONSUMPTION_TYPES:
        consumption_cap = consumption.capitalize()
        print(f"\n\n{'='*20} Processing: {consumption_cap} {'='*20}")
        
        all_regional_dfs = []
        loaded_regions = []

        # 1. LOAD EACH REGIONAL FORECAST FOR THE CURRENT CONSUMPTION TYPE
        print(f"\n[1] Searching for {consumption_cap} regional forecast files...")
        for region in REGIONS:
            # Call the function to find the correctly named file
            forecast_file_path = find_forecast_file(region, consumption, RESULTS_DIR)
            
            if forecast_file_path:
                try:
                    df_region = pd.read_csv(forecast_file_path, parse_dates=['datetime'])
                    
                    # Dynamically find the region's BCF column
                    # e.g., 'CALIFORNIA_Residential_hourly_MMcf_BCF'
                    bcf_col_name = f"{region.upper()}_{consumption_cap}_hourly_MMcf_BCF"
                    
                    if bcf_col_name not in df_region.columns:
                        print(f"    -> WARNING: BCF column '{bcf_col_name}' not found in {os.path.basename(forecast_file_path)}")
                        print(f"       Available columns: {df_region.columns.tolist()}")
                        continue

                    # Keep only the date and BCF value, and rename the column to just the region name
                    df_region_clean = df_region[['datetime', bcf_col_name]].rename(columns={bcf_col_name: region})
                    
                    all_regional_dfs.append(df_region_clean)
                    loaded_regions.append(region)
                    successful_loads.append(f"{region}-{consumption}")
                    print(f"    -> ✓ Successfully loaded: {region.capitalize()}")
                    
                except Exception as e:
                    print(f"    -> ✗ ERROR processing {region.capitalize()}: {str(e)[:100]}")
                    failed_loads.append(f"{region}-{consumption}")
            else:
                print(f"    -> ✗ File not found for {region.capitalize()}")
                failed_loads.append(f"{region}-{consumption}")

        # 2. MERGE ALL DATAFRAMES INTO ONE
        if not all_regional_dfs:
            print(f"\n✗ FATAL ERROR: No regional forecast files were found for {consumption_cap}. Skipping this type.")
            continue
            
        print(f"\n[2] Merging {len(all_regional_dfs)} regional {consumption_cap} forecasts...")
        print(f"    Regions included: {', '.join(loaded_regions)}")
        
        # Use functools.reduce to efficiently merge the list of DataFrames
        consolidated_df = reduce(lambda left, right: pd.merge(left, right, on='datetime', how='outer'), all_regional_dfs)
        consolidated_df = consolidated_df.sort_values('datetime').reset_index(drop=True)
        
        # Fill NaN with 0 (regions might have different date ranges)
        consolidated_df = consolidated_df.fillna(0)
        print(f"    -> Merging complete. Consolidated shape: {consolidated_df.shape}")
        
        # Show date range
        date_range = f"{consolidated_df['datetime'].min().strftime('%Y-%m-%d')} to {consolidated_df['datetime'].max().strftime('%Y-%m-%d')}"
        print(f"    -> Date range: {date_range}")

        # 3. CALCULATE THE TOTAL SUM
        print("\n[3] Calculating the total L48 sum...")
        # Get all columns that are region names (i.e., not 'datetime')
        regional_cols = [col for col in consolidated_df.columns if col != 'datetime']
        total_col_name = f"Total_L48_{consumption_cap}_BCF"
        consolidated_df[total_col_name] = consolidated_df[regional_cols].sum(axis=1)
        
        # Calculate some statistics
        total_mean = consolidated_df[total_col_name].mean()
        total_max = consolidated_df[total_col_name].max()
        total_min = consolidated_df[total_col_name].min()
        
        print(f"    -> '{total_col_name}' column created.")
        print(f"       Mean: {total_mean:.2f} BCF/day")
        print(f"       Max:  {total_max:.2f} BCF/day")
        print(f"       Min:  {total_min:.2f} BCF/day")

        # 4. SAVE THE FINAL CONSOLIDATED FILE
        final_output_file = os.path.join(FINAL_OUTPUT_DIR, f"consolidated_daily_{consumption}_forecasts_hybrid.csv")
        print(f"\n[4] Saving the final {consumption_cap} consolidated file...")
        try:
            consolidated_df.to_csv(final_output_file, index=False, float_format='%.4f')
            print(f"    -> ✓ Success! Final file saved to: {final_output_file}")
            
            # Also save a summary version with just datetime and total
            summary_df = consolidated_df[['datetime', total_col_name]]
            summary_file = os.path.join(FINAL_OUTPUT_DIR, f"summary_daily_{consumption}_L48_hybrid.csv")
            summary_df.to_csv(summary_file, index=False, float_format='%.4f')
            print(f"    -> ✓ Summary file saved to: {summary_file}")
            
        except Exception as e:
            print(f"    -> ✗ FATAL ERROR: Could not save the final file. Reason: {e}")

    # 5. CREATE A MASTER CONSOLIDATED FILE (ALL SECTORS COMBINED)
    print("\n" + "="*60)
    print("\n[5] Creating master consolidated file (all sectors)...")
    
    master_dfs = []
    for consumption in CONSUMPTION_TYPES:
        consumption_cap = consumption.capitalize()
        summary_file = os.path.join(FINAL_OUTPUT_DIR, f"summary_daily_{consumption}_L48_hybrid.csv")
        
        if os.path.exists(summary_file):
            df_sector = pd.read_csv(summary_file, parse_dates=['datetime'])
            # Rename the total column to include sector name
            df_sector = df_sector.rename(columns={
                f"Total_L48_{consumption_cap}_BCF": f"L48_{consumption_cap}_BCF"
            })
            master_dfs.append(df_sector)
            print(f"    -> Loaded {consumption_cap} summary")
    
    if master_dfs:
        # Merge all sector summaries
        master_df = reduce(lambda left, right: pd.merge(left, right, on='datetime', how='outer'), master_dfs)
        master_df = master_df.sort_values('datetime').reset_index(drop=True)
        master_df = master_df.fillna(0)
        
        # Calculate grand total
        sector_cols = [col for col in master_df.columns if col != 'datetime']
        master_df['Total_L48_All_Sectors_BCF'] = master_df[sector_cols].sum(axis=1)
        
        # Save master file
        master_file = os.path.join(FINAL_OUTPUT_DIR, "MASTER_daily_L48_all_sectors_hybrid.csv")
        master_df.to_csv(master_file, index=False, float_format='%.4f')
        print(f"    -> ✓ Master file saved to: {master_file}")
        
        # Print summary statistics
        print("\n--- MASTER SUMMARY STATISTICS ---")
        for col in sector_cols + ['Total_L48_All_Sectors_BCF']:
            mean_val = master_df[col].mean()
            print(f"    {col}: {mean_val:.2f} BCF/day average")

    print("\n" + "="*60)
    print("--- Aggregation Script Summary ---")
    print(f"Successful loads: {len(successful_loads)}")
    print(f"Failed loads: {len(failed_loads)}")
    
    if failed_loads:
        print("\nFailed to load:")
        for fail in failed_loads:
            print(f"  - {fail}")
    
    print("\n--- Aggregation Script Completed ---")
    print("="*60)