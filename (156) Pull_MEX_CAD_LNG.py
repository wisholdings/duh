# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import sys
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
# The directory where the HYBRID scripts saved all their results
RESULTS_DIR = r"D:\EIA_STACK\HYBRID_RESULTS"
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
    Finds the forecast file for a given region and consumption type.
    Hybrid models use simple naming: {region}_{consumption}.csv
    """
    region_upper = region_name.upper()
    
    # The directory structure for a specific region's prediction files
    predictions_dir = os.path.join(results_dir, region_upper, "Predictions")

    # Try the simple naming convention first
    simple_filename = f"{region_name}_{consumption_type}.csv"
    expected_file = os.path.join(predictions_dir, simple_filename)
    
    if os.path.exists(expected_file):
        return expected_file
    
    # Try with "_hybrid" suffix as fallback
    hybrid_filename = f"{region_name}_{consumption_type}_hybrid.csv"
    hybrid_file = os.path.join(predictions_dir, hybrid_filename)
    
    if os.path.exists(hybrid_file):
        return hybrid_file
    
    return None

def generate_summary_plots(consolidated_df, consumption_type, output_dir):
    """Generate summary plots for the consolidated forecasts."""
    consumption_cap = consumption_type.capitalize()
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(20, 12))
    
    # Get regional columns and total column
    regional_cols = [col for col in consolidated_df.columns if col != 'datetime' and 'Total' not in col]
    total_col = f"Total_L48_{consumption_cap}_BCF"
    
    # Plot 1: Total L48 consumption over time
    ax1 = axes[0]
    ax1.plot(consolidated_df['datetime'], consolidated_df[total_col], 
             linewidth=2.5, color='darkblue', label=f'Total L48 {consumption_cap}')
    ax1.fill_between(consolidated_df['datetime'], 0, consolidated_df[total_col], 
                     alpha=0.3, color='lightblue')
    ax1.set_title(f'Total L48 {consumption_cap} Natural Gas Consumption Forecast', 
                  fontsize=16, fontweight='bold')
    ax1.set_ylabel('Daily Consumption (BCF)', fontsize=12)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Add monthly average line
    df_monthly = consolidated_df.set_index('datetime')[total_col].resample('M').mean()
    ax1.plot(df_monthly.index, df_monthly.values, 
             linewidth=2, color='red', alpha=0.7, linestyle='--', label='Monthly Average')
    
    # Plot 2: Stacked area chart of regional contributions
    ax2 = axes[1]
    
    # Sort regions by average consumption for better visualization
    region_averages = {col: consolidated_df[col].mean() for col in regional_cols}
    sorted_regions = sorted(regional_cols, key=lambda x: region_averages[x], reverse=True)
    
    # Create stacked area plot
    bottom = np.zeros(len(consolidated_df))
    colors = plt.cm.Set3(np.linspace(0, 1, len(sorted_regions)))
    
    for i, region in enumerate(sorted_regions):
        ax2.fill_between(consolidated_df['datetime'], bottom, 
                        bottom + consolidated_df[region].values,
                        alpha=0.8, color=colors[i], label=region.capitalize())
        bottom += consolidated_df[region].values
    
    ax2.set_title(f'Regional Contributions to L48 {consumption_cap} Consumption', 
                  fontsize=16, fontweight='bold')
    ax2.set_ylabel('Daily Consumption (BCF)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = os.path.join(output_dir, f"consolidated_{consumption_type}_summary_plot.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_filename

# --- Main Execution Block ---
if __name__ == "__main__":
    print("="*70)
    print("   CONSOLIDATING HYBRID MODEL REGIONAL FORECASTS")
    print("="*70)
    print(f"Input Directory:  {RESULTS_DIR}")
    print(f"Output Directory: {FINAL_OUTPUT_DIR}")
    print("="*70)

    # Track overall progress
    all_successful = []
    all_failed = []
    
    # Store all consolidated dataframes for master file creation
    all_sector_dfs = {}

    # Loop through each consumption type
    for consumption in CONSUMPTION_TYPES:
        consumption_cap = consumption.capitalize()
        print(f"\n{'='*30} {consumption_cap} Sector {'='*30}")
        
        all_regional_dfs = []
        loaded_regions = []
        missing_regions = []

        # 1. LOAD EACH REGIONAL FORECAST
        print(f"\n[1] Loading {consumption_cap} regional forecasts...")
        
        for region in REGIONS:
            forecast_file_path = find_forecast_file(region, consumption, RESULTS_DIR)
            
            if forecast_file_path:
                try:
                    # Load the CSV
                    df_region = pd.read_csv(forecast_file_path, parse_dates=['datetime'])
                    
                    # Find the BCF column (format: REGION_Consumption_hourly_MMcf_BCF)
                    bcf_col_name = f"{region.upper()}_{consumption_cap}_hourly_MMcf_BCF"
                    
                    if bcf_col_name not in df_region.columns:
                        # Try to find any column with BCF in it
                        bcf_columns = [col for col in df_region.columns if 'BCF' in col and 'datetime' not in col]
                        if bcf_columns:
                            bcf_col_name = bcf_columns[0]
                            print(f"    ⚠ Using alternate column: {bcf_col_name}")
                        else:
                            print(f"    ✗ {region.capitalize()}: No BCF column found")
                            missing_regions.append(region)
                            continue

                    # Extract and rename
                    df_region_clean = df_region[['datetime', bcf_col_name]].rename(
                        columns={bcf_col_name: region}
                    )
                    
                    all_regional_dfs.append(df_region_clean)
                    loaded_regions.append(region)
                    all_successful.append(f"{region}-{consumption}")
                    print(f"    ✓ {region.capitalize()}: Loaded successfully")
                    
                except Exception as e:
                    print(f"    ✗ {region.capitalize()}: Error - {str(e)[:60]}")
                    missing_regions.append(region)
                    all_failed.append(f"{region}-{consumption}")
            else:
                print(f"    ✗ {region.capitalize()}: File not found")
                missing_regions.append(region)
                all_failed.append(f"{region}-{consumption}")

        # 2. MERGE ALL DATAFRAMES
        if not all_regional_dfs:
            print(f"\n⚠ WARNING: No files found for {consumption_cap}. Skipping...")
            continue
            
        print(f"\n[2] Merging {len(all_regional_dfs)}/{len(REGIONS)} regional forecasts...")
        
        if missing_regions:
            print(f"    Missing regions: {', '.join(missing_regions)}")
        
        # Merge using outer join to handle different date ranges
        consolidated_df = reduce(
            lambda left, right: pd.merge(left, right, on='datetime', how='outer'), 
            all_regional_dfs
        )
        consolidated_df = consolidated_df.sort_values('datetime').reset_index(drop=True)
        
        # Fill NaN with 0 (missing regions or dates)
        consolidated_df = consolidated_df.fillna(0)
        
        print(f"    Shape: {consolidated_df.shape}")
        print(f"    Date range: {consolidated_df['datetime'].min().date()} to {consolidated_df['datetime'].max().date()}")

        # 3. CALCULATE TOTAL
        print("\n[3] Calculating L48 total...")
        regional_cols = [col for col in consolidated_df.columns if col != 'datetime']
        total_col_name = f"Total_L48_{consumption_cap}_BCF"
        consolidated_df[total_col_name] = consolidated_df[regional_cols].sum(axis=1)
        
        # Calculate statistics
        stats = {
            'mean': consolidated_df[total_col_name].mean(),
            'std': consolidated_df[total_col_name].std(),
            'max': consolidated_df[total_col_name].max(),
            'min': consolidated_df[total_col_name].min(),
            'median': consolidated_df[total_col_name].median()
        }
        
        print(f"    {total_col_name}:")
        print(f"      Mean:   {stats['mean']:.2f} BCF/day")
        print(f"      Median: {stats['median']:.2f} BCF/day")
        print(f"      Std:    {stats['std']:.2f} BCF/day")
        print(f"      Range:  {stats['min']:.2f} - {stats['max']:.2f} BCF/day")

        # 4. SAVE FILES
        print(f"\n[4] Saving {consumption_cap} files...")
        
        # Full consolidated file
        full_output_file = os.path.join(FINAL_OUTPUT_DIR, f"consolidated_daily_{consumption}_hybrid.csv")
        consolidated_df.to_csv(full_output_file, index=False, float_format='%.4f')
        print(f"    ✓ Full file: {os.path.basename(full_output_file)}")
        
        # Summary file (just datetime and total)
        summary_df = consolidated_df[['datetime', total_col_name]].copy()
        summary_file = os.path.join(FINAL_OUTPUT_DIR, f"summary_L48_{consumption}_hybrid.csv")
        summary_df.to_csv(summary_file, index=False, float_format='%.4f')
        print(f"    ✓ Summary: {os.path.basename(summary_file)}")
        
        # Store for master file
        all_sector_dfs[consumption] = summary_df
        
        # Generate plots
        try:
            plot_file = generate_summary_plots(consolidated_df, consumption, FINAL_OUTPUT_DIR)
            print(f"    ✓ Plot: {os.path.basename(plot_file)}")
        except Exception as e:
            print(f"    ⚠ Plot generation failed: {str(e)[:60]}")

    # 5. CREATE MASTER FILE (ALL SECTORS COMBINED)
    print("\n" + "="*70)
    print("\n[5] Creating MASTER consolidated file...")
    
    if all_sector_dfs:
        # Merge all sectors
        master_dfs = []
        for consumption, df in all_sector_dfs.items():
            consumption_cap = consumption.capitalize()
            df_renamed = df.rename(columns={
                f"Total_L48_{consumption_cap}_BCF": f"L48_{consumption_cap}_BCF"
            })
            master_dfs.append(df_renamed)
        
        # Merge on datetime
        master_df = reduce(
            lambda left, right: pd.merge(left, right, on='datetime', how='outer'), 
            master_dfs
        )
        master_df = master_df.sort_values('datetime').reset_index(drop=True)
        master_df = master_df.fillna(0)
        
        # Calculate grand total
        sector_cols = [col for col in master_df.columns if col != 'datetime']
        master_df['Total_L48_All_Sectors_BCF'] = master_df[sector_cols].sum(axis=1)
        
        # Save master file
        master_file = os.path.join(FINAL_OUTPUT_DIR, "MASTER_L48_all_sectors_hybrid.csv")
        master_df.to_csv(master_file, index=False, float_format='%.4f')
        print(f"    ✓ Master file saved: {os.path.basename(master_file)}")
        
        # Print master statistics
        print("\n    MASTER STATISTICS (BCF/day):")
        print("    " + "-"*40)
        for col in sector_cols + ['Total_L48_All_Sectors_BCF']:
            mean_val = master_df[col].mean()
            max_val = master_df[col].max()
            min_val = master_df[col].min()
            print(f"    {col:30s}: {mean_val:6.2f} (range: {min_val:.2f}-{max_val:.2f})")

    # 6. FINAL SUMMARY
    print("\n" + "="*70)
    print("AGGREGATION COMPLETE")
    print("="*70)
    print(f"✓ Successful loads: {len(all_successful)}/{len(REGIONS)*len(CONSUMPTION_TYPES)}")
    print(f"✗ Failed loads:     {len(all_failed)}/{len(REGIONS)*len(CONSUMPTION_TYPES)}")
    
    if all_failed:
        print("\nFailed to load:")
        for fail in sorted(all_failed):
            print(f"  - {fail}")
    
    print("\nOutput files created:")
    for consumption in CONSUMPTION_TYPES:
        print(f"  ✓ consolidated_daily_{consumption}_hybrid.csv")
        print(f"  ✓ summary_L48_{consumption}_hybrid.csv")
        print(f"  ✓ consolidated_{consumption}_summary_plot.png")
    print(f"  ✓ MASTER_L48_all_sectors_hybrid.csv")
    
    print("\n" + "="*70)
    print("Script completed successfully!")
    print("="*70)