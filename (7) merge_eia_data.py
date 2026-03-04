import pandas as pd
import os
import json
import sys
import traceback
# Ensure pyarrow is installed: pip install pyarrow

# Adjust sys.path to include parent directory containing config/
# Assuming the script is run from a directory where '../config' is valid
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import settings
try:
    from config.settings import BASE_OUTPUT_DIR
except ImportError:
    print("Warning: Could not import BASE_OUTPUT_DIR from config.settings. Using current directory as base.")
    BASE_OUTPUT_DIR = "." # Default if import fails
except Exception as e:
     print(f"Warning: Error importing config ({e}). Using current directory as base.")
     BASE_OUTPUT_DIR = "."

class EIAMerger:
    """Merges historical weather/load Parquet data with EIA energy Parquet data."""

    def __init__(self, region, base_output_dir=BASE_OUTPUT_DIR):
        """
        Initialize the EIAMerger for a specific region using Parquet files.

        :param region: The region name (e.g., 'carolina', 'california').
        :param base_output_dir: Root directory for input/output files.
        """
        self.region = region.lower()
        self.base_output_dir = base_output_dir

        # Define directory paths
        self.region_dir = os.path.join(base_output_dir, self.region)
        self.final_dir = os.path.join(self.region_dir, "final")
        self.eia_dir = os.path.join(self.region_dir, "EIA")

        # --- CHANGE: File extensions to .parquet ---
        self.historical_file = os.path.join(self.final_dir, f"{self.region}_meteo_historical_with_load.parquet")
        self.eia_file = os.path.join(self.eia_dir, f"{self.region}_EIA.parquet")
        self.output_file = os.path.join(self.final_dir, f"{self.region}_meteo_historical_with_load_EIA.parquet")
        # --- End Change ---

    def merge_data(self):
        """Load Parquet files, merge, and save the result as Parquet."""
        print(f"--- Merging Parquet data for {self.region.upper()} ---")

        # Check if input files exist
        if not os.path.exists(self.historical_file):
            print(f"Error: Historical Parquet file '{self.historical_file}' not found.")
            return False
        if not os.path.exists(self.eia_file):
            print(f"Error: EIA Parquet file '{self.eia_file}' not found.")
            return False

        # Load data from Parquet files
        try:
            print(f"  Loading historical file: {self.historical_file}...")
            # --- CHANGE: Read from Parquet ---
            df_hist = pd.read_parquet(self.historical_file, engine='pyarrow')
            # --- End Change ---
            print(f"    Historical shape: {df_hist.shape}")
            # Validate date column
            if 'date' not in df_hist.columns or not pd.api.types.is_datetime64_any_dtype(df_hist['date']):
                 print(f"Error: 'date' column missing or not datetime in {self.historical_file}")
                 return False

            print(f"  Loading EIA file: {self.eia_file}...")
            # --- CHANGE: Read from Parquet ---
            df_eia = pd.read_parquet(self.eia_file, engine='pyarrow')
            # --- End Change ---
            print(f"    EIA shape: {df_eia.shape}")
            # Validate date column
            if 'date' not in df_eia.columns or not pd.api.types.is_datetime64_any_dtype(df_eia['date']):
                 print(f"Error: 'date' column missing or not datetime in {self.eia_file}")
                 return False

        except Exception as e:
            print(f"Error loading Parquet files for {self.region}: {e}")
            traceback.print_exc() # Print full traceback
            return False

        # Perform outer merge on 'date'
        try:
            print("  Performing outer merge on 'date'...")
            # Ensure date columns are suitable for merging (e.g., timezone naive)
            if df_hist['date'].dt.tz is not None: df_hist['date'] = df_hist['date'].dt.tz_localize(None)
            if df_eia['date'].dt.tz is not None: df_eia['date'] = df_eia['date'].dt.tz_localize(None)

            merged_df = pd.merge(df_hist, df_eia, on="date", how="outer")
            print(f"  Merged shape for {self.region}: {merged_df.shape}")

            # Sort by date after merging
            print("  Sorting merged data by date...")
            merged_df.sort_values(by="date", inplace=True)

            # Optional: Round float columns for consistency before saving
            float_cols = merged_df.select_dtypes(include=['float64', 'float32']).columns
            if not float_cols.empty:
                 print(f"  Rounding float columns: {list(float_cols)}")
                 merged_df[float_cols] = merged_df[float_cols].round(6) # Adjust precision if needed

        except Exception as e:
            print(f"Error merging data for {self.region}: {e}")
            traceback.print_exc() # Print full traceback
            return False

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        # Save merged data to Parquet
        try:
            # --- CHANGE: Save to Parquet ---
            print(f"  Saving merged data to Parquet '{self.output_file}'...")
            merged_df.to_parquet(self.output_file, index=False, engine='pyarrow')
            # --- End Change ---
            print(f"  Merged Parquet data saved successfully.")
            return True
        except Exception as e:
            print(f"Error saving merged Parquet data for {self.region}: {e}")
            traceback.print_exc() # Print full traceback
            return False

def main():
    """Process all regions from regions.json using Parquet files."""
    # --- Configuration Loading ---
    # Construct path relative to the script location
    config_dir = os.path.join(project_root, 'config') # project_root defined earlier
    regions_json_path = os.path.join(config_dir, 'regions.json')

    print(f"Attempting to load regions config from: {regions_json_path}")
    try:
        with open(regions_json_path, 'r') as f:
            regions = json.load(f)
        print(f"Successfully loaded regions config.")
        if not isinstance(regions, dict):
            print(f"Error: Expected regions config to be a dictionary, got {type(regions)}")
            sys.exit(1)
    except FileNotFoundError:
        print(f"FATAL ERROR: Regions configuration file not found at '{regions_json_path}'.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"FATAL ERROR: Invalid JSON in '{regions_json_path}'. Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"FATAL ERROR: An unexpected error occurred while loading regions config: {e}")
        print(traceback.format_exc())
        sys.exit(1)

    # --- Process Regions ---
    print(f"\nStarting EIA merge process (Parquet) for {len(regions)} regions...")
    regions_with_errors = []
    for region in regions.keys(): # Assumes keys are the region names needed
        # Instantiate the merger (it uses Parquet paths internally)
        # Ensure BASE_OUTPUT_DIR was loaded correctly
        if 'BASE_OUTPUT_DIR' not in globals() or not BASE_OUTPUT_DIR or BASE_OUTPUT_DIR == ".":
             print("Error: BASE_OUTPUT_DIR not properly configured. Check config/settings.py import or fallback.")
             regions_with_errors.append(f"{region} (BASE_OUTPUT_DIR Error)")
             continue # Skip this region if base dir is bad

        merger = EIAMerger(region, BASE_OUTPUT_DIR)
        try:
            success = merger.merge_data()
            if not success:
                print(f"-> Merge process reported failure for {region}.")
                regions_with_errors.append(f"{region} (Merge Failed)")
        except Exception as e:
             print(f"\n---!!! CRITICAL ERROR processing {region.upper()}: {e} !!!---")
             traceback.print_exc()
             regions_with_errors.append(f"{region} (Critical Error)")
        print("-" * 50) # Separator between regions

    # --- Final Summary ---
    print("\n" + "="*60)
    print("--- EIA Merge Script Complete (Parquet Version) ---")
    if regions_with_errors:
        print(f"Regions encountering errors ({len(regions_with_errors)}):")
        for region_err in regions_with_errors: print(f"  - {region_err}")
    else:
        print("All configured regions processed without critical errors.")
    print("="*60)


if __name__ == "__main__":
    main()