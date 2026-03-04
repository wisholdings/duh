import json
import pandas as pd
import os
import sys
import traceback
import glob
from pathlib import Path

# --- Configuration Loading ---
try:
    # Define BASE_OUTPUT_DIR directly
    BASE_OUTPUT_DIR = r"D:\EIA_STACK"  # Adjust to your actual base directory
    print("Using BASE_OUTPUT_DIR:", BASE_OUTPUT_DIR)
except Exception as e:
    print(f"Warning: Error setting BASE_OUTPUT_DIR ({e}). Using current directory as base.")
    BASE_OUTPUT_DIR = "."

try:
    # Load region configuration from config/regions.json
    # Config is at the same level as scripts
    script_dir = os.path.dirname(__file__)  # Directory of the script (scripts)
    project_root = os.path.abspath(os.path.join(script_dir, '..'))  # Parent directory (BestVersion)
    regions_config_path = os.path.join(project_root, 'config', 'regions.json')  # BestVersion/config/regions.json
    with open(regions_config_path, 'r') as f:
        regions = json.load(f)
    print(f"Successfully loaded region configuration from {regions_config_path}")
except FileNotFoundError:
    print(f"Error: Region configuration file not found at {regions_config_path}.")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: Could not decode {regions_config_path}. Please check its JSON format.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred loading region configuration: {e}")
    print(traceback.format_exc())
    sys.exit(1)

# --- CapacityMerger Class Definition ---
class CapacityMerger:
    """Merges capacity Parquet data files into the main historical Parquet file."""

    def __init__(self, region, base_output_dir=BASE_OUTPUT_DIR):
        """
        Initialize the CapacityMerger for a specific region using Parquet files.

        :param region: The region name (e.g., 'carolina', 'california').
        :param base_output_dir: Root directory (e.g., 'D:\EIA_STACK').
        """
        self.region = region.lower()
        self.base_output_dir = base_output_dir
        self.capacity_dir = os.path.join(base_output_dir, self.region, "Capacity")
        self.final_dir = os.path.join(base_output_dir, self.region, "final")

        # File extensions are .parquet
        self.main_file = os.path.join(self.final_dir, f"{self.region}_meteo_historical_with_load_EIA.parquet")
        self.output_file = os.path.join(self.final_dir, f"{self.region}_complete.parquet")

    def discover_capacity_files(self):
        """
        Dynamically discover all available capacity Parquet files in the capacity directory.
        Returns a dictionary with capacity type names as keys and file paths as values.
        """
        capacity_files = {}
        
        if not os.path.exists(self.capacity_dir):
            print(f"Warning: Capacity directory '{self.capacity_dir}' does not exist.")
            return capacity_files
        
        print(f"  Discovering capacity files in: {self.capacity_dir}")
        
        # Look for all .parquet files in the capacity directory
        parquet_pattern = os.path.join(self.capacity_dir, "*.parquet")
        parquet_files = glob.glob(parquet_pattern)
        
        if not parquet_files:
            print(f"  No .parquet files found in {self.capacity_dir}")
            return capacity_files
        
        print(f"  Found {len(parquet_files)} .parquet files")
        
        for file_path in parquet_files:
            filename = os.path.basename(file_path)
            
            # Extract capacity type from filename
            # Expected format: {region}_{capacity_type}.parquet
            if filename.startswith(f"{self.region}_") and filename.endswith(".parquet"):
                # Remove region prefix and .parquet suffix to get capacity type
                capacity_type = filename[len(f"{self.region}_"):-len(".parquet")]
                capacity_files[capacity_type] = file_path
                print(f"    Found capacity type: {capacity_type} -> {filename}")
            else:
                print(f"    Skipping file (doesn't match expected pattern): {filename}")
        
        print(f"  Total capacity types discovered: {list(capacity_files.keys())}")
        return capacity_files

    def validate_capacity_file(self, file_path, capacity_type):
        """
        Validate that a capacity file has the required structure.
        Returns the DataFrame if valid, None otherwise.
        """
        try:
            print(f"    Validating {capacity_type} file: {os.path.basename(file_path)}")
            df_cap = pd.read_parquet(file_path, engine='pyarrow')
            
            # Check if file is empty
            if df_cap.empty:
                print(f"    Warning: Capacity file '{file_path}' is empty. Skipping {capacity_type}.")
                return None
            
            # Check for date column
            if 'date' not in df_cap.columns:
                print(f"    Warning: 'date' column missing in '{file_path}'. Skipping {capacity_type}.")
                return None
            
            # Check if date column is datetime
            if not pd.api.types.is_datetime64_any_dtype(df_cap['date']):
                print(f"    Warning: 'date' column is not datetime in '{file_path}'. Attempting conversion...")
                try:
                    df_cap['date'] = pd.to_datetime(df_cap['date'])
                except Exception as e:
                    print(f"    Error: Could not convert 'date' column to datetime in '{file_path}': {e}")
                    return None
            
            # Ensure capacity date is timezone-naive
            if df_cap['date'].dt.tz is not None:
                print(f"    Converting '{file_path}' date column to timezone-naive.")
                df_cap['date'] = df_cap['date'].dt.tz_localize(None)

            # Check for duplicates in capacity file before merge
            if df_cap['date'].duplicated().any():
                print(f"    Warning: Duplicate dates found in '{file_path}'. Keeping last entry.")
                df_cap = df_cap.drop_duplicates(subset=['date'], keep='last')
            
            print(f"    Validation successful for {capacity_type} - Shape: {df_cap.shape}")
            print(f"    Columns: {list(df_cap.columns)}")
            print(f"    Date range: {df_cap['date'].min()} to {df_cap['date'].max()}")
            
            return df_cap
            
        except Exception as e:
            print(f"    Error validating capacity file '{file_path}': {e}")
            traceback.print_exc()
            return None

    def merge_capacity_data(self):
        """Merge all available capacity Parquet files into the main historical Parquet file."""
        print(f"--- Merging capacity data for {self.region.upper()} (Parquet) ---")

        # Check if main file exists
        if not os.path.exists(self.main_file):
            print(f"Error: Main Parquet file '{self.main_file}' not found.")
            return False

        # Load the main historical file from Parquet
        try:
            print(f"  Loading main file: {self.main_file}...")
            df_main = pd.read_parquet(self.main_file, engine='pyarrow')
            print(f"    Loaded main file with shape: {df_main.shape}")
            print(f"    Main file columns: {list(df_main.columns)}")

            # Validate date column
            if 'date' not in df_main.columns or not pd.api.types.is_datetime64_any_dtype(df_main['date']):
                print(f"Error: 'date' column missing or not datetime in {self.main_file}")
                return False
            
            # Ensure main date is timezone-naive for merging consistency
            if df_main['date'].dt.tz is not None:
                print("    Converting main file 'date' column to timezone-naive.")
                df_main['date'] = df_main['date'].dt.tz_localize(None)
            
            print(f"    Main file date range: {df_main['date'].min()} to {df_main['date'].max()}")

        except Exception as e:
            print(f"Error loading main Parquet file for {self.region}: {e}")
            traceback.print_exc()
            return False

        # Discover all available capacity files
        capacity_files = self.discover_capacity_files()
        
        if not capacity_files:
            print(f"  No capacity files found for {self.region}. Saving main file as complete file.")
            try:
                # Ensure output directory exists
                os.makedirs(self.final_dir, exist_ok=True)
                
                # Save the main data as complete (no capacity data to merge)
                print(f"  Saving main data as complete file: {self.output_file}")
                df_main.to_parquet(self.output_file, index=False, engine='pyarrow')
                print(f"    Saved successfully with shape: {df_main.shape}")
                return True
            except Exception as e:
                print(f"Error saving main file as complete for {self.region}: {e}")
                traceback.print_exc()
                return False

        # Merge each discovered capacity file
        merge_occurred = False
        successful_merges = 0
        
        for capacity_type, file_path in capacity_files.items():
            print(f"\n  Processing capacity type: {capacity_type}")
            
            # Validate and load capacity file
            df_cap = self.validate_capacity_file(file_path, capacity_type)
            if df_cap is None:
                print(f"    Skipping {capacity_type} due to validation failure.")
                continue

            try:
                print(f"    Merging {capacity_type}...")
                rows_before_merge = len(df_main)
                columns_before_merge = len(df_main.columns)
                
                df_main = pd.merge(df_main, df_cap, on="date", how="outer")
                
                rows_after_merge = len(df_main)
                columns_after_merge = len(df_main.columns)
                new_columns = columns_after_merge - columns_before_merge
                
                print(f"    Merge completed:")
                print(f"      Rows: {rows_before_merge} -> {rows_after_merge} ({rows_after_merge - rows_before_merge:+d})")
                print(f"      Columns: {columns_before_merge} -> {columns_after_merge} (+{new_columns})")

                # Check if merge drastically increased rows
                if rows_after_merge > rows_before_merge + len(df_cap):
                    print(f"    Warning: Row count increased more than expected. Check for date misalignments.")
                
                merge_occurred = True
                successful_merges += 1

            except Exception as e:
                print(f"    Error merging {capacity_type} file '{file_path}': {e}")
                traceback.print_exc()
                print(f"    Continuing merge process despite error with {capacity_type}.")

        print(f"\n  Merge summary:")
        print(f"    Total capacity files found: {len(capacity_files)}")
        print(f"    Successful merges: {successful_merges}")
        print(f"    Final dataframe shape: {df_main.shape}")

        # Sort by date after all merges
        print("  Sorting final dataframe by date...")
        df_main.sort_values("date", inplace=True)

        # Ensure output directory exists
        os.makedirs(self.final_dir, exist_ok=True)

        # Save the merged data to Parquet
        try:
            print(f"\n  Saving merged data to Parquet '{self.output_file}'...")
            print(f"    Final columns: {list(df_main.columns)}")
            
            # Round float columns
            float_cols = df_main.select_dtypes(include=['float64', 'float32']).columns
            if not float_cols.empty:
                print(f"    Rounding {len(float_cols)} float columns to 6 decimal places")
                df_main[float_cols] = df_main[float_cols].round(6)

            df_main.to_parquet(self.output_file, index=False, engine='pyarrow')
            print(f"    Merged data saved successfully with shape: {df_main.shape}")
            
            # Verify saved file
            try:
                verification_df = pd.read_parquet(self.output_file, engine='pyarrow')
                print(f"    Verification: Saved file has shape {verification_df.shape}")
                del verification_df
            except Exception as e:
                print(f"    Warning: Could not verify saved file: {e}")
            
            return True
            
        except Exception as e:
            print(f"Error saving merged Parquet data for {self.region}: {e}")
            traceback.print_exc()
            return False

# --- Main Execution Block ---
if __name__ == "__main__":
    print("===== Starting Dynamic Capacity Merging Process =====")
    if not regions:
        print("No regions found in config/regions.json. Nothing to process.")
        sys.exit(0)

    success_count = 0
    total_count = len(regions)

    for region in regions.keys():
        print(f"\nProcessing region: {region}")
        try:
            merger = CapacityMerger(region, BASE_OUTPUT_DIR)
            success = merger.merge_capacity_data()
            if success:
                success_count += 1
                print(f"✓ Successfully processed {region}.")
            else:
                print(f"✗ Failed to process {region}.")
            print("-" * 50)
        except Exception as e:
            print(f"[ERROR] Failed to process {region}: {e}")
            print(traceback.format_exc())
            print("-" * 50)

    print(f"\n===== Capacity Merging Process Finished =====")
    print(f"Success rate: {success_count}/{total_count} regions processed successfully")