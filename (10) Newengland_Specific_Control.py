import json
import os
import sys
import pandas as pd
import numpy as np
import traceback
import gc
from typing import Dict, Optional, List, Any
from datetime import datetime

# Ensure pyarrow is installed: pip install pyarrow

# --- Path Setup ---
# Assuming the script is in a 'scripts' directory and config is in a sibling 'config' directory
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
except NameError:
    # Handle case where __file__ is not defined (e.g., interactive notebook)
    script_dir = os.getcwd()
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))


if project_root not in sys.path:
    sys.path.append(project_root)

print(f"Script directory: {script_dir}")
print(f"Calculated project root: {project_root}")
print(f"sys.path updated to include: {project_root}")

# --- Configuration Loading ---
BASE_OUTPUT_DIR = None
regions = {}

try:
    from config.settings import BASE_OUTPUT_DIR
    print("Successfully imported BASE_OUTPUT_DIR from config.settings.")

    config_dir = os.path.join(project_root, 'config')
    regions_config_path = os.path.join(config_dir, 'regions.json')

    if not os.path.exists(regions_config_path):
        raise FileNotFoundError(f"Regions config not found at: {regions_config_path}")

    with open(regions_config_path, 'r') as f:
        regions = json.load(f)
    print(f"Successfully loaded region configuration from {regions_config_path}")

except ImportError as e:
    print(f"Error: Could not import BASE_OUTPUT_DIR from config.settings.")
    print(f"Details: {e}")
    sys.exit(1)
except FileNotFoundError as e:
    print(f"Error loading configuration file: {e}")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"Error decoding JSON configuration file: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during configuration loading: {e}")
    print(traceback.format_exc())
    sys.exit(1)

if not BASE_OUTPUT_DIR or BASE_OUTPUT_DIR == ".":
    print("Error: BASE_OUTPUT_DIR is not properly configured. Cannot proceed.")
    sys.exit(1)

print(f"Using Base Output Directory for data: {BASE_OUTPUT_DIR}")

# --- Constants ---
DATE_COL = "date"
REGION = "newengland"
REGION_UPPER = "NEWENGLAND"

# --- NEWENGLANDDataProcessor Class Definition ---
class NEWENGLANDDataProcessor:
    def __init__(self, base_dir: str = BASE_OUTPUT_DIR):
        """
        Initialize the NEWENGLANDDataProcessor using Parquet files.

        Args:
            base_dir: Base directory for file paths (default: BASE_OUTPUT_DIR).
        """
        if not isinstance(base_dir, str) or not base_dir:
            raise ValueError("base_dir must be a non-empty string.")

        self.region = REGION
        self.region_upper = REGION_UPPER
        self.base_dir = base_dir
        self.final_dir = os.path.join(self.base_dir, self.region, "final")

        # Input and output files
        self.input_file = os.path.join(self.final_dir, f"{self.region}_complete.parquet")
        self.output_file = os.path.join(self.final_dir, f"{self.region}_complete_finished.parquet")
        self.date_col = DATE_COL

        # Ensure the final output directory exists
        os.makedirs(self.final_dir, exist_ok=True)

    def load_data(self) -> Optional[pd.DataFrame]:
        """Load the input Parquet file."""
        if not os.path.exists(self.input_file):
            print(f"ERROR: Input Parquet file not found: {self.input_file}")
            return None

        print(f"Loading data from Parquet: {self.input_file}")
        df: Optional[pd.DataFrame] = None
        
        try:
            df = pd.read_parquet(self.input_file, engine='pyarrow')
            print(f"  Loaded {len(df)} rows and {len(df.columns)} columns.")

            # Validate date column
            if self.date_col not in df.columns:
                print(f"ERROR: '{self.date_col}' column not found in input file.")
                return None

            if not pd.api.types.is_datetime64_any_dtype(df[self.date_col]):
                print(f"Warning: '{self.date_col}' column not datetime type, attempting conversion.")
                initial_rows = len(df)
                try:
                    df[self.date_col] = pd.to_datetime(df[self.date_col], errors='coerce')
                    df.dropna(subset=[self.date_col], inplace=True)
                    if len(df) < initial_rows:
                        print(f"  Dropped {initial_rows - len(df)} rows with invalid dates during conversion.")
                    print(f"  Successfully converted '{self.date_col}' column to datetime.")
                except Exception as conv_e:
                    print(f"ERROR: Could not convert '{self.date_col}' column to datetime: {conv_e}")
                    if df is not None:
                        del df
                    gc.collect()
                    return None

            # Ensure date column is timezone-naive
            if df[self.date_col].dt.tz is not None:
                print(f"  Converting '{self.date_col}' column to timezone-naive.")
                df[self.date_col] = df[self.date_col].dt.tz_localize(None)

            # Sort and drop duplicates
            df.sort_values(self.date_col, inplace=True)
            rows_before_drop = len(df)
            df = df.drop_duplicates(subset=[self.date_col])
            rows_after_drop = len(df)
            if rows_before_drop > rows_after_drop:
                print(f"  Dropped {rows_before_drop - rows_after_drop} duplicate date rows.")
            print(f"  Rows after sorting and dropping duplicates: {len(df)}")

            if df.empty:
                print("Warning: DataFrame is empty after cleaning and deduplication.")

            return df

        except Exception as e:
            print(f"Error loading or processing Parquet file {self.input_file}: {e}")
            traceback.print_exc()
            if df is not None:
                del df
            gc.collect()
            return None

    def fill_NEWENGLAND_specific_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill NEWENGLAND-specific data requirements:
        1. NEWENGLAND_BAT_MW, NEWENGLAND_SNB_MW: zeros until 2025-03-17 22:00
        2. NEWENGLAND_PS_MW: zeros until 2025-03-17 22:00
        """
        if df is None or df.empty:
            print("Skipping data filling for empty or None DataFrame.")
            return df

        print("\nApplying NEWENGLAND-specific data filling...")
        df_out = df.copy()

        # Define cutoff dates
        cutoff_2025_03_17 = pd.to_datetime('2025-01-14 22:00:00')

        # *** MODIFIED: Removed NEWENGLAND_GEO_MW from this list ***
        # Columns to fill with zeros until the cutoff
        early_columns = ['NEWENGLAND_BAT_MW']
        
        # Column to fill with zeros until the cutoff
        late_column = 'NEWENGLAND_PS_MW'

        # Fill early columns with zeros until the cutoff
        for col in early_columns:
            if col in df_out.columns:
                print(f"  Processing {col}...")
                # Find mask for dates before cutoff
                before_cutoff_mask = df_out[self.date_col] < cutoff_2025_03_17
                # Find mask for null/blank values before cutoff
                null_before_cutoff = before_cutoff_mask & (df_out[col].isnull() | (df_out[col] == ''))
                
                if null_before_cutoff.any():
                    count_filled = null_before_cutoff.sum()
                    print(f"    Filling {count_filled} null/blank values with 0 before {cutoff_2025_03_17}")
                    df_out.loc[null_before_cutoff, col] = 0.0
                else:
                    print(f"    No null/blank values found before {cutoff_2025_03_17}")
            else:
                print(f"  Warning: Column {col} not found in DataFrame")

        # Fill late column with zeros until the cutoff
        if late_column in df_out.columns:
            print(f"  Processing {late_column}...")
            before_cutoff_mask = df_out[self.date_col] < cutoff_2025_03_17
            null_before_cutoff = before_cutoff_mask & (df_out[late_column].isnull() | (df_out[late_column] == ''))
            
            if null_before_cutoff.any():
                count_filled = null_before_cutoff.sum()
                print(f"    Filling {count_filled} null/blank values with 0 before {cutoff_2025_03_17}")
                df_out.loc[null_before_cutoff, late_column] = 0.0
            else:
                print(f"    No null/blank values found before {cutoff_2025_03_17}")
        else:
            print(f"  Warning: Column {late_column} not found in DataFrame")

        # *** MODIFIED: Removed the entire forward-fill section for NEWENGLAND_GEO_MW ***

        print("NEWENGLAND-specific data filling complete.")
        return df_out

    def save_data(self, df: pd.DataFrame) -> bool:
        """Save the updated DataFrame to Parquet. Returns True on success."""
        if df is None or df.empty:
            print("No data to save.")
            return False

        print(f"\nSaving updated data to Parquet: {self.output_file}")
        save_successful = False
        
        try:
            print("  Rounding all numerical columns...")
            numeric_cols = df.select_dtypes(include='number').columns
            
            # Ensure 'is_analogous_forecast' is not rounded if it exists
            if 'is_analogous_forecast' in numeric_cols:
                numeric_cols = numeric_cols.drop('is_analogous_forecast')

            if not numeric_cols.empty:
                print(f"  Rounding numeric columns: {list(numeric_cols)} to 6 decimal places.")
                df.loc[:, numeric_cols] = df[numeric_cols].round(6)
            else:
                print("   No numeric columns found to round.")

            df.to_parquet(self.output_file, index=False, engine='pyarrow')
            print("  Parquet file saved successfully.")
            save_successful = True
            
        except Exception as e:
            print(f"ERROR: Failed to save Parquet file {self.output_file}: {e}")
            traceback.print_exc()
            save_successful = False
        finally:
            if 'df' in locals() and df is not None:
                del df
            gc.collect()
            return save_successful

    def run(self):
        """Execute the NEWENGLAND data processing."""
        print(f"\n--- Processing NEWENGLAND Data ---")
        df: Optional[pd.DataFrame] = None
        df_filled: Optional[pd.DataFrame] = None
        save_successful = False

        try:
            # 1. Load Data
            df = self.load_data()
            if df is None or df.empty:
                print(f"ERROR: Failed to load data or loaded data is empty for {self.region}. Aborting.")
                return

            # 2. Apply NEWENGLAND-specific data filling
            df_filled = self.fill_NEWENGLAND_specific_data(df)
            if df_filled is None or df_filled.empty:
                print(f"ERROR: Data filling failed for {self.region}. Aborting.")
                return

            # 3. Save Data
            save_successful = self.save_data(df_filled)

            if save_successful:
                print(f"\n--- Processing completed successfully for {self.region} ---")
            else:
                print(f"\n--- Processing for {self.region} completed, but the final file was NOT saved. ---")

        except Exception as e:
            print(f"ERROR: An unhandled exception occurred while processing {self.region}: {e}")
            traceback.print_exc()
        finally:
            # Explicit memory cleanup
            if 'df' in locals() and df is not None:
                del df
            if 'df_filled' in locals() and df_filled is not None:
                del df_filled
            gc.collect()
            print(f"\n--- Data Processor finished for {self.region} ---")

# --- Main Execution Block ---
if __name__ == "__main__":
    print("===== Starting NEWENGLAND-Specific Data Processing =====")

    if not regions:
        print("No regions loaded from config/regions.json. Nothing to process.")
        sys.exit(0)

    # Check if NEWENGLAND is in the regions config
    if REGION not in regions:
        print(f"NEWENGLAND not found in regions config. Available regions: {list(regions.keys())}")
        sys.exit(1)

    print(f"Processing NEWENGLAND region specifically")
    print(f"Using Base Output Directory for data: {BASE_OUTPUT_DIR}")

    processor: Optional[NEWENGLANDDataProcessor] = None

    try:
        # Instantiate the NEWENGLAND-specific processor
        processor = NEWENGLANDDataProcessor(base_dir=BASE_OUTPUT_DIR)
        
        # Execute the data processing
        processor.run()

    except ValueError as e:
        print(f"NEWENGLAND (Config/Init Error): {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"NEWENGLAND (Critical Unhandled Error): {e}")
        traceback.print_exc()
    finally:
        if processor is not None:
            del processor
        gc.collect()

    print("\n" + "="*60)
    print("--- NEWENGLAND Data Processing Complete ---")
    print("="*60)