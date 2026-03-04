import os
import sys
import pandas as pd
import numpy as np
import traceback
import gc
from typing import Optional, List

# Ensure pyarrow is installed: pip install pyarrow

# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

if project_root not in sys.path:
    sys.path.append(project_root)

print(f"Script directory: {script_dir}")
print(f"Calculated project root (parent of script dir): {project_root}")
print(f"sys.path updated to include: {project_root}")

# --- Configuration Loading ---
BASE_OUTPUT_DIR = None

try:
    from config.settings import BASE_OUTPUT_DIR
    print("Successfully imported BASE_OUTPUT_DIR from config.settings.")
except ImportError as e:
    print(f"Error: Could not import BASE_OUTPUT_DIR from config.settings.")
    print(f"Details: {e}")
    sys.exit(1)

if not BASE_OUTPUT_DIR or BASE_OUTPUT_DIR == ".":
    print("Error: BASE_OUTPUT_DIR is not properly configured. Cannot proceed.")
    sys.exit(1)

print(f"Using Base Output Directory for data: {BASE_OUTPUT_DIR}")

# --- Constants ---
DATE_COL = "date"
REGION = "Southwest"

# --- SouthwestDataWrangler Class Definition ---
class SouthwestDataWrangler:
    def __init__(self, base_dir: str = BASE_OUTPUT_DIR):
        """
        Initialize the SouthwestDataWrangler.

        Args:
            base_dir: Base directory for file paths (default: BASE_OUTPUT_DIR).
        """
        if not isinstance(base_dir, str) or not base_dir:
            raise ValueError("base_dir must be a non-empty string.")

        self.region = REGION
        self.base_dir = base_dir
        self.final_dir = os.path.join(self.base_dir, self.region, "final")
        self.date_col = DATE_COL
        
        # Input and output files
        self.input_file = os.path.join(self.final_dir, f"{self.region}_complete_finished.parquet")
        self.output_file = os.path.join(self.final_dir, f"{self.region}_complete_finished.parquet")

        # Define column combinations for data wrangling
        self.mw_combinations = {
            "SOUTHWEST_OTH_MW": ["SOUTHWEST_OIL_MW", "SOUTHWEST_OTH_MW","SOUTHWEST_GEO_MW"],
            "SOUTHWEST_BAT_MW": ["SOUTHWEST_BAT_MW","SOUTHWEST_UES_MW","SOUTHWEST_SNB_MW","SOUTHWEST_PS_MW"]
        }
        
        self.capacity_combinations = {
            "Southwest_Total_Capacity_Other": [
                "Southwest_Total_Capacity_Other", 
                "Southwest_Total_Capacity_Oil",
                "Southwest_Total_Capacity_Geothermal",
                "Southwest_Total_Capacity_Biomass",

            ],
                "Southwest_Total_Capacity_Batteries": [
                "Southwest_Total_Capacity_Solar_Batteries", 
                "Southwest_Total_Capacity_Batteries",
                "Southwest_Total_Capacity_Pumped_Storage"

            ],

        }

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

            # Show date range
            if pd.api.types.is_datetime64_any_dtype(df[self.date_col]):
                print(f"  Date range: {df[self.date_col].min()} to {df[self.date_col].max()}")
            
            return df

        except Exception as e:
            print(f"Error loading Parquet file {self.input_file}: {e}")
            traceback.print_exc()
            if df is not None:
                del df
            gc.collect()
            return None

    def combine_columns(self, df: pd.DataFrame, target_col: str, source_cols: List[str], 
                       operation: str = "sum") -> pd.DataFrame:
        """
        Combine multiple columns into a target column.
        
        Args:
            df: DataFrame to operate on
            target_col: Name of the target column to create/update
            source_cols: List of source column names to combine
            operation: Type of operation ('sum', 'mean', etc.)
        """
        print(f"\n  Combining columns for {target_col}:")
        
        # Check which source columns exist
        existing_cols = [col for col in source_cols if col in df.columns]
        missing_cols = [col for col in source_cols if col not in df.columns]
        
        print(f"    Source columns found: {existing_cols}")
        if missing_cols:
            print(f"    Source columns missing: {missing_cols}")
        
        if not existing_cols:
            print(f"    Warning: No source columns found for {target_col}. Setting to NaN.")
            df[target_col] = np.nan
            return df
        
        # Convert columns to numeric, handling any non-numeric values
        for col in existing_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Perform the combination operation
        if operation == "sum":
            # Sum across columns, handling NaNs properly
            df[target_col] = df[existing_cols].sum(axis=1, skipna=True)
            
            # If all values in a row are NaN, the result should be NaN
            all_nan_mask = df[existing_cols].isna().all(axis=1)
            df.loc[all_nan_mask, target_col] = np.nan
            
        else:
            print(f"    Warning: Operation '{operation}' not implemented. Using sum.")
            df[target_col] = df[existing_cols].sum(axis=1, skipna=True)
            all_nan_mask = df[existing_cols].isna().all(axis=1)
            df.loc[all_nan_mask, target_col] = np.nan
        
        # Show statistics
        non_null_count = df[target_col].notna().sum()
        total_count = len(df)
        
        print(f"    Result: {target_col} created with {non_null_count}/{total_count} non-null values")
        
        if non_null_count > 0:
            stats = df[target_col].describe()
            print(f"    Statistics: Min={stats['min']:.2f}, Max={stats['max']:.2f}, Mean={stats['mean']:.2f}")
        
        return df

    def remove_source_columns(self, df: pd.DataFrame, combinations: dict) -> pd.DataFrame:
        """
        Remove original source columns after they've been combined.
        Only removes columns that were actually combined (not target columns that were overwritten).
        """
        print(f"\nRemoving original source columns...")
        
        columns_to_remove = set()
        
        for target_col, source_cols in combinations.items():
            for source_col in source_cols:
                # Only remove if it's not the target column itself
                if source_col != target_col and source_col in df.columns:
                    columns_to_remove.add(source_col)
        
        if columns_to_remove:
            print(f"  Removing columns: {sorted(list(columns_to_remove))}")
            df = df.drop(columns=list(columns_to_remove))
            print(f"  Removed {len(columns_to_remove)} columns")
        else:
            print("  No columns to remove")
        
        return df

    def wrangle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform all data wrangling operations."""
        if df is None or df.empty:
            print("Skipping data wrangling for empty or None DataFrame.")
            return df

        print("\n=== Starting Data Wrangling ===")
        df_out = df.copy()
        
        print(f"Starting with {len(df_out.columns)} columns")
        
        # 1. Combine MW columns
        print("\n--- Combining MW Columns ---")
        for target_col, source_cols in self.mw_combinations.items():
            df_out = self.combine_columns(df_out, target_col, source_cols, "sum")
        
        # 2. Combine Capacity columns
        print("\n--- Combining Capacity Columns ---")
        for target_col, source_cols in self.capacity_combinations.items():
            df_out = self.combine_columns(df_out, target_col, source_cols, "sum")
        
        # 3. Remove original source columns (but keep target columns that were overwritten)
        all_combinations = {**self.mw_combinations, **self.capacity_combinations}
        df_out = self.remove_source_columns(df_out, all_combinations)
        
        print(f"\nEnding with {len(df_out.columns)} columns")
        print("=== Data Wrangling Complete ===")
        
        return df_out

    def save_data(self, df: pd.DataFrame) -> bool:
        """Save the wrangled DataFrame to Parquet. Returns True on success."""
        if df is None or df.empty:
            print("No data to save.")
            return False

        print(f"\nSaving wrangled data to Parquet: {self.output_file}")
        save_successful = False
        
        try:
            print("  Rounding all numerical columns...")
            numeric_cols = df.select_dtypes(include='number').columns
            numeric_cols = numeric_cols.drop(['is_analogous_forecast'], errors='ignore')

            if not numeric_cols.empty:
                print(f"  Rounding {len(numeric_cols)} numeric columns to 6 decimal places.")
                df.loc[:, numeric_cols] = df[numeric_cols].round(6)
            else:
                print("   No numeric columns found to round.")

            df.to_parquet(self.output_file, index=False, engine='pyarrow')
            print("  Parquet file saved successfully.")
            print(f"  Final dataset: {len(df)} rows, {len(df.columns)} columns")
            
            # Display file size
            file_size = os.path.getsize(self.output_file)
            file_size_mb = file_size / (1024 * 1024)
            print(f"  File size: {file_size_mb:.2f} MB")
            
            save_successful = True
            
        except Exception as e:
            print(f"ERROR: Failed to save Parquet file {self.output_file}: {e}")
            traceback.print_exc()
            save_successful = False
        finally:
            return save_successful

    def run(self):
        """Execute the Southwest data wrangling process."""
        print(f"\n--- Starting Southwest Data Wrangling ---")
        df: Optional[pd.DataFrame] = None
        df_wrangled: Optional[pd.DataFrame] = None
        save_successful = False

        try:
            # 1. Load Data
            df = self.load_data()
            if df is None or df.empty:
                print(f"ERROR: Failed to load data or loaded data is empty. Aborting.")
                return

            # 2. Perform Data Wrangling
            df_wrangled = self.wrangle_data(df)
            if df_wrangled is None or df_wrangled.empty:
                print(f"ERROR: Data wrangling failed. Aborting.")
                return

            # 3. Save Data
            save_successful = self.save_data(df_wrangled)

            if save_successful:
                print(f"\n--- Data wrangling completed successfully ---")
                print(f"Input:  {self.input_file}")
                print(f"Output: {self.output_file}")
                
                # Show summary of transformations
                print(f"\nTransformations Applied:")
                print(f"  1. SOUTHWEST_OTH_MW = SOUTHWEST_OIL_MW + SOUTHWEST_OTH_MW + SOUTHWEST_GEO_MW")
                print(f"  2. SOUTHWEST_BATTERIES_MW = SOUTHWEST_BAT_MW + SOUTHWEST_PS_MW + SOUTHWEST_SNB_MW")
                print(f"  3. SOUTHWEST_Total_Capacity_Other = sum of Other + Oil + Geothermal capacities")
                print(f"  4. Removed original source columns")
            else:
                print(f"\n--- Data wrangling completed, but the final file was NOT saved. ---")

        except Exception as e:
            print(f"ERROR: An unhandled exception occurred during data wrangling: {e}")
            traceback.print_exc()
        finally:
            # Explicit memory cleanup
            if 'df' in locals() and df is not None:
                del df
            if 'df_wrangled' in locals() and df_wrangled is not None:
                del df_wrangled
            gc.collect()
            print(f"\n--- Data Wrangler finished ---")

# --- Main Execution Block ---
if __name__ == "__main__":
    print("===== Starting Southwest Data Wrangling Process =====")
    print(f"Input file: Southwest_complete_with_interchange.parquet")
    print(f"Output file: Southwest_complete_wrangled.parquet")
    print(f"Using Base Output Directory: {BASE_OUTPUT_DIR}")

    wrangler: Optional[SouthwestDataWrangler] = None

    try:
        # Instantiate the Southwest data wrangler
        wrangler = SouthwestDataWrangler(base_dir=BASE_OUTPUT_DIR)
        
        # Execute the data wrangling process
        wrangler.run()

    except ValueError as e:
        print(f"Southwest Data Wrangler (Config/Init Error): {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"Southwest Data Wrangler (Critical Error): {e}")
        traceback.print_exc()
    finally:
        if wrangler is not None:
            del wrangler
        gc.collect()

    print("\n" + "="*60)
    print("--- Southwest Data Wrangling Process Complete ---")
    print("="*60)