import os
import sys
import pandas as pd
import numpy as np
import traceback
import gc
from typing import Optional
from datetime import datetime

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
REGION = "midatlantic"
CUTOFF_DATE = "2027-12-31 23:00:00"

# --- MidatlanticDataCleaner Class Definition ---
class MidatlanticDataCleaner:
    def __init__(self, base_dir: str = BASE_OUTPUT_DIR):
        """
        Initialize the MidatlanticDataCleaner.

        Args:
            base_dir: Base directory for file paths (default: BASE_OUTPUT_DIR).
        """
        if not isinstance(base_dir, str) or not base_dir:
            raise ValueError("base_dir must be a non-empty string.")

        self.region = REGION
        self.base_dir = base_dir
        self.final_dir = os.path.join(self.base_dir, self.region, "final")
        
        # Input and output files
        self.input_file = os.path.join(self.final_dir, f"{self.region}_complete_finished.parquet")
        self.output_file = os.path.join(self.final_dir, f"{self.region}_complete_finished.parquet")
        self.date_col = DATE_COL
        self.cutoff_date = pd.to_datetime(CUTOFF_DATE)

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
                try:
                    df[self.date_col] = pd.to_datetime(df[self.date_col], errors='coerce')
                    df.dropna(subset=[self.date_col], inplace=True)
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

            # Sort by date
            df.sort_values(self.date_col, inplace=True)
            df.reset_index(drop=True, inplace=True)
            print(f"  Data sorted by date and index reset.")

            return df

        except Exception as e:
            print(f"Error loading or processing Parquet file {self.input_file}: {e}")
            traceback.print_exc()
            if df is not None:
                del df
            gc.collect()
            return None

    def forward_fill_non_Midatlantic_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Forward fill missing data in columns that DON'T start with 'MIDATLANTIC'.
        """
        if df is None or df.empty:
            print("Skipping forward fill for empty or None DataFrame.")
            return df

        print("\nApplying forward fill to non-Midatlantic columns...")
        df_out = df.copy()

        # Identify columns that don't start with 'MIDATLANTIC' (excluding the date column)
        non_Midatlantic_cols = [col for col in df_out.columns 
                              if not col.startswith('MIDATLANTIC') and col != self.date_col]
        
        print(f"Found {len(non_Midatlantic_cols)} non-Midatlantic columns to process:")
        for col in non_Midatlantic_cols[:10]:  # Show first 10 for brevity
            print(f"  - {col}")
        if len(non_Midatlantic_cols) > 10:
            print(f"  ... and {len(non_Midatlantic_cols) - 10} more columns")

        if not non_Midatlantic_cols:
            print("  No non-Midatlantic columns found to forward fill.")
            return df_out

        # Apply forward fill to each non-Midatlantic column
        for col in non_Midatlantic_cols:
            if col in df_out.columns:
                # Count missing values before forward fill
                missing_before = df_out[col].isnull().sum()
                
                if missing_before > 0:
                    # Apply forward fill
                    df_out[col] = df_out[col].fillna(method='ffill')
                    
                    # Count missing values after forward fill
                    missing_after = df_out[col].isnull().sum()
                    filled_count = missing_before - missing_after
                    
                    if filled_count > 0:
                        print(f"  {col}: Forward filled {filled_count} missing values "
                              f"({missing_after} still missing)")
                    else:
                        print(f"  {col}: No values could be forward filled "
                              f"({missing_after} still missing)")
                else:
                    print(f"  {col}: No missing values found")

        print("Forward fill operation complete.")
        return df_out

    def truncate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Truncate data at the specified cutoff date (2027-12-31 23:00:00).
        Remove all data that comes after this date.
        """
        if df is None or df.empty:
            print("Skipping data truncation for empty or None DataFrame.")
            return df

        print(f"\nTruncating data at {self.cutoff_date}...")
        
        # Count rows before truncation
        rows_before = len(df)
        
        # Filter data to keep only rows at or before the cutoff date
        df_truncated = df[df[self.date_col] <= self.cutoff_date].copy()
        
        # Count rows after truncation
        rows_after = len(df_truncated)
        rows_removed = rows_before - rows_after
        
        if rows_removed > 0:
            print(f"  Removed {rows_removed} rows after {self.cutoff_date}")
            print(f"  Remaining rows: {rows_after}")
            
            # Show the date range of remaining data
            if not df_truncated.empty:
                min_date = df_truncated[self.date_col].min()
                max_date = df_truncated[self.date_col].max()
                print(f"  Date range of remaining data: {min_date} to {max_date}")
        else:
            print(f"  No rows found after {self.cutoff_date}. No truncation needed.")
            print(f"  All {rows_after} rows retained.")

        print("Data truncation complete.")
        return df_truncated

    def save_data(self, df: pd.DataFrame) -> bool:
        """Save the cleaned DataFrame to Parquet. Returns True on success."""
        if df is None or df.empty:
            print("No data to save.")
            return False

        print(f"\nSaving cleaned data to Parquet: {self.output_file}")
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
            save_successful = True
            
        except Exception as e:
            print(f"ERROR: Failed to save Parquet file {self.output_file}: {e}")
            traceback.print_exc()
            save_successful = False
        finally:
            return save_successful

    def run(self):
        """Execute the Midatlantic data cleaning process."""
        print(f"\n--- Starting Midatlantic Data Cleaning ---")
        df: Optional[pd.DataFrame] = None
        df_filled: Optional[pd.DataFrame] = None
        df_truncated: Optional[pd.DataFrame] = None
        save_successful = False

        try:
            # 1. Load Data
            df = self.load_data()
            if df is None or df.empty:
                print(f"ERROR: Failed to load data or loaded data is empty. Aborting.")
                return

            # 2. Forward fill non-Midatlantic columns
            df_filled = self.forward_fill_non_Midatlantic_columns(df)
            if df_filled is None or df_filled.empty:
                print(f"ERROR: Forward fill operation failed. Aborting.")
                return

            # 3. Truncate data at cutoff date
            df_truncated = self.truncate_data(df_filled)
            if df_truncated is None or df_truncated.empty:
                print(f"WARNING: Data truncation resulted in empty DataFrame.")
                return

            # 4. Save Data
            save_successful = self.save_data(df_truncated)

            if save_successful:
                print(f"\n--- Data cleaning completed successfully ---")
                print(f"Output file: {self.output_file}")
            else:
                print(f"\n--- Data cleaning completed, but the final file was NOT saved. ---")

        except Exception as e:
            print(f"ERROR: An unhandled exception occurred during data cleaning: {e}")
            traceback.print_exc()
        finally:
            # Explicit memory cleanup
            if 'df' in locals() and df is not None:
                del df
            if 'df_filled' in locals() and df_filled is not None:
                del df_filled
            if 'df_truncated' in locals() and df_truncated is not None:
                del df_truncated
            gc.collect()
            print(f"\n--- Data Cleaner finished ---")

# --- Main Execution Block ---
if __name__ == "__main__":
    print("===== Starting Midatlantic Data Cleaning Process =====")
    print(f"Input file: Midatlantic_complete_finished.parquet")
    print(f"Output file: Midatlantic_complete_cleaned.parquet")
    print(f"Cutoff date: {CUTOFF_DATE}")
    print(f"Using Base Output Directory: {BASE_OUTPUT_DIR}")

    cleaner: Optional[MidatlanticDataCleaner] = None

    try:
        # Instantiate the Midatlantic data cleaner
        cleaner = MidatlanticDataCleaner(base_dir=BASE_OUTPUT_DIR)
        
        # Execute the data cleaning process
        cleaner.run()

    except ValueError as e:
        print(f"Midatlantic Data Cleaner (Config/Init Error): {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"Midatlantic Data Cleaner (Critical Unhandled Error): {e}")
        traceback.print_exc()
    finally:
        if cleaner is not None:
            del cleaner
        gc.collect()

    print("\n" + "="*60)
    print("--- Midatlantic Data Cleaning Process Complete ---")
    print("="*60)