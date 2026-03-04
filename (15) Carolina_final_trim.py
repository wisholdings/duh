import os
import sys
import pandas as pd
import numpy as np
import traceback
import gc
from datetime import datetime, timedelta
from typing import Optional

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
REGION = "carolina"

# --- CarolinaBackfillCleaner Class Definition ---
class CarolinaBackfillCleaner:
    def __init__(self, base_dir: str = BASE_OUTPUT_DIR):
        """
        Initialize the CarolinaBackfillCleaner.

        Args:
            base_dir: Base directory for file paths (default: BASE_OUTPUT_DIR).
        """
        if not isinstance(base_dir, str) or not base_dir:
            raise ValueError("base_dir must be a non-empty string.")

        self.region = REGION
        self.base_dir = base_dir
        self.final_dir = os.path.join(self.base_dir, self.region, "final")
        self.date_col = DATE_COL
        
        # Calculate reference date (2 days before today at 23:00)
        today = datetime.now().date()
        reference_date_obj = today - timedelta(days=2)
        self.reference_date = pd.to_datetime(f"{reference_date_obj} 23:00:00")
        
        # Input and output files
        self.input_file = os.path.join(self.final_dir, f"{self.region}_complete_finished.parquet")
        self.output_file = os.path.join(self.final_dir, f"{self.region}_complete_finished.parquet")

        # Columns to clean after reference date
        self.oth_columns = ["CAROLINA_OTH_RATIO", "CAROLINA_OTH_MW"]

        # Ensure the final output directory exists
        os.makedirs(self.final_dir, exist_ok=True)
        
        print(f"Reference date (2 days before today at 23:00): {self.reference_date}")

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

            # Ensure date column is datetime
            if not pd.api.types.is_datetime64_any_dtype(df[self.date_col]):
                print(f"Converting '{self.date_col}' column to datetime...")
                df[self.date_col] = pd.to_datetime(df[self.date_col], errors='coerce')
                df.dropna(subset=[self.date_col], inplace=True)

            # Ensure timezone-naive
            if df[self.date_col].dt.tz is not None:
                print(f"Converting '{self.date_col}' column to timezone-naive...")
                df[self.date_col] = df[self.date_col].dt.tz_localize(None)

            # Sort by date
            df.sort_values(self.date_col, inplace=True)
            df.reset_index(drop=True, inplace=True)
            
            # Show date range
            print(f"  Original date range: {df[self.date_col].min()} to {df[self.date_col].max()}")
            
            return df
            
        except Exception as e:
            print(f"Error loading Parquet file {self.input_file}: {e}")
            traceback.print_exc()
            if df is not None:
                del df
            gc.collect()
            return None

    def backfill_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Backfill all values before the reference date (2 days before today)."""
        if df is None or df.empty:
            print("Skipping backfill for empty or None DataFrame.")
            return df

        print(f"\nBackfilling all data before {self.reference_date}...")
        df_out = df.copy()
        
        # Create mask for dates before reference date
        before_reference_mask = df_out[self.date_col] < self.reference_date
        rows_before_reference = before_reference_mask.sum()
        
        print(f"  Found {rows_before_reference} rows before reference date")
        
        if rows_before_reference == 0:
            print("  No rows found before reference date. No backfill needed.")
            return df_out
        
        # Get all columns except date column
        data_columns = [col for col in df_out.columns if col != self.date_col]
        
        print(f"  Backfilling {len(data_columns)} data columns...")
        
        # Track backfill statistics
        backfill_stats = {}
        
        for col in data_columns:
            # Count missing values before backfill (only in the before-reference period)
            missing_before = df_out.loc[before_reference_mask, col].isnull().sum()
            
            if missing_before > 0:
                # Apply backfill (backward fill) to the entire column
                df_out[col] = df_out[col].fillna(method='bfill')
                
                # Count missing values after backfill (only in the before-reference period)
                missing_after = df_out.loc[before_reference_mask, col].isnull().sum()
                filled_count = missing_before - missing_after
                
                if filled_count > 0:
                    backfill_stats[col] = {
                        'filled': filled_count,
                        'still_missing': missing_after
                    }
        
        # Show summary of backfill operations
        if backfill_stats:
            print(f"  Backfill summary for {len(backfill_stats)} columns:")
            for col, stats in list(backfill_stats.items())[:10]:  # Show first 10
                print(f"    {col}: filled {stats['filled']}, {stats['still_missing']} still missing")
            if len(backfill_stats) > 10:
                print(f"    ... and {len(backfill_stats) - 10} more columns")
        else:
            print("  No missing values found to backfill")
        
        print("Backfill operation complete.")
        return df_out

    def clean_oth_data_after_reference(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean CAROLINA_OTH_RATIO and CAROLINA_OTH_MW data after reference date."""
        if df is None or df.empty:
            print("Skipping OTH data cleaning for empty or None DataFrame.")
            return df

        print(f"\nCleaning OTH data after {self.reference_date}...")
        df_out = df.copy()
        
        # Create mask for dates after reference date
        after_reference_mask = df_out[self.date_col] > self.reference_date
        rows_after_reference = after_reference_mask.sum()
        
        print(f"  Found {rows_after_reference} rows after reference date")
        
        if rows_after_reference == 0:
            print("  No rows found after reference date. No cleaning needed.")
            return df_out
        
        # Clean each OTH column
        for col in self.oth_columns:
            if col in df_out.columns:
                # Count non-null values before cleaning
                non_null_before = df_out.loc[after_reference_mask, col].notna().sum()
                
                if non_null_before > 0:
                    # Set values to NaN after reference date
                    df_out.loc[after_reference_mask, col] = np.nan
                    print(f"    Cleared {non_null_before} values from '{col}' after {self.reference_date}")
                else:
                    print(f"    No values to clear from '{col}' after {self.reference_date}")
            else:
                print(f"    Warning: Column '{col}' not found in DataFrame")
        
        print("OTH data cleaning complete.")
        return df_out

    def save_data(self, df: pd.DataFrame) -> bool:
        """Save the processed DataFrame to Parquet. Returns True on success."""
        if df is None or df.empty:
            print("No data to save.")
            return False

        print(f"\nSaving processed data to Parquet: {self.output_file}")
        save_successful = False
        
        try:
            print("  Rounding numerical columns...")
            numeric_cols = df.select_dtypes(include='number').columns
            numeric_cols = numeric_cols.drop(['is_analogous_forecast'], errors='ignore')

            if not numeric_cols.empty:
                print(f"  Rounding {len(numeric_cols)} numeric columns to 6 decimal places.")
                df.loc[:, numeric_cols] = df[numeric_cols].round(6)
            else:
                print("  No numeric columns found to round.")

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
        """Execute the Carolina backfill and cleaning process."""
        print(f"\n--- Starting Carolina Backfill and OTH Cleaning ---")
        df: Optional[pd.DataFrame] = None
        df_backfilled: Optional[pd.DataFrame] = None
        df_cleaned: Optional[pd.DataFrame] = None
        save_successful = False

        try:
            # 1. Load Data
            df = self.load_data()
            if df is None or df.empty:
                print(f"ERROR: Failed to load data or data is empty. Aborting.")
                return

            # 2. Backfill data before reference date
            df_backfilled = self.backfill_data(df)
            if df_backfilled is None or df_backfilled.empty:
                print(f"ERROR: Backfill operation failed. Aborting.")
                return

            # 3. Clean OTH data after reference date
            df_cleaned = self.clean_oth_data_after_reference(df_backfilled)
            if df_cleaned is None or df_cleaned.empty:
                print(f"ERROR: OTH cleaning operation failed. Aborting.")
                return

            # 4. Save Data
            save_successful = self.save_data(df_cleaned)

            if save_successful:
                print(f"\n--- Carolina backfill and cleaning completed successfully ---")
                print(f"Input:  {self.input_file}")
                print(f"Output: {self.output_file}")
                print(f"Reference date: {self.reference_date}")
                print(f"Operations performed:")
                print(f"  1. Backfilled all data before {self.reference_date}")
                print(f"  2. Cleared OTH data after {self.reference_date}")
            else:
                print(f"\n--- Processing completed, but file save failed ---")

        except Exception as e:
            print(f"ERROR: An unhandled exception occurred: {e}")
            traceback.print_exc()
        finally:
            # Explicit memory cleanup
            for df_var in ['df', 'df_backfilled', 'df_cleaned']:
                if df_var in locals() and locals()[df_var] is not None:
                    del locals()[df_var]
            gc.collect()
            print(f"\n--- Carolina Backfill Cleaner finished ---")

# --- Main Execution Block ---
if __name__ == "__main__":
    print("===== Starting Carolina Backfill and OTH Cleaning Process =====")
    
    # Calculate and display reference date
    today = datetime.now().date()
    reference_date = f"{today - timedelta(days=2)} 23:00:00"
    print(f"Today: {today}")
    print(f"Reference date (2 days before at 23:00): {reference_date}")
    print(f"Input file: carolina_complete_finished.parquet")
    print(f"Output file: carolina_complete_backfilled.parquet")
    print(f"Using Base Output Directory: {BASE_OUTPUT_DIR}")

    cleaner: Optional[CarolinaBackfillCleaner] = None

    try:
        # Instantiate the cleaner
        cleaner = CarolinaBackfillCleaner(base_dir=BASE_OUTPUT_DIR)
        
        # Execute the cleaning process
        cleaner.run()

    except ValueError as e:
        print(f"Carolina Backfill Cleaner (Config/Init Error): {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"Carolina Backfill Cleaner (Critical Error): {e}")
        traceback.print_exc()
    finally:
        if cleaner is not None:
            del cleaner
        gc.collect()

    print("\n" + "="*60)
    print("--- Carolina Backfill and OTH Cleaning Complete ---")
    print("="*60)