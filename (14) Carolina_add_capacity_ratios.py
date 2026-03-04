import os
import sys
import pandas as pd
import numpy as np
import traceback
import gc
from typing import Optional, List, Dict, Any

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

# --- CarolinaRatioCalculator Class Definition ---
class CarolinaRatioCalculator:
    def __init__(self, base_dir: str = BASE_OUTPUT_DIR):
        """
        Initialize the CarolinaRatioCalculator.

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

        # Define ratio configurations
        self.ratio_configs = self._get_ratio_configs()

        # Ensure the final output directory exists
        os.makedirs(self.final_dir, exist_ok=True)

    def _get_ratio_configs(self) -> List[Dict[str, Any]]:
        """
        Define configurations for calculating capacity ratios.
        """
        configs = [
            # Base Load Generation [0,1]
            {
                "mw_col": "CAROLINA_NG_MW",
                "capacity_col": "Carolina_Total_Capacity_Naturalgas",
                "ratio_col": "CAROLINA_NG_RATIO",
                "label": "Natural Gas",
                "min_ratio": 0.0,
                "max_ratio": 1.0,
                "category": "base_load"
            },
            {
                "mw_col": "CAROLINA_COL_MW",
                "capacity_col": "Carolina_Total_Capacity_Coal",
                "ratio_col": "CAROLINA_COAL_RATIO",
                "label": "Coal",
                "min_ratio": 0.0,
                "max_ratio": 1.0,
                "category": "base_load"
            },
            {
                "mw_col": "CAROLINA_NUC_MW",
                "capacity_col": "Carolina_Total_Capacity_Nuclear",
                "ratio_col": "CAROLINA_NUCLEAR_RATIO",
                "label": "Nuclear",
                "min_ratio": 0.0,
                "max_ratio": 1.0,
                "category": "base_load"
            },
            
            # Renewable Generation [0,1]
            {
                "mw_col": "CAROLINA_SUN_MW",
                "capacity_col": "Carolina_Total_Capacity_Solar",
                "ratio_col": "CAROLINA_SOLAR_RATIO",
                "label": "Solar",
                "min_ratio": 0.0,
                "max_ratio": 1.0,
                "category": "renewable"
            },
            # {
            #     "mw_col": "CAROLINA_WND_MW",
            #     "capacity_col": "Carolina_Total_Capacity_Wind",
            #     "ratio_col": "CAROLINA_WIND_RATIO",
            #     "label": "Wind",
            #     "min_ratio": 0.0,
            #     "max_ratio": 1.0,
            #     "category": "renewable"
            # },
            {
                "mw_col": "CAROLINA_WAT_MW",
                "capacity_col": "Carolina_Total_Capacity_Hydroelectric",
                "ratio_col": "CAROLINA_HYDRO_RATIO",
                "label": "Hydro",
                "min_ratio": 0.0,
                "max_ratio": 1.0,
                "category": "renewable"
            },
            
            # Other Generation [-1,1] (can be negative)
            {
                "mw_col": "CAROLINA_OTH_MW",
                "capacity_col": "Carolina_Total_Capacity_Other",
                "ratio_col": "CAROLINA_OTH_RATIO",
                "label": "Other",
                "min_ratio": -1.0,
                "max_ratio": 1.0,
                "category": "other"
            },
            
            # Battery/Storage [-1,1] (can charge/discharge)
            
            # Interchange [-1,1] (self-scaled)
            {
                "mw_col": "CAROLINA_INTERCHANGE_MW",
                "capacity_col": None,  # No capacity column - self-scaled
                "ratio_col": "CAROLINA_INTERCHANGE_RATIO",
                "label": "Interchange",
                "min_ratio": -1.0,
                "max_ratio": 1.0,
                "category": "self_scaled"
            },
                                    {
                "mw_col": "CAROLINA_PS_MW",
                "capacity_col": "Carolina_Total_Capacity_Pumped_Storage",
                "ratio_col": "CAROLINA_PS_RATIO",
                "label": "Natural Gas",
                "min_ratio": -1.0,
                "max_ratio": 1.0,
                "category": "storage"
            },
            {
                "mw_col": "CAROLINA_BAT_MW",
                "capacity_col": "Carolina_Total_Capacity_Batteries",
                "ratio_col": "CAROLINA_BATTERIES_RATIO",
                "label": "Batteries",
                "min_ratio": -1.0,
                "max_ratio": 1.0,
                "category": "storage"
            },
        ]
        return configs

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

    def validate_columns(self, df: pd.DataFrame) -> bool:
        """Check for required input columns for ratio calculations."""
        if df is None or df.empty:
            print("Cannot validate columns on empty or None DataFrame.")
            return False

        required_cols = []
        for config in self.ratio_configs:
            required_cols.append(config["mw_col"])
            # Only add capacity column if it exists (not None for self-scaled)
            if config["capacity_col"] is not None:
                required_cols.append(config["capacity_col"])

        df_columns_set = set(df.columns)
        missing_cols = [col for col in required_cols if col not in df_columns_set]

        if missing_cols:
            print(f"Warning: Missing columns for ratio calculation: {', '.join(missing_cols)}")
            return False
        else:
            print("  All required columns for ratio calculations found.")
            return True

    def calculate_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate capacity ratios for each generation type."""
        if df is None or df.empty:
            print("Skipping ratio calculation on empty or None DataFrame.")
            return pd.DataFrame(columns=[self.date_col])

        df_out = df.copy()
        print("\n=== Calculating Capacity Ratios ===")

        for config in self.ratio_configs:
            mw_col = config["mw_col"]
            capacity_col = config["capacity_col"]
            ratio_col = config["ratio_col"]
            label = config["label"]
            min_ratio = config["min_ratio"]
            max_ratio = config["max_ratio"]
            category = config["category"]

            print(f"\nCalculating '{ratio_col}' ({label} - {category})...")

            # Check if MW column exists
            if mw_col not in df_out.columns:
                print(f"  Warning: MW column '{mw_col}' not found. Skipping {label}.")
                df_out[ratio_col] = np.nan
                continue

            # Handle self-scaled category (no capacity column)
            if category == "self_scaled":
                # Convert MW column to numeric
                df_out.loc[:, mw_col] = pd.to_numeric(df_out[mw_col], errors='coerce')
                
                # Count NaN values
                nan_mw = df_out[mw_col].isnull().sum()
                if nan_mw > 0:
                    print(f"  Found {nan_mw} non-numeric/null values in '{mw_col}'")
                
                # Get non-null values for scaling
                valid_values = df_out[mw_col].dropna()
                
                if valid_values.empty:
                    print(f"  Warning: No valid values in '{mw_col}'. Setting ratios to NaN.")
                    df_out[ratio_col] = np.nan
                    continue
                
                # Find min and max values for scaling
                min_val = valid_values.min()
                max_val = valid_values.max()
                
                print(f"  Data range: {min_val:.2f} to {max_val:.2f} MW")
                
                # Handle edge case where min == max
                if min_val == max_val:
                    print(f"  Warning: All values are the same ({min_val:.2f}). Setting ratios to 0.")
                    df_out[ratio_col] = 0.0
                    # Set NaNs where original MW was NaN
                    df_out.loc[df_out[mw_col].isnull(), ratio_col] = np.nan
                    continue
                
                # Calculate the maximum absolute value for symmetric scaling
                max_abs = max(abs(min_val), abs(max_val))
                
                print(f"  Scaling by max absolute value: {max_abs:.2f} MW")
                
                # Scale to [-1, 1] based on maximum absolute value
                df_out.loc[:, ratio_col] = df_out[mw_col] / max_abs
                
                # Ensure values are within [-1, 1] (should be by construction, but safety check)
                df_out.loc[:, ratio_col] = df_out[ratio_col].clip(lower=-1.0, upper=1.0)
                
                # Show final statistics
                if df_out[ratio_col].notna().any():
                    final_stats = df_out[ratio_col].describe()
                    print(f"  Final stats: Min={final_stats['min']:.4f}, Max={final_stats['max']:.4f}, "
                          f"Mean={final_stats['mean']:.4f}, Non-null={df_out[ratio_col].notna().sum()}")
                else:
                    print(f"  No valid ratios calculated for {label}")
                
                continue
            
            # Standard capacity-based calculation
            if capacity_col not in df_out.columns:
                print(f"  Warning: Capacity column '{capacity_col}' not found. Skipping {label}.")
                df_out[ratio_col] = np.nan
                continue

            # Convert columns to numeric
            df_out.loc[:, mw_col] = pd.to_numeric(df_out[mw_col], errors='coerce')
            df_out.loc[:, capacity_col] = pd.to_numeric(df_out[capacity_col], errors='coerce')

            # Count NaN values
            nan_mw = df_out[mw_col].isnull().sum()
            nan_cap = df_out[capacity_col].isnull().sum()
            if nan_mw > 0:
                print(f"  Found {nan_mw} non-numeric/null values in '{mw_col}'")
            if nan_cap > 0:
                print(f"  Found {nan_cap} non-numeric/null values in '{capacity_col}'")

            # Handle capacity denominator based on category
            capacity_denominator = df_out[capacity_col].copy()
            
            if category in ["storage", "other"]:
                # For storage and other, only avoid exact zero (can handle negative capacity)
                zero_cap_mask = capacity_denominator == 0
                if zero_cap_mask.any():
                    print(f"  Warning: Found {zero_cap_mask.sum()} zero capacity values. Setting to NaN.")
                    capacity_denominator[zero_cap_mask] = np.nan
            else:
                # For standard generation, capacity must be positive
                non_positive_cap_mask = capacity_denominator <= 0
                if non_positive_cap_mask.any():
                    print(f"  Warning: Found {non_positive_cap_mask.sum()} zero/negative capacity values. Setting to NaN.")
                    capacity_denominator[non_positive_cap_mask] = np.nan

            # Calculate ratio: MW / Capacity
            df_out.loc[:, ratio_col] = df_out[mw_col] / capacity_denominator

            # Handle special cases where MW was 0 but division resulted in NaN
            fill_mask = (df_out[mw_col] == 0) & (df_out[ratio_col].isnull())
            num_filled = fill_mask.sum()
            if num_filled > 0:
                print(f"  Filling {num_filled} NaN ratios (where MW=0) with 0")
                df_out.loc[fill_mask, ratio_col] = 0

            # Show NaN count after calculation
            nans_after = df_out[ratio_col].isnull().sum()
            if nans_after > 0:
                print(f"  {nans_after} NaN values remaining in '{ratio_col}'")

            # Apply ratio limits (clipping)
            if df_out[ratio_col].notna().any():
                original_min = df_out[ratio_col].min()
                original_max = df_out[ratio_col].max()
                
                # Clip values to the specified range
                df_out.loc[:, ratio_col] = df_out[ratio_col].clip(lower=min_ratio, upper=max_ratio)
                
                # Report clipping if it occurred
                if not np.isnan(original_min) and original_min < min_ratio:
                    clipped_count = (df_out[ratio_col] == min_ratio).sum()
                    print(f"  Clipped {clipped_count} values from below {min_ratio} to {min_ratio}")
                if not np.isnan(original_max) and original_max > max_ratio:
                    clipped_count = (df_out[ratio_col] == max_ratio).sum()
                    print(f"  Clipped {clipped_count} values from above {max_ratio} to {max_ratio}")

                # Show final statistics
                final_stats = df_out[ratio_col].describe()
                print(f"  Final stats: Min={final_stats['min']:.4f}, Max={final_stats['max']:.4f}, "
                      f"Mean={final_stats['mean']:.4f}, Non-null={df_out[ratio_col].notna().sum()}")
            else:
                print(f"  No valid ratios calculated for {label}")

        print("\n=== Ratio Calculations Complete ===")
        return df_out

    def save_data(self, df: pd.DataFrame) -> bool:
        """Save the DataFrame with ratios to Parquet. Returns True on success."""
        if df is None or df.empty:
            print("No data to save.")
            return False

        print(f"\nSaving data with ratios to Parquet: {self.output_file}")
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
        """Execute the Carolina ratio calculation process."""
        print(f"\n--- Starting Carolina Capacity Ratio Calculations ---")
        df: Optional[pd.DataFrame] = None
        df_with_ratios: Optional[pd.DataFrame] = None
        save_successful = False

        try:
            # 1. Load Data
            df = self.load_data()
            if df is None or df.empty:
                print(f"ERROR: Failed to load data or data is empty. Aborting.")
                return

            # 2. Validate Columns
            self.validate_columns(df)

            # 3. Calculate Ratios
            df_with_ratios = self.calculate_ratios(df)
            if df_with_ratios is None or df_with_ratios.empty:
                print(f"ERROR: Ratio calculation failed. Aborting.")
                return

            # 4. Save Data
            save_successful = self.save_data(df_with_ratios)

            if save_successful:
                print(f"\n--- Carolina ratio calculations completed successfully ---")
                print(f"Input:  {self.input_file}")
                print(f"Output: {self.output_file}")
                
                # Show summary of ratios calculated
                print(f"\nRatios Calculated:")
                for config in self.ratio_configs:
                    ratio_range = f"[{config['min_ratio']}, {config['max_ratio']}]"
                    print(f"  {config['ratio_col']}: {config['label']} {ratio_range}")
            else:
                print(f"\n--- Ratio calculations completed, but file save failed ---")

        except Exception as e:
            print(f"ERROR: An unhandled exception occurred: {e}")
            traceback.print_exc()
        finally:
            # Explicit memory cleanup
            if 'df' in locals() and df is not None:
                del df
            if 'df_with_ratios' in locals() and df_with_ratios is not None:
                del df_with_ratios
            gc.collect()
            print(f"\n--- Carolina Ratio Calculator finished ---")

# --- Main Execution Block ---
if __name__ == "__main__":
    print("===== Starting Carolina Capacity Ratio Calculation Process =====")
    print(f"Input file: carolina_complete_wrangled.parquet")
    print(f"Output file: carolina_complete_with_ratios.parquet")
    print(f"Using Base Output Directory: {BASE_OUTPUT_DIR}")

    calculator: Optional[CarolinaRatioCalculator] = None

    try:
        # Instantiate the calculator
        calculator = CarolinaRatioCalculator(base_dir=BASE_OUTPUT_DIR)
        
        # Execute the ratio calculation process
        calculator.run()

    except ValueError as e:
        print(f"Carolina Ratio Calculator (Config/Init Error): {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"Carolina Ratio Calculator (Critical Error): {e}")
        traceback.print_exc()
    finally:
        if calculator is not None:
            del calculator
        gc.collect()

    print("\n" + "="*60)
    print("--- Carolina Capacity Ratio Calculation Complete ---")
    print("="*60)