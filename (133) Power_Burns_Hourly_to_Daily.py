import os
import sys
import pandas as pd
from datetime import datetime
import traceback
import warnings

# --- Configuration Loading (Consistent with your previous script) ---
try:
    # Try to find the script's own directory first
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from config.settings import BASE_OUTPUT_DIR
    print(f"Loaded BASE_OUTPUT_DIR from config: {BASE_OUTPUT_DIR}")
except (ImportError, NameError, FileNotFoundError) as e:
    print(f"Could not import BASE_OUTPUT_DIR from config.settings ({e}). Using default.")
    # FALLBACK - IMPORTANT: Update this if your actual default path is different
    BASE_OUTPUT_DIR = "D:\\EIA_STACK"
    print(f"Using default BASE_OUTPUT_DIR: {BASE_OUTPUT_DIR}")

# --- File Configuration ---
HOURLY_COMBINED_INPUT_FILENAME = "final_power_burns_multi_features.csv"
DAILY_SUMMED_OUTPUT_FILENAME = "daily_power_burns_multi_features_summed.csv"

# --- Column Name Definitions ---
HOURLY_DATE_COL = 'date'  # From the input file
# This suffix identifies the columns we want to sum (from your previous script's output)
HOURLY_ESTIMATE_SUFFIX = "_Hourly_MMCF_Multi"
# This will be the new suffix for the daily summed columns in the output
DAILY_SUM_SUFFIX = "_Daily_MMCF_Multi_Sum"
DAILY_DATE_COL_OUTPUT = 'Date' # The name of the daily date column in the output

# --- Pandas Options ---
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

def aggregate_hourly_to_daily_sum():
    """
    Loads the hourly combined gas estimates, aggregates them to daily sums,
    and saves the result to a new CSV file.
    """
    print("\n=====================================================================")
    print("=== Aggregating Hourly Regional Gas Estimates to Daily Sums ===")
    print("=====================================================================")

    input_filepath = os.path.join(BASE_OUTPUT_DIR, HOURLY_COMBINED_INPUT_FILENAME)
    output_filepath = os.path.join(BASE_OUTPUT_DIR, DAILY_SUMMED_OUTPUT_FILENAME)

    print(f"Input hourly file: {input_filepath}")
    print(f"Output daily file: {output_filepath}")

    # 1. Load Hourly Data
    if not os.path.exists(input_filepath):
        print(f"ERROR: Input file not found: {input_filepath}")
        sys.exit(1)

    try:
        print(f"Loading hourly data from '{os.path.basename(input_filepath)}'...")
        df_hourly = pd.read_csv(input_filepath)
        
        # Ensure the date column is parsed correctly
        if HOURLY_DATE_COL not in df_hourly.columns:
            print(f"ERROR: Expected date column '{HOURLY_DATE_COL}' not found in the input file.")
            sys.exit(1)
        df_hourly[HOURLY_DATE_COL] = pd.to_datetime(df_hourly[HOURLY_DATE_COL], errors='coerce')
        
        # Drop rows where date parsing failed
        initial_rows = len(df_hourly)
        df_hourly.dropna(subset=[HOURLY_DATE_COL], inplace=True)
        if len(df_hourly) < initial_rows:
            print(f"    Dropped {initial_rows - len(df_hourly)} rows with invalid date entries.")

        if df_hourly.empty:
            print("ERROR: No valid data after loading and date parsing. Exiting.")
            sys.exit(1)
        print(f"Successfully loaded {len(df_hourly)} hourly records.")

    except Exception as e:
        print(f"ERROR: Could not load or parse the input file '{input_filepath}': {e}")
        traceback.print_exc()
        sys.exit(1)

    # 2. Identify Regional Estimate Columns to Sum
    estimate_cols_to_sum = []
    for col in df_hourly.columns:
        if col.endswith(HOURLY_ESTIMATE_SUFFIX):
            estimate_cols_to_sum.append(col)

    if not estimate_cols_to_sum:
        print(f"WARNING: No columns found with the suffix '{HOURLY_ESTIMATE_SUFFIX}' to aggregate.")
        print("Ensure the input file contains the correct columns from the previous script.")
        # Decide if you want to exit or create an empty/minimal output
        # For now, let's proceed and it will likely create a file with just a date column if no data cols
    else:
        print(f"Identified {len(estimate_cols_to_sum)} columns for daily summation: {estimate_cols_to_sum}")

    # Convert identified columns to numeric, coercing errors to NaN (which sum handles as 0)
    for col in estimate_cols_to_sum:
        df_hourly[col] = pd.to_numeric(df_hourly[col], errors='coerce')


    # 3. Create Daily Date Column for Grouping
    # .dt.normalize() sets the time part to 00:00:00, effectively giving the date
    df_hourly[DAILY_DATE_COL_OUTPUT] = df_hourly[HOURLY_DATE_COL].dt.normalize()

    # 4. Perform Aggregation
    agg_dict = {}
    output_col_mapping = {} # To store new column names

    for col_name in estimate_cols_to_sum:
        agg_dict[col_name] = 'sum'
        # Create the new column name for the daily output
        new_col_name = col_name.replace(HOURLY_ESTIMATE_SUFFIX, DAILY_SUM_SUFFIX)
        output_col_mapping[col_name] = new_col_name
    
    if not agg_dict:
        print("No columns to aggregate. Output will only contain dates if any hourly data existed.")
        # Create a DataFrame with unique dates if df_hourly is not empty
        if not df_hourly.empty:
            df_daily_summed = pd.DataFrame({DAILY_DATE_COL_OUTPUT: df_hourly[DAILY_DATE_COL_OUTPUT].unique()})
            df_daily_summed.sort_values(by=DAILY_DATE_COL_OUTPUT, inplace=True)
        else:
            df_daily_summed = pd.DataFrame(columns=[DAILY_DATE_COL_OUTPUT]) # Empty df if no data
    else:
        print("Aggregating data to daily sums...")
        df_daily_summed = df_hourly.groupby(DAILY_DATE_COL_OUTPUT, as_index=False).agg(agg_dict)
        
        # Rename columns to the new daily sum names
        df_daily_summed.rename(columns=output_col_mapping, inplace=True)
        print("Aggregation complete.")

    # Ensure the daily date column is the first column if it's not already
    if DAILY_DATE_COL_OUTPUT in df_daily_summed.columns:
        cols = [DAILY_DATE_COL_OUTPUT] + [col for col in df_daily_summed.columns if col != DAILY_DATE_COL_OUTPUT]
        df_daily_summed = df_daily_summed[cols]

    # 5. Add 'date_published' column
    df_daily_summed['date_published'] = datetime.now().strftime('%Y-%m-%d')

    # 6. Save Output
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        
        # Prepare float format for summed columns
        float_cols = [col for col in df_daily_summed.columns if col.endswith(DAILY_SUM_SUFFIX)]
        float_format_dict = {col: '%.3f' for col in float_cols}

        df_daily_summed.to_csv(
            output_filepath, 
            index=False, 
            date_format='%Y-%m-%d' # For the DAILY_DATE_COL_OUTPUT
            # Using float_format directly on to_csv might not work for specific columns,
            # but pandas usually handles this well. If precision issues, round before saving.
        )
        # If you need specific precision for float columns and date_format for date:
        # for col in float_cols:
        #    if col in df_daily_summed:
        #        df_daily_summed[col] = df_daily_summed[col].round(3)
        # df_daily_summed.to_csv(output_filepath, index=False, date_format='%Y-%m-%d')


        print(f"Successfully saved daily summed gas estimates to: {output_filepath}")
        print(f"Output file contains {len(df_daily_summed)} daily records and {len(df_daily_summed.columns)} columns.")

    except Exception as e:
        print(f"ERROR: Could not save the output file '{output_filepath}': {e}")
        traceback.print_exc()
        sys.exit(1)

    print("\n=====================================================================")
    print("=== Daily Aggregation Process Finished ===")
    print("=====================================================================")

if __name__ == "__main__":
    aggregate_hourly_to_daily_sum()