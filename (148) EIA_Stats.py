import os
import json
import sys
import pandas as pd
import traceback
import time
from datetime import datetime
import requests
import warnings
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# --- SQL Server Configuration ---
DB_CONNECTION_STRING = (
)
DB_SCHEMA = "dbo"
SQL_TABLE_NAME = "PRODUCTION_EIA_SUPPLY_STATISTICS"

# --- EIA API Configuration ---
API_KEY = "" # Your EIA API Key
EIA_API_URL = "https://api.eia.gov/v2/natural-gas/sum/sndm/data/"

# --- File Configuration ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from config.settings import BASE_OUTPUT_DIR
    print(f"Loaded BASE_OUTPUT_DIR from config: {BASE_OUTPUT_DIR}")
except (ImportError, NameError, FileNotFoundError) as e:
    print(f"Could not import BASE_OUTPUT_DIR from config.settings ({e}). Using default.")
    BASE_OUTPUT_DIR = "D:\\EIA_STACK\\LOWER48FUNDIES" # EXAMPLE: Change this path
    print(f"Using default BASE_OUTPUT_DIR: {BASE_OUTPUT_DIR}")

CSV_OUTPUT_FILENAME = "eia_supply_statistics_us_total.csv"

# Disable InsecureRequestWarning if verify=False is necessary
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

PROCESS_CODE_TO_COLUMN_NAME_MAP = {
    "VG4": "Balancing_Item_Bcf",
    "VGM": "Marketed_Production_Wet_Bcf",
    "FPD": "Dry_Production_Bcf",
    "STN": "Net_Withdrawals_Storage_Bcf",
    "VG9": "Extraction_Loss_GasEq_Bcf",
    "IMN": "Net_Imports_Bcf",
    "OVI": "Supplemental_Fuels_Input_Bcf",
    "VC0": "Total_Consumption_Bcf",
    "FGW": "Gross_Withdrawals_Bcf"
}


def fetch_eia_supply_data(api_key, process_codes, start_date_str="2019-01"): # Added start_date_str parameter
    """
    Fetches monthly data for specified process codes.
    Uses the global EIA_API_URL.
    Args:
        api_key (str): Your EIA API key.
        process_codes (list): List of process codes to fetch.
        start_date_str (str): The start date for fetching data (YYYY-MM format).
                              Defaults to "2019-01". Set to None to fetch all history.
    """
    offset = 0
    limit = 5000
    all_data_dfs = []
    print(f"  Fetching EIA supply statistics using URL: {EIA_API_URL}")
    if start_date_str:
        print(f"  Start date for data: {start_date_str}")
    else:
        print(f"  Fetching all available historical data (no start date specified).")
    print(f"  For process codes: {process_codes}")

    facets_process_params = {f"facets[process][{i}]": code for i, code in enumerate(process_codes)}

    while True:
        params = {
            "api_key": api_key,
            "frequency": "monthly",
            "data[0]": "value",
            # "facets[duoarea][0]": "NUS", # Removed to match working URL, sndm implies US
            **facets_process_params,
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "offset": offset,
            "length": limit
        }

        # Add start date if provided
        if start_date_str:
            params["start"] = start_date_str
        
        # 'end' is not specified, so it will fetch up to the latest available data.
        # If you wanted to specify an end date, you'd add: params["end"] = "YYYY-MM"

        try:
            response = requests.get(EIA_API_URL, params=params, verify=False, timeout=120)
            print(f"    Request URL: {response.url}") 

            if response.status_code == 200:
                data = response.json()
                if 'response' in data and 'data' in data['response']:
                    page_data = data['response']['data']
                    if not page_data:
                        print("    No more data found from API.")
                        break
                    df_page = pd.DataFrame(page_data)
                    print(f"    Fetched {len(df_page)} records (offset {offset})...")
                    all_data_dfs.append(df_page)
                    offset += limit
                    if len(df_page) < limit:
                        print("    Last page of data fetched.")
                        break
                else:
                    print(f"    API response format unexpected. Data: {str(data)[:500]}")
                    return pd.DataFrame()
            elif response.status_code == 404:
                 print(f"    Warning: EIA API returned 404 (Not Found). URL: {response.url}")
                 print(f"    Response: {response.text[:500]}")
                 break
            elif response.status_code == 400:
                print(f"    ERROR: Bad Request (400) from EIA. URL: {response.url}")
                print(f"    Response: {response.text[:500]}")
                return pd.DataFrame()
            else:
                print(f"    ERROR: Failed to fetch data. Status: {response.status_code}, URL: {response.url}")
                print(f"    Response: {response.text[:500]}")
                return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            print(f"    ERROR: Network or request error: {e}")
            return pd.DataFrame()
        except json.JSONDecodeError as e:
            print(f"    ERROR: Could not decode JSON response: {e}. Response text: {response.text[:500]}...")
            return pd.DataFrame()
        except Exception as e:
             print(f"    ERROR: An unexpected error occurred during fetch: {e}")
             traceback.print_exc()
             return pd.DataFrame()

    if not all_data_dfs:
        print("  No dataframes were fetched from API.")
        return pd.DataFrame()

    combined_df = pd.concat(all_data_dfs, ignore_index=True)
    print(f"  Finished fetching. Total rows fetched before processing: {len(combined_df)}")
    return combined_df

def process_fetched_data(df_raw):
    """
    Processes the raw fetched data.
    """
    if df_raw.empty:
        return pd.DataFrame()

    print("  Processing fetched data...")
    required_pivot_cols = ['period', 'process', 'value']
    if not all(col in df_raw.columns for col in required_pivot_cols):
        print(f"    Critical columns for pivoting {required_pivot_cols} are missing. Available: {df_raw.columns.tolist()}")
        return pd.DataFrame()

    df_raw['datetime'] = pd.to_datetime(df_raw['period'], errors='coerce')
    df_raw['value'] = pd.to_numeric(df_raw['value'], errors='coerce')
    df_raw.dropna(subset=['datetime', 'value', 'process'], inplace=True)

    if df_raw.empty:
        print("  DataFrame is empty after type conversion and NaN drop.")
        return pd.DataFrame()

    if 'duoarea' in df_raw.columns:
        unique_duoareas = df_raw['duoarea'].unique()
        if len(unique_duoareas) > 1:
            print(f"    Warning: Multiple 'duoarea' values found: {unique_duoareas}. Pivoting might produce unexpected results if data is not for a single area per period/process.")
        elif len(unique_duoareas) == 1:
            print(f"    Data is for 'duoarea': {unique_duoareas[0]}")
    else:
        print(f"    'duoarea' column not present in fetched data (normal for this endpoint).")


    df_pivot = df_raw.pivot_table(index='datetime', columns='process', values='value')
    df_pivot.rename(columns=PROCESS_CODE_TO_COLUMN_NAME_MAP, inplace=True)

    for target_col in PROCESS_CODE_TO_COLUMN_NAME_MAP.values():
        if target_col not in df_pivot.columns:
            df_pivot[target_col] = pd.NA

    df_pivot.reset_index(inplace=True)
    df_pivot.sort_values(by='datetime', inplace=True)

    print(f"  Data processed and pivoted. Shape: {df_pivot.shape}")
    return df_pivot


def upload_to_sql_overwrite(df, table_name, db_connection_string, schema_name):
    """
    Uploads a DataFrame to a SQL table, overwriting the table if it exists.
    """
    if df.empty:
        print(f"  DataFrame is empty. Skipping SQL upload for table '{schema_name}.{table_name}'.")
        return False
    if not db_connection_string:
        print("  DB_CONNECTION_STRING is not configured. Skipping SQL upload.")
        return False
    engine = None
    try:
        print(f"\n  Attempting to upload data to SQL table: {schema_name}.{table_name}")
        engine = create_engine(db_connection_string)
        with engine.connect() as connection:
            print(f"    Successfully connected to database for table {schema_name}.[{table_name}].")
        
        df.to_sql(
            name=table_name,
            con=engine,
            schema=schema_name,
            if_exists='replace',
            index=False,
            method='multi',
            chunksize=1000
        )
        print(f"  Successfully overwrote and uploaded {len(df)} rows to table '{schema_name}.{table_name}'.")
        return True
    except SQLAlchemyError as e_sql:
        print(f"  SQLAlchemyError during SQL upload to '{schema_name}.{table_name}': {e_sql}")
        traceback.print_exc()
        return False
    except Exception as e_gen:
        print(f"  An unexpected error during SQL upload to '{schema_name}.{table_name}': {e_gen}")
        traceback.print_exc()
        return False
    finally:
        if engine:
            engine.dispose()
            print("    SQLAlchemy engine disposed after upload attempt.")

def main():
    print("\n=====================================================================")
    print("=== Starting EIA Natural Gas Supply Statistics Data Processing ===")
    print("=====================================================================")
    start_total_time = time.time()
    csv_success_flag = False
    sql_upload_success_flag = False

    process_codes_to_fetch = ["FGW", "FPD", "IMN", "OVI", "STN", "VC0", "VG4", "VG9", "VGM"]
    
    # Call fetch_eia_supply_data with the desired start date
    # If you want to make it configurable, you could pass it as a command-line argument
    # or read from a config file. For now, it's hardcoded in the function call.
    # The function `fetch_eia_supply_data` now has a default "2019-01" for start_date_str
    # so calling it without the third argument will use that default.
    # Or you can be explicit:
    # raw_data_df = fetch_eia_supply_data(API_KEY, process_codes_to_fetch, start_date_str="2019-01")

    try:
        raw_data_df = fetch_eia_supply_data(API_KEY, process_codes_to_fetch) # Uses default start_date_str="2019-01"

        if raw_data_df.empty:
            print("  No data fetched from EIA. Exiting.")
        else:
            processed_data_df = process_fetched_data(raw_data_df)

            if processed_data_df.empty:
                print("  Data processing resulted in an empty DataFrame. Exiting.")
            else:
                output_csv_path = os.path.join(BASE_OUTPUT_DIR, CSV_OUTPUT_FILENAME)
                os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
                try:
                    processed_data_df.to_csv(output_csv_path, index=False, float_format='%.3f', date_format='%Y-%m-%d')
                    print(f"  Successfully saved processed data to CSV: {output_csv_path}")
                    csv_success_flag = True
                except Exception as e_csv:
                    print(f"  ERROR saving data to CSV '{output_csv_path}': {e_csv}")
                    traceback.print_exc()

                if csv_success_flag:
                    print(f"\n--- Starting SQL Upload Process for {SQL_TABLE_NAME} ---")
                    sql_upload_success_flag = upload_to_sql_overwrite(
                        processed_data_df,
                        SQL_TABLE_NAME,
                        DB_CONNECTION_STRING,
                        DB_SCHEMA
                    )
    except Exception as e:
        print(f"  CRITICAL ERROR during main processing: {e}")
        traceback.print_exc()

    end_total_time = time.time()
    total_duration = end_total_time - start_total_time
    print("\n\n=============================================")
    print("========== Processing Summary ==========")
    if csv_success_flag:
        print(f"Supply statistics data processed and saved to CSV '{CSV_OUTPUT_FILENAME}' successfully.")
    else:
        print("Supply statistics data CSV processing FAILED or produced no data.")

    if DB_CONNECTION_STRING:
        if sql_upload_success_flag:
            print(f"Data successfully uploaded and overwritten in SQL table '{DB_SCHEMA}.{SQL_TABLE_NAME}'.")
        elif csv_success_flag:
            print(f"SQL upload FAILED for table '{DB_SCHEMA}.{SQL_TABLE_NAME}'. Check logs.")
    else:
        print("SQL upload skipped as DB_CONNECTION_STRING is not configured.")

    print(f"\nTotal execution time: {total_duration:.2f} seconds.")
    print("=============================================")

    final_success = csv_success_flag
    if DB_CONNECTION_STRING:
        final_success = final_success and sql_upload_success_flag

    if not final_success:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()