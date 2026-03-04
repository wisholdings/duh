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

# --- SQL Server Configuration (ADD THESE) ---
DB_CONNECTION_STRING = (
)
DB_SCHEMA = "dbo" # Define your schema if not dbo
SQL_TABLE_NAME = "NATGAS_Pipeline_Losses_US" # Define the target SQL table name

# --- Configuration Loading ---
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

# --- EIA API Configuration ---
API_KEY = "" # Your EIA API Key

# Disable InsecureRequestWarning if verify=False is necessary
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

def fetch_pipeline_losses_data(api_key, start_date="2019-01"):
    """
    Fetches monthly 'Pipeline and Other Losses' (VGP) data for the entire US (NUS).
    """
    api_url = "https://api.eia.gov/v2/natural-gas/cons/sum/data/"
    offset = 0
    limit = 5000
    combined_df = pd.DataFrame()
    today_date_str = datetime.now().strftime("%Y-%m")
    print(f"  Fetching 'Pipeline and Other Losses' (VGP) data for US (NUS) from {start_date} to {today_date_str}")

    while True:
        params = {
            "api_key": api_key,
            "frequency": "monthly",
            "data[0]": "value",
            "facets[duoarea][0]": "NUS",
            "facets[process][0]": "VGP", # Process code for Pipeline and Other Losses
            "start": start_date,
            "end": today_date_str,
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "offset": offset,
            "length": limit
        }

        try:
            response = requests.get(api_url, params=params, verify=False, timeout=60)
            if response.status_code == 200:
                data = response.json()
                if 'response' in data and 'data' in data['response']:
                    page_data = data['response']['data']
                    if not page_data:
                        print("    No more data found from API.")
                        break
                    df = pd.DataFrame(page_data)
                    print(f"    Fetched {len(df)} records (offset {offset})...")
                    expected_cols = ['period', 'duoarea', 'process-name', 'process', 'value']
                    if not all(col in df.columns for col in expected_cols):
                        print(f"    Warning: Missing expected columns in API response batch. Skipping batch.")
                        print(f"    Found columns: {df.columns.tolist()}")
                        print(f"    Expected columns: {expected_cols}")
                        offset += limit
                        continue
                    df_filtered = df[['period', 'duoarea', 'process-name', 'process', 'value']].copy()
                    df_filtered = df_filtered[df_filtered['process'] == 'VGP'] # Ensure it's VGP
                    if not df_filtered.empty:
                        df_filtered['period'] = pd.to_datetime(df_filtered['period'], errors='coerce')
                        df_filtered['duoarea'] = df_filtered['duoarea'].astype(str)
                        df_filtered['value'] = pd.to_numeric(df_filtered['value'], errors='coerce')
                        df_filtered.dropna(subset=['period', 'value'], inplace=True)
                        combined_df = pd.concat([combined_df, df_filtered], ignore_index=True)
                    else:
                         print("    No 'VGP' (Pipeline and Other Losses) data in this batch, or data filtered out.")
                    offset += limit
                else:
                    print(f"    API response format unexpected. Data: {data}")
                    break
            elif response.status_code == 404:
                 print(f"    Warning: Data not found (404) at offset {offset}.")
                 break
            else:
                print(f"    ERROR: Failed to fetch data. Status: {response.status_code}, Response: {response.text}")
                break
        except requests.exceptions.RequestException as e:
            print(f"    ERROR: Network or request error: {e}")
            break
        except json.JSONDecodeError as e:
            print(f"    ERROR: Could not decode JSON response: {e}. Response: {response.text[:500]}...")
            break
        except Exception as e:
             print(f"    ERROR: An unexpected error occurred during fetch: {e}")
             traceback.print_exc()
             break
    print(f"  Finished fetching. Total relevant rows fetched: {len(combined_df)}")
    return combined_df

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
            # Drop table if exists (part of 'replace' logic, but can be explicit)
            # Note: For some DBs/drivers, 'replace' might not drop and recreate constraints correctly.
            # Being explicit with DROP can be safer if specific DDL is not being managed by SQLAlchemy's 'replace'.
            # However, pandas to_sql 'replace' should handle this for simple cases.
            # connection.execute(text(f"DROP TABLE IF EXISTS {schema_name}.[{table_name}]"))
            # connection.commit()
            # print(f"    Ensured table {schema_name}.[{table_name}] is dropped if it existed.")
            pass # Relying on if_exists='replace'

        # The 'replace' option will typically:
        # 1. DROP the table if it exists.
        # 2. CREATE a new table based on the DataFrame's schema.
        # 3. INSERT the data.
        df.to_sql(
            name=table_name,
            con=engine,
            schema=schema_name,
            if_exists='replace', # This will drop the table first if it exists, then create and insert
            index=False,       # Do not write DataFrame index as a column
            method='multi',    # Use multi-value insert for efficiency with MSSQL
            chunksize=1000     # Adjust as needed
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


# --- Main Processing Loop ---
print("\n=====================================================================")
print("=== Starting EIA Natural Gas Pipeline & Other Losses Data Processing ===")
print("=====================================================================")
start_total_time = time.time()
csv_success_flag = False
sql_upload_success_flag = False

try:
    us_total_base_dir = os.path.join(BASE_OUTPUT_DIR, "us_total")
    final_dir = os.path.join(us_total_base_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    print(f"  Output directory for CSV: {final_dir}")

    pipeline_loss_df_raw = fetch_pipeline_losses_data(API_KEY, start_date="2019-01")

    if pipeline_loss_df_raw.empty:
        print(f"  No 'Pipeline and Other Losses' data fetched or processed. Exiting.")
    else:
        print(f"  Preparing data for saving...")
        df_to_save = pipeline_loss_df_raw[['period', 'value']].copy()
        output_col_name = "US_Pipe_Other_Losses_MMcf" # This will be the column name in the CSV and SQL table
        df_to_save.rename(columns={'period': 'datetime', 'value': output_col_name}, inplace=True)
        df_to_save.sort_values(by='datetime', inplace=True)
        
        # Ensure 'datetime' is just date part if it has time, for consistency if SQL type is Date
        # df_to_save['datetime'] = pd.to_datetime(df_to_save['datetime']).dt.date # Uncomment if SQL target is Date

        output_filename = "natural_gas_pipe_losses_us_total.csv"
        output_filepath = os.path.join(final_dir, output_filename)
        print(f"  Saving monthly pipeline loss data to: {output_filepath}")
        df_to_save.to_csv(output_filepath, index=False, float_format='%.3f')
        print(f"  Successfully saved data to CSV.")
        csv_success_flag = True

        # --- Upload to SQL Server ---
        if csv_success_flag: # Only attempt upload if CSV save was successful
            print(f"\n--- Starting SQL Upload Process for {SQL_TABLE_NAME} ---")
            # df_to_save already has 'datetime' and 'US_Pipe_Other_Losses_MMcf' columns
            # The data types will be inferred by to_sql: datetime64[ns] -> DATETIME2, float64 -> FLOAT
            sql_upload_success_flag = upload_to_sql_overwrite(
                df_to_save, 
                SQL_TABLE_NAME, 
                DB_CONNECTION_STRING, 
                DB_SCHEMA
            )

except Exception as e:
    print(f"  ERROR during main processing: {e}")
    traceback.print_exc()

# --- End of Processing Summary ---
end_total_time = time.time()
total_duration = end_total_time - start_total_time

print("\n\n=============================================")
print("========== Processing Summary ==========")
if csv_success_flag:
    print("Pipeline and Other Losses data processed and saved to CSV successfully.")
else:
    print("Pipeline and Other Losses data CSV processing FAILED or produced no data.")

if DB_CONNECTION_STRING: # Only report SQL status if it was configured
    if sql_upload_success_flag:
        print(f"Data successfully uploaded and overwritten in SQL table '{DB_SCHEMA}.{SQL_TABLE_NAME}'.")
    elif csv_success_flag : # If CSV was fine, but SQL failed
        print(f"SQL upload FAILED for table '{DB_SCHEMA}.{SQL_TABLE_NAME}'. Check logs above.")
    # If csv_success_flag is false, sql_upload_success_flag will also be false or not attempted.
else:
    print("SQL upload skipped as DB_CONNECTION_STRING is not configured.")


print(f"\nTotal execution time: {total_duration:.2f} seconds.")
print("=============================================")

# Exit code depends on whether at least CSV was successful if SQL wasn't configured.
# If SQL is configured, then both CSV and SQL should succeed.
final_success = False
if DB_CONNECTION_STRING:
    final_success = csv_success_flag and sql_upload_success_flag
else:
    final_success = csv_success_flag


if not final_success:
    sys.exit(1)
else:
    sys.exit(0)