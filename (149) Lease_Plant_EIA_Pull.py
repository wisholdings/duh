import os
import json
import sys
import pandas as pd
import traceback
import time
from datetime import datetime
import requests
import warnings

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
    # --- !! SET YOUR DEFAULT BASE DIRECTORY HERE IF CONFIG FAILS !! ---
    # Changed default path slightly to avoid conflict if running multiple scripts with failed config
    BASE_OUTPUT_DIR = "D:\\EIA_STACK_PIPELOSS" # EXAMPLE: Change this path
    print(f"Using default BASE_OUTPUT_DIR: {BASE_OUTPUT_DIR}")

# --- EIA API Configuration ---
API_KEY = "" # Your EIA API Key

# Disable InsecureRequestWarning if verify=False is necessary
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

def fetch_lease_losses_data(api_key, start_date="2019-01"):
    """
    Fetches monthly '
     
       and Other Losses' (VGL) data for the entire US (NUS).

    Args:
        api_key (str): Your EIA API key.
        start_date (str): The start date for fetching data (YYYY-MM format).

    Returns:
        pandas.DataFrame: DataFrame with 'period', 'duoarea', 'process-name', 'value'
                          for the specified process, or empty DataFrame on error.
    """
    # API endpoint for Natural Gas Summary (includes lease losses under 'cons' -> 'sum')
    api_url = "https://api.eia.gov/v2/natural-gas/cons/sum/data/"
    offset = 0
    limit = 5000 # Max records per request
    combined_df = pd.DataFrame()
    today_date_str = datetime.now().strftime("%Y-%m")
    print(f"  Fetching 'lease and Other Losses' (VGL) data for US (NUS) from {start_date} to {today_date_str}")

    while True:
        params = {
            "api_key": api_key,
            "frequency": "monthly",
            "data[0]": "value",
            "facets[duoarea][0]": "NUS",   # Duoarea for Total US
            "facets[process][0]": "VGL", # Process code for lease and Other Losses
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
                    if not page_data: # No more data left to fetch
                        print("    No more data found from API.")
                        break

                    df = pd.DataFrame(page_data)
                    print(f"    Fetched {len(df)} records (offset {offset})...")

                    expected_cols = ['period', 'duoarea', 'process-name', 'process', 'value'] # 'process' (code) is also returned
                    if not all(col in df.columns for col in expected_cols):
                        print(f"    Warning: Missing expected columns in API response batch. Skipping batch.")
                        print(f"    Found columns: {df.columns.tolist()}")
                        print(f"    Expected columns: {expected_cols}")
                        offset += limit
                        continue

                    # Keep only necessary columns (including 'process' to verify it's VGL)
                    df_filtered = df[['period', 'duoarea', 'process-name', 'process', 'value']].copy()

                    # Explicitly check if the fetched data is indeed for 'VGL'
                    # This is a safeguard as we requested it, but good to confirm.
                    df_filtered = df_filtered[df_filtered['process'] == 'VGL']


                    if not df_filtered.empty:
                        df_filtered['period'] = pd.to_datetime(df_filtered['period'], errors='coerce')
                        df_filtered['duoarea'] = df_filtered['duoarea'].astype(str)
                        df_filtered['value'] = pd.to_numeric(df_filtered['value'], errors='coerce')
                        df_filtered.dropna(subset=['period', 'value'], inplace=True)

                        combined_df = pd.concat([combined_df, df_filtered], ignore_index=True)
                    else:
                         print("    No 'VGL' (lease and Other Losses) data in this batch, or data filtered out.")

                    offset += limit

                else:
                    print(f"    API response format unexpected. Data: {data}")
                    break

            elif response.status_code == 404:
                 print(f"    Warning: Data not found (404) at offset {offset}. May indicate no data for this period/filter.")
                 break
            else:
                print(f"    ERROR: Failed to fetch data. Status: {response.status_code}, Response: {response.text}")
                break
        except requests.exceptions.RequestException as e:
            print(f"    ERROR: Network or request error: {e}")
            break
        except json.JSONDecodeError as e:
            print(f"    ERROR: Could not decode JSON response: {e}")
            print(f"    Response Text: {response.text[:500]}...")
            break
        except Exception as e:
             print(f"    ERROR: An unexpected error occurred during fetch: {e}")
             traceback.print_exc()
             break

    print(f"  Finished fetching. Total relevant rows fetched: {len(combined_df)}")
    return combined_df


# --- Main Processing Loop ---
print("\n=====================================================================")
print("=== Starting EIA Natural Gas lease & Other Losses Data Processing ===")
print("=====================================================================")
start_total_time = time.time()
success_flag = False

try:
    # --- Define Paths ---
    # Since it's US total, no sub-region folder is created by default in this script
    # Output directly to BASE_OUTPUT_DIR or a specific subfolder if desired.
    # For consistency with your other scripts, let's make a "us_total" subfolder.
    us_total_base_dir = os.path.join(BASE_OUTPUT_DIR, "us_total") # Output to a 'us_total' subdir
    final_dir = os.path.join(us_total_base_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    print(f"  Output directory: {final_dir}")

    # --- Fetch Data ---
    # The API documentation implies data for losses might not go back as far as other series.
    # Let's try from 2001-01 which is a common start for many EIA series.
    # Adjust if you know the specific availability. For this process "VGL", 2019 might be too recent.
    # The provided X-Params example had null for start/end, implying all available data.
    # Let's fetch from 2001-01 as a reasonable starting point for historical data.
    # If you want *all* data, you can try omitting start/end or setting start to a very early date.
    # However, the API might have limits or default ranges if start/end are null.
    # For safety, we'll set a start.
    lease_loss_df_raw = fetch_lease_losses_data(API_KEY, start_date="2019-01")


    if lease_loss_df_raw.empty:
        print(f"  No 'lease and Other Losses' data fetched or processed. Exiting.")
    else:
        # --- Data is already monthly, no further summation needed by duoarea or process ---
        # We are directly fetching the specific process for NUS.
        # Just need to select and rename columns for the output CSV.
        print(f"  Preparing data for saving...")
        
        df_to_save = lease_loss_df_raw[['period', 'value']].copy()
        
        # Rename columns for clarity in the final CSV
        output_col_name = "US_Lease_Other_Losses_MMcf"
        df_to_save.rename(columns={'period': 'datetime', 'value': output_col_name}, inplace=True)

        # Sort by date
        df_to_save.sort_values(by='datetime', inplace=True)

        # --- Save Locally ---
        output_filename = "natural_gas_lease_losses_us_total.csv"
        output_filepath = os.path.join(final_dir, output_filename)
        print(f"  Saving monthly lease loss data to: {output_filepath}")
        df_to_save.to_csv(output_filepath, index=False, float_format='%.3f')
        print(f"  Successfully saved data.")
        success_flag = True

except Exception as e:
    print(f"  ERROR during processing: {e}")
    traceback.print_exc()

# --- End of Processing Summary ---
end_total_time = time.time()
total_duration = end_total_time - start_total_time

print("\n\n=============================================")
print("========== Processing Summary ==========")
if success_flag:
    print("lease and Other Losses data processed successfully.")
else:
    print("lease and Other Losses data processing FAILED or produced no data.")
print(f"\nTotal execution time: {total_duration:.2f} seconds.")
print("=============================================")

if not success_flag:
    sys.exit(1)
else:
    sys.exit(0)