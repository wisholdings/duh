import os
import pandas as pd
import datetime
import time
import openmeteo_requests
import requests_cache
from retry_requests import retry
from sqlalchemy import create_engine, text, Integer, Float, Date
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

# --- Configuration ---

API_KEY = ""  # Verify this key
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

OPENMETEO_ARCHIVE_URL = "https://customer-archive-api.open-meteo.com/v1/archive"
OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Define locations with airport codes
alberta_locations = {
    "CYEG": {"latitude": 53.3097, "longitude": -113.58},  # Edmonton AB
    "CYMM": {"latitude": 56.6533, "longitude": -111.22},  # Fort McMurray AB
    "CYQF": {"latitude": 52.1822, "longitude": -113.89},  # Red Deer AB
    "CYQU": {"latitude": 55.1794, "longitude": -118.89},  # Grand Prairie AB
    "CYXJ": {"latitude": 56.2381, "longitude": -120.74},  # Fort St. John BC
    "CYYC": {"latitude": 51.1139, "longitude": -114.02},  # Calgary AB
    "CYYE": {"latitude": 58.8364, "longitude": -122.60},  # Fort Nelson BC
    "CZPC": {"latitude": 49.5206, "longitude": -113.99}   # Pincher Creek AB
}

# Define weather variables (for fetching raw data)
daily_weather_variables = [
    "temperature_2m_mean"  # Needed for HDD/CDD calculation
]

INITIAL_HISTORICAL_START_DATE = datetime.date(2015, 1, 1)
today = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)
today_str = today.strftime('%Y-%m-%d')
yesterday_str = yesterday.strftime('%Y-%m-%d')

# New: Define recent past range to cover potential gaps
RECENT_PAST_DAYS = 2  # Fetch 5 days back to ensure today-2 and today-1 are included
recent_past_start_date = today - datetime.timedelta(days=RECENT_PAST_DAYS)
recent_past_start_date_str = recent_past_start_date.strftime('%Y-%m-%d')

MAX_OPENMETEO_NO_DATA_RETRIES = 3
OPENMETEO_NO_DATA_RETRY_DELAY_SECONDS = 15

# --- Database Setup ---

LOCAL_DB_PATH = "./x.db"
DB_CONNECTION_STRING = f"sqlite:///{LOCAL_DB_PATH}"
sql_table_name = "y"

# Define SQL columns for HDD and CDD
wide_sql_columns_definitions = []
wide_sql_columns_definitions.append("Date DATE NOT NULL")
wide_sql_columns_definitions.append("Date_Modified DATE NOT NULL")
for location_code in alberta_locations.keys():
    for var_name in ['HDD', 'CDD']:
        wide_col_name = f"{location_code}_{var_name}"
        wide_sql_columns_definitions.append(f"{wide_col_name} REAL")
wide_sql_columns_definitions.append("PRIMARY KEY (Date, Date_Modified)")

columns_definition_sql = ",\n    ".join(wide_sql_columns_definitions)
create_table_sql = f"""
CREATE TABLE IF NOT EXISTS "{sql_table_name}" (
{columns_definition_sql}
)
"""

def get_sqlalchemy_dtype_for_wide_column(column_name):
    if column_name in ['Date', 'Date_Modified']:
        return Date
    if column_name.endswith('_HDD') or column_name.endswith('_CDD'):
        return Float
    print(f"Warning: Could not infer SQLAlchemy dtype for '{column_name}'. Using Float.")
    return Float

# --- Main Process ---

engine = None
all_weather_data_long_format_current = []
all_historical_data_for_normals = []

try:
    print("Creating SQLAlchemy engine...")
    engine = create_engine(DB_CONNECTION_STRING, echo=False)
    with engine.connect() as connection:
        print("Database connection successful.")
        connection.execute(text(create_table_sql))
        connection.commit()
        print(f"Table '{sql_table_name}' checked/created.")

        historical_start_date_str = INITIAL_HISTORICAL_START_DATE.strftime('%Y-%m-%d')
        # Change: Extend historical fetch to today to maximize archive data
        historical_end_date_str = today_str  # Changed from yesterday_str
        forecast_start_date_str = today_str
        forecast_end_date_date = today + datetime.timedelta(days=15)
        forecast_end_date_str = forecast_end_date_date.strftime('%Y-%m-%d')

        print(f"\nHistorical range: {historical_start_date_str} to {historical_end_date_str}")
        print(f"Recent past range: {recent_past_start_date_str} to {yesterday_str}")
        print(f"Forecast range: {forecast_start_date_str} to {forecast_end_date_str}")

        # --- Fetch Weather Data ---
        for location_name_key, coords in alberta_locations.items():
            cleaned_location_name = location_name_key  # Use airport code directly
            print(f"\nProcessing weather data for {location_name_key}...")

            # Historical Data (Archive API)
            retries = 0
            historical_df_current = pd.DataFrame()
            historical_df_for_normals = pd.DataFrame()
            while retries < MAX_OPENMETEO_NO_DATA_RETRIES:
                print(f"Fetching Historical data for {location_name_key} (Attempt {retries + 1})...")
                params_hist = {
                    "latitude": coords['latitude'],
                    "longitude": coords['longitude'],
                    "start_date": historical_start_date_str,
                    "end_date": historical_end_date_str,
                    "daily": daily_weather_variables,
                    "timezone": "auto",
                    "apikey": API_KEY
                }
                try:
                    responses_hist = openmeteo.weather_api(OPENMETEO_ARCHIVE_URL, params=params_hist,verify=False)
                    response_hist = responses_hist[0]
                    daily_hist = response_hist.Daily()
                    if daily_hist.TimeEnd() <= daily_hist.Time():
                        print(f"No historical data for {location_name_key}.")
                        retries += 1
                        time.sleep(OPENMETEO_NO_DATA_RETRY_DELAY_SECONDS)
                        continue
                    date_index_hist = pd.date_range(
                        start=pd.to_datetime(daily_hist.Time(), unit="s", utc=True),
                        end=pd.to_datetime(daily_hist.TimeEnd(), unit="s", utc=True),
                        freq=pd.Timedelta(seconds=daily_hist.Interval()),
                        inclusive="left"
                    )
                    historical_data = {"Date": [d.date() for d in date_index_hist]}
                    for i, var_name in enumerate(daily_weather_variables):
                        historical_data[var_name] = daily_hist.Variables(i).ValuesAsNumpy()
                    historical_df_for_normals = pd.DataFrame(data=historical_data)
                    historical_df_for_normals['location_name'] = cleaned_location_name
                    # Filter historical data for current insertion up to yesterday
                    historical_df_current = historical_df_for_normals[historical_df_for_normals['Date'] <= yesterday]
                    all_historical_data_for_normals.append(historical_df_for_normals)
                    print(f"Processed {len(historical_df_current)} days for {location_name_key}.")
                    break
                except Exception as e:
                    print(f"Error fetching historical data: {e}")
                    retries += 1
                    if retries == MAX_OPENMETEO_NO_DATA_RETRIES:
                        print(f"Max retries reached for {location_name_key}.")
                        break
                    time.sleep(OPENMETEO_NO_DATA_RETRY_DELAY_SECONDS)

            # New: Recent Past Data (Forecast API to cover today-5 to yesterday)
            retries = 0
            recent_past_df = pd.DataFrame()
            while retries < MAX_OPENMETEO_NO_DATA_RETRIES:
                print(f"Fetching Recent Past data for {location_name_key} from {recent_past_start_date_str} to {yesterday_str} (Attempt {retries + 1})...")
                params_recent = {
                    "latitude": coords['latitude'],
                    "longitude": coords['longitude'],
                    "start_date": recent_past_start_date_str,
                    "end_date": yesterday_str,
                    "daily": daily_weather_variables,
                    "timezone": "auto",
                    "apikey": API_KEY
                }
                try:
                    responses_recent = openmeteo.weather_api(OPENMETEO_FORECAST_URL, params=params_recent,verify=False)
                    response_recent = responses_recent[0]
                    daily_recent = response_recent.Daily()
                    if daily_recent.TimeEnd() <= daily_recent.Time():
                        print(f"No recent past data for {location_name_key}.")
                        retries += 1
                        time.sleep(OPENMETEO_NO_DATA_RETRY_DELAY_SECONDS)
                        continue
                    date_index_recent = pd.date_range(
                        start=pd.to_datetime(daily_recent.Time(), unit="s", utc=True),
                        end=pd.to_datetime(daily_recent.TimeEnd(), unit="s", utc=True),
                        freq=pd.Timedelta(seconds=daily_recent.Interval()),
                        inclusive="left"
                    )
                    recent_data = {"Date": [d.date() for d in date_index_recent]}
                    for i, var_name in enumerate(daily_weather_variables):
                        recent_data[var_name] = daily_recent.Variables(i).ValuesAsNumpy()
                    recent_past_df = pd.DataFrame(data=recent_data)
                    recent_past_df['location_name'] = cleaned_location_name
                    print(f"Processed {len(recent_past_df)} days of recent past data.")
                    break
                except Exception as e:
                    print(f"Error fetching recent past data: {e}")
                    retries += 1
                    if retries == MAX_OPENMETEO_NO_DATA_RETRIES:
                        print(f"Max retries reached for {location_name_key}.")
                        break
                    time.sleep(OPENMETEO_NO_DATA_RETRY_DELAY_SECONDS)

            # Forecast Data
            retries = 0
            forecast_df = pd.DataFrame()
            while retries < MAX_OPENMETEO_NO_DATA_RETRIES:
                print(f"Fetching Forecast data for {location_name_key} (Attempt {retries + 1})...")
                params_fcst = {
                    "latitude": coords['latitude'],
                    "longitude": coords['longitude'],
                    "start_date": forecast_start_date_str,
                    "end_date": forecast_end_date_str,
                    "daily": daily_weather_variables,
                    "timezone": "auto",
                    "apikey": API_KEY
                }
                try:
                    responses_fcst = openmeteo.weather_api(OPENMETEO_FORECAST_URL, params=params_fcst,verify=False)
                    response_fcst = responses_fcst[0]
                    daily_fcst = response_fcst.Daily()
                    if daily_fcst.TimeEnd() <= daily_fcst.Time():
                        print(f"No forecast data for {location_name_key}.")
                        retries += 1
                        time.sleep(OPENMETEO_NO_DATA_RETRY_DELAY_SECONDS)
                        continue
                    date_index_fcst = pd.date_range(
                        start=pd.to_datetime(daily_fcst.Time(), unit="s", utc=True),
                        end=pd.to_datetime(daily_fcst.TimeEnd(), unit="s", utc=True),
                        freq=pd.Timedelta(seconds=daily_fcst.Interval()),
                        inclusive="left"
                    )
                    forecast_data = {"Date": [d.date() for d in date_index_fcst]}
                    for i, var_name in enumerate(daily_weather_variables):
                        forecast_data[var_name] = daily_fcst.Variables(i).ValuesAsNumpy()
                    forecast_df = pd.DataFrame(data=forecast_data)
                    forecast_df['location_name'] = cleaned_location_name
                    print(f"Processed {len(forecast_df)} days of forecast data.")
                    break
                except Exception as e:
                    print(f"Error fetching forecast data: {e}")
                    retries += 1
                    if retries == MAX_OPENMETEO_NO_DATA_RETRIES:
                        print(f"Max retries reached for {location_name_key}.")
                        break
                    time.sleep(OPENMETEO_NO_DATA_RETRY_DELAY_SECONDS)

            # Combine Historical, Recent Past, and Forecast
            combined_loc_df_current = pd.concat([historical_df_current, recent_past_df, forecast_df], ignore_index=True)
            if not combined_loc_df_current.empty:
                # Remove duplicates based on Date and location_name
                combined_loc_df_current = combined_loc_df_current.drop_duplicates(
                    subset=['Date', 'location_name'], keep='last'
                ).reset_index(drop=True)
                combined_loc_df_current['Date_Modified'] = today
                all_weather_data_long_format_current.append(combined_loc_df_current)
                print(f"Added {len(combined_loc_df_current)} rows for {location_name_key}.")

        # --- Calculate HDD and CDD ---
        combined_current_long_df = pd.DataFrame()
        if all_weather_data_long_format_current:
            combined_current_long_df = pd.concat(all_weather_data_long_format_current, ignore_index=True)
            combined_current_long_df['HDD'] = combined_current_long_df['temperature_2m_mean'].apply(lambda x: max(0, 18 - x))
            combined_current_long_df['CDD'] = combined_current_long_df['temperature_2m_mean'].apply(lambda x: max(0, x - 18))
            combined_current_long_df = combined_current_long_df.drop_duplicates(
                subset=['location_name', 'Date', 'Date_Modified']
            ).reset_index(drop=True)
            print("\nHDD and CDD calculated.")

        # --- Pivot Current Data ---
        current_data_wide_df = pd.DataFrame()
        if not combined_current_long_df.empty:
            print("\nPivoting current weather data...")
            current_pivot_df = combined_current_long_df.pivot(
                index=['Date', 'Date_Modified'],
                columns='location_name',
                values=['HDD', 'CDD']
            )
            current_pivot_df.columns = [f"{loc}_{var}" for var, loc in current_pivot_df.columns]
            current_data_wide_df = current_pivot_df.reset_index()
            print(f"Pivoted {len(current_data_wide_df)} rows.")
            current_data_wide_df.to_csv('stats.csv', index=False)
            print("Saved to stats.csv.")

        # --- Calculate Normals ---
        daily_normals_wide_df = pd.DataFrame()
        if all_historical_data_for_normals:
            print("\nCalculating daily normals...")
            combined_historical_for_normals_long_df = pd.concat(all_historical_data_for_normals, ignore_index=True)
            combined_historical_for_normals_long_df['HDD'] = combined_historical_for_normals_long_df['temperature_2m_mean'].apply(lambda x: max(0, 18 - x))
            combined_historical_for_normals_long_df['CDD'] = combined_historical_for_normals_long_df['temperature_2m_mean'].apply(lambda x: max(0, x - 18))
            combined_historical_for_normals_long_df['month'] = combined_historical_for_normals_long_df['Date'].apply(lambda d: d.month)
            combined_historical_for_normals_long_df['day'] = combined_historical_for_normals_long_df['Date'].apply(lambda d: d.day)
            daily_normals_long_df_averaged = combined_historical_for_normals_long_df.groupby(
                ['location_name', 'month', 'day']
            )[['HDD', 'CDD']].mean().reset_index()

            normals_start_date = today + datetime.timedelta(days=16)
            normals_end_date = normals_start_date + pd.DateOffset(years=3) - datetime.timedelta(days=1)
            normals_date_range = pd.date_range(start=normals_start_date, end=normals_end_date, freq='D').date
            future_dates_df = pd.DataFrame({'Date': normals_date_range})
            future_dates_df['month'] = future_dates_df['Date'].apply(lambda d: d.month)
            future_dates_df['day'] = future_dates_df['Date'].apply(lambda d: d.day)

            all_future_date_locations = []
            for _, row in future_dates_df.iterrows():
                for loc_name in alberta_locations.keys():
                    all_future_date_locations.append({
                        'Date': row['Date'],
                        'month': row['month'],
                        'day': row['day'],
                        'location_name': loc_name
                    })
            all_future_date_locations_df = pd.DataFrame(all_future_date_locations)

            normals_long_df_filled = pd.merge(
                all_future_date_locations_df,
                daily_normals_long_df_averaged,
                on=['location_name', 'month', 'day'],
                how='left'
            )
            normals_long_df_filled.drop(columns=['month', 'day'], inplace=True)
            normals_long_df_filled['Date_Modified'] = today

            normals_pivot_df = normals_long_df_filled.pivot(
                index=['Date', 'Date_Modified'],
                columns='location_name',
                values=['HDD', 'CDD']
            )
            normals_pivot_df.columns = [f"{loc}_{var}" for var, loc in normals_pivot_df.columns]
            daily_normals_wide_df = normals_pivot_df.reset_index()
            print(f"Calculated {len(daily_normals_wide_df)} rows of normals.")

        # --- Combine Data ---
        final_wide_df = pd.DataFrame()
        if not current_data_wide_df.empty and not daily_normals_wide_df.empty:
            final_wide_df = pd.concat([current_data_wide_df, daily_normals_wide_df], ignore_index=True)
            print(f"Combined {len(final_wide_df)} rows.")
        elif not current_data_wide_df.empty:
            final_wide_df = current_data_wide_df
            print("Only current data available.")
        elif not daily_normals_wide_df.empty:
            final_wide_df = daily_normals_wide_df
            print("Only normals data available.")

        # --- Insert Data ---
        if not final_wide_df.empty:
            print(f"\nInserting {len(final_wide_df)} rows into '{sql_table_name}'...")
            actual_wide_dtypes_map = {col: get_sqlalchemy_dtype_for_wide_column(col) for col in final_wide_df.columns}
            try:
                final_wide_df.to_sql(
                    name=sql_table_name,
                    con=engine,
                    if_exists='append',
                    index=False,
                    dtype=actual_wide_dtypes_map
                )
                print("Data inserted.")
            except IntegrityError as ex:
                print(f"IntegrityError: {ex}")
                print("Some rows may already exist.")
            except Exception as e:
                print(f"Error during insertion: {e}")

        # --- Cleanup ---
        print(f"\nCleaning up table '{sql_table_name}'...")
        try:
            with engine.connect() as cleanup_connection:
                cutoff_date_row = cleanup_connection.execute(
                    text(f"SELECT Date_Modified FROM \"{sql_table_name}\" ORDER BY Date_Modified DESC LIMIT 1 OFFSET 1")
                ).fetchone()
                if cutoff_date_row:
                    cutoff_date = cutoff_date_row[0]
                    print(f"Deleting rows older than {cutoff_date}.")
                    delete_result = cleanup_connection.execute(
                        text(f"DELETE FROM \"{sql_table_name}\" WHERE Date_Modified < :cutoff_date"),
                        {'cutoff_date': cutoff_date}
                    )
                    cleanup_connection.commit()
                    print(f"Deleted {delete_result.rowcount} rows.")
                else:
                    print("No cleanup needed.")
        except Exception as e:
            print(f"Cleanup error: {e}")

except Exception as e:
    print(f"An error occurred: {e}")
    raise
finally:
    if engine:
        engine.dispose()
    print("Script finished.")