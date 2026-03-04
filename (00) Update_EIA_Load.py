import logging
import io
import requests
import concurrent.futures
from datetime import datetime, timedelta, date, time
import http.client
import numpy as np
import ssl
import certifi
import pandas as pd
from io import StringIO, BytesIO
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import ResourceNotFoundError
import time as time_module
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
from itertools import combinations
import urllib.request
from dateutil.relativedelta import relativedelta
from functools import reduce
import pytz
import re
import sqlalchemy
from sqlalchemy import create_engine, text

# Define MST timezone (America/Denver for accurate local time handling)
mst_tz = pytz.timezone("Etc/GMT+7")

# API key for EIA
api_key = ""

# Define the start date for fetching data
start_date = "2025-07-01T00"

# Database connection with optimized settings (pymssql compatible)
DB_CONNECTION_STRING = (
)

# Create engine with connection pooling
engine = create_engine(
    DB_CONNECTION_STRING,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,
    connect_args={'timeout': 60}
)

# Create a persistent session for reuse
session = requests.Session()
session.verify = False  # Keep your SSL setting

# -------------------------
# Helper Functions
# -------------------------

def fetch_existing_data():
    """Fetch existing data to avoid duplicates - optimized with caching."""
    query = "SELECT DATETIME FROM PRODUCTION_REGION_LOAD_HOURLY"
    try:
        existing_data = pd.read_sql(query, engine)
        existing_data['DATETIME'] = pd.to_datetime(existing_data['DATETIME'])
        return set(existing_data['DATETIME'])
    except Exception as e:
        print(f"Warning: Error fetching existing data: {e}")
        print("Continuing without duplicate checking...")
        return set()

def _sql_get_primary_keys(engine, table_name):
    """Retrieve primary key columns from the SQL table."""
    try:
        with engine.connect() as conn:
            query = text(
                """
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                WHERE TABLE_NAME = :table_name
                    AND OBJECTPROPERTY(OBJECT_ID(CONSTRAINT_SCHEMA + '.' + CONSTRAINT_NAME), 'IsPrimaryKey') = 1
                """
            )
            result = conn.execute(query, {"table_name": table_name})
            rows = result.fetchall()
            return [row[0] for row in rows]
    except Exception as e:
        print(f"Warning: Error retrieving primary keys: {e}")
        return ['DATETIME']  # Default assumption

def sql_batch_upsert_to_database(engine, table_name, data, chunk_size=10000):
    """Perform batch upsert into SQL database using MERGE statement - optimized."""
    if data.empty:
        return
    
    primary_keys = _sql_get_primary_keys(engine, table_name)
    
    def format_value_vec(values):
        """Format values for SQL insertion - vectorized for better performance."""
        formatted = []
        for value in values:
            if pd.isna(value):
                formatted.append("NULL")
            elif isinstance(value, pd.Timestamp):
                # Convert to MST and format as SQL-compatible string
                formatted.append(f"'{value.astimezone(mst_tz).strftime('%Y-%m-%d %H:%M:%S')}'")
            elif isinstance(value, str):
                formatted.append(f"""'{value.replace("'", "''")}'""")
            else:
                formatted.append(str(value))
        return formatted
    
    columns = data.columns.tolist()
    rowcounter = 0
    
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i + chunk_size]
        if chunk.empty:
            continue
        
        print(f"Upserting chunk for {data.columns[1]}: {chunk['DATETIME'].min()} to {chunk['DATETIME'].max()}")
        
        # Use vectorized operations for better performance
        values_list = []
        for row in chunk.values:
            formatted_row = format_value_vec(row)
            values_list.append(f"({', '.join(formatted_row)})")
        
        values_clause = ", ".join(values_list)
        
        merge_query = f"""
        MERGE INTO {table_name} AS target
        USING (VALUES {values_clause}) AS source ({', '.join(columns)})
        ON {" AND ".join([f"target.{key} = source.{key}" for key in primary_keys])}
        WHEN MATCHED THEN
            UPDATE SET {", ".join([f"target.{col} = source.{col}" for col in columns if col not in primary_keys])}
        WHEN NOT MATCHED THEN
            INSERT ({", ".join(columns)})
            VALUES ({", ".join([f"source.{col}" for col in columns])});
        """
        
        try:
            with engine.connect() as sql_connection:
                with sql_connection.begin() as transaction:
                    result = sql_connection.execute(text(merge_query))
                    rowcounter += result.rowcount
        except Exception as e:
            print(f"Error in upsert chunk: {e}")
            continue
    
    print(f"Total upserted/updated rows in {table_name}: {rowcounter}")

def fetch_hourly_region_data(api_key, respondent, start_date, session):
    """Fetch all hourly region data from EIA API from start_date to the latest available - optimized."""
    api_url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    all_data = []
    offset = 0
    page_size = 5000

    while True:
        params = {
            "api_key": api_key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": respondent,
            "start": start_date,
            # No 'end' parameter to fetch all available data
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": page_size,
        }
        
        response = session.get(api_url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if "response" in data and "data" in data["response"]:
                page_data = data["response"]["data"]
                if not page_data:
                    break
                all_data.extend(page_data)
                offset += page_size
            else:
                print(f"Data format mismatch or empty data for {respondent}.")
                break
        else:
            print(f"Failed to fetch data for {respondent}: {response.status_code}")
            break
    
    if all_data:
        df = pd.DataFrame(all_data)
        print(f"Fetched {len(df)} records for {respondent}")
        return df
    else:
        return pd.DataFrame()

def process_region_data(df, respondent_name, desired_end_date_mst):
    """Process raw EIA data, trimming to the desired end date in MST - optimized."""
    # Convert timestamps more efficiently
    df['DateTime'] = pd.to_datetime(df['period'], utc=True).dt.tz_convert(mst_tz)
    df['DATETIME'] = df['DateTime'].dt.floor('h')
    
    # Filter for demand data only
    df = df[df['type-name'] == 'Demand']
    
    # Define start date
    start_date_mst = mst_tz.localize(datetime(2025, 6, 1, 0, 0, 0))
    
    print(f"Raw MST range before filter for {respondent_name}: {df['DATETIME'].min()} to {df['DATETIME'].max()}")
    print(f"Filtering between {start_date_mst} and {desired_end_date_mst} for {respondent_name}")
    
    # Apply date filters
    df = df[(df['DATETIME'] >= start_date_mst) & (df['DATETIME'] <= desired_end_date_mst)]
    
    print(f"Filtered range for {respondent_name}: {df['DATETIME'].min()} to {df['DATETIME'].max()} (MST)")
    
    # Drop unnecessary columns
    df.drop(
        columns=['period', 'respondent', 'respondent-name', 'type', 'value-units', 'DateTime', 'type-name'],
        inplace=True,
        errors='ignore'
    )
    
    # Rename value column
    column_name = f"{respondent_name.upper()}_LOAD"
    df.rename(columns={'value': column_name}, inplace=True)
    
    # Remove duplicates more efficiently
    df.drop_duplicates(subset=['DATETIME'], inplace=True)
    
    print(f"Final processed range for {respondent_name}: {df['DATETIME'].min()} to {df['DATETIME'].max()}")
    
    return df

def fetch_regions_parallel(api_key, respondents, start_date_utc, session, max_workers=6):
    """Fetch region data for multiple respondents in parallel."""
    region_data = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all region requests
        future_to_respondent = {
            executor.submit(fetch_hourly_region_data, api_key, respondent_code, start_date_utc, session): (respondent_code, respondent_name)
            for respondent_code, respondent_name in respondents.items()
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_respondent):
            respondent_code, respondent_name = future_to_respondent[future]
            
            try:
                df_raw = future.result()
                if not df_raw.empty:
                    region_data[respondent_name] = {
                        'data': df_raw,
                        'code': respondent_code
                    }
                else:
                    print(f"No data fetched for {respondent_name}.")
            except Exception as exc:
                print(f"Region {respondent_name} generated an exception: {exc}")
    
    return region_data

def fetch_and_append_region_data(api_key, respondents):
    """Fetch all data and trim to desired end date - optimized with parallel processing."""
    # Calculate time ranges
    now_mst = datetime.now(mst_tz)
    print(f"Current time in MST: {now_mst}")
    
    day_before_yesterday = now_mst.date() - timedelta(days=2)
    desired_end_date_mst = mst_tz.localize(datetime.combine(day_before_yesterday, time(23, 0)))
    
    start_date = mst_tz.localize(datetime(2025, 6, 1, 0, 0, 0))
    start_date_utc = start_date.astimezone(pytz.UTC).strftime("%Y-%m-%dT%H")
    
    print(f"Fetching all data from {start_date_utc} (UTC) onward")
    print(f"Desired end date in MST: {desired_end_date_mst}")
    
    # Step 1: Fetch all regions in parallel
    print(f"\nFetching data for all {len(respondents)} regions in parallel...")
    region_data = fetch_regions_parallel(api_key, respondents, start_date_utc, session, max_workers=8)
    
    # Step 2: Process and upload each region's data
    print(f"\nProcessing and uploading data for {len(region_data)} regions...")
    for respondent_name, region_info in region_data.items():
        df_raw = region_info['data']
        respondent_code = region_info['code']
        
        try:
            print(f"Processing data for {respondent_name} (code: {respondent_code})...")
            df_processed = process_region_data(df_raw, respondent_name, desired_end_date_mst)
            
            if not df_processed.empty:
                sql_batch_upsert_to_database(engine, "PRODUCTION_REGION_LOAD_HOURLY", df_processed)
                print(f"✅ Successfully processed {respondent_name}")
            else:
                print(f"No new/valid data after processing for {respondent_name}.")
                
        except Exception as e:
            print(f"❌ Error processing {respondent_name}: {e}")
            continue

# -------------------------
# Main Execution
# -------------------------
respondents = {
    "TEX": "texas",
    "NW": "northwest",
    "CAL": "california",
    "CAR": "carolina",
    "CENT": "central",
    "FLA": "florida",
    "MIDA": "midatlantic",
    "MIDW": "midwest",
    "NE": "newengland",
    "NY": "newyork",
    "SE": "southeast",
    "SW": "southwest",
    "TEN": "tennessee",
}

print(f"Starting region load data collection for {len(respondents)} regions...")
fetch_and_append_region_data(api_key, respondents)

# Close the session
session.close()
print("\n🎉 All region load data updated with parallel processing!")
print("Note: Data processed without time zone conversions, filtered to day before yesterday 23:00 MST.")