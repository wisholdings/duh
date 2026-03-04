import logging
import io
import requests
import concurrent.futures
from datetime import datetime, timedelta, date
import http.client
import numpy as np
import ssl
import certifi
import pandas as pd
from io import StringIO, BytesIO
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import ResourceNotFoundError
import time
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
from sqlalchemy import create_engine, text, MetaData, Table, Column, DateTime, Float, inspect

# Define MST timezone (America/Denver for accurate local time handling)
mst_tz = pytz.timezone("America/Denver")

# API key for EIA
api_key = ""

# Define the start date for fetching data
start_date = "2025-07-01T00"

# Calculate "day before yesterday at 23:00" in local Denver time (MST)
now_local = datetime.now(mst_tz)
day_before_yesterday = now_local.date() - timedelta(days=2)
desired_end_time_local = mst_tz.localize(datetime.combine(day_before_yesterday, datetime.strptime("23:00", "%H:%M").time()))

# Fetch all data up to the current UTC time
end_date = datetime.now(pytz.UTC).strftime("%Y-%m-%dT%H")

print(f"API end_date (UTC): {end_date}")

# Common fuel types
fuel_types = {
    "NG": "natural_gas",
    "WND": "wind",
    "SUN": "solar",
    "COL": "coal",
    "NUC": "nuclear",
    "WAT": "hydro",
    "OTH": "other",
    "OIL": "oil",
    "BAT": "bat",
    "GEO": "geo",
    "OES": "oes",
    "PS": "ps",
    "SNB": "snb",
}

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

# List of region configs
region_configs = [
    {
        "respondent_code": "TEN",
        "respondent_name": "tennessee",
        "table_name": "PRODUCTION_TENNESSEE_ENERGY_EIA_HOURLY",
    }
]

# -------------------------
# Helper Functions
# -------------------------

def create_table_if_not_exists(engine, table_name, region_name):
    """Create table if it doesn't exist with proper schema for energy data."""
    try:
        inspector = inspect(engine)
        
        if inspector.has_table(table_name):
            print(f"[INFO] Table {table_name} already exists. Checking column structure...")
            ensure_all_columns_exist(engine, table_name, region_name)
            return
        
        print(f"[INFO] Creating table {table_name}...")
        
        # Define the table schema
        metadata = MetaData()
        
        # Create columns dynamically based on fuel types and region
        columns = [
            Column('DATETIME', DateTime, primary_key=True, nullable=False),
        ]
        
        # Add fuel type columns
        for fuel_abbrev in fuel_types.keys():
            col_name = f"{region_name.upper()}_{fuel_abbrev}_MW"
            columns.append(Column(col_name, Float, nullable=True))
        
        table = Table(table_name, metadata, *columns)
        
        # Create the table
        metadata.create_all(engine)
        print(f"[SUCCESS] Table {table_name} created successfully.")
        
        # Optionally create an index on DATETIME for better query performance
        try:
            with engine.connect() as conn:
                index_query = f"CREATE INDEX IX_{table_name}_DATETIME ON {table_name} (DATETIME)"
                conn.execute(text(index_query))
                conn.commit()
                print(f"[INFO] Index created on DATETIME column for {table_name}")
        except Exception as e:
            print(f"[WARNING] Could not create index on {table_name}: {e}")
                
    except Exception as e:
        print(f"[ERROR] Failed to create table {table_name}: {e}")
        print(f"[INFO] Attempting to continue without table creation...")
        # Don't raise the exception, let the script continue

def ensure_all_columns_exist(engine, table_name, region_name):
    """Ensure all required columns exist in the table, add missing ones."""
    try:
        inspector = inspect(engine)
        existing_columns = [col['name'].upper() for col in inspector.get_columns(table_name)]
        
        # Define expected columns based on fuel types
        expected_columns = ['DATETIME']
        for fuel_abbrev in fuel_types.keys():
            col_name = f"{region_name.upper()}_{fuel_abbrev}_MW"
            expected_columns.append(col_name)
        
        # Find missing columns
        missing_columns = [col for col in expected_columns if col not in existing_columns]
        
        if missing_columns:
            print(f"[INFO] Adding missing columns to {table_name}: {missing_columns}")
            
            with engine.connect() as conn:
                for col_name in missing_columns:
                    if col_name != 'DATETIME':  # Don't try to add the primary key column
                        try:
                            alter_query = f"ALTER TABLE {table_name} ADD {col_name} FLOAT NULL"
                            conn.execute(text(alter_query))
                            print(f"[SUCCESS] Added column {col_name} to {table_name}")
                        except Exception as e:
                            print(f"[WARNING] Could not add column {col_name} to {table_name}: {e}")
                conn.commit()
        else:
            print(f"[INFO] All required columns already exist in {table_name}")
    except Exception as e:
        print(f"[WARNING] Could not check/add columns for {table_name}: {e}")

def _sql_get_primary_keys(engine, table_name):
    """Retrieve the primary keys for the specified table."""
    try:
        with engine.connect() as conn:
            query = sqlalchemy.text(
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
        print(f"[WARNING] Could not get primary keys for {table_name}: {e}")
        return ['DATETIME']  # Default assumption

def sql_batch_upsert_to_database(engine, table_name, data, chunk_size=25000):
    """Perform an upsert (INSERT/UPDATE) using a MERGE statement on the specified table."""
    if data.empty:
        return
    
    # Ensure datetime column is properly parsed
    for column in data.columns:
        if "datetime" in column.lower():
            data[column] = pd.to_datetime(data[column], errors="coerce")

    primary_keys = _sql_get_primary_keys(engine, table_name)

    def format_value_vec(values):
        """Format values for SQL insertion."""
        formatted = []
        for value in values:
            if pd.isna(value):
                formatted.append("NULL")
            elif isinstance(value, pd.Timestamp):
                formatted.append(f"'{value.strftime('%Y-%m-%d %H:%M:%S')}'")
            elif isinstance(value, str):
                escaped_value = value.replace("'", "''")
                formatted.append(f"'{escaped_value}'")
            else:
                formatted.append(str(value))
        return formatted

    columns = data.columns.tolist()
    rowcounter = 0

    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i + chunk_size]
        if chunk.empty:
            continue

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
            with engine.connect() as conn:
                with conn.begin():
                    result = conn.execute(sqlalchemy.text(merge_query))
                    rowcounter += result.rowcount
        except sqlalchemy.exc.IntegrityError as e:
            print(f"[ERROR] Integrity error during upsert to {table_name}: {e}")
            print(f"Problematic chunk range: {chunk['DATETIME'].min()} to {chunk['DATETIME'].max()}")
            continue
        except Exception as e:
            print(f"[ERROR] SQL execution error for {table_name}: {e}")
            print(f"Problematic chunk range: {chunk['DATETIME'].min()} to {chunk['DATETIME'].max()}")
            continue

    print(f"[UPSERT] {table_name} - Rows affected: {rowcounter}")

def fetch_hourly_data(api_key, fuel_type, start, end, respondent, session):
    """Fetches hourly generation data from EIA for a single fuel_type and respondent."""
    api_url = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
    all_data = []
    offset = 0
    page_size = 5000

    while True:
        params = {
            "api_key": api_key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[fueltype][]": fuel_type,
            "facets[respondent][]": respondent,
            "start": start,
            "end": end,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": page_size
        }

        response = session.get(api_url, params=params, timeout=30)
        if response.status_code != 200:
            print(f"[ERROR] Failed to fetch data for {fuel_type}: {response.status_code}")
            break

        data = response.json()
        if 'response' in data and 'data' in data['response']:
            page_data = data['response']['data']
            if not page_data:
                break
            all_data.extend(page_data)
            offset += page_size
        else:
            break

    if all_data:
        df = pd.DataFrame(all_data)
        print(f"Fetched {len(df)} records for {fuel_type} in {respondent}")
        return df
    else:
        return pd.DataFrame()

def fetch_fuel_data_parallel(api_key, fuel_types, start, end, respondent, session, max_workers=5):
    """Fetch data for multiple fuel types in parallel."""
    raw_dataframes = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all fuel type requests
        future_to_fuel = {
            executor.submit(fetch_hourly_data, api_key, ft_abbrev, start, end, respondent, session): ft_abbrev
            for ft_abbrev in fuel_types.keys()
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_fuel):
            fuel_abbrev = future_to_fuel[future]
            try:
                raw_df = future.result()
                if not raw_df.empty:
                    raw_df['fuel_type'] = fuel_abbrev
                    raw_dataframes.append(raw_df)
                else:
                    print(f"No data returned for fuel {fuel_abbrev} in region {respondent}.")
            except Exception as exc:
                print(f"Fuel {fuel_abbrev} generated an exception: {exc}")
    
    return raw_dataframes

def process_combined_data(combined_df, r_name, desired_end_time_local):
    """Process combined DataFrame with optimized operations."""
    # Convert 'period' to UTC datetime
    combined_df['utc_datetime'] = pd.to_datetime(combined_df['period'], utc=True)
    
    # Remove duplicates more efficiently
    combined_df = combined_df.drop_duplicates(subset=['utc_datetime', 'fuel_type'], keep='last')
    
    # Convert to local Denver time
    combined_df['datetime'] = combined_df['utc_datetime'].dt.tz_convert(mst_tz)
    
    # Filter data more efficiently
    combined_df = combined_df[combined_df['datetime'] <= desired_end_time_local]
    
    # Create pivot column name more efficiently
    combined_df['col_name'] = (r_name + '_' + combined_df['fuel_type'].str.lower() + '_mw')
    
    # Pivot the data
    pivoted_df = combined_df.pivot_table(
        index='datetime', 
        columns='col_name', 
        values='value', 
        aggfunc='last'  # Use last value in case of duplicates
    ).reset_index()
    
    # Ensure all expected columns exist (13 fuel types for Tennessee)
    required_cols = [f"{r_name}_{ft.lower()}_mw" for ft in fuel_types.keys()]
    
    for col in required_cols:
        if col not in pivoted_df.columns:
            pivoted_df[col] = pd.NA
    
    # Select and reorder columns
    final_cols = ['datetime'] + required_cols
    merged_df = pivoted_df[final_cols].copy()
    
    # Convert column names to uppercase
    merged_df.columns = merged_df.columns.str.upper()
    
    # Handle duplicates more efficiently
    merged_df = merged_df.sort_values('DATETIME').drop_duplicates(subset=['DATETIME'], keep='last')
    
    return merged_df

# -------------------------
# Main Loop Over Regions
# -------------------------
print(f"Fetching data from {start_date} to {end_date} (interpreted as local Denver time, up to now).")

for region in region_configs:
    r_code = region["respondent_code"]
    r_name = region["respondent_name"]
    tbl_name = region["table_name"]

    print(f"\n=== Processing region: {r_name.upper()} ({r_code}), Table: {tbl_name} ===")

    # Step 0: Create table if it doesn't exist
    create_table_if_not_exists(engine, tbl_name, r_name)

    # Step 1: Fetch all raw data for the region in parallel
    print(f"Fetching data for {r_name} using parallel processing...")
    raw_dataframes = fetch_fuel_data_parallel(
        api_key, fuel_types, start_date, end_date, r_code, session, max_workers=6
    )

    if not raw_dataframes:
        print(f"[WARNING] No data frames for region {r_name}, skipping upsert.")
        continue

    # Step 2: Combine all raw data into a single DataFrame
    print(f"Combining {len(raw_dataframes)} DataFrames...")
    combined_df = pd.concat(raw_dataframes, ignore_index=True)

    # Step 3: Process the combined DataFrame
    print(f"Processing combined data for {r_name}...")
    merged_df = process_combined_data(combined_df, r_name, desired_end_time_local)
    
    print(f"Final data range: {merged_df['DATETIME'].min()} to {merged_df['DATETIME'].max()}")
    print(f"Final data shape: {merged_df.shape}")

    # Step 4: Upsert into SQL database
    print(f"Upserting data to database...")
    sql_batch_upsert_to_database(engine, tbl_name, merged_df)
    print(f"Data successfully upserted into {tbl_name}.\n")

# Close the session
session.close()
print("All region data updated (interpreted as local Denver time, filtered to 23:00 MST day before yesterday).")