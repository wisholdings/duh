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
from itertools import combinations
import urllib.request
from dateutil.relativedelta import relativedelta
from functools import reduce
import pytz
import re
import sqlalchemy
from sqlalchemy import create_engine, text

# Define MST timezone (America/Denver for accurate local time handling)
mst_tz = pytz.timezone("America/Denver")

# API key for EIA
api_key = ""

# Define the start date for fetching data
start_date = "2025-07-01T00"

# Calculate current time for fetching all available data
now_local = datetime.now(mst_tz)

# For filtering, still use "day before yesterday at 23:00" in local Denver time (MST)
day_before_yesterday = now_local.date() - timedelta(days=2)
desired_end_time_local = mst_tz.localize(datetime.combine(day_before_yesterday, datetime.strptime("23:00", "%H:%M").time()))

# Fetch all data up to the current UTC time (no end date restriction for API call)
end_date = None  # Let API return all available data

print(f"API start_date: {start_date}, end_date: {end_date if end_date else 'current'}")

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

# Combined list of all region configs
region_configs = [
    # Southeast regions
    {
        "respondent_code": "NE",
        "respondent_name": "newengland",
        "table_name": "PRODUCTION_NEWENGLAND_ENERGY_EIA_HOURLY",
    },
    {
        "respondent_code": "SW",
        "respondent_name": "southwest",
        "table_name": "PRODUCTION_SOUTHWEST_ENERGY_EIA_HOURLY",
    },
    {
        "respondent_code": "NW",
        "respondent_name": "northwest",
        "table_name": "PRODUCTION_NORTHWEST_ENERGY_EIA_HOURLY",
    },
    {
        "respondent_code": "CENT",
        "respondent_name": "central",
        "table_name": "PRODUCTION_CENTRAL_ENERGY_EIA_HOURLY",
    },
    {
        "respondent_code": "NY",
        "respondent_name": "newyork",
        "table_name": "PRODUCTION_NEWYORK_ENERGY_EIA_HOURLY",
    },
    {
        "respondent_code": "CAR",
        "respondent_name": "carolina",
        "table_name": "PRODUCTION_CAROLINA_ENERGY_EIA_HOURLY",
    },
    {
        "respondent_code": "FLA",
        "respondent_name": "florida",
        "table_name": "PRODUCTION_FLORIDA_ENERGY_EIA_HOURLY",
    },
    {
        "respondent_code": "SE",
        "respondent_name": "southeast",
        "table_name": "PRODUCTION_SOUTHEAST_ENERGY_EIA_HOURLY",
    },
    {
        "respondent_code": "TEN",
        "respondent_name": "tennessee",
        "table_name": "PRODUCTION_TENNESSEE_ENERGY_EIA_HOURLY",
    },
    # Other regions
    {
        "respondent_code": "TEX",
        "respondent_name": "texas",
        "table_name": "PRODUCTION_TEXAS_ENERGY_EIA_HOURLY",
    },
    {
        "respondent_code": "CAL",
        "respondent_name": "california",
        "table_name": "PRODUCTION_CALIFORNIA_ENERGY_EIA_HOURLY",
    },
    {
        "respondent_code": "MIDA",
        "respondent_name": "midatlantic",
        "table_name": "PRODUCTION_MIDATLANTIC_ENERGY_EIA_HOURLY",
    },
    {
        "respondent_code": "MIDW",
        "respondent_name": "midwest",
        "table_name": "PRODUCTION_MIDWEST_ENERGY_EIA_HOURLY",
    }
]

# -------------------------
# Helper Functions
# -------------------------

def create_table_if_not_exists(engine, table_name, region_configs):
    """Create the interchange table if it doesn't exist."""
    
    try:
        # Build column definitions for all regions
        region_columns = []
        for region in region_configs:
            col_name = f"{region['respondent_name'].upper()}_INTERCHANGE_MW"
            region_columns.append(f"{col_name} FLOAT NULL")
        
        # Combine all columns
        all_columns = ["DATETIME DATETIME NOT NULL PRIMARY KEY"] + region_columns
        columns_sql = ",\n    ".join(all_columns)
        
        create_table_sql = f"""
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{table_name}' AND xtype='U')
        CREATE TABLE {table_name} (
            {columns_sql}
        )
        """
        
        with engine.connect() as conn:
            with conn.begin():
                conn.execute(sqlalchemy.text(create_table_sql))
        print(f"✅ Table {table_name} created or already exists")
        return True
    except Exception as e:
        print(f"❌ Error creating table {table_name}: {e}")
        print(f"[INFO] Attempting to continue without table creation...")
        return True  # Continue anyway

def fast_sql_server_upsert(engine, table_name, data, chunk_size=5000):
    """
    Fast upsert using SQL Server's MERGE statement with larger chunks.
    Optimized for better performance with connection pooling.
    """
    if data.empty:
        return 0
    
    # Ensure datetime column is properly parsed
    for column in data.columns:
        if "datetime" in column.lower():
            data[column] = pd.to_datetime(data[column], errors="coerce")

    def format_value_for_sql(value):
        """Format a single value for SQL insertion."""
        if pd.isna(value):
            return "NULL"
        elif isinstance(value, pd.Timestamp):
            return f"'{value.strftime('%Y-%m-%d %H:%M:%S')}'"
        elif isinstance(value, str):
            escaped_value = value.replace("'", "''")
            return f"'{escaped_value}'"
        else:
            return str(value)

    columns = data.columns.tolist()
    total_processed = 0

    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i + chunk_size]
        if chunk.empty:
            continue

        # Build VALUES clause for the source data using vectorized operations
        values_list = []
        for row in chunk.values:
            formatted_values = [format_value_for_sql(val) for val in row]
            values_list.append(f"({', '.join(formatted_values)})")
        
        values_clause = ",\n        ".join(values_list)
        
        # Build column list for the source table
        source_columns = ", ".join(columns)
        
        # Build the MERGE statement
        merge_sql = f"""
        MERGE {table_name} AS target
        USING (
            VALUES 
                {values_clause}
        ) AS source ({source_columns})
        ON target.DATETIME = source.DATETIME
        WHEN MATCHED THEN
            UPDATE SET {', '.join([f"{col} = source.{col}" for col in columns[1:]])}
        WHEN NOT MATCHED THEN
            INSERT ({source_columns})
            VALUES ({', '.join([f"source.{col}" for col in columns])});
        """

        try:
            with engine.connect() as conn:
                with conn.begin():
                    result = conn.execute(sqlalchemy.text(merge_sql))
                    total_processed += len(chunk)
        except Exception as e:
            print(f"    ❌ Error in MERGE operation for chunk {i//chunk_size + 1}: {e}")
            # Fall back to smaller chunks for this batch
            for j in range(i, min(i + chunk_size, len(data)), 100):
                small_chunk = data.iloc[j:j+100]
                try:
                    small_values_list = []
                    for row in small_chunk.values:
                        formatted_values = [format_value_for_sql(val) for val in row]
                        small_values_list.append(f"({', '.join(formatted_values)})")
                    
                    small_values_clause = ",\n        ".join(small_values_list)
                    
                    small_merge_sql = f"""
                    MERGE {table_name} AS target
                    USING (
                        VALUES 
                            {small_values_clause}
                    ) AS source ({source_columns})
                    ON target.DATETIME = source.DATETIME
                    WHEN MATCHED THEN
                        UPDATE SET {', '.join([f"{col} = source.{col}" for col in columns[1:]])}
                    WHEN NOT MATCHED THEN
                        INSERT ({source_columns})
                        VALUES ({', '.join([f"source.{col}" for col in columns])});
                    """
                    
                    with engine.connect() as conn:
                        with conn.begin():
                            conn.execute(sqlalchemy.text(small_merge_sql))
                            total_processed += len(small_chunk)
                except Exception as small_e:
                    print(f"    Error in small chunk: {small_e}")
                    continue

    return total_processed

def fetch_interchange_data(api_key, start, end, respondent, session):
    """Fetches hourly total interchange data from EIA for a single respondent using region-data API."""
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
            "facets[type][]": "TI",  # Total Interchange
            "start": start,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": page_size
        }
        
        # Only add end parameter if it's not None
        if end is not None:
            params["end"] = end

        response = session.get(api_url, params=params, timeout=30)
        if response.status_code != 200:
            print(f"[ERROR] Failed to fetch interchange data for {respondent}: {response.status_code}")
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
        print(f"Fetched {len(df)} interchange records for {respondent}")
        return df
    else:
        return pd.DataFrame()

def fetch_regions_parallel(api_key, start_date, end_date, region_configs, session, max_workers=6):
    """Fetch interchange data for multiple regions in parallel."""
    region_data = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all region requests
        future_to_region = {
            executor.submit(fetch_interchange_data, api_key, start_date, end_date, region['respondent_code'], session): region
            for region in region_configs
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_region):
            region = future_to_region[future]
            r_code = region['respondent_code']
            r_name = region['respondent_name']
            
            try:
                interchange_df = future.result()
                if not interchange_df.empty:
                    region_data[r_name] = {
                        'data': interchange_df,
                        'region': region
                    }
                else:
                    print(f"No interchange data returned for region {r_name}.")
            except Exception as exc:
                print(f"Region {r_name} generated an exception: {exc}")
    
    return region_data

def process_region_data(region_name, interchange_df, desired_end_time_local):
    """Process a single region's interchange data."""
    # Process this region's data
    print(f"Processing {len(interchange_df)} records for {region_name}...")
    
    # Convert 'period' to UTC datetime
    interchange_df['utc_datetime'] = pd.to_datetime(interchange_df['period'], utc=True)
    
    # Remove duplicates based on 'utc_datetime' to handle DST transitions
    interchange_df = interchange_df.drop_duplicates(subset=['utc_datetime'], keep='last')
    
    # Convert to local Denver time (MST/MDT)
    interchange_df['datetime'] = interchange_df['utc_datetime'].dt.tz_convert(mst_tz)
    
    # Filter data up to the desired end time
    interchange_df = interchange_df[interchange_df['datetime'] <= desired_end_time_local]
    
    print(f"Filtered {region_name} data range: {interchange_df['datetime'].min()} to {interchange_df['datetime'].max()}")
    
    # Create single-region DataFrame for database
    region_df = pd.DataFrame({
        'DATETIME': interchange_df['datetime'],
        f'{region_name.upper()}_INTERCHANGE_MW': interchange_df['value']
    })
    
    # Handle datetime formatting and deduplication
    region_df = region_df.sort_values('DATETIME').drop_duplicates(subset=['DATETIME'], keep='last')
    
    return region_df

# -------------------------
# Main Loop to Collect All Import/Export Data
# -------------------------
print(f"Fetching interchange data from {start_date} to current available data (filtered to day before yesterday 23:00 MST).")

# Step 1: Create the table if it doesn't exist
table_name = "PRODUCTION_IMPORT_EXPORT_EIA"
print(f"\nCreating table {table_name} if it doesn't exist...")
if not create_table_if_not_exists(engine, table_name, region_configs):
    print("Failed to create table, but continuing anyway.")

# Step 2: Fetch all regions in parallel
print(f"\nFetching interchange data for all {len(region_configs)} regions in parallel...")
region_data = fetch_regions_parallel(api_key, start_date, end_date, region_configs, session, max_workers=8)

# Step 3: Process and upload each region's data
print(f"\nProcessing and uploading data for {len(region_data)} regions...")
for region_name, region_info in region_data.items():
    interchange_df = region_info['data']
    region = region_info['region']
    
    try:
        # Process the region's data
        region_df = process_region_data(region_name, interchange_df, desired_end_time_local)
        
        if not region_df.empty:
            print(f"Uploading {len(region_df)} records for {region_name} to database...")
            
            # Upload this region's data using fast MERGE-based upsert
            total_processed = fast_sql_server_upsert(engine, table_name, region_df, chunk_size=10000)
            print(f"✅ Successfully uploaded {region_name} interchange data ({total_processed} records processed)")
        else:
            print(f"No processed data for {region_name} after filtering.")
            
    except Exception as e:
        print(f"❌ Error processing/uploading {region_name} data: {e}")
        continue

# Close the session
session.close()

print("\n🎉 All regional interchange data processed and uploaded!")
print("Note: Data from different regions will be merged/combined in the database based on matching timestamps.")
print("Import/export data collection completed with parallel processing.")