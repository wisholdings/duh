import os
import pandas as pd
import datetime
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---

# Local SQLite Database Configuration
LOCAL_DB_PATH = "./x.db"
DB_CONNECTION_STRING = f"sqlite:///{LOCAL_DB_PATH}"

sql_table_name = "x"

# Define locations with airport codes (matching weather_data_fetch.py)
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

# Define weights for AESO_HDD and AESO_CDD
hdd_weights = {
    'CYYC': 0.5119,
    'CYEG': 0.4188,
    'CYQF': 0.0269,
    'CYMM': 0.0232,
    'CYQU': 0.0179,
    'CYXJ': 0.0007,
    'CYYE': 0.0006,
    'CZPC': 0.0000
}

cdd_weights = {
    'CYEG': 0.4065,
    'CYYC': 0.3150,
    'CYQF': 0.2023,
    'CZPC': 0.0495,
    'CYQU': 0.0191,
    'CYYE': 0.0076,
    'CYMM': 0.0000,
    'CYXJ': 0.0000
}

# Output CSV file name (matching List_Blobs.py and calculate_aeso.py)
OUTPUT_CSV_FILE = "stats.csv"

# --- Main Process ---

engine = None
try:
    logging.info(f"Connecting to local database: {LOCAL_DB_PATH}")
    engine = create_engine(DB_CONNECTION_STRING)

    # --- Load Data ---
    # Construct the list of columns to select
    columns_to_select = ['Date', 'Date_Modified']
    hdd_cols = []
    cdd_cols = []
    for location_code in alberta_locations.keys():
        hdd_col_name = f"{location_code}_HDD"
        cdd_col_name = f"{location_code}_CDD"
        columns_to_select.extend([f'"{hdd_col_name}"', f'"{cdd_col_name}"'])
        hdd_cols.append(hdd_col_name)
        cdd_cols.append(cdd_col_name)

    logging.info(f"Loading data from table: {sql_table_name}")
    query = f'SELECT {", ".join(columns_to_select)} FROM "{sql_table_name}"'

    df = pd.read_sql_query(query, engine)

    if df.empty:
        logging.error("No data loaded from the database. Exiting.")
        exit()

    logging.info(f"Successfully loaded {len(df)} rows.")

    # --- Filter for Most Recent Date_Modified per Date ---
    logging.info("Filtering data to keep only the most recent Date_Modified for each date...")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date_Modified'] = pd.to_datetime(df['Date_Modified'])
    idx_most_recent = df.groupby('Date')['Date_Modified'].idxmax()
    df_filtered = df.loc[idx_most_recent].reset_index(drop=True)
    logging.info(f"Filtered down to {len(df_filtered)} rows.")

    # --- Process Data ---
    # Calculate AESO_HDD and AESO_CDD as weighted sums
    logging.info("Calculating AESO_HDD and AESO_CDD...")
    df_filtered['AESO_HDD'] = 0.0
    df_filtered['AESO_CDD'] = 0.0
    for city in alberta_locations.keys():
        hdd_col = f"{city}_HDD"
        cdd_col = f"{city}_CDD"
        df_filtered['AESO_HDD'] += df_filtered[hdd_col] * hdd_weights[city]
        df_filtered['AESO_CDD'] += df_filtered[cdd_col] * cdd_weights[city]
    df_filtered['AESO_HDD'] = df_filtered['AESO_HDD'].round(2)
    df_filtered['AESO_CDD'] = df_filtered['AESO_CDD'].round(2)

    # Fill NaNs with 0 for HDD and CDD
    df_filtered[hdd_cols + cdd_cols + ['AESO_HDD', 'AESO_CDD']] = df_filtered[
        hdd_cols + cdd_cols + ['AESO_HDD', 'AESO_CDD']
    ].fillna(0)

    # --- Final DataFrame and Export ---
    logging.info(f"Preparing data for export to {OUTPUT_CSV_FILE}...")
    # Select columns in the order expected by List_Blobs.py
    final_columns = ['Date', 'AESO_HDD', 'AESO_CDD']
    for location_code in alberta_locations.keys():
        final_columns.extend([f"{location_code}_HDD", f"{location_code}_CDD"])
    df_output = df_filtered[final_columns]

    # Sort by date
    df_output = df_output.sort_values(by='Date').reset_index(drop=True)

    # Format Date as string (e.g., '26-May-25')
    df_output['Date'] = df_output['Date'].dt.strftime('%d-%b-%y')

    # Export to CSV
    df_output.to_csv(OUTPUT_CSV_FILE, index=False)
    logging.info(f"\nData successfully processed and exported to '{OUTPUT_CSV_FILE}'.")
    logging.info("First few rows of output:")
    print(df_output.head())

except SQLAlchemyError as ex:
    logging.error(f"A database error occurred: {ex}")
except Exception as e:
    logging.error(f"An unexpected error occurred: {e}")
finally:
    if engine:
        engine.dispose()
        logging.info("Database connection closed.")