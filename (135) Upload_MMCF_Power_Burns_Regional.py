# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
from datetime import date
import time
from sqlalchemy import create_engine, MetaData, Table, Column, DateTime, Date, Float, inspect, and_, or_
from sqlalchemy.exc import SQLAlchemyError

# --- SQL Server Configuration ---
DB_CONNECTION_STRING = (
)
# NEW: A new table name appropriate for the wide-format regional data
TABLE_NAME = "FORECAST_POWER_BURNS_REGIONAL_ST"
BASE_OUTPUT_DIR = "D:\\EIA_STACK"
# NEW: The input file is now the consolidated forecast
INPUT_FILE = os.path.join(BASE_OUTPUT_DIR, "consolidated_daily_forecasts.csv")

# List of regions to define expected columns
REGIONS = [
    "california", "carolina", "central", "florida", "midatlantic", "midwest",
    "newengland", "newyork", "northwest", "southeast", "southwest", "tennessee", "texas"
]

# ------------------------------
# Functions (Modified and Reused)
# ------------------------------

def create_sql_table_if_not_exists(engine, table_name, df_columns):
    """
    MODIFIED: Creates the SQL table dynamically based on the DataFrame's columns.
    It preserves the composite primary key (datetime, date_published).
    """
    metadata = MetaData()
    
    # Start with the primary key columns
    columns = [
        Column('datetime', DateTime, primary_key=True),
        Column('date_published', Date, primary_key=True),
    ]
    
    # Add a Float column for every other column in the DataFrame
    for col_name in df_columns:
        if col_name not in ['datetime', 'date_published']:
            columns.append(Column(col_name, Float))
            
    table = Table(table_name, metadata, *columns, extend_existing=True)
    
    try:
        metadata.create_all(engine, tables=[table])
        print(f"Table '{table_name}' ensured to exist with {len(columns)} columns.")
    except SQLAlchemyError as e:
        print(f"Error creating or verifying table '{table_name}': {e}")
        raise

def delete_existing_rows_batched(engine, table_name, df_pk_columns, batch_size=500):
    """
    UNCHANGED: This function is robust and reusable. It deletes rows from the target
    SQL table based on the primary key combinations in the provided DataFrame.
    """
    if df_pk_columns.empty:
        return 0
    metadata = MetaData()
    try:
        table = Table(table_name, metadata, autoload_with=engine)
    except SQLAlchemyError as e:
        print(f"Error autoloading table '{table_name}' for deletion: {e}")
        raise
        
    unique_pk_list = [tuple(x) for x in df_pk_columns[['datetime', 'date_published']].drop_duplicates().to_numpy()]
    total_deleted_rows = 0
    
    with engine.connect() as connection:
        for i in range(0, len(unique_pk_list), batch_size):
            batch_pks = unique_pk_list[i:i + batch_size]
            conditions = [and_(table.c.datetime == dt, table.c.date_published == dp) for dt, dp in batch_pks]
            if not conditions: continue

            delete_stmt = table.delete().where(or_(*conditions))
            try:
                result = connection.execute(delete_stmt)
                connection.commit()
                total_deleted_rows += result.rowcount
            except SQLAlchemyError as e:
                connection.rollback()
                print(f"Error deleting batch of rows from '{table_name}': {e}")
                raise
        
        if total_deleted_rows > 0:
            print(f"Successfully deleted {total_deleted_rows} existing conflicting rows.")
    return total_deleted_rows

def upload_dataframe_to_sql(engine, table_name, df):
    """
    UNCHANGED: This function is also robust and reusable.
    """
    try:
        if df.empty:
            print("Input DataFrame is empty, no rows to upload.")
            return 0

        delete_existing_rows_batched(engine, table_name, df[['datetime', 'date_published']])

        df.to_sql(name=table_name, con=engine, if_exists='append', index=False, method='multi', chunksize=200)
        print(f"Successfully uploaded {len(df)} rows to table '{table_name}'.")
        return len(df)
    except SQLAlchemyError as e:
        print(f"Error during SQL operations for table '{table_name}': {e}")
        raise

# ------------------------------
# Main Processing Function
# ------------------------------

def upload_consolidated_forecast_to_sql():
    """
    MODIFIED: Reads the consolidated forecast, prepares it for a wide SQL table,
    and uploads the data.
    """
    print("\n" + "="*60)
    print("--- Starting Consolidated Regional Forecast Upload to SQL ---")
    print("="*60)
    engine = None
    
    try:
        # --- Read the Input File ---
        print(f"Reading consolidated forecast file: {INPUT_FILE}")
        if not os.path.exists(INPUT_FILE):
            raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

        df = pd.read_csv(INPUT_FILE)
        print(f"Read {len(df)} rows from the consolidated file.")

        # --- Validate and Prepare Data ---
        expected_columns = ['date'] + REGIONS + ['Total_L48_Power_Burn_BCF']
        if not all(col in df.columns for col in expected_columns):
            raise ValueError("Consolidated CSV is missing expected regional or total columns.")

        # Create a copy to avoid SettingWithCopyWarning
        df_upload = df[expected_columns].copy()

        # RENAME columns for SQL clarity and consistency
        df_upload.rename(columns={'date': 'datetime'}, inplace=True)
        # Add a '_bcf' suffix to all value columns
        for col in REGIONS + ['Total_L48_Power_Burn_BCF']:
            df_upload.rename(columns={col: f"{col.lower()}_bcf"}, inplace=True)
        
        # PROCESS datetime column
        df_upload['datetime'] = pd.to_datetime(df_upload['datetime'], errors='coerce').dt.floor('D')
        df_upload.dropna(subset=['datetime'], inplace=True)
        
        # PROCESS all value columns
        value_cols = [col for col in df_upload.columns if col not in ['datetime']]
        for col in value_cols:
            df_upload[col] = pd.to_numeric(df_upload[col], errors='coerce').fillna(0)
        
        # ADD the publication date
        df_upload['date_published'] = pd.Timestamp.now().date()
        print(f"Set 'date_published' to {df_upload['date_published'].iloc[0]} for all rows.")

        # DEDUPLICATE based on the composite primary key
        df_upload.drop_duplicates(subset=['datetime', 'date_published'], keep='last', inplace=True)
        
        if df_upload.empty:
            print("No valid data to upload after cleaning. Exiting.")
            sys.exit(0)
        
        print(f"Prepared {len(df_upload)} unique rows for SQL upload.")

        # --- Database Operations ---
        engine = create_engine(DB_CONNECTION_STRING)
        
        # Create table dynamically based on the final prepared DataFrame columns
        create_sql_table_if_not_exists(engine, TABLE_NAME, df_upload.columns)
        
        # Upload the data
        upload_dataframe_to_sql(engine, TABLE_NAME, df_upload)

    except (FileNotFoundError, ValueError, SQLAlchemyError, Exception) as e:
        print(f"\nFATAL ERROR: A failure occurred during the process.")
        print(f"  -> Reason: {e}")
        sys.exit(1)
    finally:
        if engine:
            engine.dispose()
            print("SQLAlchemy engine disposed.")

    print("\n" + "="*60)
    print("--- Forecast Upload Script Completed Successfully ---")
    print("="*60)

# --- Execute the Script ---
if __name__ == "__main__":
    upload_consolidated_forecast_to_sql()