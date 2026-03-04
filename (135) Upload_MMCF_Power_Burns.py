# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np
import time
from datetime import date
from sqlalchemy import create_engine, MetaData, Table, Column, DateTime, Date, Float
from sqlalchemy.exc import SQLAlchemyError

# --- SQL Server Configuration ---
DB_CONNECTION_STRING = (
)
# CORRECT: Using the original table name as requested
TABLE_NAME = "PRODUCTION_POWER_BURNS_ST"
BASE_OUTPUT_DIR = "D:\\EIA_STACK"
# CORRECT: Using the consolidated forecast as the input
INPUT_FILE = os.path.join(BASE_OUTPUT_DIR, "power_burns_daily_small_model.csv")

# ------------------------------
# Functions (Reverted to original, simple version)
# ------------------------------

def create_sql_table_if_not_exists(engine, table_name):
    """
    Creates the original, simple SQL Server table with the correct schema
    if it doesn't already exist.
    """
    metadata = MetaData()
    table = Table(
        table_name,
        metadata,
        Column('datetime', DateTime, primary_key=True),
        Column('date_published', Date, primary_key=True),
        Column('Daily_MMCF', Float),
        extend_existing=True
    )
    metadata.create_all(engine, tables=[table])
    print(f"Table '{table_name}' ensured to exist in SQL Server.")

def upload_dataframe_to_sql(engine, table_name, df):
    """
    Deletes rows from the SQL table with primary keys matching those in the DataFrame,
    then uploads the new data.
    """
    if df.empty:
        print("Input DataFrame is empty, no rows to upload.")
        return 0

    # This delete logic is robust and works perfectly for this task
    from sqlalchemy import and_, or_
    with engine.connect() as connection:
        metadata = MetaData()
        table = Table(table_name, metadata, autoload_with=engine)
        
        pk_list = [tuple(x) for x in df[['datetime', 'date_published']].drop_duplicates().to_numpy()]
        total_deleted = 0
        
        # Batch deletion for safety and performance
        batch_size = 500
        for i in range(0, len(pk_list), batch_size):
            batch = pk_list[i:i + batch_size]
            conditions = [and_(table.c.datetime == dt, table.c.date_published == dp) for dt, dp in batch]
            if conditions:
                delete_stmt = table.delete().where(or_(*conditions))
                trans = connection.begin()
                try:
                    result = connection.execute(delete_stmt)
                    trans.commit()
                    total_deleted += result.rowcount
                except SQLAlchemyError as e:
                    trans.rollback()
                    raise e
        if total_deleted > 0:
            print(f"Successfully deleted {total_deleted} existing conflicting rows.")

    # Upload the new/updated data from the DataFrame
    df.to_sql(
        name=table_name,
        con=engine,
        if_exists='append',
        index=False,
        method='multi',
        chunksize=500
    )
    print(f"Successfully uploaded {len(df)} rows to table '{table_name}'.")
    return len(df)

# ------------------------------
# Main Execution
# ------------------------------
def upload_total_burn_to_sql():
    print("\n" + "="*60)
    print("--- Uploading Total L48 Power Burn Forecast to SQL ---")
    print("="*60)
    
    engine = None
    try:
        # --- Read the Consolidated Input File ---
        print(f"Reading consolidated forecast: {INPUT_FILE}")
        if not os.path.exists(INPUT_FILE):
            raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

        df = pd.read_csv(INPUT_FILE)
        print(f"Read {len(df)} rows from consolidated file.")

        # --- Validate and Prepare Data for the ORIGINAL Table ---
        required_col = 'L48_Power_Burns'
        if 'date' not in df.columns or required_col not in df.columns:
            raise KeyError(f"Input CSV must contain 'date' and '{required_col}' columns.")

        # 1. Select ONLY the columns we need
        df_upload = df[['date', required_col]].copy()
        
        # 2. Rename columns to match the target SQL table schema
        df_upload.rename(columns={
            'date': 'datetime',
            required_col: 'Daily_MMCF'
        }, inplace=True)

        # 3. *** CRITICAL STEP: Convert from BCF to MMCF ***
        print("Converting 'Total_L48_Power_Burn_BCF' to 'Daily_MMCF' (multiplying by 1000)...")
        df_upload['Daily_MMCF'] = df_upload['Daily_MMCF'] * 1000.0

        # 4. Perform standard data cleaning from your original script
        df_upload['datetime'] = pd.to_datetime(df_upload['datetime'], errors='coerce').dt.floor('D')
        df_upload.dropna(subset=['datetime'], inplace=True)
        df_upload['Daily_MMCF'] = pd.to_numeric(df_upload['Daily_MMCF'], errors='coerce').fillna(0)
        df_upload['Daily_MMCF'] = df_upload['Daily_MMCF'].round(0).astype(np.int64)
        
        # 5. Add the publication date
        df_upload['date_published'] = pd.Timestamp.now().date()
        print(f"Set 'date_published' to {df_upload['date_published'].iloc[0]} for all rows.")

        # 6. Deduplicate to ensure data integrity
        df_upload.drop_duplicates(subset=['datetime', 'date_published'], keep='last', inplace=True)
        
        if df_upload.empty:
            print("No valid data to upload after cleaning. Exiting.")
            sys.exit(0)
        
        print(f"Prepared {len(df_upload)} unique rows for SQL upload.")

        # --- Database Operations ---
        engine = create_engine(DB_CONNECTION_STRING)
        
        # Ensure the original, simple table exists
        create_sql_table_if_not_exists(engine, TABLE_NAME)
        
        # Upload the prepared data
        upload_dataframe_to_sql(engine, TABLE_NAME, df_upload)

    except (FileNotFoundError, KeyError, SQLAlchemyError, Exception) as e:
        print(f"\nFATAL ERROR: A failure occurred during the process.")
        print(f"  -> Reason: {e}")
        sys.exit(1)
    finally:
        if engine:
            engine.dispose()
            print("SQLAlchemy engine disposed.")
    
    print("\n" + "="*60)
    print("--- L48 Burn Upload Script Completed Successfully ---")
    print("="*60)


if __name__ == "__main__":
    upload_total_burn_to_sql()