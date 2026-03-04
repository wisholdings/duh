# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
from datetime import date
from sqlalchemy import create_engine, MetaData, Table, Column, DateTime, Date, Float, String, inspect, and_, or_
from sqlalchemy.exc import SQLAlchemyError

# --- SQL Server Configuration ---
DB_CONNECTION_STRING = (
)
BASE_OUTPUT_DIR = r"D:\EIA_STACK"

# List of regions to define expected columns
REGIONS = [
    "california", "carolina", "central", "florida", "midatlantic", "midwest",
    "newengland", "newyork", "northwest", "southeast", "southwest", "tennessee", "texas"
]

# --- List of consumption types to process ---
CONSUMPTION_TYPES = ["residential", "commercial", "industrial"]

# ------------------------------
# Database Functions
# ------------------------------

def create_sql_table_if_not_exists(engine, table_name, df_columns):
    """Creates the SQL table dynamically based on the DataFrame's columns."""
    metadata = MetaData()
    columns = [
        Column('datetime', DateTime, primary_key=True),
        Column('date_published', Date, primary_key=True),
        Column('forecast_model', String(50)),  # Add model type column
    ]
    for col_name in df_columns:
        if col_name not in ['datetime', 'date_published', 'forecast_model']:
            columns.append(Column(col_name, Float))
            
    table = Table(table_name, metadata, *columns, extend_existing=True)
    try:
        metadata.create_all(engine, tables=[table])
        print(f"    ✓ Table '{table_name}' ensured to exist with {len(columns)} columns.")
    except SQLAlchemyError as e:
        print(f"    ✗ Error creating or verifying table '{table_name}': {e}")
        raise

def delete_existing_rows_batched(engine, table_name, df_pk_columns, batch_size=500):
    """Deletes rows from the target SQL table based on primary key combinations."""
    if df_pk_columns.empty:
        return 0
    metadata = MetaData()
    try:
        table = Table(table_name, metadata, autoload_with=engine)
    except SQLAlchemyError as e:
        print(f"    ✗ Error autoloading table '{table_name}' for deletion: {e}")
        raise
        
    unique_pk_list = [tuple(x) for x in df_pk_columns[['datetime', 'date_published']].drop_duplicates().to_numpy()]
    total_deleted_rows = 0
    
    with engine.connect() as connection:
        for i in range(0, len(unique_pk_list), batch_size):
            batch_pks = unique_pk_list[i:i + batch_size]
            conditions = [and_(table.c.datetime == dt, table.c.date_published == dp) for dt, dp in batch_pks]
            if not conditions: 
                continue

            delete_stmt = table.delete().where(or_(*conditions))
            try:
                with connection.begin() as trans:
                    result = connection.execute(delete_stmt)
                total_deleted_rows += result.rowcount
            except SQLAlchemyError as e:
                print(f"    ✗ Error deleting batch of rows from '{table_name}': {e}")
                raise
        
        if total_deleted_rows > 0:
            print(f"    ✓ Deleted {total_deleted_rows} existing conflicting rows.")
    return total_deleted_rows

def upload_dataframe_to_sql(engine, table_name, df):
    """Uploads a pandas DataFrame to a SQL table."""
    try:
        if df.empty:
            print("    ⚠ Input DataFrame is empty, no rows to upload.")
            return 0

        delete_existing_rows_batched(engine, table_name, df[['datetime', 'date_published']])

        df.to_sql(name=table_name, con=engine, if_exists='append', index=False, method='multi', chunksize=500)
        print(f"    ✓ Successfully uploaded {len(df)} rows to table '{table_name}'.")
        return len(df)
    except SQLAlchemyError as e:
        print(f"    ✗ Error during SQL operations for table '{table_name}': {e}")
        raise

def verify_upload(engine, table_name, expected_rows):
    """Verify the upload by checking row count and date range."""
    try:
        query = f"""
        SELECT COUNT(*) as row_count, 
               MIN(datetime) as min_date, 
               MAX(datetime) as max_date,
               MIN(date_published) as publish_date,
               MIN(forecast_model) as model_type
        FROM {table_name}
        WHERE date_published = (SELECT MAX(date_published) FROM {table_name})
        """
        result = pd.read_sql(query, con=engine)
        
        if not result.empty:
            row_count = result.iloc[0]['row_count']
            min_date = result.iloc[0]['min_date']
            max_date = result.iloc[0]['max_date']
            publish_date = result.iloc[0]['publish_date']
            model_type = result.iloc[0]['model_type']
            
            print(f"\n    📊 Verification:")
            print(f"      Rows uploaded: {row_count} (expected: {expected_rows})")
            print(f"      Date range: {min_date} to {max_date}")
            print(f"      Publish date: {publish_date}")
            print(f"      Model type: {model_type}")
            
            if row_count == expected_rows:
                print(f"      ✓ Upload verified successfully!")
            else:
                print(f"      ⚠ Row count mismatch!")
    except Exception as e:
        print(f"    ⚠ Could not verify upload: {str(e)[:60]}")

# ------------------------------
# Main Processing Function
# ------------------------------

def process_and_upload_forecasts():
    """
    Loops through each consumption type, reads the corresponding consolidated forecast,
    prepares it, and uploads it to the HYBRID SQL tables.
    """
    
    print("\n" + "="*80)
    print("   UPLOADING HYBRID FORECASTS TO SQL SERVER")
    print("="*80)
    print(f"Database: gasstosbxarmservus2001")
    print(f"Source Directory: {BASE_OUTPUT_DIR}")
    print("="*80)
    
    upload_summary = []
    
    for consumption_type in CONSUMPTION_TYPES:
        consumption_cap = consumption_type.capitalize()
        
        # --- UPLOAD TO HYBRID TABLES ---
        TABLE_NAME = f"FORECAST_{consumption_type.upper()}_BURNS_REGIONAL_HYBRID"
        INPUT_FILE = os.path.join(BASE_OUTPUT_DIR, f"consolidated_daily_{consumption_type}_forecasts_hybrid.csv")
        TOTAL_BCF_COL = f"Total_L48_{consumption_cap}_BCF"
        
        print(f"\n{'='*40}")
        print(f"Processing: {consumption_cap} Sector")
        print(f"{'='*40}")
        print(f"📁 Input File: {os.path.basename(INPUT_FILE)}")
        print(f"🗃️  Target Table: {TABLE_NAME}")
        print("")
        
        engine = None
        try:
            # --- Read the Input File ---
            print(f"[1] Reading consolidated forecast file...")
            if not os.path.exists(INPUT_FILE):
                print(f"    ✗ Input file not found. Skipping {consumption_cap}.")
                upload_summary.append((consumption_cap, "FAILED", "File not found"))
                continue

            df = pd.read_csv(INPUT_FILE, parse_dates=['datetime'])
            print(f"    ✓ Read {len(df)} rows from file")

            # --- Validate and Prepare Data ---
            print(f"\n[2] Validating and preparing data...")
            
            # Handle missing regions gracefully
            available_regions = [r for r in REGIONS if r in df.columns]
            missing_regions = [r for r in REGIONS if r not in df.columns]
            
            if missing_regions:
                print(f"    ⚠ Missing regions: {', '.join(missing_regions)}")
                print(f"    ✓ Using {len(available_regions)} available regions")
            
            # Build expected columns list
            expected_columns = ['datetime'] + available_regions + [TOTAL_BCF_COL]
            
            # Check for total column
            if TOTAL_BCF_COL not in df.columns:
                print(f"    ✗ ERROR: Missing total column '{TOTAL_BCF_COL}'")
                upload_summary.append((consumption_cap, "FAILED", "Missing total column"))
                continue
            
            df_upload = df[expected_columns].copy()

            # RENAME columns for SQL clarity
            rename_dict = {}
            for region in available_regions:
                rename_dict[region] = f"{region}_bcf"
            rename_dict[TOTAL_BCF_COL] = f"total_l48_{consumption_type}_bcf"
            
            df_upload.rename(columns=rename_dict, inplace=True)
            
            # Fill missing regions with 0
            for region in REGIONS:
                col_name = f"{region}_bcf"
                if col_name not in df_upload.columns:
                    df_upload[col_name] = 0.0
            
            # PROCESS datetime column
            df_upload['datetime'] = pd.to_datetime(df_upload['datetime'], errors='coerce').dt.floor('D')
            df_upload.dropna(subset=['datetime'], inplace=True)
            
            # PROCESS all value columns
            value_cols = [col for col in df_upload.columns if col not in ['datetime']]
            for col in value_cols:
                df_upload[col] = pd.to_numeric(df_upload[col], errors='coerce').fillna(0)
            
            # ADD metadata columns
            df_upload['date_published'] = pd.Timestamp.now().date()
            df_upload['forecast_model'] = 'HYBRID'
            
            print(f"    ✓ Set 'date_published' to {df_upload['date_published'].iloc[0]}")
            print(f"    ✓ Set 'forecast_model' to HYBRID")

            # DEDUPLICATE based on the composite primary key
            original_rows = len(df_upload)
            df_upload.drop_duplicates(subset=['datetime', 'date_published'], keep='last', inplace=True)
            if len(df_upload) < original_rows:
                print(f"    ⚠ Removed {original_rows - len(df_upload)} duplicate rows")
            
            if df_upload.empty:
                print("    ✗ No valid data to upload after cleaning.")
                upload_summary.append((consumption_cap, "FAILED", "No valid data"))
                continue
            
            print(f"    ✓ Prepared {len(df_upload)} unique rows for upload")
            
            # Show data preview
            print(f"\n[3] Data Preview:")
            print(f"    Date range: {df_upload['datetime'].min().date()} to {df_upload['datetime'].max().date()}")
            total_col = f"total_l48_{consumption_type}_bcf"
            if total_col in df_upload.columns:
                mean_val = df_upload[total_col].mean()
                max_val = df_upload[total_col].max()
                min_val = df_upload[total_col].min()
                print(f"    Total L48 stats:")
                print(f"      Mean: {mean_val:.2f} BCF/day")
                print(f"      Range: {min_val:.2f} - {max_val:.2f} BCF/day")

            # --- Database Operations ---
            print(f"\n[4] Uploading to SQL Server...")
            engine = create_engine(DB_CONNECTION_STRING)
            
            # Ensure all regional columns are present (fill with 0 if missing)
            for region in REGIONS:
                col_name = f"{region}_bcf"
                if col_name not in df_upload.columns:
                    df_upload[col_name] = 0.0
                    print(f"    ⚠ Added missing column {col_name} with 0 values")
            
            create_sql_table_if_not_exists(engine, TABLE_NAME, df_upload.columns)
            rows_uploaded = upload_dataframe_to_sql(engine, TABLE_NAME, df_upload)
            
            # Verify the upload
            verify_upload(engine, TABLE_NAME, len(df_upload))
            
            upload_summary.append((consumption_cap, "SUCCESS", f"{rows_uploaded} rows"))

        except (FileNotFoundError, ValueError, SQLAlchemyError, Exception) as e:
            print(f"\n    ✗ FATAL ERROR for {consumption_cap}:")
            print(f"      {str(e)[:200]}")
            upload_summary.append((consumption_cap, "FAILED", str(e)[:50]))
        finally:
            if engine:
                engine.dispose()
                print("    ✓ Database connection closed")

    # --- Final Summary ---
    print("\n" + "="*80)
    print("UPLOAD SUMMARY")
    print("="*80)
    
    print("\nTarget Tables:")
    for consumption_type in CONSUMPTION_TYPES:
        table_name = f"FORECAST_{consumption_type.upper()}_BURNS_REGIONAL_HYBRID"
        print(f"  • {table_name}")
    
    print("\nResults:")
    for sector, status, details in upload_summary:
        status_symbol = "✓" if status == "SUCCESS" else "✗"
        print(f"  {status_symbol} {sector:12s}: {status:8s} - {details}")
    
    success_count = sum(1 for _, status, _ in upload_summary if status == "SUCCESS")
    print(f"\nTotal: {success_count}/{len(CONSUMPTION_TYPES)} successful uploads")
    
    print("\n" + "="*80)
    print("✅ Script completed!")
    print("="*80)

# --- Execute the Script ---
if __name__ == "__main__":
    process_and_upload_forecasts()