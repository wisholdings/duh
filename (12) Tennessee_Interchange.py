import os
import sys
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text
import traceback
import gc
from datetime import datetime

# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

if project_root not in sys.path:
    sys.path.append(project_root)

print(f"Script directory: {script_dir}")
print(f"Calculated project root (parent of script dir): {project_root}")
print(f"sys.path updated to include: {project_root}")

# --- Configuration Loading ---
BASE_OUTPUT_DIR = None

try:
    from config.settings import BASE_OUTPUT_DIR
    print("Successfully imported BASE_OUTPUT_DIR from config.settings.")
except ImportError as e:
    print(f"Error: Could not import BASE_OUTPUT_DIR from config.settings.")
    print(f"Details: {e}")
    sys.exit(1)

if not BASE_OUTPUT_DIR or BASE_OUTPUT_DIR == ".":
    print("Error: BASE_OUTPUT_DIR is not properly configured. Cannot proceed.")
    sys.exit(1)

print(f"Using Base Output Directory for data: {BASE_OUTPUT_DIR}")

# --- Constants ---
REGION = "tennessee"
TABLE_NAME = "PRODUCTION_IMPORT_EXPORT_EIA"
TENNESSEE_COLUMN = "TENNESSEE_INTERCHANGE_MW"
DATE_COL = "date"
START_DATE_FILTER = "2019-01-01 00:00:00"

# Database connection string (from your original script)
DB_CONNECTION_STRING = (
)

# --- TennesseeInterchangeMerger Class Definition ---
class TennesseeInterchangeMerger:
    def __init__(self, base_dir: str = BASE_OUTPUT_DIR):
        """
        Initialize the TennesseeInterchangeMerger.

        Args:
            base_dir: Base directory for file paths (default: BASE_OUTPUT_DIR).
        """
        if not isinstance(base_dir, str) or not base_dir:
            raise ValueError("base_dir must be a non-empty string.")

        self.region = REGION
        self.base_dir = base_dir
        self.final_dir = os.path.join(self.base_dir, self.region, "final")
        self.table_name = TABLE_NAME
        self.Tennessee_column = TENNESSEE_COLUMN
        self.date_col = DATE_COL
        self.start_date_filter = pd.to_datetime(START_DATE_FILTER)
        
        # Input and output files
        self.input_file = os.path.join(self.final_dir, f"{self.region}_complete_finished.parquet")
        self.output_file = os.path.join(self.final_dir, f"{self.region}_complete_finished.parquet")
        
        # Database connection
        self.connection_string = DB_CONNECTION_STRING
        self.engine = None

        # Ensure the final output directory exists
        os.makedirs(self.final_dir, exist_ok=True)

    def connect_to_database(self) -> bool:
        """Establish database connection."""
        try:
            print("Connecting to database...")
            self.engine = create_engine(self.connection_string)
            
            # Test the connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            
            print("✅ Database connection established successfully.")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to database: {e}")
            traceback.print_exc()
            return False

    def load_existing_parquet(self) -> pd.DataFrame:
        """Load the existing Tennessee_complete_finished.parquet file."""
        if not os.path.exists(self.input_file):
            print(f"❌ Input Parquet file not found: {self.input_file}")
            return pd.DataFrame()

        try:
            print(f"Loading existing Parquet file: {self.input_file}")
            df = pd.read_parquet(self.input_file, engine='pyarrow')
            
            print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns from Parquet")
            
            # Validate and process date column
            if self.date_col not in df.columns:
                print(f"❌ '{self.date_col}' column not found in Parquet file.")
                return pd.DataFrame()

            # Ensure date column is datetime
            if not pd.api.types.is_datetime64_any_dtype(df[self.date_col]):
                print(f"Converting '{self.date_col}' column to datetime...")
                df[self.date_col] = pd.to_datetime(df[self.date_col], errors='coerce')
                df.dropna(subset=[self.date_col], inplace=True)

            # Ensure timezone-naive
            if df[self.date_col].dt.tz is not None:
                print(f"Converting '{self.date_col}' column to timezone-naive...")
                df[self.date_col] = df[self.date_col].dt.tz_localize(None)

            # Sort by date
            df.sort_values(self.date_col, inplace=True)
            df.reset_index(drop=True, inplace=True)
            
            # Show date range
            print(f"   Parquet date range: {df[self.date_col].min()} to {df[self.date_col].max()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading Parquet file: {e}")
            traceback.print_exc()
            return pd.DataFrame()

    def download_Tennessee_interchange(self) -> pd.DataFrame:
        """Download Tennessee interchange data from the database with date filtering."""
        if self.engine is None:
            print("❌ No database connection available.")
            return pd.DataFrame()

        try:
            print(f"Downloading Tennessee interchange data from {self.table_name}...")
            print(f"Filtering data from {self.start_date_filter} onwards...")
            
            # SQL query to get Tennessee data with date filter
            query = f"""
            SELECT 
                DATETIME,
                {self.Tennessee_column}
            FROM {self.table_name}
            WHERE {self.Tennessee_column} IS NOT NULL
              AND DATETIME >= '{self.start_date_filter.strftime('%Y-%m-%d %H:%M:%S')}'
            ORDER BY DATETIME ASC
            """
            
            print(f"Executing query: {query}")
            
            # Execute query and load into DataFrame
            df = pd.read_sql_query(query, self.engine)
            
            if df.empty:
                print("⚠️  No Tennessee interchange data found in the database for the specified date range.")
                return df
            
            # Ensure DATETIME column is timezone-naive
            if df['DATETIME'].dt.tz is not None:
                df['DATETIME'] = df['DATETIME'].dt.tz_localize(None)
            
            print(f"✅ Successfully downloaded {len(df)} interchange records")
            print(f"   Date range: {df['DATETIME'].min()} to {df['DATETIME'].max()}")
            
            # Display basic statistics
            if not df[self.Tennessee_column].empty:
                stats = df[self.Tennessee_column].describe()
                print(f"   Interchange data statistics:")
                print(f"     Min: {stats['min']:.2f} MW")
                print(f"     Max: {stats['max']:.2f} MW") 
                print(f"     Mean: {stats['mean']:.2f} MW")
                print(f"     Std: {stats['std']:.2f} MW")
                
                # Count of non-null values
                non_null_count = df[self.Tennessee_column].notna().sum()
                print(f"     Non-null values: {non_null_count}/{len(df)} ({100*non_null_count/len(df):.1f}%)")
            
            return df
            
        except Exception as e:
            print(f"❌ Error downloading interchange data: {e}")
            traceback.print_exc()
            return pd.DataFrame()

    def merge_data(self, parquet_df: pd.DataFrame, interchange_df: pd.DataFrame) -> pd.DataFrame:
        """Merge the Parquet data with interchange data using outer join."""
        if parquet_df.empty:
            print("❌ No Parquet data to merge with.")
            return pd.DataFrame()
        
        if interchange_df.empty:
            print("⚠️  No interchange data to merge. Adding empty interchange column.")
            parquet_df[self.Tennessee_column] = None
            return parquet_df

        try:
            print("\nMerging Parquet data with interchange data...")
            
            # Prepare interchange data for merge
            interchange_merge_df = interchange_df.copy()
            interchange_merge_df = interchange_merge_df.rename(columns={'DATETIME': self.date_col})
            
            print(f"Parquet data: {len(parquet_df)} rows")
            print(f"Interchange data: {len(interchange_merge_df)} rows")
            
            # Perform outer join
            merged_df = pd.merge(
                parquet_df, 
                interchange_merge_df, 
                on=self.date_col, 
                how='outer'
            )
            
            print(f"✅ Merge completed: {len(merged_df)} total rows")
            
            # Sort by date and reset index
            merged_df.sort_values(self.date_col, inplace=True)
            merged_df.reset_index(drop=True, inplace=True)
            
            # Show merge statistics
            interchange_col_stats = merged_df[self.Tennessee_column].describe()
            non_null_interchange = merged_df[self.Tennessee_column].notna().sum()
            
            print(f"   Final date range: {merged_df[self.date_col].min()} to {merged_df[self.date_col].max()}")
            print(f"   Interchange column statistics:")
            print(f"     Non-null values: {non_null_interchange}/{len(merged_df)} ({100*non_null_interchange/len(merged_df):.1f}%)")
            if non_null_interchange > 0:
                print(f"     Min: {interchange_col_stats['min']:.2f} MW")
                print(f"     Max: {interchange_col_stats['max']:.2f} MW")
                print(f"     Mean: {interchange_col_stats['mean']:.2f} MW")
            
            return merged_df
            
        except Exception as e:
            print(f"❌ Error merging data: {e}")
            traceback.print_exc()
            return pd.DataFrame()

    def save_merged_parquet(self, df: pd.DataFrame) -> bool:
        """Save the merged DataFrame to Parquet file."""
        if df.empty:
            print("⚠️  No merged data to save.")
            return False

        try:
            print(f"\nSaving merged data to Parquet: {self.output_file}")
            
            # Round numerical columns
            print("  Rounding numerical columns...")
            numeric_cols = df.select_dtypes(include='number').columns
            numeric_cols = numeric_cols.drop(['is_analogous_forecast'], errors='ignore')

            if not numeric_cols.empty:
                print(f"  Rounding {len(numeric_cols)} numeric columns to 6 decimal places.")
                df.loc[:, numeric_cols] = df[numeric_cols].round(6)
            
            # Save to Parquet
            df.to_parquet(self.output_file, index=False, engine='pyarrow')
            
            print(f"✅ Merged Parquet file saved successfully.")
            print(f"   File location: {self.output_file}")
            print(f"   Records saved: {len(df)}")
            print(f"   Columns: {len(df.columns)}")
            
            # Display file size
            file_size = os.path.getsize(self.output_file)
            file_size_mb = file_size / (1024 * 1024)
            print(f"   File size: {file_size_mb:.2f} MB")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving merged Parquet file: {e}")
            traceback.print_exc()
            return False

    def close_connection(self):
        """Close the database connection."""
        if self.engine is not None:
            try:
                self.engine.dispose()
                print("Database connection closed.")
            except Exception as e:
                print(f"Error closing database connection: {e}")

    def run(self):
        """Execute the Tennessee interchange data merge process."""
        print(f"\n--- Starting Tennessee Interchange Data Merge ---")
        parquet_df = pd.DataFrame()
        interchange_df = pd.DataFrame()
        merged_df = pd.DataFrame()
        success = False

        try:
            # 1. Load existing Parquet file
            parquet_df = self.load_existing_parquet()
            if parquet_df.empty:
                print("❌ Failed to load existing Parquet file. Aborting.")
                return

            # 2. Connect to Database
            if not self.connect_to_database():
                print("❌ Failed to establish database connection. Aborting.")
                return

            # 3. Download Tennessee Interchange Data
            interchange_df = self.download_Tennessee_interchange()
            # Note: We continue even if interchange_df is empty to add an empty column

            # 4. Merge Data
            merged_df = self.merge_data(parquet_df, interchange_df)
            if merged_df.empty:
                print("❌ Data merge failed. Aborting.")
                return

            # 5. Save Merged Parquet
            success = self.save_merged_parquet(merged_df)

            if success:
                print(f"\n✅ Tennessee interchange data merge completed successfully!")
                print(f"📁 Output file: {self.output_file}")
            else:
                print(f"\n❌ Merge completed but file save failed.")

        except Exception as e:
            print(f"❌ An unhandled exception occurred: {e}")
            traceback.print_exc()
        finally:
            # Cleanup
            self.close_connection()
            for df_var in ['parquet_df', 'interchange_df', 'merged_df']:
                if df_var in locals() and not locals()[df_var].empty:
                    del locals()[df_var]
            gc.collect()
            print(f"\n--- Tennessee Interchange Merge Process Finished ---")

# --- Main Execution Block ---
if __name__ == "__main__":
    print("===== Starting Tennessee Interchange Data Merge =====")
    print(f"Input Parquet: Tennessee_complete_finished.parquet")
    print(f"Database table: {TABLE_NAME}")
    print(f"Tennessee column: {TENNESSEE_COLUMN}")
    print(f"Date filter: From {START_DATE_FILTER} onwards")
    print(f"Output Parquet: Tennessee_complete_with_interchange.parquet")
    
    merger = None

    try:
        # Instantiate the merger
        merger = TennesseeInterchangeMerger(base_dir=BASE_OUTPUT_DIR)
        
        # Execute the merge process
        merger.run()

    except ValueError as e:
        print(f"Tennessee Interchange Merger (Config/Init Error): {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"Tennessee Interchange Merger (Critical Error): {e}")
        traceback.print_exc()
    finally:
        if merger is not None:
            del merger
        gc.collect()

    print("\n" + "="*60)
    print("--- Tennessee Interchange Data Merge Complete ---")
    print("="*60)