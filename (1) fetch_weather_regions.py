# scripts/(1) fetch_weather_regions.py

import json
from datetime import datetime, timedelta
import os
import sys
import traceback

# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)
print(f"Project root added to sys.path: {project_root}")

# --- Imports after path setup ---
try:
    from src.weather_fetcher import WeatherDataFetcher # Your modified fetcher class
except ImportError:
    print(f"FATAL ERROR: Could not import WeatherDataFetcher from src.")
    print(f"Ensure '{os.path.join(project_root, 'src', 'weather_fetcher.py')}' exists and sys.path is correct.")
    sys.exit(1)

try:
    from config.settings import API_KEY, BASE_OUTPUT_DIR # Import settings
except ImportError:
    print(f"FATAL ERROR: Could not import settings from config.settings.")
    print(f"Ensure '{os.path.join(project_root, 'config', 'settings.py')}' exists and sys.path is correct.")
    sys.exit(1)


# --- Configuration Loading ---
config_dir = os.path.join(project_root, 'config')
regions_json_path = os.path.join(config_dir, 'regions.json')
variables_json_path = os.path.join(config_dir, 'variables.json')

print(f"Attempting to load regions config from: {regions_json_path}")
try:
    with open(regions_json_path, 'r') as f:
        regions = json.load(f)
    print(f"Successfully loaded regions config.")
except FileNotFoundError:
    print(f"FATAL ERROR: Regions configuration file not found at '{regions_json_path}'.")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"FATAL ERROR: Invalid JSON in '{regions_json_path}'. Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred while loading '{regions_json_path}'. Error: {e}")
    print(traceback.format_exc())
    sys.exit(1)

print(f"Attempting to load variables config from: {variables_json_path}")
try:
    with open(variables_json_path, 'r') as f:
        variables = json.load(f)
    print(f"Successfully loaded variables config.")
    if not isinstance(variables, list):
         print(f"WARNING: Expected 'variables.json' to contain a JSON list of strings. Found type: {type(variables)}")
except FileNotFoundError:
    print(f"FATAL ERROR: Variables configuration file not found at '{variables_json_path}'.")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"FATAL ERROR: Invalid JSON in '{variables_json_path}'. Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred while loading '{variables_json_path}'. Error: {e}")
    print(traceback.format_exc())
    sys.exit(1)


# --- Timezone Mapping ---
REGION_TIMEZONE_MAP = {
    "southwest": "America/Denver",       # Mountain Time
    "southeast": "America/New_York",     # Eastern Time
    "newyork": "America/New_York",       # Eastern Time
    "northwest": "America/Los_Angeles",  # Pacific Time
    "newengland": "America/New_York",    # Eastern Time
    "midwest": "America/Chicago",        # Central Time
    "midatlantic": "America/New_York",   # Eastern Time
    "florida": "America/New_York",       # Eastern Time
    "central": "America/Chicago",        # Central Time
    "california": "America/Los_Angeles", # Pacific Time
    "carolina": "America/New_York",      # Eastern Time
    "tennessee": "America/Chicago",      # Central Time
    "texas": "America/Chicago",          # Central Time
}
ALLOWED_TIMEZONES = {
    "America/New_York",      # Eastern
    "America/Chicago",       # Central
    "America/Denver",        # Mountain
    "America/Los_Angeles",   # Pacific
}


# --- Time settings ---
start_date = datetime(2010, 1, 1)
end_date = datetime.now() - timedelta(days=3)
batch_months = 36

# --- Initialize fetcher ---
print(f"Initializing WeatherDataFetcher. Output base directory: {BASE_OUTPUT_DIR}")
try:
    fetcher = WeatherDataFetcher(API_KEY, BASE_OUTPUT_DIR)
except Exception as e:
    print(f"FATAL ERROR: Failed to initialize WeatherDataFetcher. Error: {e}")
    print(traceback.format_exc())
    sys.exit(1)

# --- Process each region ---
print("\nStarting weather data processing...")
all_regions_processed_successfully = True

for region_name, locations_data in regions.items():
    print(f"\nProcessing Region: {region_name}")
    region_success = True

    # 1. Get Timezone
    region_timezone = REGION_TIMEZONE_MAP.get(region_name)
    if not region_timezone:
        print(f"  [ERROR] Timezone not defined for region '{region_name}' in REGION_TIMEZONE_MAP. Skipping this region.")
        all_regions_processed_successfully = False
        region_success = False
        continue

    if region_timezone not in ALLOWED_TIMEZONES:
         print(f"  [WARNING] Timezone '{region_timezone}' for region '{region_name}' is not in the ALLOWED_TIMEZONES set. Proceeding...")

    print(f"  Using timezone: {region_timezone}")

    # 2. Extract Locations
    if not isinstance(locations_data, list):
        print(f"  [ERROR] Expected a list of locations for region '{region_name}', but got type {type(locations_data)}. Skipping.")
        all_regions_processed_successfully = False
        region_success = False
        continue

    locations = []
    try:
        locations = [(loc['lat'], loc['lon'], loc['code']) for loc in locations_data]
        if not locations:
             print(f"  [ERROR] No locations found in the list for region '{region_name}'. Skipping.")
             all_regions_processed_successfully = False
             region_success = False
             continue
        print(f"  Found {len(locations)} locations for this region.")
    except (KeyError, TypeError) as e:
        print(f"  [ERROR] Invalid format in location list for region '{region_name}'. Ensure each item is a dict with 'lat', 'lon', 'code'. Error: {e}. Skipping.")
        all_regions_processed_successfully = False
        region_success = False
        continue

    # 3. Fetch Forecast Data
    try:
        print(f"  Fetching forecast data...")
        # CHANGE: Pass parquet_filename instead of csv_filename
        forecast_parquet_filename = f"{region_name}_meteo_forecast.parquet"
        fetcher.fetch_forecast(
            region=region_name,
            locations=locations,
            variables=variables,
            timezone_str=region_timezone,
            past_days=3,
            forecast_days=16,
            parquet_filename=forecast_parquet_filename # Pass the parquet filename
        )
        print(f"  Forecast fetch initiated for {region_name} -> {forecast_parquet_filename}.")
    except Exception as e:
        print(f"  [ERROR] Unhandled exception during forecast fetch for {region_name}. Error: {e}")
        print(traceback.format_exc())
        all_regions_processed_successfully = False
        region_success = False

    # 4. Fetch Historical Data
    try:
        print(f"  Fetching historical data...")
        # CHANGE: Pass parquet_filename instead of csv_filename
        hist_db_filename = f"{region_name}_meteo_weather.sqlite"
        hist_parquet_filename = f"{region_name}_meteo_weather.parquet"
        fetcher.fetch_historical_in_batches(
            region=region_name,
            locations=locations,
            variables=variables,
            timezone_str=region_timezone,
            start_date=start_date,
            end_date=end_date,
            batch_months=batch_months,
            db_filename=hist_db_filename, # Keep db filename
            parquet_filename=hist_parquet_filename # Pass the parquet filename
        )
        print(f"  Historical fetch initiated for {region_name} -> {hist_db_filename} & {hist_parquet_filename}.")
    except Exception as e:
        print(f"  [ERROR] Unhandled exception during historical fetch for {region_name}. Error: {e}")
        print(traceback.format_exc())
        all_regions_processed_successfully = False
        region_success = False

    if region_success:
        print(f"Finished processing Region: {region_name}")
    else:
        print(f"Finished processing Region: {region_name} with ERRORS.")


# --- Final Summary ---
print("\n-------------------------------------")
if all_regions_processed_successfully:
    print("All regions processed successfully. Output files are in Parquet format.")
else:
    print("Processing finished, but ERRORS occurred in one or more regions. Please review logs.")
print("-------------------------------------")

# Optional: Exit with non-zero status if errors occurred
# if not all_regions_processed_successfully:
#     sys.exit(1)