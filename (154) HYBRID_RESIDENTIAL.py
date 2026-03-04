# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os
import warnings
import traceback

warnings.filterwarnings('ignore')

# --- Configuration ---
BASE_DATA_DIR = r"D:\EIA_STACK"
RESULTS_DIR = r"D:\EIA_STACK\HYBRID_RESULTS"

REGIONS = [
    "california", "carolina", "central", "florida", "midatlantic", "midwest",
    "newengland", "newyork", "northwest", "southeast", "southwest", "tennessee", "texas"
]
CONSUMPTION_TYPES = ["residential"]

# --- FEATURE ENGINEERING ---

def create_features(df):
    """Creates all necessary features for the hybrid model."""
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    
    # Cyclical features for the RF model
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
    
    # Temperature features for the Linear Regression model
    temp_cols = [col for col in df.columns if 'temperature' in col.lower()]
    if temp_cols:
        avg_temp = df[temp_cols].mean(axis=1)
        df['HDD'] = np.maximum(0, 65 - avg_temp)
        df['HDD_squared'] = df['HDD'] ** 2 # Key for non-linearity
    else:
        df['HDD'] = 0
        df['HDD_squared'] = 0
        
    return df

# --- ANALYSIS FUNCTION ---
def print_monthly_analysis(df, target_col, forecast_start_date):
    """
    Prints a monthly summary of consumption, HDD, and the response factor.
    """
    print("\n[5] Monthly Consumption vs. HDD Analysis")
    print("-" * 70)
    
    analysis_df = df.copy()
    analysis_df.set_index('datetime', inplace=True)
    
    # Resample by month and sum the relevant columns
    monthly_agg = analysis_df.resample('M').agg({
        target_col: 'sum',
        'HDD': 'sum'
    })
    
    # Calculate BCF and the response factor
    monthly_agg['BCF'] = monthly_agg[target_col] / 1000.0
    # Handle months with zero HDD to avoid division by zero
    monthly_agg['BCF_per_HDD'] = (monthly_agg['BCF'] / monthly_agg['HDD']).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Add a column to distinguish historical vs. forecast periods
    monthly_agg['Type'] = np.where(monthly_agg.index < forecast_start_date, 'Historical', 'Forecast')
    
    # Filter to show the last 6 months of history and the full forecast
    display_df = monthly_agg[monthly_agg.index >= (forecast_start_date - pd.DateOffset(months=6))]

    print(f"{'Month':<12} {'Type':<12} {'Total BCF':>12} {'Total HDD':>12} {'BCF per HDD':>15}")
    print("-" * 70)

    for index, row in display_df.iterrows():
        month_str = index.strftime('%Y-%m')
        print(f"{month_str:<12} {row['Type']:<12} {row['BCF']:>12.2f} {row['HDD']:>12.1f} {row['BCF_per_HDD']:>15.4f}")
    print("-" * 70)


# --- FINAL HYBRID FORECASTING FUNCTION ---

def run_final_hybrid_forecast(train_data, predict_data, target_col):
    """
    Final robust hybrid model:
    1. Linear Regression models the smoothed daily trend based on temperature.
    2. Random Forest models the remaining hourly/weekly patterns (residuals).
    """
    print("    -> Using Final Hybrid Model (LR on Smoothed Trend -> RF on Hourly Pattern)...")

    # 1. Prepare data and features
    full_df = pd.concat([train_data, predict_data], ignore_index=True)
    full_df = create_features(full_df)
    
    temp_features = ['HDD', 'HDD_squared']
    pattern_features = ['hour_sin', 'hour_cos', 'dayofyear_sin', 'dayofyear_cos', 'dayofweek']

    train_mask = full_df[target_col].notna()
    predict_mask = full_df[target_col].isna()

    # 2. Create a SMOOTHED daily target for the Linear Regression model
    y_smooth = full_df[target_col].rolling(window=24, center=True, min_periods=12).mean()
    y_smooth.fillna(method='ffill', inplace=True)
    y_smooth.fillna(method='bfill', inplace=True)
    
    # 3. Train the TEMPERATURE model (Linear Regression) on the SMOOTHED target
    print("    -> Training Linear model on SMOOTH daily trend...")
    lr_temp_model = LinearRegression()
    lr_temp_model.fit(full_df[train_mask][temp_features], y_smooth[train_mask])
    
    daily_trend_forecast = lr_temp_model.predict(full_df[predict_mask][temp_features])
    
    # 4. Calculate the HOURLY PATTERN residuals for the training period
    daily_trend_train = lr_temp_model.predict(full_df[train_mask][temp_features])
    residuals_train = full_df[train_mask][target_col] - daily_trend_train
    
    # 5. Train the PATTERN model (Random Forest) on the clean residuals
    print("    -> Training RF model on clean hourly patterns...")
    rf_pattern_model = RandomForestRegressor(
        n_estimators=150, max_depth=16, min_samples_leaf=10,
        random_state=42, n_jobs=-1
    )
    rf_pattern_model.fit(full_df[train_mask][pattern_features], residuals_train)
    
    # 6. Generate Final Forecast
    print("    -> Generating final forecast...")
    pattern_forecast = rf_pattern_model.predict(full_df[predict_mask][pattern_features])
    final_forecast = daily_trend_forecast + pattern_forecast

    return np.maximum(0, final_forecast), full_df # Return full_df for analysis

def run_pipeline(region_name, consumption_type, base_dir, results_dir):
    """Main pipeline to load, forecast, save, and plot results."""
    region_upper = region_name.upper()
    consumption_cap = consumption_type.capitalize()
    print("\n" + "="*80)
    print(f"--- Processing: {region_upper} - {consumption_cap} ---")
    print("="*80)

    target_col = f"{region_upper}_{consumption_cap}_hourly_MMcf"
    input_dir = os.path.join(base_dir, region_name, "L48_Predictions")
    train_file = os.path.join(input_dir, f"{region_name}_{consumption_type}_train.csv")
    predict_file = os.path.join(input_dir, f"{region_name}_{consumption_type}_predict.csv")

    predictions_dir = os.path.join(results_dir, region_upper, "Predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    
    file_suffix = f"{region_name}_{consumption_type}_final_hybrid_forecast"
    daily_output_file = os.path.join(predictions_dir, f"{file_suffix}.csv")
    plot_file_daily = os.path.join(predictions_dir, f"{file_suffix}_plot.png")
    
    print(f"\n[1] Loading data...")
    try:
        df_train = pd.read_csv(train_file, parse_dates=['datetime'])
        df_predict = pd.read_csv(predict_file, parse_dates=['datetime'])
    except FileNotFoundError:
        print(f"    -> WARNING: Files not found. Skipping.")
        return False
    
    print("[2] Running final hybrid forecast...")
    predictions, full_df_with_features = run_final_hybrid_forecast(df_train, df_predict, target_col)
    
    print("[3] Processing and saving results...")
    df_pred_output = df_predict[['datetime']].copy()
    df_pred_output[f"{target_col}_Predicted"] = predictions
    
    df_final_output = pd.concat([df_train[['datetime', target_col]], df_pred_output], ignore_index=True)
    df_final_output[target_col] = df_final_output[target_col].fillna(df_final_output[f"{target_col}_Predicted"])
    
    df_daily = df_final_output.set_index('datetime').resample('D')[target_col].sum().reset_index()
    df_daily[f'{target_col}_BCF'] = df_daily[target_col] / 1000.0
    df_daily.to_csv(daily_output_file, index=False, float_format='%.4f')
    print(f"    -> Daily predictions saved to {daily_output_file}")
    
    # Combine final results with features for analysis
    final_df_for_analysis = df_final_output.merge(full_df_with_features.drop(columns=[target_col], errors='ignore'), on='datetime')
    
    forecast_start_date = pd.to_datetime(df_predict['datetime'].min())
    
    # This now happens before plotting
    print_monthly_analysis(final_df_for_analysis, target_col, forecast_start_date)
    
    print("[4] Generating plots...")
    df_daily['datetime'] = pd.to_datetime(df_daily['datetime'])
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), gridspec_kw={'height_ratios': [2, 1]})

    ax1.plot(df_daily[df_daily['datetime'] < forecast_start_date]['datetime'], 
             df_daily[df_daily['datetime'] < forecast_start_date][f'{target_col}_BCF'], 
             label='Historical', color='black', linewidth=1.2)
    ax1.plot(df_daily[df_daily['datetime'] >= forecast_start_date]['datetime'], 
             df_daily[df_daily['datetime'] >= forecast_start_date][f'{target_col}_BCF'], 
             label='Forecast', color='blue', linewidth=1.5)
    ax1.axvline(x=forecast_start_date, color='red', linestyle='--', label='Forecast Start', linewidth=2)
    ax1.set_title(f'{region_upper} - {consumption_cap} - Final Hybrid Forecast', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Daily Consumption (BCF)')
    ax1.legend()

    zoom_start = forecast_start_date - pd.Timedelta(days=90)
    df_zoom = df_daily[df_daily['datetime'] >= zoom_start]
    
    ax2.plot(df_zoom[df_zoom['datetime'] < forecast_start_date]['datetime'], 
             df_zoom[df_zoom['datetime'] < forecast_start_date][f'{target_col}_BCF'], 
             label='Historical', color='black', marker='o', markersize=3, alpha=0.7)
    ax2.plot(df_zoom[df_zoom['datetime'] >= forecast_start_date]['datetime'], 
             df_zoom[df_zoom['datetime'] >= forecast_start_date][f'{target_col}_BCF'], 
             label='Forecast', color='blue')
    ax2.axvline(x=forecast_start_date, color='red', linestyle='--')
    ax2.set_title('Zoomed View: Last 90 Days + Forecast', fontsize=14)
    ax2.set_ylabel('Daily Consumption (BCF)')
    ax2.set_xlabel('Date')
    
    plt.tight_layout()
    plt.savefig(plot_file_daily, dpi=300)
    plt.close()
    print(f"    -> Plot saved to {plot_file_daily}")
    
    return True

if __name__ == "__main__":
    for region in REGIONS:
        for consumption in CONSUMPTION_TYPES:
            try:
                run_pipeline(region, consumption, BASE_DATA_DIR, RESULTS_DIR)
            except Exception as e:
                print(f"\n!!! ERROR processing {region} - {consumption}: {str(e)}")
                traceback.print_exc()