# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
BASE_DATA_DIR = r"D:\EIA_STACK"
RESULTS_DIR = r"D:\EIA_STACK\HYBRID_RESULTS"  # Same folder as residential

REGIONS = [
    "california", "carolina", "central", "florida", "midatlantic", "midwest",
    "newengland", "newyork", "northwest", "southeast", "southwest", "tennessee", "texas"
]
CONSUMPTION_TYPES = ["commercial", "industrial"]

# Model parameters
SEASONAL_PERIOD = 8760  # Yearly seasonality for hourly data
TREND_PERIOD = 168  # Weekly trend component

def extract_seasonal_components(series, period=8760):
    """Extract trend, seasonal, and residual components with multiple fallback methods."""
    series = series.fillna(method='ffill').fillna(method='bfill').fillna(series.mean())
    
    # Method 1: Try STL if we have enough data
    if len(series) >= period * 2:
        try:
            stl = STL(series, seasonal=13, trend=period//2+1 if period//2+1 < len(series) else len(series)//3)
            result = stl.fit()
            return result.trend, result.seasonal, result.resid
        except Exception as e:
            print(f"       STL failed: {e}")
    
    # Method 2: Classical seasonal decomposition
    if len(series) >= period:
        try:
            result = seasonal_decompose(series, model='additive', period=period, extrapolate_trend='freq')
            return result.trend, result.seasonal, result.resid
        except Exception as e:
            print(f"       Classical decomposition failed: {e}")
    
    # Method 3: Multi-scale moving average decomposition
    print("       Using multi-scale moving average decomposition")
    
    # Extract trend using multiple scales
    if len(series) >= 720:  # 30 days
        trend = series.rolling(window=720, center=True, min_periods=360).mean()
    elif len(series) >= 168:  # 7 days
        trend = series.rolling(window=168, center=True, min_periods=84).mean()
    else:
        trend = series.rolling(window=24, center=True, min_periods=12).mean()
    
    # Fill trend edges
    trend = trend.fillna(method='bfill').fillna(method='ffill')
    
    # Extract seasonal patterns at multiple scales
    detrended = series - trend
    
    # Daily pattern (24 hours)
    daily_seasonal = np.zeros(len(series))
    if len(series) >= 24:
        for hour in range(24):
            hour_data = [detrended.iloc[i] for i in range(hour, len(detrended), 24) if i < len(detrended)]
            if hour_data:
                for i in range(hour, len(series), 24):
                    if i < len(series):
                        daily_seasonal[i] = np.nanmean(hour_data)
    
    # Weekly pattern (168 hours)
    weekly_seasonal = np.zeros(len(series))
    if len(series) >= 168:
        for hour in range(168):
            hour_data = [detrended.iloc[i] for i in range(hour, len(detrended), 168) if i < len(detrended)]
            if hour_data:
                for i in range(hour, len(series), 168):
                    if i < len(series):
                        weekly_seasonal[i] = np.nanmean(hour_data)
    
    # Combine seasonal components
    seasonal = pd.Series(daily_seasonal + weekly_seasonal * 0.5, index=series.index)
    
    # Calculate residuals
    resid = series - trend - seasonal
    
    return trend, seasonal, resid

def get_seasonal_forecast_commercial_industrial(historical_data, n_periods, seasonal_period, sector_type):
    """Generate seasonal forecast specifically tuned for commercial/industrial sectors."""
    
    # Commercial and industrial have different patterns than residential
    if sector_type == "commercial":
        # Commercial: Strong weekday/weekend pattern, business hours effect
        workday_factor = 1.2  # Higher on weekdays
        weekend_factor = 0.6  # Lower on weekends
        business_hours_factor = 1.3  # 8am-6pm higher
        night_factor = 0.7  # Night hours lower
    else:  # industrial
        # Industrial: More consistent, less weather-dependent
        workday_factor = 1.1
        weekend_factor = 0.8
        business_hours_factor = 1.15
        night_factor = 0.85
    
    # Method 1: If we have full year(s) of data
    if len(historical_data) >= seasonal_period:
        n_years = len(historical_data) // seasonal_period
        forecasts = []
        
        for i in range(n_periods):
            # Get same hour from previous years
            historical_values = []
            for year in range(min(n_years, 3)):
                idx = len(historical_data) - (year + 1) * seasonal_period + i
                if 0 <= idx < len(historical_data):
                    historical_values.append(historical_data.iloc[idx])
            
            if historical_values:
                # Weighted average (more recent years get more weight)
                weights = [0.5, 0.3, 0.2][:len(historical_values)]
                forecast_value = np.average(historical_values, weights=weights[:len(historical_values)])
            else:
                # Use same period from last available cycle
                last_cycle_idx = i % len(historical_data)
                forecast_value = historical_data.iloc[last_cycle_idx]
            
            forecasts.append(forecast_value)
    
    # Method 2: Use weekly patterns with business logic
    elif len(historical_data) >= 168 * 4:  # 4 weeks
        weekly_period = 168
        forecasts = []
        
        # Calculate average pattern for each hour of the week
        weekly_pattern = np.zeros(weekly_period)
        for hour in range(weekly_period):
            hour_values = [historical_data.iloc[i] for i in range(hour, len(historical_data), weekly_period)]
            if hour_values:
                weekly_pattern[hour] = np.nanmean(hour_values)
        
        # Apply business logic adjustments
        for hour in range(weekly_period):
            day_of_week = hour // 24
            hour_of_day = hour % 24
            
            # Weekday vs weekend
            if day_of_week < 5:  # Monday-Friday
                weekly_pattern[hour] *= workday_factor
            else:  # Weekend
                weekly_pattern[hour] *= weekend_factor
            
            # Business hours vs off-hours
            if 8 <= hour_of_day <= 18:
                weekly_pattern[hour] *= business_hours_factor
            else:
                weekly_pattern[hour] *= night_factor
        
        # Generate forecast
        for i in range(n_periods):
            forecasts.append(weekly_pattern[i % weekly_period])
    
    # Method 3: Daily patterns with sector-specific adjustments
    else:
        daily_period = 24
        forecasts = []
        
        # Calculate average pattern for each hour of the day
        daily_pattern = np.zeros(daily_period)
        for hour in range(daily_period):
            hour_values = [historical_data.iloc[i] for i in range(hour, len(historical_data), daily_period)]
            if hour_values:
                daily_pattern[hour] = np.nanmean(hour_values)
        
        # Apply business hours logic
        for hour in range(daily_period):
            if 8 <= hour <= 18:
                daily_pattern[hour] *= business_hours_factor
            else:
                daily_pattern[hour] *= night_factor
        
        # Generate forecast
        for i in range(n_periods):
            forecasts.append(daily_pattern[i % daily_period])
    
    return np.array(forecasts)

def create_time_features(df, date_col='datetime'):
    """Create time-based features."""
    dt = pd.to_datetime(df[date_col])
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)
    df['day_sin'] = np.sin(2 * np.pi * dt.dt.dayofyear / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * dt.dt.dayofyear / 365.25)
    df['week_sin'] = np.sin(2 * np.pi * dt.dt.isocalendar().week / 52)
    df['week_cos'] = np.cos(2 * np.pi * dt.dt.isocalendar().week / 52)
    df['month_sin'] = np.sin(2 * np.pi * dt.dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * dt.dt.month / 12)
    
    # Categorical features
    df['hour'] = dt.dt.hour
    df['dayofweek'] = dt.dt.dayofweek
    df['dayofyear'] = dt.dt.dayofyear
    df['weekofyear'] = dt.dt.isocalendar().week
    df['month'] = dt.dt.month
    df['quarter'] = dt.dt.quarter
    df['year'] = dt.dt.year
    df['weekend'] = (dt.dt.dayofweek >= 5).astype(int)
    
    # Business indicators (important for commercial/industrial)
    df['is_business_hour'] = ((dt.dt.hour >= 8) & (dt.dt.hour <= 18)).astype(int)
    df['is_weekday'] = (dt.dt.dayofweek < 5).astype(int)
    
    return df

def create_weather_features(df, weather_cols):
    """Create weather features."""
    if not weather_cols:
        return df
    
    df['temp_mean'] = df[weather_cols].mean(axis=1)
    df['temp_range'] = df[weather_cols].max(axis=1) - df[weather_cols].min(axis=1)
    
    # Heating/cooling degree days (less important for industrial)
    base_temp = 65
    df['heating_degrees'] = np.maximum(0, base_temp - df['temp_mean'])
    df['cooling_degrees'] = np.maximum(0, df['temp_mean'] - base_temp)
    df['temp_squared'] = df['temp_mean'] ** 2
    
    return df

def hybrid_forecast(train_data, predict_periods, target_col, sector_type, weather_data=None):
    """
    Hybrid forecasting approach adapted for commercial/industrial sectors.
    """
    
    # Step 1: Decompose the series
    print("    -> Decomposing time series...")
    train_series = train_data[target_col]
    
    # Choose decomposition period based on data length
    if len(train_series) >= SEASONAL_PERIOD:
        decomp_period = SEASONAL_PERIOD
        print(f"       Using yearly seasonality ({SEASONAL_PERIOD} hours)")
    elif len(train_series) >= 168 * 4:
        decomp_period = 168  # Weekly
        print(f"       Using weekly seasonality (168 hours)")
    else:
        decomp_period = 24  # Daily
        print(f"       Using daily seasonality (24 hours)")
    
    trend, seasonal, residual = extract_seasonal_components(train_series, decomp_period)
    
    # Step 2: Forecast trend (more conservative for commercial/industrial)
    print("    -> Forecasting trend component...")
    x_train = np.arange(len(trend)).reshape(-1, 1)
    
    # Use robust linear regression (no polynomial for stability)
    trend_model = HuberRegressor()
    trend_model.fit(x_train, trend.fillna(method='bfill').fillna(method='ffill'))
    
    x_future = np.arange(len(trend), len(trend) + predict_periods).reshape(-1, 1)
    trend_forecast = trend_model.predict(x_future)
    
    # Strong damping for commercial/industrial (more stable demand)
    damping_factor = 0.95 ** np.arange(len(trend_forecast))
    trend_mean = trend.tail(168).mean() if len(trend) >= 168 else trend.mean()
    trend_forecast = trend_forecast * damping_factor + trend_mean * (1 - damping_factor)
    
    # Step 3: Seasonal forecast with sector-specific patterns
    print(f"    -> Forecasting seasonal component ({sector_type})...")
    seasonal_forecast = get_seasonal_forecast_commercial_industrial(
        seasonal, predict_periods, decomp_period, sector_type
    )
    
    # Adjust amplitude based on sector
    if sector_type == "commercial":
        amplitude_factor = 1.2  # More variation in commercial
    else:  # industrial
        amplitude_factor = 0.8  # Less variation in industrial
    
    recent_amplitude = seasonal.tail(min(decomp_period, len(seasonal))).std()
    historical_amplitude = seasonal.std()
    if historical_amplitude > 0:
        seasonal_forecast = seasonal_forecast * (recent_amplitude / historical_amplitude) * amplitude_factor
    
    # Step 4: Model residuals (less important for industrial)
    residual_forecast = np.zeros(predict_periods)
    
    if weather_data is not None and sector_type == "commercial":
        # Commercial is more weather-sensitive than industrial
        print("    -> Modeling residual component with weather data...")
        try:
            # Simple mean reversion for residuals
            residual_mean = residual.mean()
            residual_std = residual.std()
            # Add small random noise that reverts to mean
            residual_forecast = np.random.normal(residual_mean, residual_std * 0.5, predict_periods)
            # Dampen over time
            decay = np.exp(-np.arange(predict_periods) / (predict_periods / 3))
            residual_forecast = residual_forecast * decay
        except:
            residual_forecast = np.zeros(predict_periods)
    
    # Step 5: Combine forecasts
    combined_forecast = trend_forecast + seasonal_forecast + residual_forecast
    
    # Ensure non-negative
    combined_forecast = np.maximum(0, combined_forecast)
    
    # Apply sector-specific bounds
    if sector_type == "industrial":
        # Industrial tends to be more stable
        forecast_mean = combined_forecast.mean()
        forecast_std = combined_forecast.std()
        max_deviation = 2.0  # Standard deviations
        combined_forecast = np.clip(
            combined_forecast,
            forecast_mean - max_deviation * forecast_std,
            forecast_mean + max_deviation * forecast_std
        )
    
    return combined_forecast, {
        'trend': trend_forecast,
        'seasonal': seasonal_forecast,
        'residual': residual_forecast
    }

def run_hybrid_forecast(region_name, consumption_type, base_dir, results_dir):
    """Run the hybrid forecasting pipeline for commercial/industrial."""
    region_upper = region_name.upper()
    consumption_cap = consumption_type.capitalize()
    print("\n" + "="*80)
    print(f"--- Processing: {region_upper} - {consumption_cap} ---")
    print("="*80)

    target_col = f"{region_upper}_{consumption_cap}_hourly_MMcf"
    input_dir = os.path.join(base_dir, region_name, "L48_Predictions")
    train_file = os.path.join(input_dir, f"{region_name}_{consumption_type}_train.csv")
    predict_file = os.path.join(input_dir, f"{region_name}_{consumption_type}_predict.csv")

    models_dir = os.path.join(results_dir, region_upper, "Models")
    predictions_dir = os.path.join(results_dir, region_upper, "Predictions")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    # Use same naming convention as the N-HiTS script
    file_suffix = f"{region_name}_{consumption_type}"
    daily_output_file = os.path.join(predictions_dir, f"{file_suffix}.csv")
    hourly_output_file = os.path.join(predictions_dir, f"{file_suffix}_hourly.csv")
    plot_file_daily = os.path.join(predictions_dir, f"{file_suffix}_daily_plot.png")
    plot_file_monthly = os.path.join(predictions_dir, f"{file_suffix}_monthly_plot.png")
    components_plot_file = os.path.join(predictions_dir, f"{file_suffix}_components_plot.png")
    
    print(f"\n[1] Loading data...")
    try:
        df_train = pd.read_csv(train_file, parse_dates=['datetime'])
        df_predict = pd.read_csv(predict_file, parse_dates=['datetime'])
        print(f"    -> Train data: {len(df_train)} rows")
        print(f"    -> Predict data: {len(df_predict)} rows")
    except FileNotFoundError:
        print(f"    -> WARNING: Train/Predict files not found. Skipping.")
        return
    
    print("[2] Feature Engineering...")
    # Get weather columns
    weather_cols = [col for col in df_train.columns if '_temperature_2m' in col.lower()]
    for col in weather_cols:
        df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
        df_predict[col] = pd.to_numeric(df_predict[col], errors='coerce')
    
    # Create features
    df_train = create_time_features(df_train)
    df_train = create_weather_features(df_train, weather_cols)
    
    df_predict = create_time_features(df_predict)
    df_predict = create_weather_features(df_predict, weather_cols)
    
    print("[3] Running hybrid forecast...")
    # Get weather data for prediction period if available
    weather_features = None
    if weather_cols and consumption_type == "commercial":
        # Commercial is more weather-sensitive
        weather_features = df_predict[['temp_mean', 'heating_degrees', 'cooling_degrees']].fillna(
            df_train[['temp_mean', 'heating_degrees', 'cooling_degrees']].mean()
        )
    
    # Run hybrid forecast
    predictions, components = hybrid_forecast(
        df_train,
        len(df_predict),
        target_col,
        consumption_type,
        weather_features
    )
    
    print("[4] Processing results...")
    # Create output dataframe
    df_pred_output = df_predict[['datetime']].copy()
    df_pred_output[f"{target_col}_Predicted"] = predictions
    
    # Combine with historical data
    df_final_output = pd.concat([
        df_train[['datetime', target_col]],
        df_pred_output
    ], ignore_index=True)
    
    # Fill target column
    df_final_output[target_col] = df_final_output[target_col].fillna(
        df_final_output[f"{target_col}_Predicted"])
    
    # Save hourly output
    df_final_output.to_csv(hourly_output_file, index=False, float_format='%.4f')
    print(f"    -> Hourly predictions saved to {hourly_output_file}")
    
    print("[5] Aggregating to daily data...")
    df_daily = df_final_output.set_index('datetime').resample('D')[target_col].sum().reset_index()
    df_daily[f'{target_col}_BCF'] = df_daily[target_col] / 1000.0
    df_daily.to_csv(daily_output_file, index=False, float_format='%.4f')
    print(f"    -> Daily predictions saved to {daily_output_file}")
    
    print("[6] Generating plots...")
    forecast_start_date = df_predict['datetime'].min()
    df_daily_hist = df_daily[df_daily['datetime'] < forecast_start_date]
    df_daily_pred = df_daily[df_daily['datetime'] >= forecast_start_date]
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Main forecast plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
    
    ax1.plot(df_daily_hist['datetime'], df_daily_hist[f'{target_col}_BCF'], 
             label='Historical Daily', color='black', linewidth=1.5, alpha=0.8)
    ax1.plot(df_daily_pred['datetime'], df_daily_pred[f'{target_col}_BCF'], 
             label=f'Predicted Daily ({consumption_cap})', color='blue', linewidth=2)
    ax1.axvline(x=forecast_start_date, color='red', linestyle='--', 
                label='Forecast Start', linewidth=2)
    ax1.set_title(f'{region_upper} - {consumption_cap} - Daily Consumption Forecast (Hybrid Model)', 
                  fontsize=16, fontweight='bold')
    ax1.set_ylabel('Daily Consumption (BCF)', fontsize=12)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Zoomed view
    zoom_start = forecast_start_date - pd.Timedelta(days=90)
    df_zoom = df_daily[df_daily['datetime'] >= zoom_start]
    df_zoom_hist = df_zoom[df_zoom['datetime'] < forecast_start_date]
    df_zoom_pred = df_zoom[df_zoom['datetime'] >= forecast_start_date]
    
    ax2.plot(df_zoom_hist['datetime'], df_zoom_hist[f'{target_col}_BCF'], 
             label='Historical', color='black', linewidth=1.5, marker='o', markersize=3)
    ax2.plot(df_zoom_pred['datetime'], df_zoom_pred[f'{target_col}_BCF'], 
             label='Predicted', color='blue', linewidth=1.5, marker='s', markersize=3)
    ax2.axvline(x=forecast_start_date, color='red', linestyle='--', 
                label='Forecast Start', linewidth=2)
    ax2.set_title('Zoomed View: Last 90 Days + Forecast', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Daily Consumption (BCF)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_file_daily, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Monthly comparison
    df_daily['year'] = df_daily['datetime'].dt.year
    df_daily['month'] = df_daily['datetime'].dt.month
    df_monthly = df_daily.groupby(['year', 'month'])[f'{target_col}_BCF'].sum().reset_index()
    df_monthly['month_name'] = df_monthly['month'].apply(
        lambda x: pd.to_datetime(str(x), format='%m').strftime('%b'))
    
    plt.figure(figsize=(16, 8))
    sns.barplot(data=df_monthly, x='month_name', y=f'{target_col}_BCF', 
                hue='year', palette='viridis')
    plt.title(f'{region_upper} - {consumption_cap} - Monthly Consumption (Hybrid)', 
              fontsize=16, fontweight='bold')
    plt.ylabel('Total Monthly Consumption (BCF)', fontsize=12)
    plt.xlabel('Month', fontsize=12)
    plt.legend(title='Year', loc='best')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(plot_file_monthly, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    -> Plots saved successfully.")
    
    # Summary statistics
    if len(df_daily_hist) > 30 and len(df_daily_pred) > 0:
        recent_hist = df_daily_hist.tail(30)[f'{target_col}_BCF'].mean()
        pred_mean = df_daily_pred[f'{target_col}_BCF'].mean()
        hist_std = df_daily_hist.tail(30)[f'{target_col}_BCF'].std()
        pred_std = df_daily_pred[f'{target_col}_BCF'].std()
        
        print(f"\n[7] Summary Statistics:")
        print(f"    -> Recent 30-day historical average: {recent_hist:.2f} BCF/day (±{hist_std:.2f})")
        print(f"    -> Predicted average: {pred_mean:.2f} BCF/day (±{pred_std:.2f})")
        print(f"    -> Predicted std dev: {pred_std:.2f} BCF/day")
        
        # Check if seasonal pattern is preserved
        if pred_std > 0.05:  # Has variation
            print(f"    -> ✓ Pattern preserved in forecast")
        else:
            print(f"    -> ⚠ Warning: Forecast may be too flat")

# --- Main Execution ---
if __name__ == "__main__":
    print("="*80)
    print("--- HYBRID SEASONAL FORECASTING PIPELINE ---")
    print("--- Commercial & Industrial Sectors ---")
    print(f"--- Base Data Directory: {BASE_DATA_DIR} ---")
    print(f"--- Results Directory: {RESULTS_DIR} ---")
    print("="*80)
    
    successful_regions = []
    failed_regions = []
    
    for region in REGIONS:
        for consumption in CONSUMPTION_TYPES:
            try:
                run_hybrid_forecast(region, consumption, BASE_DATA_DIR, RESULTS_DIR)
                successful_regions.append(f"{region}-{consumption}")
            except Exception as e:
                print(f"\n!!! ERROR processing {region} - {consumption}: {str(e)}")
                failed_regions.append(f"{region}-{consumption}")
                import traceback
                traceback.print_exc()
                continue
    
    print("\n" + "="*80)
    print("--- PIPELINE SUMMARY ---")
    print(f"Successfully processed: {len(successful_regions)} regions")
    if successful_regions:
        print(f"  Successful: {', '.join(successful_regions)}")
    if failed_regions:
        print(f"  Failed: {', '.join(failed_regions)}")
    print("="*80)