import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# --- Configuration ---
EXISTING_DATA_PATH = 'INDIAmerged_hourly_load_weather.csv'
OUTPUT_SYNTHETIC_PATH = 'india_synthetic_load_2001_2017.csv'
OUTPUT_COMPLETE_PATH = 'india_complete_load_2001_2025.csv'
PLOT_TIMESERIES_PATH = 'load_timeseries_2001_2025.png'
PLOT_HEATMAP_PATH = 'load_heatmap_2001_2025.png'

# Historical Constraints
PEAK_2001_TARGET = 78000  # ~78 GW
PEAK_2017_TARGET = 160000 # ~160 GW
GROWTH_RATE_2001_2010 = 0.07 # 7% growth (approx)
GROWTH_RATE_2011_2017 = 0.05 # 5% growth (approx)

# Diwali Dates (Approximate for major impact days)
DIWALI_DATES = {
    2001: '2001-11-14', 2002: '2002-11-04', 2003: '2003-10-25', 2004: '2004-11-12',
    2005: '2005-11-01', 2006: '2006-10-21', 2007: '2007-11-09', 2008: '2008-10-27',
    2009: '2009-10-17', 2010: '2010-11-05', 2011: '2011-10-26', 2012: '2012-11-13',
    2013: '2013-11-03', 2014: '2014-10-23', 2015: '2015-11-11', 2016: '2016-10-30',
    2017: '2017-10-19'
}

def load_and_preprocess(filepath):
    """Loads existing data and preprocesses timestamps."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    # Flexible column name handling for timestamp
    time_col = [c for c in df.columns if 'time' in c.lower()][0]
    df[time_col] = pd.to_datetime(df[time_col])
    # Ensure no duplicates
    df = df.drop_duplicates(subset=[time_col]).sort_values(by=time_col)
    # Set index to timestamp
    df = df.set_index(time_col)
    
    # Identify load column
    load_col = [c for c in df.columns if 'load' in c.lower()][0]
    
    print(f"Data loaded. Range: {df.index.min()} to {df.index.max()}")
    return df, load_col

def extract_patterns(df, year=2018):
    """Extracts seasonal and daily patterns from a reference year."""
    print(f"Extracting patterns from reference year: {year}...")
    ref_df = df[df.index.year == year].copy()
    
    # 1. Annual Seasonality (Day of Year) - Normalized by yearly mean
    daily_avg = ref_df.resample('D').mean(numeric_only=True)
    yearly_mean = daily_avg['load'].mean().item()  # Scalar
    
    # Handle leap years in pattern extraction logic later, but for now get 365 profile
    # We smooth it to avoid overfitting to one year's specific weather anomalies too much
    # Using a 7-day rolling average for smoothness
    daily_pattern = daily_avg.rolling(window=7, center=True, min_periods=1).mean()
    daily_factors = daily_pattern / yearly_mean
    
    # 2. Daily Cycles (Hour of Day) - Normalized by daily mean
    ref_df['hour'] = ref_df.index.hour
    ref_df['dayofweek'] = ref_df.index.dayofweek
    ref_df['month'] = ref_df.index.month
    
    # Create hourly profile for each month/daytype (weekday/weekend)
    ref_df['is_weekend'] = ref_df['dayofweek'] >= 5
    
    hourly_factors = ref_df.groupby(['month', 'is_weekend', 'hour']).mean(numeric_only=True)
    # Normalize by the monthly mean for that specific day type
    monthly_means = ref_df.groupby(['month', 'is_weekend']).mean(numeric_only=True)
    
    return daily_factors, hourly_factors, monthly_means, yearly_mean

def generate_weather(start_date, end_date, ref_weather_df):
    """Clones 2018 weather data patterns with injected noise."""
    print("Generating synthetic weather data...")
    dates = pd.date_range(start=start_date, end=end_date, freq='h')
    synthetic_weather = pd.DataFrame(index=dates)
    
    # Reference 2018 weather
    ref_2018 = ref_weather_df[ref_weather_df.index.year == 2018].copy()
    
    # We map formatted Month-Day-Hour to the reference year
    # This aligns leap years reasonably well (Feb 29 might map to Feb 28 or March 1 depending on logic, 
    # but simplest is to map day-of-year mod 365)
    
    # Only numeric columns from weather
    ref_weather_cols = [c for c in ref_weather_df.select_dtypes(include=[np.number]).columns if c not in ['load']]
    
    # Check if any columns remain
    if not ref_weather_cols:
        print("Warning: No numeric weather columns found. Skipping weather generation details.")
        return pd.DataFrame(index=dates)

    for col in ref_weather_cols:
        # Create a mapping dict: (month, day, hour) -> value
        # Handling leap year logic: if ref is not leap, Feb 29?
        # 2018 is not a leap year. 
        # Strategy: Map DayOfYear to 2018 DayOfYear. 
        # For Leap Years in synthetic (e.g. 2004, 2008, 2012, 2016), Feb 29 will map to Feb 28 or Mar 1 of 2018.
        
        mapping = ref_2018[col].groupby([ref_2018.index.month, ref_2018.index.day, ref_2018.index.hour]).first()
        
        def get_val(row):
            try:
                val = mapping.get((row.month, row.day, row.hour))
                if val is None:
                    # Fallback for leap seconds or Feb 29 if not in mapping
                     val = mapping.get((2, 28, row.hour)) 
                return val
            except:
                return 0
        
        # Optimize: Construct a pattern array indexable by doy*24 + hour
        # This is slow with apply. Let's match by 'dayofyear'
        
        # Simplified approach: Tile the 2018 data
        # But we need to handle noise.
        pass # Actual implementation below
        
    # Vectorized approach for weather generation
    # Extract reference values
    ref_values = ref_2018[ref_weather_cols].values
    
    # Determine number of repeats needed
    total_hours_needed = len(dates)
    hours_in_ref = len(ref_values)
    
    # This simple tiling won't align dates correctly.
    # Correct approach: Map each generated timestamp to a timestamp in 2018
    
    # Create a mapping series
    # Timestamp -> 2018-MM-DD HH:MM:SS
    # Taking care of Feb 29
    mapped_dates = dates.map(lambda x: x.replace(year=2018) if not (x.month == 2 and x.day == 29) else x.replace(year=2018, month=2, day=28))
    
    # Reindex ref_2018 to cover all needed lookup times
    # Note: mapped_dates includes all timestamps. We can look them up in ref_2018
    
    # Create lookup table
    ref_2018_lookup = ref_2018.copy()
    # Ensure unique index
    ref_2018_lookup = ref_2018_lookup[~ref_2018_lookup.index.duplicated(keep='first')]
    
    # Perform lookup
    vals = ref_2018_lookup.loc[mapped_dates, ref_weather_cols]
    
    # Add noise
    # Temp noise: +/- 2-3 degrees. 
    # Other cols: +/- 5% relative
    
    noisy_vals = vals.copy()
    
    for col in noisy_vals.columns:
        if 'temperature' in col.lower():
            noise = np.random.normal(0, 1.5, size=len(noisy_vals)) # std dev 1.5 gives mostly +/- 3 range
            noisy_vals[col] += noise
        elif 'humidity' in col.lower() or 'cloud' in col.lower():
            noise = np.random.normal(0, 2, size=len(noisy_vals))
            noisy_vals[col] = (noisy_vals[col] + noise).clip(0, 100)
        else:
            noise_pct = np.random.normal(0, 0.05, size=len(noisy_vals))
            noisy_vals[col] = noisy_vals[col] * (1 + noise_pct)
            
    synthetic_weather = pd.DataFrame(noisy_vals.values, index=dates, columns=ref_weather_cols)
    return synthetic_weather

def apply_historical_events(df, load_col):
    """Injects specific historical events like blackouts and festive spikes."""
    print("Injecting historical events (Blackouts, Diwali)...")
    
    # 1. 2012 Blackout (July 30-31)
    # The Grid collapses were severe. 
    # July 30: 02:30 AM to ~4 PM in parts.
    # July 31: 01:00 PM collapse.
    
    # Event 1: July 30, 2012
    mask_2012_1 = (df.index.year == 2012) & (df.index.month == 7) & (df.index.day == 30) & (df.index.hour >= 2) & (df.index.hour <= 16)
    df.loc[mask_2012_1, load_col] *= 0.6  # 40% drop
    
    # Event 2: July 31, 2012
    mask_2012_2 = (df.index.year == 2012) & (df.index.month == 7) & (df.index.day == 31) & (df.index.hour >= 13) & (df.index.hour <= 20)
    df.loc[mask_2012_2, load_col] *= 0.5  # 50% drop
    
    # 2. Diwali Spikes
    for year, date_str in DIWALI_DATES.items():
        if year not in df.index.year: continue
        
        diwali_date = pd.to_datetime(date_str)
        # Evening peak (6 PM - 10 PM)
        mask_diwali = (df.index.year == year) & (df.index.month == diwali_date.month) & (df.index.day == diwali_date.day) & (df.index.hour >= 18) & (df.index.hour <= 22)
        
        # Historical spikes were often due to lighting, but industrial load drops. 
        # Net effect: Usually a slight drop in total (industry off) but residential spike.
        # However, user requested "Festive season spikes". We will add 5-10%.
        df.loc[mask_diwali, load_col] *= 1.10
        
    return df

def generate_synthetic_load(start_date, end_date, patterns, ref_load_mean, load_col_name, weather_df):
    """Generates synthetic load using backward projection."""
    print(f"Generating synthetic load from {start_date} to {end_date}...")
    
    daily_factors_series, hourly_factors_grouped, monthly_means, yearly_mean_2018 = patterns
    
    dates = pd.date_range(start=start_date, end=end_date, freq='h')
    synth_df = pd.DataFrame(index=dates)
    synth_df[load_col_name] = np.nan
    
    # Pre-compute growth factors per year
    # Base is 2018 start.
    # 2017 = 2018 / (1+g)
    # ...
    
    years = sorted(list(set(dates.year)), reverse=True)
    
    # Initialize baseline load with 2018 mean
    current_annual_mean = yearly_mean_2018
    
    annual_means = {}
    
    # We step back from 2018 
    # Reference: 2018. 
    # Calculation for 2017:
    for year in range(2017, 2000, -1):
        if year >= 2011:
            growth = GROWTH_RATE_2011_2017
        else:
            growth = GROWTH_RATE_2001_2010
            
        # Going backwards: Load_Prev = Load_Curr / (1 + growth)
        current_annual_mean = current_annual_mean / (1 + growth)
        annual_means[year] = current_annual_mean
        
    print(f"Projected Annual Means (MW): 2001={annual_means[2001]:.0f}, 2017={annual_means[2017]:.0f}")

    # Generate Loop
    # Vectorization strategy:
    # 1. Base Load = Annual Mean for that year
    # 2. Apply Monthly/Daily seasonality
    # 3. Apply Hourly profile
    
    # Assign Annual Mean
    synth_df['year'] = synth_df.index.year
    synth_df['annual_base'] = synth_df['year'].map(annual_means)
    
    # Assign Daily Factor (Seasonality)
    # Map DayOfYear to 2018 DayOfYear factors
    # Handling Leap Years for DayOfYear mapping
    # 2018 is not leap. Dates > 59 (Feb 28) in leap year need adjustment?
    # Simple map: dayofyear. Clip to 365.
    
    synth_df['dayofyear'] = np.clip(synth_df.index.dayofyear, 1, 365)
    
    # Convert daily_factors_series to dataframe/dict for mapping
    # Note: daily_factors_series index is DateTime index of 2018. We need DayOfYear index.
    doy_factors = daily_factors_series.groupby(daily_factors_series.index.dayofyear).mean()
    # We are using the 'load' column from the series
    doy_factors = doy_factors[load_col_name]
    
    synth_df['daily_factor'] = synth_df['dayofyear'].map(doy_factors)
    
    # Assign Hourly Factor
    synth_df['month'] = synth_df.index.month
    synth_df['hour'] = synth_df.index.hour
    synth_df['dayofweek'] = synth_df.index.dayofweek
    synth_df['is_weekend'] = synth_df['dayofweek'] >= 5
    
    # We need to look up hourly factors from 'hourly_factors_grouped'
    # Shape of hourly_factors_grouped: Index=['month', 'is_weekend', 'hour'], Columns=['load', ...]
    
    # Reset index to make merging easier
    hourly_factors_reset = hourly_factors_grouped[[load_col_name]].reset_index()
    hourly_factors_reset.rename(columns={load_col_name: 'hourly_factor_val'}, inplace=True)
    
    # But wait, 'hourly_factors_grouped' values are raw MW means, not normalized factors yet!
    # In 'extract_patterns', we computed:
    # hourly_factors = ref_df.groupby(['month', 'is_weekend', 'hour']).mean()
    # We need to normalize these by the Monthly Mean to get a shape factor
    
    monthly_means_reset = monthly_means[[load_col_name]].reset_index()
    monthly_means_reset.rename(columns={load_col_name: 'monthly_mean_val'}, inplace=True)
    
    merged_factors = pd.merge(hourly_factors_reset, monthly_means_reset, on=['month', 'is_weekend'])
    merged_factors['hourly_shape_factor'] = merged_factors['hourly_factor_val'] / merged_factors['monthly_mean_val']
    
    # Now merge this factor back to synth_df
    synth_df = pd.merge(synth_df.reset_index(), 
                        merged_factors[['month', 'is_weekend', 'hour', 'hourly_shape_factor']], 
                        on=['month', 'is_weekend', 'hour'], 
                        how='left').set_index('index')
    
    
    # Compute Raw Load
    # Formula: AnnualBase * DailySeasonalityFactor * HourlyShapeFactor
    # Note: DailySeasonalityFactor captures "Winter vs Summer".
    # HourlyShapeFactor captures "Day vs Night" (relative to that month's average).
    # There is a slight double counting of seasonality if not careful, because HourlyShapeFactor is normalized by MonthlyMean.
    # daily_factor is normalized by YearlyMean.
    # AnnualBase is the ProjectedYearlyMean.
    
    # Refined Formula:
    # We treat AnnualBase as the "Level".
    # We essentially want: Load ~ Level * (SeasonalVar) * (DailyVar)
    
    # Construct:
    # 1. Adjusted Annual Base for Day of Year: AnnualBase * daily_factor 
    # (This gives the Daily Average Load for that specific day)
    
    # 2. Hourly Shape:
    # We have 'hourly_shape_factor' which is DayTime/MonthlyMean.
    # We should multiply the Daily Average Load by 'hourly_shape_factor'.
    # Assumption: The intra-day shape normalized by monthly mean is a good proxy for intra-day shape normalized by daily mean.
    # Valid enough.
    
    synth_df['synthetic_load'] = synth_df['annual_base'] * synth_df['daily_factor'] * synth_df['hourly_shape_factor']
    
    # Add Random Noise (2-3%)
    noise = np.random.normal(0, 0.025, size=len(synth_df))
    synth_df[load_col_name] = synth_df['synthetic_load'] * (1 + noise)
    
    # Merge with weather
    # weather_df has same index
    synth_df = pd.concat([synth_df[[load_col_name]], weather_df], axis=1)
    
    return synth_df

def create_visualizations(combined_df, load_col):
    """Generates requested plots."""
    print("Generating visualizations...")
    
    # 1. Time Series Plot
    plt.figure(figsize=(15, 7))
    
    # Resample to daily for cleaner plot
    daily_df = combined_df.resample('D')[load_col].mean()
    
    # Split
    synthetic = daily_df['2001':'2017']
    actual = daily_df['2018':]
    
    plt.plot(synthetic.index, synthetic.values, linestyle='--', label='Synthetic (2001-2017)', alpha=0.8)
    plt.plot(actual.index, actual.values, linestyle='-', label='Actual (2018-2025)', alpha=0.8)
    
    plt.axvline(pd.Timestamp('2018-01-01'), color='r', linestyle=':', label='Transition (2018)')
    
    plt.title('India Electricity Load: Synthetic (2001-2017) + Actual (2018-2025)')
    plt.ylabel('Average Daily Load (MW)')
    plt.xlabel('Year')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(PLOT_TIMESERIES_PATH)
    plt.close()
    
    # 2. Heatmap
    # Pivot table: Index=Year, Columns=Month
    monthly_df = combined_df.resample('ME')[load_col].mean()
    heatmap_data = pd.DataFrame({
        'Year': monthly_df.index.year,
        'Month': monthly_df.index.month,
        'Load': monthly_df.values
    })
    heatmap_pivot = heatmap_data.pivot(index='Year', columns='Month', values='Load')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_pivot, cmap='viridis', annot=False)
    plt.title('Monthly Average Load Heatmap (2001-2025)')
    plt.savefig(PLOT_HEATMAP_PATH)
    plt.close()
    
    print(f"Plots saved to {PLOT_TIMESERIES_PATH} and {PLOT_HEATMAP_PATH}")

def validate_data(synth_df, actual_df, load_col):
    """Validation checks."""
    print("Running validation checks...")
    
    # 1. Transition Smoothness
    avail_synth_end = synth_df[load_col].dropna().tail(24*7) # Last week
    avail_actual_start = actual_df[load_col].dropna().head(24*7) # First week
    
    if len(avail_synth_end) > 0 and len(avail_actual_start) > 0:
        mean_synth = avail_synth_end.mean()
        mean_actual = avail_actual_start.mean()
        diff_pct = abs(mean_synth - mean_actual) / mean_actual * 100
        print(f"Transition Check (2017-2018):")
        print(f"  Last Week 2017 Mean: {mean_synth:.2f} MW")
        print(f"  First Week 2018 Mean: {mean_actual:.2f} MW")
        print(f"  Difference: {diff_pct:.2f}% (Target: < 10%)")
    
    # 2. Peak Check
    peak_2001 = synth_df.loc['2001'][load_col].max()
    print(f"2001 Peak Load: {peak_2001:.2f} MW (Target: ~78,000 MW)")
    
    # 3. Stats
    print("\nSummary Statistics by Year:")
    combined = pd.concat([synth_df, actual_df])
    stats = combined[load_col].resample('YE').agg(['min', 'max', 'mean'])
    print(stats)
    
    return stats

def main():
    # 1. Load Data
    existing_df, load_col = load_and_preprocess(EXISTING_DATA_PATH)
    
    # 2. Extract Patterns
    daily_factors, hourly_factors, monthly_means, yearly_mean = extract_patterns(existing_df, year=2018)
    
    # 3. Generate Synthetic Weather
    start_date = '2001-01-01 00:00:00'
    end_date = '2017-12-31 23:00:00'
    weather_df = generate_weather(start_date, end_date, existing_df)
    
    # 4. Generate Synthetic Load
    synth_df = generate_synthetic_load(
        start_date, end_date, 
        (daily_factors, hourly_factors, monthly_means, yearly_mean), 
        yearly_mean, load_col, weather_df
    )
    
    # 5. Apply Events
    synth_df = apply_historical_events(synth_df, load_col)
    
    # 6. Save Synthetic
    print(f"Saving synthetic data to {OUTPUT_SYNTHETIC_PATH}...")
    synth_df.to_csv(OUTPUT_SYNTHETIC_PATH)
    
    # 7. Merge and Save Complete
    # Align columns
    # Ensure existing_df has same columns as synth_df (or vice versa)
    # synth_df has index name 'index' or 'timestamp'
    synth_df.index.name = existing_df.index.name
    
    # Select common columns
    common_cols = [c for c in synth_df.columns if c in existing_df.columns]
    combined_df = pd.concat([synth_df[common_cols], existing_df[common_cols]]).sort_index()
    
    print(f"Saving complete dataset to {OUTPUT_COMPLETE_PATH}...")
    combined_df.to_csv(OUTPUT_COMPLETE_PATH)
    
    # 8. Visualize & Validate
    create_visualizations(combined_df, load_col)
    validate_data(synth_df, existing_df, load_col)
    
    print("Done!")

if __name__ == "__main__":
    main()
