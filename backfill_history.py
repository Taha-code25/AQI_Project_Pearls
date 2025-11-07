# backfill_historical.py
import requests
import pandas as pd
from datetime import datetime, timedelta
import hopsworks
import os
import time

CITY = "Karaci"
LAT, LON = 24.8607,  67.0011
BATCH_DAYS = 90

def fetch_air_quality_batch(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch 90-day batch of air-quality data"""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "us_aqi,pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "timezone": "Asia/Karachi"
    }
    r = requests.get(url, params=params, timeout=60)
    if r.status_code != 200:
        print(f"AQ Error ({start_date} to {end_date}):", r.text)
        return pd.DataFrame()
    data = r.json()
    if "hourly" not in data:
        return pd.DataFrame()
    df = pd.DataFrame(data["hourly"])
    df["timestamp"] = pd.to_datetime(df["time"])
    return df

def fetch_weather_full_year() -> pd.DataFrame:
    """Fetch full year of weather using archive API"""
    end = datetime.now().date()
    start = end - timedelta(days=365)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "timezone": "Asia/Kolkata"
    }
    r = requests.get(url, params=params, timeout=60)
    if r.status_code != 200:
        print("Weather Error:", r.text)
        return pd.DataFrame()
    data = r.json()
    if "hourly" not in data:
        return pd.DataFrame()
    df = pd.DataFrame(data["hourly"])
    df["timestamp"] = pd.to_datetime(df["time"])
    return df

def backfill():
    print("Backfilling 1 year of hourly data...")
    
    # 1. Weather: one call
    print("Fetching full-year weather...")
    wx_df = fetch_weather_full_year()
    if wx_df.empty:
        print("Weather failed. Aborting.")
        return

    # 2. Air-Quality: 90-day batches
    print("Fetching air-quality in 90-day batches...")
    all_aq = []
    end = datetime.now().date()
    batches = 0
    while end - timedelta(days=BATCH_DAYS) > end - timedelta(days=365):
        start = end - timedelta(days=BATCH_DAYS)
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")
        print(f"  → {start_str} to {end_str}")
        batch = fetch_air_quality_batch(start_str, end_str)
        if not batch.empty:
            all_aq.append(batch)
        end = start - timedelta(days=1)
        time.sleep(1)  # Be gentle
        batches += 1
        if batches >= 5:  # Safety break
            break

    if not all_aq:
        print("No air-quality data. Aborting.")
        return

    aq_df = pd.concat(all_aq, ignore_index=True)
    print(f"Fetched {len(aq_df)} air-quality rows")

    df = pd.merge(aq_df, wx_df, on="timestamp", how="inner")
    if df.empty:
        print("No overlap.")
        return

    # Drop any lingering time_x/time_y
    df = df.drop(columns=[col for col in df.columns if col.startswith("time") and col != "timestamp"], errors="ignore")

    # Engineer features
    df["city"] = CITY
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["timestamp_unix"] = df["timestamp"].astype('int64') // 10**9
    df["aqi"] = df["us_aqi"]
    df = df.sort_values("timestamp")
    df["aqi_lag1"] = df["aqi"].shift(1)
    df["aqi_change_rate"] = (df["aqi"] - df["aqi_lag1"]) / (df["aqi_lag1"] + 1e-6)
    df = df.dropna()

    # FINAL COLUMNS (exact match to schema)
    cols = [
        "city", "timestamp", "timestamp_unix", "aqi", "aqi_lag1", "aqi_change_rate",
        "hour", "day", "month", "dayofweek",
        "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation",
        "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"
    ]
    df = df[cols]

    # Upload
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["city", "timestamp_unix"],
        event_time="timestamp",
        online_enabled=True
    )
    fg.insert(df)
    print(f"Backfilled {len(df):,} rows")

if __name__ == "__main__":
    backfill()