# feature_pipeline.py
import requests
import pandas as pd
from datetime import datetime
import hopsworks
import os

# PDSYoYJGLZLbLEkZ.kF4XKAmi3fynDB3cCQBqHggdxJJv5SbwjmhGPqCdODIo2pOMmdZvru26hQEeyBdx

CITY = "Karachi"
LAT, LON = 24.8607,  67.0011

def fetch_air_quality():
    """Only pollutants + us_aqi"""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "us_aqi,pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "forecast_days": 3,
        "timezone": "Asia/Karachi"
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        print("AirQuality API Error:", r.text)
        return pd.DataFrame()
    data = r.json()
    if "hourly" not in data:
        print("No hourly in AirQuality:", data)
        return pd.DataFrame()
    df = pd.DataFrame(data["hourly"])
    df["timestamp"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"], errors="ignore")
    return df

def fetch_weather():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "forecast_days": 3,
        "timezone": "Asia/Kolkata"
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        print("Weather API Error:", r.text)
        return pd.DataFrame()
    data = r.json()
    if "hourly" not in data:
        print("No hourly in Weather:", data)
        return pd.DataFrame()
    df = pd.DataFrame(data["hourly"])
    df["timestamp"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"], errors="ignore")
    return df

def engineer_features():
    print(f"[{datetime.now()}] Fetching data...")
    aq_df = fetch_air_quality()
    wx_df = fetch_weather()

    if aq_df.empty or wx_df.empty:
        print("One of the APIs failed. Skipping.")
        return pd.DataFrame()

    # Merge on timestamp
    df = pd.merge(aq_df, wx_df, on="timestamp", how="inner")
    if df.empty:
        print("No overlapping timestamps.")
        return pd.DataFrame()

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
    df = df.dropna(subset=["aqi_change_rate"])

    cols = [
        "city", "timestamp", "timestamp_unix","aqi", "aqi_lag1", "aqi_change_rate",
        "hour", "day", "month", "dayofweek",
        "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation",
        "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"
    ]
    return df[cols]

def main():
    df = engineer_features()
    if df.empty:
        print("No data to insert.")
        return

    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["city", "timestamp_unix"],
        event_time="timestamp",
        online_enabled=True
    )
    fg.insert(df, write_options={"wait_for_job": True})
    print(f"Inserted {len(df)} rows")

if __name__ == "__main__":
    main()