# app.py
import streamlit as st
import hopsworks
import pandas as pd
import plotly.graph_objects as go
import joblib
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("🟣 Pearls AQI Predictor - Karachi")
st.markdown("**72-Hour Forecast | Live Data | 100% Serverless**")

@st.cache_resource
def load_all():
    project = hopsworks.login(api_key_value=st.secrets["HOPSWORKS_API_KEY"])
    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_features", 1)
    mr = project.get_model_registry()
    model = mr.get_model("aqi_predictor", 1)
    model_dir = model.download()
    clf = joblib.load(f"{model_dir}/aqi_model.pkl")
    return fg, clf

fg, model = load_all()

# GET LIVE DATA
@st.cache_data(ttl=300)
def get_live():
    try:
        df = fg.read(online=True)
        if df.empty:
            df = fg.read().tail(1)
        return df.iloc[-1]
    except:
        return None

row = get_live()
if row is None:
    st.error("No data. Run: python feature_pipeline.py")
    st.stop()

current_aqi = int(row["aqi"])

# FORECAST
@st.cache_data(ttl=3600)
def forecast():
    times = [datetime.now() + timedelta(hours=i) for i in range(1, 73)]
    preds = []
    last = current_aqi
    feature_names = [c for c in row.index if c not in ["aqi","timestamp","city","timestamp_unix","us_aqi"]]
    
    for t in times:
        X = pd.DataFrame([{k: row[k] for k in feature_names}])
        X["hour"] = t.hour
        X["day"] = t.day
        X["month"] = t.month
        X["dayofweek"] = t.weekday()
        X["aqi_lag1"] = last
        X["aqi_change_rate"] = 0
        pred = model.predict(X)[0]
        preds.append(round(pred))
        last = pred
    return pd.DataFrame({"time": times, "aqi": preds})

forecast_df = forecast()

# UI
col1, col2 = st.columns([1, 2])

with col1:
    st.metric("Live AQI", current_aqi)
    colors = ["#00e400","#ffff00","#ff7e00","#ff0000","#8f3f97","#7e0023"]
    status = ["Good","Moderate","Unhealthy(S)","Unhealthy","Very Unhealthy","Hazardous"]
    idx = min(current_aqi//51, 5)
    st.markdown(f"**{status[idx]}**", unsafe_allow_html=True)
    st.markdown(f"<span style='color:{colors[idx]}'>●</span>", unsafe_allow_html=True)
    if current_aqi > 150: st.error("🚨 HAZARDOUS - Stay indoors!")
    elif current_aqi > 100: st.warning("⚠️ Limit outdoor time")

with col2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df.time, y=forecast_df.aqi, 
                            mode='lines+markers', line_color='purple', name='Forecast'))
    fig.add_vline(x=datetime.now(), line_dash="dash", line_color="gray")
    fig.update_layout(title="Next 72 Hours", height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.success("**Pipeline**: Open-Meteo → Hopsworks → GitHub Actions → Streamlit")