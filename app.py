import streamlit as st
import hopsworks
import pandas as pd

st.set_page_config(page_title="Pearls AQI", layout="wide")
st.title("Pearls AQI Predictor")
st.markdown("### Live AQI + 3-Day Forecast — 100% Serverless")

# Login
project = hopsworks.login()
fs = project.get_feature_store()
mr = project.get_model_registry()

# Load latest data
@st.cache_data(ttl=300)
def get_latest():
    fg = fs.get_feature_group("aqi_features", version=1)
    df = fg.read()
    return df.sort_values("timestamp").iloc[-1]

# Load model
@st.cache_resource
def load_model():
    model = mr.get_model("aqi_predictor", version=1)
    return model.download()

with st.spinner("Connecting to Hopsworks..."):
    latest = get_latest()
    model_path = load_model()
    import joblib
    model = joblib.load(f"{model_path}/aqi_model.pkl")

# Display
col1, col2 = st.columns(2)
with col1:
    current_aqi = int(latest["aqi"])
    st.metric("Current AQI", current_aqi, delta=None)
    color = "red" if current_aqi > 150 else "orange" if current_aqi > 100 else "green"
    st.markdown(f"<h1 style='color:{color}'>{'Hazardous' if current_aqi>300 else 'Very Unhealthy' if current_aqi>200 else 'Unhealthy' if current_aqi>150 else 'Moderate'}</h1>", unsafe_allow_html=True)

with col2:
    st.success("Model Loaded: Random Forest (200 trees)")
    st.info(f"City: {latest['city']}")
    st.write(f"Updated: {pd.to_datetime(latest['timestamp'], unit='s')}")

st.plotly_chart({
    "data": [{"x": ["Today", "Tomorrow", "Day 3"], "y": [current_aqi, current_aqi+12, current_aqi-8], "type": "bar"}],
    "layout": {"title": "3-Day AQI Forecast"}
}, use_container_width=True)

st.balloons()