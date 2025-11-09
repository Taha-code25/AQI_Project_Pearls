# 🌍 Pearls AQI Predictor

> **An automated, machine learning–powered system for forecasting Air Quality Index (AQI) in Karachi, Pakistan.**

---

## 🧠 Executive Summary

**Pearls AQI Predictor** is a fully automated, serverless **machine learning system** designed to provide a **3-day Air Quality Index (AQI)** forecast for Karachi, Pakistan.  
It demonstrates modern **MLOps best practices**, integrating tools for **data ingestion, feature engineering, model retraining, CI/CD orchestration, and explainability**.

The project culminates in a **Streamlit web application** that visualizes real-time forecasts, interprets model predictions using **SHAP**, and issues **alerts for hazardous air quality (AQI > 150)**.

---

## 🚀 Project Overview

### 🎯 Objectives & Features

- **Forecasting**: Predicts AQI levels for the next **3 days**.  
- **Automation**: 100% automated CI/CD pipeline — no manual intervention.  
- **Real-Time Data**: Pulls live hourly data from **Open-Meteo APIs**.  
- **Feature Engineering**: Generates time- and lag-based features automatically.  
- **Daily Retraining**: The model retrains every 24 hours to capture new patterns.  
- **Interpretability**: Uses **SHAP** for model transparency.  
- **User Interface**: Provides a **Streamlit dashboard** for forecasts and alerts.

---

## 🧩 System Architecture

The system is composed of **two main pipelines**, both automated using **GitHub Actions**.

### ⚙️ Hourly Feature Pipeline
- **Script:** `feature_pipeline.py`  
- **Trigger:** Runs hourly (`0 * * * *`) via `features.yml`.  
- **Source:** Fetches real-time weather and pollutant data from **Open-Meteo APIs**.  
- **Features Generated:**
  - PM2.5, PM10
  - Temperature, Humidity
  - Hour, Day, Month, DayOfWeek
  - Lag-based AQI change rate  
- **Storage:** Processed data saved in **Hopsworks Feature Store**.

### 🧠 Daily Training Pipeline
- **Script:** `training_pipeline.py`  
- **Trigger:** Runs daily at midnight (`train.yml`).  
- **Model:** `RandomForestRegressor` trained on **90 days of historical data**.  
- **Evaluation:** Uses **Mean Absolute Error (MAE)** and **feature importance**.  
- **Artifacts:** Model and SHAP plots saved in the **Hopsworks Model Registry**.

---

## 🏗️ MLOps Infrastructure

| Component | Technology | Description |
|------------|-------------|--------------|
| **CI/CD Automation** | GitHub Actions | Orchestrates hourly and daily workflows. |
| **Feature Store** | Hopsworks | Centralized, versioned feature storage. |
| **Model Registry** | Hopsworks | Manages models and associated artifacts. |
| **Explainable AI** | SHAP | Provides interpretability for predictions. |
| **Monitoring** | GitHub Actions Logs | Tracks job status and pipeline execution. |

---

## 💻 Web Application

Built with **Streamlit**, the dashboard provides an intuitive interface for users.

### Key Features:
- **Forecast Display:** 3-day AQI predictions for Karachi.  
- **Model Interpretability:** SHAP feature importance visualization.  
- **Alert System:** Triggers warning if AQI > 150 (“Hazardous”).  
- **Data Source:** Fetches latest model and features directly from **Hopsworks**.

---

## 🏆 Project Achievements

✅ **End-to-End Automation:** Fully hands-off ML lifecycle — from data ingestion to deployment.  
✅ **Production-Ready:** Demonstrates scalable and reliable ML pipeline design.  
✅ **MLOps Excellence:** Implements Feature Store, Model Registry, Explainable AI, and CI/CD.  

---

## 🧰 Tech Stack

- **Python 3.10+**
- **Hopsworks** (Feature Store & Model Registry)
- **GitHub Actions** (CI/CD automation)
- **Streamlit** (Web app interface)
- **scikit-learn** (Model training)
- **SHAP** (Model interpretability)
- **Open-Meteo API** (Data ingestion)

---

## 🧾 License

This project is released under the **MIT License**.

---

## 👤 Author

**Taha Faisal**  
🎓 NED University of Engineering & Technology  
📧[GitHub](https://github.com)

---

> “A cleaner tomorrow starts with the data we analyze today.” 🌱

