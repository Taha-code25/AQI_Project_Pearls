import os
import joblib
import hopsworks
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()
fg = fs.get_feature_group(name="aqi_features", version=1)

df = fg.read()
print(f"Loaded {len(df):,} rows")

TARGET = "aqi"
feature_cols = [
    col for col in df.columns
    if col not in {TARGET, "timestamp", "city", "timestamp_unix", "us_aqi"}
]

X = df[feature_cols]
y = df[TARGET]


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

models = {
    "RandomForest": RandomForestRegressor(
        n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
    ),
    "Ridge": Ridge(alpha=1.0, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42
    ),
}

best_name = None
best_model = None
best_metrics = {"rmse": np.inf}

for name, model in models.items():
    print(f"\nTraining {name} …")
    model.fit(X_train, y_train)
    pred = model.predict(X_val)

    rmse = root_mean_squared_error(y_val, pred)
    mae  = mean_absolute_error(y_val, pred)
    r2   = r2_score(y_val, pred)

    print(f"  RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.3f}")

    if rmse < best_metrics["rmse"]:
        best_metrics = {"rmse": rmse, "mae": mae, "r2": r2}
        best_name = name
        best_model = model

print(f"\nBest model: **{best_name}** (RMSE = {best_metrics['rmse']:.2f})")

print("\nComputing SHAP values …")
sample = X_val.sample(200, random_state=42)

if "Tree" in best_name:
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(sample, check_additivity=False)
else:
    explainer = shap.Explainer(best_model, X_train)
    shap_values = explainer(sample, check_additivity=False).values

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, sample, show=False)
plt.tight_layout()
shap_path = "shap_summary.png"
plt.savefig(shap_path)
plt.close()
print(f"SHAP plot saved → {shap_path}")

model_dir = "model"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "aqi_model.pkl")
joblib.dump(best_model, model_path)
print(f"Model saved → {model_path}")

mr = project.get_model_registry()
model_obj = mr.python.create_model(
    name="aqi_predictor",
    metrics=best_metrics,
    description=f"Best model: {best_name}. Predicts US AQI.",
    input_example=X_train.iloc[:1].to_dict(orient="records"),
)

model_obj.save(model_path)  
print("Model registered in Hopsworks")

print("Upload shap_summary.png to model in Hopsworks UI for visualization")