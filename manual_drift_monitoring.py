import pandas as pd
import numpy as np
import requests
import zipfile
import io
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import ensemble
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import ks_2samp, ttest_ind
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Create static directory if not exists
os.makedirs("static", exist_ok=True)

# Download and read dataset
content = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip").content
with zipfile.ZipFile(io.BytesIO(content)) as arc:
    raw_data = pd.read_csv(arc.open("hour.csv"), 
                            header=0, 
                            sep=',', 
                            parse_dates=['dteday'], 
                            index_col='dteday')

# Define columns
target = 'cnt'
numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'hr', 'weekday']
categorical_features = ['season', 'holiday', 'workingday']

# Split reference and current data
reference = raw_data.loc['2011-01-01 00:00:00':'2011-01-28 23:00:00']
current = raw_data.loc['2011-01-29 00:00:00':'2011-02-28 23:00:00']

# Train regression model
regressor = ensemble.RandomForestRegressor(random_state=0, n_estimators=50)
regressor.fit(reference[numerical_features + categorical_features], reference[target])
reference['prediction'] = regressor.predict(reference[numerical_features + categorical_features])
current['prediction'] = regressor.predict(current[numerical_features + categorical_features])

# Plot and save regression performance
mse = mean_squared_error(reference[target], reference['prediction'])
r2 = r2_score(reference[target], reference['prediction'])

plt.figure()
plt.scatter(reference[target], reference['prediction'], alpha=0.5)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title(f"Regression Performance\nRMSE={np.sqrt(mse):.2f}, R2={r2:.2f}")
plt.savefig("static/regression_perf_ref.png")

# Perform target drift test
t_stat, p_val = ttest_ind(reference[target], current[target], equal_var=False)
plt.figure()
plt.hist(reference[target], bins=30, alpha=0.5, label="Reference")
plt.hist(current[target], bins=30, alpha=0.5, label="Current")
plt.legend()
plt.title(f"Target Drift: t-test\np={p_val:.4f}")
plt.savefig("static/target_drift_cnt.png")




# Feature drift (example: temp)
for col in numerical_features:
    stat, p = ks_2samp(reference[col], current[col])
    plt.figure()
    plt.hist(reference[col], bins=30, alpha=0.5, label="Reference")
    plt.hist(current[col], bins=30, alpha=0.5, label="Current")
    plt.title(f"Feature Drift: {col}\np = {p:.4f}")
    plt.legend()
    plt.savefig(f"static/feature_drift_{col}.png")
    
    
with open("static/index.html", "w") as f:
    f.write("""
    <html>
    <head><title>Data Drift Dashboard</title></head>
    <body>
        <h2>Regression Performance</h2>
        <img src="regression_perf_ref.png" width="600"><br>

        <h2>Target Drift</h2>
        <img src="target_drift_cnt.png" width="600"><br>

        <h2>Feature Drift</h2>
        <img src="feature_drift_temp.png" width="600"><br>
        <img src="feature_drift_atemp.png" width="600"><br>
        <img src="feature_drift_hum.png" width="600"><br>
        <img src="feature_drift_windspeed.png" width="600"><br>
        <img src="feature_drift_hr.png" width="600"><br>
        <img src="feature_drift_weekday.png" width="600"><br>
    </body>
    </html>
    """)



# Serve static files with FastAPI
app = FastAPI()
app.mount("/", StaticFiles(directory="static", html=True), name="static")
