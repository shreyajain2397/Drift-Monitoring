# 🚲 Real-Time Data Drift Monitoring for Bike Rental Forecasting

## 🧠 Project Overview

This project focuses on building an **end-to-end monitoring system** for a regression model predicting **bike rental demand** using historical data. It tracks **model performance**, **feature drift**, and **target drift** over time and serves visualizations using **FastAPI** — without relying on specialized libraries like `Evidently`.

---

## 🎯 Objectives

- Predict hourly bike rental count (`cnt`) using environmental and time features.
- Monitor **model performance** over time.
- Detect and visualize **data drift** (feature shift) and **target drift** (label shift).
- Serve dashboards using a lightweight FastAPI app.

---

## 📊 Dataset

**Source**: [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)  
- Records hourly bike rentals in Washington D.C. for 2011–2012.
- Includes features like temperature, humidity, windspeed, hour of day, etc.

---

## 🛠️ Tech Stack

- Python (Pandas, Scikit-learn, Matplotlib, Scipy)
- FastAPI (Static file serving)
- Statistical Drift Detection (Kolmogorov–Smirnov Test, t-test)

---

## 📈 Modeling Approach

- Target: `cnt` (number of rentals)
- Model: `RandomForestRegressor` from Scikit-learn
- Features:
  - **Numerical**: `temp`, `atemp`, `hum`, `windspeed`, `hr`, `weekday`
  - **Categorical**: `season`, `holiday`, `workingday`
- Split data into:
  - 📘 **Reference Window**: Jan 1–28, 2011 (training + baseline)
  - 📕 **Current Window**: Jan 29–Feb 28, 2011 (simulated production data)
 
- ![image](https://github.com/user-attachments/assets/70034ba7-c798-489b-a37f-f82278f753d7)

---

## 🧪 Drift Detection Logic

### 🔁 Feature Drift
- Used **Kolmogorov–Smirnov test** to compare reference vs current distributions.
- Drift is visualized using histograms for each numerical feature.

- ![image](https://github.com/user-attachments/assets/2ce8f2da-78ab-403c-b981-f3a4dd38adcb)

- ![image](https://github.com/user-attachments/assets/03594bec-d254-4422-b6a7-519f463ba18f)

- ![image](https://github.com/user-attachments/assets/37e849f5-74ee-40e8-a88d-856d8308e3f3)

- ![image](https://github.com/user-attachments/assets/5baaf867-feae-42e5-b30b-5dd99c0ad942)

- ![image](https://github.com/user-attachments/assets/520130dd-3f46-442a-b239-359eb0686ea7)






### 🎯 Target Drift
- Applied **t-test** between `cnt` in reference and current windows.
- Significant p-value indicates potential model decay.

  ![image](https://github.com/user-attachments/assets/a0657232-813f-4d2a-af3d-8de3d642f78b)


### 📊 Regression Performance
- RMSE and R² score computed for predictions vs actuals in the reference set.

---

## 📸 Sample Visualizations

All plots are saved in the `static/` directory and viewable via a browser.

| Metric                   | Visualization                 |
|--------------------------|-------------------------------|
| Model Performance        | `regression_perf_ref.png`     |
| Target Drift (cnt)       | `target_drift_cnt.png`        |
| Feature Drift (temp)     | `feature_drift_temp.png`      |
| Feature Drift (humidity) | `feature_drift_hum.png`       |
| ...                      | One plot per numerical feature|






---

## 🌐 FastAPI Dashboard

The static dashboards are served using FastAPI:

```bash
uvicorn manual_drift_dashboard:app --reload


## Key Learnings

Built a lightweight, extensible model monitoring framework without external drift libraries.

Demonstrated how data shifts can impact model performance over time.

Used classical statistics (KS-test, t-test) to assess data quality and stability.

Served results through a live dashboard for observability.
