import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# LOAD DATASETS
# ======================================================

da = pd.read_csv("dailyActivity_merged.csv")
wt = pd.read_csv("weightLogInfo_merged.csv")
hs = pd.read_csv("hourlySteps_merged.csv")
hc = pd.read_csv("hourlyCalories_merged.csv")

# ======================================================
# CLEAN DATA
# ======================================================

da = da[(da["TotalSteps"] > 0) & (da["Calories"] > 0)]

# ======================================================
# BASIC STATS
# ======================================================

print("\n=== DATASET OVERVIEW ===")
print("Rows:", len(da))
print("Avg Steps:", round(da["TotalSteps"].mean()))
print("Avg Calories:", round(da["Calories"].mean()))
print("Avg Sedentary:", round(da["SedentaryMinutes"].mean()))

# ======================================================
# BMI ANALYSIS
# ======================================================

wt["BMI"] = pd.to_numeric(wt["BMI"], errors="coerce")
print("\nAvg BMI:", round(wt["BMI"].mean(), 2))

# ======================================================
# LINEAR REGRESSION MODEL
# ======================================================

X = np.column_stack([
    np.ones(len(da)),
    da["TotalSteps"],
    da["VeryActiveMinutes"],
    da["SedentaryMinutes"]
])

y = da["Calories"].values

beta = np.linalg.inv(X.T @ X) @ (X.T @ y)

B0, B1, B2, B3 = beta

print("\n=== MODEL COEFFICIENTS ===")
print("Intercept:", round(B0, 2))
print("Steps coef:", round(B1, 5))
print("Active coef:", round(B2, 5))
print("Sedentary coef:", round(B3, 5))

# ======================================================
# PREDICTION FUNCTION
# ======================================================

def predict(s, a, d):
    return B0 + B1*s + B2*a + B3*d

print("\nExample Prediction (9000 steps):")
print(round(predict(9000, 30, 800), 2))

# ======================================================
# MODEL EVALUATION
# ======================================================

y_pred = X @ beta

r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
rmse = np.sqrt(np.mean((y - y_pred)**2))

print("\nR² Score:", round(r2, 4))
print("RMSE:", round(rmse, 2))

# ======================================================
# 📊 GRAPH 1 — Steps vs Calories
# ======================================================

plt.figure(figsize=(6,4))
plt.scatter(da["TotalSteps"], da["Calories"], alpha=0.4, color="blue")
plt.title("Steps vs Calories")
plt.xlabel("Total Steps")
plt.ylabel("Calories")
plt.grid()
plt.show()

# ======================================================
# 📊 GRAPH 2 — Active Minutes vs Calories
# ======================================================

plt.figure(figsize=(6,4))
plt.scatter(da["VeryActiveMinutes"], da["Calories"], alpha=0.5, color="green")
plt.title("Very Active Minutes vs Calories")
plt.xlabel("Active Minutes")
plt.ylabel("Calories")
plt.grid()
plt.show()

# ======================================================
# 📊 GRAPH 3 — Hourly Steps Pattern
# ======================================================

hs["Hour"] = pd.to_datetime(hs["ActivityHour"]).dt.hour
hourly_steps = hs.groupby("Hour")["StepTotal"].mean()

plt.figure(figsize=(7,4))
plt.bar(hourly_steps.index, hourly_steps.values, color="skyblue")
plt.title("Hourly Steps Pattern")
plt.xlabel("Hour of Day")
plt.ylabel("Avg Steps")
plt.grid()
plt.show()

# ======================================================
# 📊 GRAPH 4 — Hourly Calories Pattern
# ======================================================

hc["Hour"] = pd.to_datetime(hc["ActivityHour"]).dt.hour
hourly_cal = hc.groupby("Hour")["Calories"].mean()

plt.figure(figsize=(7,4))
plt.plot(hourly_cal.index, hourly_cal.values, marker="o", color="orange")
plt.title("Hourly Calories Burned")
plt.xlabel("Hour of Day")
plt.ylabel("Calories")
plt.grid()
plt.show()

# ======================================================
# 📊 GRAPH 5 — Step Buckets vs Calories
# ======================================================

bins = [0, 5000, 8000, 12000, 50000]
labels = ["<5k", "5k-8k", "8k-12k", "12k+"]

da["StepGroup"] = pd.cut(da["TotalSteps"], bins=bins, labels=labels)

bucket_avg = da.groupby("StepGroup")["Calories"].mean()

plt.figure(figsize=(6,4))
plt.bar(bucket_avg.index.astype(str), bucket_avg.values, color="purple")
plt.title("Step Buckets vs Calories")
plt.xlabel("Step Range")
plt.ylabel("Avg Calories")
plt.grid()
plt.show()

# ======================================================
print("\nALL GRAPHS GENERATED SUCCESSFULLY ✅")
