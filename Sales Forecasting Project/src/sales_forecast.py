import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

print("🚀 Sales Forecasting Script Started")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "sales.csv")

print("Looking for data at:", DATA_PATH)

# Check file exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ sales.csv not found at {DATA_PATH}")

# Load data
df = pd.read_csv(DATA_PATH)
print("✅ Data loaded. Shape:", df.shape)
print(df.head())

# Prepare time series
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

# Plot actual sales
plt.figure()
plt.plot(df.index, df["sales"])
plt.title("Monthly Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()   # 👈 IMPORTANT

# Train ARIMA model
model = ARIMA(df["sales"], order=(1, 1, 1))
model_fit = model.fit()

# Forecast next 6 periods
forecast = model_fit.forecast(steps=6)
print("\n📈 Forecast for next 6 periods:")
print(forecast)

# Plot forecast
plt.figure()
plt.plot(df.index, df["sales"], label="Actual")
future_dates = pd.date_range(start=df.index[-1], periods=7, freq="MS")[1:]
plt.plot(future_dates, forecast, linestyle="dashed", label="Forecast")
plt.legend()
plt.title("Sales Forecast")
plt.show()   # 👈 IMPORTANT

print("✅ Script finished successfully")
