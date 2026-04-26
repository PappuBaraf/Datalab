# DAV_Practical/q7_time_series.py
# Question 7: Demonstration of all components of Time Series and Stationary test and Covert non stationary data into stationary in Python.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# 1. Generate Non-Stationary Time Series Data
np.random.seed(42)
# FIX: Changed freq='M' to freq='ME' to support newer versions of pandas
dates = pd.date_range('2022-01-01', periods=100, freq='ME')
data = np.linspace(10, 50, 100) + np.sin(np.linspace(0, 20, 100)) * 10 + np.random.normal(0, 2, 100)
ts = pd.Series(data, index=dates)

# 2. Components Demonstration
decomposition = seasonal_decompose(ts, model='additive', period=12)
decomposition.plot()
plt.suptitle('Time Series Components', y=1.02)
plt.show()

# 3. Stationarity Test (ADF Test)
def test_stationarity(timeseries, title):
    result = adfuller(timeseries.dropna())
    print(f"--- ADF Test: {title} ---")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print("Is Stationary?:", "Yes" if result[1] <= 0.05 else "No", "\n")

test_stationarity(ts, "Original Data")

# 4. Convert to Stationary (Differencing)
ts_diff = ts.diff().dropna()
test_stationarity(ts_diff, "First Differenced Data")

# Plot Transformed Data
plt.plot(ts_diff, label='Differenced (Stationary)')
plt.title('Stationary Time Series Data')
plt.legend()
plt.show()
