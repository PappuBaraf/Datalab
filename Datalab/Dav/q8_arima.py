# DAV_Practical/q8_arima.py
# Question 8: Implement ARIMA and forecast time series values visualize ACF,PACF, Residual plot in Python. [cite: 2]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Generate Sample Time Series Data
np.random.seed(0)
dates = pd.date_range('2020-01-01', periods=100, freq='D')
ts = pd.Series(np.random.randn(100).cumsum(), index=dates)

# 1. Implement ARIMA (p=1, d=1, q=1)
model = ARIMA(ts, order=(1, 1, 1))
fitted_model = model.fit()

print("--- ARIMA Model Summary ---")
print(fitted_model.summary().tables[1])

# 2. Forecast Values
forecast = fitted_model.forecast(steps=10)

# 3. Visualize ACF, PACF, and Residuals
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

# ACF
plot_acf(ts.diff().dropna(), ax=axes[0], lags=20, title="Autocorrelation Function (ACF)")

# PACF
plot_pacf(ts.diff().dropna(), ax=axes[1], lags=20, title="Partial Autocorrelation Function (PACF)")

# Residuals
residuals = fitted_model.resid
axes[2].plot(residuals)
axes[2].set_title("Model Residuals")

plt.tight_layout()
plt.show()

# Visualize Forecast
plt.figure(figsize=(10,4))
plt.plot(ts, label='Original')
plt.plot(forecast, color='red', label='Forecast (10 steps)')
plt.title('ARIMA Forecasting')
plt.legend()
plt.show()
