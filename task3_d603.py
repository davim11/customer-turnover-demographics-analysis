import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load dataset and fix dates
df = pd.read_csv('teleco_time_series.csv')
df['Day'] = pd.date_range(start='2021-01-01', periods=731, freq='D')  # Set realistic dates
df.set_index('Day', inplace=True)
print("First few rows of the dataset:")
print(df.head())

# D1: Line graph visualization
# Adjust the dates to start from 2021-01-01
df['Day'] = pd.date_range(start='2021-01-01', periods=731, freq='D')  # Set realistic dates
df.set_index('Day', inplace=True)

plt.figure(figsize=(10, 6))  # Set plot size
plt.plot(df.index, df['Revenue'], label='Daily Revenue', color='blue')  # Plot revenue over time
plt.title('Daily Revenue Over Time (First Two Years)')  # Title
plt.xlabel('Date')  # X-axis label
plt.ylabel('Revenue (Million $)')  # Y-axis label
plt.legend()  # Show legend
plt.grid(True)  # Add grid for readability
plt.show()  # Display the plot

# D2: Time step formatting
time_diffs = df.index.to_series().diff().dropna()  # Calculate differences between days
print("Time step differences (should be 1 day if no gaps):")
print(time_diffs.value_counts())  # Check for consistent 1-day steps
print(f"Total length of the series: {len(df)} days")  # Confirm 731 days
missing_values = df['Revenue'].isnull().sum()  # Count missing values
print(f"Number of missing values: {missing_values}")

# D3: Evaluate stationarity
result = adfuller(df['Revenue'])  # Run ADF test
print('ADF Statistic:', result[0])  # Test statistic
print('p-value:', result[1])  # p-value to check stationarity
print('Critical Values:', result[4])  # Critical values for comparison
if result[1] < 0.05:
    print("The series is stationary (p-value < 0.05).")
else:
    print("The series is non-stationary (p-value >= 0.05).")
    
# D4: Prepare the data and Training Test Split
if missing_values > 0:  # If there are missing values
    df['Revenue'] = df['Revenue'].interpolate()  # Fill them with interpolation
    print("Missing values interpolated.")
    
train_size = 585  # Explicitly set to 585
train, test = df.iloc[:train_size], df.iloc[train_size:]
print(f"Training set size: {len(train)} days")
print(f"Test set size: {len(test)} days")

train.to_csv('task3_train_teleco_time_series.csv')
test.to_csv('task3_test_teleco_time_series.csv')
print("Training set saved as 'task3_train_teleco_time_series.csv'.")
print("Test set saved as 'task3_test_teleco_time_series.csv'.")

# D5: Save the cleaned dataset
df.to_csv('task3_cleaned_teleco_time_series.csv')
print("Cleaned dataset saved as 'task3_cleaned_teleco_time_series.csv'.")

# E: Analyze the Time Series Dataset
# Load the training data
train = pd.read_csv('task3_train_teleco_time_series.csv', index_col='Day', parse_dates=True)

# E1: Analyze the time series
train = pd.read_csv('task3_train_teleco_time_series.csv', index_col='Day', parse_dates=True)

# Decomposition of the original training data
decomposition = seasonal_decompose(train['Revenue'], model='additive', period=30)
fig = decomposition.plot()
fig.set_size_inches(10, 8)
plt.suptitle('Decomposition of Daily Revenue (Training Data)', y=1.02)
plt.show()

# ADF test on residuals
residuals = decomposition.resid
result_resid = adfuller(residuals.dropna())
print("ADF Test on Residuals:")
print('ADF Statistic:', result_resid[0])
print('p-value:', result_resid[1])
print('Critical Values:', result_resid[4])
if result_resid[1] < 0.05:
    print("Residuals are stationary (no trends).")
else:
    print("Residuals are non-stationary (trends remain).")

# ACF plot of the original training data
plt.figure(figsize=(10, 4))
plot_acf(train['Revenue'], lags=50)
plt.title('Autocorrelation Function (ACF) of Daily Revenue')
plt.show()

# Spectral density plot of the original training data
frequencies = np.fft.fftfreq(len(train))
positive_freqs = frequencies[frequencies > 0]
periodogram = np.abs(np.fft.fft(train['Revenue']))**2 / len(train)
positive_periodogram = periodogram[frequencies > 0]
plt.figure(figsize=(10, 4))
plt.plot(1/positive_freqs, positive_periodogram, label='Spectral Density')
plt.xlabel('Period (Days)')
plt.ylabel('Power')
plt.title('Spectral Density of Daily Revenue')
plt.xlim(0, 100)
plt.legend()
plt.grid(True)
plt.show()

# New analysis: Differenced training data
train_diff = train['Revenue'].diff().dropna()  # First difference to remove trend
decomposition_diff = seasonal_decompose(train_diff, model='additive', period=30)
fig = decomposition_diff.plot()
fig.set_size_inches(10, 8)
plt.suptitle('Decomposition of Differenced Daily Revenue (Training Data)', y=1.02)
plt.show()

# E2: Identify ARIMA model
# Since the series is non-stationary, use differencing (d=1)
# Try ARIMA(1,1,1) as a baseline
model_111 = ARIMA(train['Revenue'], order=(1, 1, 1))
fit_111 = model_111.fit()
print("ARIMA(1,1,1) Summary:")
print(fit_111.summary())
print("AIC for ARIMA(1,1,1):", fit_111.aic)

# Try ARIMA(2,1,1) for comparison
model_211 = ARIMA(train['Revenue'], order=(2, 1, 1))
fit_211 = model_211.fit()
print("ARIMA(2,1,1) Summary:")
print(fit_211.summary())
print("AIC for ARIMA(2,1,1):", fit_211.aic)

# Select the model with the lowest AIC
best_model = fit_111 if fit_111.aic < fit_211.aic else fit_211
best_order = (1, 1, 1) if fit_111.aic < fit_211.aic else (2, 1, 1)
print(f"Selected ARIMA{best_order} with AIC: {best_model.aic}")

# E3: Perform forecast
# Combine training and test data into a single dataset
full_data = pd.concat([train, test])

# Fit the ARIMA(1,1,1) model on the entire dataset
model_full = ARIMA(full_data['Revenue'], order=(1, 1, 1))
fit_full = model_full.fit()
print("ARIMA(1,1,1) Model Summary (Fitted on Full Data):")
print(fit_full.summary())

# Forecast 90 days starting from the end of the test set (2023-01-01 to 2023-03-31)
forecast_steps = 90
forecast = fit_full.forecast(steps=forecast_steps)
forecast_index = pd.date_range(start='2023-01-01', periods=forecast_steps, freq='D')
forecast_series = pd.Series(forecast, index=forecast_index)
forecast_obj = fit_full.get_forecast(steps=forecast_steps)
conf_int = forecast_obj.conf_int(alpha=0.05)
conf_int.index = forecast_index
print("Forecast Values (first 5):")
print(forecast_series.head())
print("Confidence Intervals (first 5):")
print(conf_int.head())

# Plot the full dataset and the forecast
plt.figure(figsize=(10, 6))
plt.plot(full_data.index, full_data['Revenue'], label='Historical Data', color='blue')
plt.plot(forecast_series.index, forecast_series, label='Forecast', color='purple', linewidth=2, linestyle='--')
plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='purple', alpha=0.1, label='95% Confidence Interval')
plt.title('ARIMA(1,1,1) Forecast Starting from End of Test Set')
plt.xlabel('Date')
plt.ylabel('Revenue (Million $)')
plt.legend()
plt.grid(True)
plt.show()

# F1: Model Evaluation
# Fit the ARIMA(1,1,1) model on the training data
model_eval = ARIMA(train['Revenue'], order=(1, 1, 1))
fit_eval = model_eval.fit()

# Forecast the test period
forecast_eval = fit_eval.forecast(steps=len(test))
forecast_index_eval = test.index
forecast_series_eval = pd.Series(forecast_eval, index=forecast_index_eval)

# Calculate MAE
mae = np.mean(np.abs(forecast_series_eval - test['Revenue']))
print(f"Mean Absolute Error (MAE) of the forecast on the test set: {mae:.2f} million dollars")

# Plot the forecast vs actual test data
plt.figure(figsize=(10, 6))
plt.plot(train.index, train['Revenue'], label='Training Data', color='blue')
plt.plot(test.index, test['Revenue'], label='Test Data', color='green')
plt.plot(forecast_series_eval.index, forecast_series_eval, label='Forecast', color='red', linewidth=2)
plt.title('ARIMA(1,1,1) Forecast vs Actual Test Data (Model Evaluation)')
plt.xlabel('Date')
plt.ylabel('Revenue (Million $)')
plt.legend()
plt.grid(True)
plt.show()
