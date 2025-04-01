# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 23:00:23 2025

@author: grace
"""

"""
Baseline Multiple Linear Regression Model
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Sort data by Year for time-series modeling
trade_summary = merged_df.sort_values(by=["Year"])

# Create lag features (1-year, 2-year, 3-year lags)
for lag in range(1, 4):
    trade_summary[f"Trade_Value_Lag{lag}"] = trade_summary.groupby(["Partner", "Indicator Type"])["Trade_Value"].shift(lag)

# Drop rows with NaN values due to lag creation
trade_summary = trade_summary.dropna()
features = [
    "Trade_Value_Lag1", 
    "Trade_Value_Lag2", 
    "Trade_Value_Lag3", 
    "GDP_normalized", 
    "Exchange Rate", 
    "IdealPointDistance",
    "agree"
]

X = trade_summary[features]
y = trade_summary["Trade_Value"]

# Split data into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Train Multiple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

# Display key statistics of trade values
print("Mean Trade Value:", y.mean())
print("Median Trade Value:", y.median())
print("Min Trade Value:", y.min())
print("Max Trade Value:", y.max())
print("Variance:", y.var())

# Display model coefficients
coefficients = pd.DataFrame({"Feature": features, "Coefficient": model.coef_})
print(coefficients)
