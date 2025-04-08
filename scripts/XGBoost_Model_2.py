# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 22:09:32 2025

@author: grace
"""
# XGBoost edited with prof huang comments

#%%

#data preparation functions

import pandas as pd
import ast

def process_unga_data():
    unga = pd.read_csv("data/cleaned data/unga_voting_3.csv")

    # Split the 'CountryPair' column into 'Country1' and 'Country2'
    unga['CountryPair'] = unga['CountryPair'].apply(lambda x: ast.literal_eval(x))
    unga[['Country1', 'Country2']] = pd.DataFrame(unga['CountryPair'].tolist(), index=unga.index)

    # Filter data for years 1989 to 2021
    unga = unga[(unga['year'] >= 1989) & (unga['year'] <= 2021)]

    # Filter the data for Singapore-related pairs
    unga_sg = unga[(unga['Country1'] == 'Singapore') | (unga['Country2'] == 'Singapore')]

    # Create 'Partner' column to identify the non-Singapore country
    unga_sg['Partner'] = unga_sg.apply(lambda row: row['Country1'] if row['Country1'] != 'Singapore' else row['Country2'], axis=1)

    # Select relevant columns
    unga_sg = unga_sg[['agree', 'year', 'IdealPointDistance', 'Country1', 'Country2', 'Partner']]

    # Ensure 'year' is of integer type
    unga_sg['year'] = unga_sg['year'].astype(int)

    return unga_sg

#reclean GDP data
def process_gdp_data():
    # Load the GDP data
    gdp = pd.read_csv("data/raw data/GDP.csv", header = 2)
    gdp = gdp.drop(gdp.columns[[1, 2, 3]], axis=1)

    gdp_long = gdp.melt(id_vars=['Country Name'], var_name='Year', value_name='GDP')
    gdp_long = gdp_long.dropna(subset=['GDP'])

    countries = pd.read_csv("data/raw data/COW-country-codes.csv")

    gdp_long = countries.merge(gdp_long, left_on='StateNme', right_on='Country Name', how='inner')
    gdp_long = gdp_long.drop(gdp_long.columns[[0, 1, 2]], axis=1)
    gdp_long['Year'] = gdp_long['Year'].astype(int)
    gdp_long['Country Name'] = gdp_long['Country Name'].astype(str)
    # Return the cleaned and reshaped GDP data
    return gdp_long

#reclean exchange rate data
def process_exrate_data():
    exrate = pd.read_csv("data/raw data/exchange_rate.csv", header = 4)
    exrate = exrate.drop(exrate.columns[[1, 2, 3]], axis=1)
    exrate_long = exrate.melt(id_vars=['Country Name'], var_name='Year', value_name='Exchange Rate (per US$)')
    exrate_long = exrate_long.dropna()
    exrate_long['Year'] = exrate_long['Year'].astype(int)
    exrate_long['Country Name'] = exrate_long['Country Name'].astype(str)

    return exrate_long
def process_FTA_data():
    fta = pd.read_csv("data/cleaned data/adjusted_fta_data_2.csv")

    # Filter where either country or partner country is SG or SGP
    fta_sg = fta[
        (fta['Partner Country'].isin(['SG', 'SGP']))
    ]
    fta_sg['Country Code'] = fta_sg.apply(
        lambda row: row['Country'] if row['Country'] not in ['SG', 'SGP'] else row['Partner Country'],
        axis=1
    )

    fta_sg = fta_sg.drop(columns=["Country", "Partner Country"])

    # ISO3 to country name mapping
    iso3_to_country = {
        'CHN': 'China',
        'HKG': 'Hong Kong',
        'JPN': 'Japan',
        'KOR': 'Korea',
        'MYS': 'Malaysia',
        'SAU': 'Saudi Arabia',
        'THA': 'Thailand',
        'USA': 'United States',
        'IDN': 'Indonesia'
    }

    fta_sg['Country'] = fta_sg['Country Code'].replace(iso3_to_country)
    fta_sg = fta_sg.sort_values(by='Year')

    return fta_sg

#%%

#data merging for XGBoost

import numpy as np

trade_data = pd.read_csv("data/cleaned data/10 years Trade Product Data.csv")
unga_data = process_unga_data()
gdp_data = process_gdp_data()
exrate_data = process_exrate_data()
fta_data = process_FTA_data()

# Clean up column names (strip whitespace)
trade_data.columns = trade_data.columns.str.strip()
gdp_data.columns = gdp_data.columns.str.strip()
exrate_data.columns = exrate_data.columns.str.strip()
fta_data.columns = fta_data.columns.str.strip()
# Rename columns to ensure consistent merging
trade_data = trade_data.rename(columns={"Year": "year"})
trade_data = trade_data.rename(columns={"Country": "Partner"})
gdp_data = gdp_data.rename(columns={"Year": "year"})
exrate_data = exrate_data.rename(columns={"Year": "year"})
fta_data = fta_data.rename(columns={"Year": "year"})

# Merge datasets
merged_data = pd.merge(unga_data, trade_data, how='left', left_on=['year', 'Partner'], right_on=['year', 'Partner'])
merged_data = pd.merge(merged_data, gdp_data, how='left', left_on=['Partner', 'year'], right_on=['Country Name', 'year'])
merged_data = pd.merge(merged_data, exrate_data, how='left', left_on=['Partner', 'year'], right_on=['Country Name', 'year'])
merged_data = pd.merge(merged_data, fta_data, how='left', left_on=['Partner', 'year'], right_on=['Country', 'year'])

# Debug: Check columns after merging
print("Columns after merging:", merged_data.columns)

# Convert HS_Code to string and handle NaN values
if "HS Code" in merged_data.columns:
    merged_data["HS Code"] = merged_data["HS Code"].astype(str)  # Ensure it's a string
    merged_data["HS_Section"] = merged_data["HS Code"].str[:2]  # first 2 digits of HS code

    # Check for any NaN or invalid values
    merged_data["HS_Section"] = merged_data["HS_Section"].fillna('00')  # Fill NaN with '00' or any placeholder you prefer
else:
    print("'HS_Code' column is missing from merged data.")
    merged_data["HS_Section"] = '00'  # Handle the case where HS_Code is missing

# Lag features for Imports
merged_data["Imports_Lag1"] = merged_data.groupby(["Partner"])['Imports'].shift(1)
merged_data["Imports_Lag2"] = merged_data.groupby(["Partner"])['Imports'].shift(2)
merged_data["Imports_Lag3"] = merged_data.groupby(["Partner"])['Imports'].shift(3)

# Lag features for GDP
merged_data["GDP_Lag1"] = merged_data.groupby(["Partner"])['GDP'].shift(1)

# drop empty data
merged_data = merged_data.dropna()


X = merged_data[["Imports_Lag1", "Imports_Lag2", "Imports_Lag3","IdealPointDistance", "GDP_Lag1",
                 'Exchange Rate (per US$)', 'Adjusted_value', 'HS Code', 'Imports']]

# Log-transform target (Imports)
y = np.log(merged_data["Imports"])

#%%

#XGBoost with time series split and regularisation

from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np


# Make sure you haven't used rmse_scores or r2_scores as function names above
rmse_scores = []
r2_scores = []

tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    print(f"\nFold {fold + 1}")

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1,     # L1 regularization
        reg_lambda=2,    # L2 regularization
        random_state=42,
        objective="reg:squarederror"
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    rmse_scores.append(rmse)
    r2_scores.append(r2)

    print(f"RMSE: {rmse:.4f} | R²: {r2:.4f}")
    
#%%

#plot predicted vs actual

import matplotlib.pyplot as plt


all_preds = []
all_actuals = []

for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    # ... existing code ...
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Store for combined plot
    all_preds.extend(preds)
    all_actuals.extend(y_test.values)

    # Optional: Plot for each fold
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, preds, alpha=0.5)
    plt.xlabel("Actual Log Imports")
    plt.ylabel("Predicted Log Imports")
    plt.title(f"Fold {fold + 1}: Predicted vs Actual")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(all_actuals, all_preds, alpha=0.5)
plt.xlabel("Actual Log Imports")
plt.ylabel("Predicted Log Imports")
plt.title("All Folds: Predicted vs Actual")
plt.plot([min(all_actuals), max(all_actuals)], [min(all_actuals), max(all_actuals)], color='red', linestyle='--')
plt.tight_layout()
plt.show()
