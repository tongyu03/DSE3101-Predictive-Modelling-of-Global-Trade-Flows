# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 22:03:53 2025

@author: grace
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
sgp_trade_raw = "data/en_SGP_AllYears_WITS_Trade_Summary.CSV"

sgp_trade_cleaned = pd.read_csv(sgp_trade_raw)

# convert to long format
sgp_trade_cleaned = sgp_trade_cleaned.melt(id_vars=["Reporter", "Partner", "Product categories", "Indicator Type", "Indicator"], 
                  var_name="Year", value_name="Trade_Value")

sgp_trade_cleaned["Year"] = pd.to_numeric(sgp_trade_cleaned["Year"], errors='coerce')

sgp_trade_cleaned = sgp_trade_cleaned.dropna()

sgp_trade_cleaned = sgp_trade_cleaned[~sgp_trade_cleaned.apply(lambda row: row.astype(str).str.contains("\.\.\.").any(), axis=1)]

# Define the indicators to keep
relevant_indicators = [
    "Import(US$ Mil)", 
    "Export(US$ Mil)", 
    "Imports (in US$ Mil)", 
    "Exports (in US$ Mil)", 
    "Trade (US$ Mil)-Top 5 Export Partner", 
    "Trade (US$ Mil)-Top 5 Import Partner"
]

sgp_trade_cleaned = sgp_trade_cleaned[sgp_trade_cleaned["Indicator"].isin(relevant_indicators)]

sgp_trade_cleaned = sgp_trade_cleaned[sgp_trade_cleaned["Product categories"] == "All Products"]

sgp_trade_cleaned = sgp_trade_cleaned[sgp_trade_cleaned["Partner"] != "World"]
#print(sgp_trade_cleaned.head())

trade_summary = sgp_trade_cleaned.groupby(["Year", "Partner", "Indicator Type"])["Trade_Value"].sum().reset_index()


# Plot trade volume over time
plt.figure(figsize=(12, 6))

# Loop through unique partners and indicator types
for partner in trade_summary["Partner"].unique():
    for indicator_type in trade_summary["Indicator Type"].unique():
        subset = trade_summary[(trade_summary["Partner"] == partner) & (trade_summary["Indicator Type"] == indicator_type)]
        plt.plot(subset["Year"], subset["Trade_Value"], label=f"{partner} - {indicator_type}")

# Customize the plot
plt.xlabel("Year")
plt.ylabel("Trade Volume (US$ Mil)")
plt.title("Trade Volume Over Time by Partner and Indicator Type")
plt.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.grid(True)

# Show the plot
plt.show()


#%%
'''

Created on Sat Mar 22 02:17:44 2025
@author: shannen

'''
import ast
#integrate unga_voting script into model
unga = pd.read_csv("data/unga_voting.csv")

#split pait
unga['CountryPair'] = unga['CountryPair'].apply(lambda x: ast.literal_eval(x))
unga[['Country1', 'Country2']] = pd.DataFrame(unga['CountryPair'].tolist(), index=unga.index)
unga = unga.drop(columns=['CountryPair'])

#filter
unga = unga[(unga['year'] >= 1989) & (unga['year'] <= 2021)]
unga_sg = unga[(unga['Country1'] == 'Singapore') | (unga['Country2'] == 'Singapore')]
unga_sg = unga_sg[['agree', 'year', 'IdealPointDistance', 'Country1', 'Country2']]

#standardisation
unga_sg['year'] = unga_sg['year'].astype(int)
trade_summary['Year'] = trade_summary['Year'].astype(int)

unga_sg['Partner'] = unga_sg.apply(lambda row: row['Country1'] if row['Country1'] != 'Singapore' else row['Country2'], axis=1)

#join
merged_df = pd.merge(unga_sg, trade_summary, how='left', left_on=['Partner', 'year'], right_on=['Partner', 'Year'])
merged_df = merged_df.drop(columns=['Country1', 'Country2', 'Year'])
print(merged_df.head())

#integrate GDP script 1.py
gdp = pd.read_csv("data/Processed_GDP.csv")
countries = pd.read_csv("data/countrylegend.csv")
gdp = pd.merge(gdp, countries[['alpha-3', 'name']], how='left', left_on='Country Code', right_on='alpha-3')
gdp = gdp.drop(columns=['alpha-3'])
merged_df = pd.merge(merged_df, gdp, how='left', left_on =['year', 'Partner'], right_on=['Year', 'name'])
merged_df = merged_df.drop(columns=['name'])

# time-series training
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# Sort by Year for time-series modeling
trade_summary = merged_df.sort_values(by=["year"])

# Create lag features (1-year, 2-year, 3-year lags)
trade_summary["Trade_Value_Lag1"] = trade_summary.groupby(["Partner", "Indicator Type"])["Trade_Value"].shift(1)
trade_summary["Trade_Value_Lag2"] = trade_summary.groupby(["Partner", "Indicator Type"])["Trade_Value"].shift(2)
trade_summary["Trade_Value_Lag3"] = trade_summary.groupby(["Partner", "Indicator Type"])["Trade_Value"].shift(3)

# Drop rows with NaN values (since lags introduce missing values)
trade_summary = trade_summary.dropna()

# Display processed dataset
print(trade_summary.head())

# Define features (lagged trade values) and target variable
X = trade_summary[["Trade_Value_Lag1", "Trade_Value_Lag2", "Trade_Value_Lag3"]]
y = trade_summary["Trade_Value"]

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)


# Train an XGBoost Regressor
model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

print("Mean Trade Value:", y.mean())
print("Median Trade Value:", y.median())
print("Min Trade Value:", y.min())
print("Max Trade Value:", y.max())




























