# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 22:58:50 2025

@author: grace
"""



#%%

# reorganised updated script by grace
# part1: data cleaning

import pandas as pd
import ast

def process_trade_data():
    # Load the CSV file
    sgp_trade_raw = "data/en_SGP_AllYears_WITS_Trade_Summary.CSV"
    sgp_trade_cleaned = pd.read_csv(sgp_trade_raw)

    # Convert to long format
    sgp_trade_cleaned = sgp_trade_cleaned.melt(id_vars=["Reporter", "Partner", "Product categories", "Indicator Type", "Indicator"], 
                                              var_name="Year", value_name="Trade_Value")

    # Convert 'Year' to numeric and handle errors
    sgp_trade_cleaned["Year"] = pd.to_numeric(sgp_trade_cleaned["Year"], errors='coerce')

    # Drop rows with missing values
    sgp_trade_cleaned = sgp_trade_cleaned.dropna()

    # Remove rows that contain '...' in any of the columns
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

    # Filter based on relevant indicators
    sgp_trade_cleaned = sgp_trade_cleaned[sgp_trade_cleaned["Indicator"].isin(relevant_indicators)]

    # Keep only rows where Product categories are "All Products"
    sgp_trade_cleaned = sgp_trade_cleaned[sgp_trade_cleaned["Product categories"] == "All Products"]

    # Remove rows where Partner is "World"
    sgp_trade_cleaned = sgp_trade_cleaned[sgp_trade_cleaned["Partner"] != "World"]

    # Group by Year, Partner, and Indicator Type and sum Trade_Value
    trade_summary = sgp_trade_cleaned.groupby(["Year", "Partner", "Indicator Type"])["Trade_Value"].sum().reset_index()

    return trade_summary

# Function to load and preprocess UNGA data
def process_unga_data():
    unga = pd.read_csv("data/unga_voting.csv")
    
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

process_unga_data()

#reclean GDP data
def process_gdp_data():
    # Load the GDP data
    gdp = pd.read_csv("data/GDP.csv", header = 2)
    gdp = gdp.drop(gdp.columns[[1, 2, 3]], axis=1)

    gdp_long = gdp.melt(id_vars=['Country Name'], var_name='Year', value_name='GDP')
    gdp_long = gdp_long.dropna(subset=['GDP'])

    countries = pd.read_csv("data/COW-country-codes.csv")

    gdp_long = countries.merge(gdp_long, left_on='StateNme', right_on='Country Name', how='inner')
    gdp_long = gdp_long.drop(gdp_long.columns[[0, 1, 2]], axis=1)
    gdp_long['Year'] = gdp_long['Year'].astype(int)
    gdp_long['Country Name'] = gdp_long['Country Name'].astype(str)
    # Return the cleaned and reshaped GDP data
    return gdp_long

process_gdp_data()

#dummy chunk
exrate = pd.read_csv("data/exchange_rate.csv", header = 4)
exrate = exrate.drop(exrate.columns[[1, 2, 3]], axis=1)
exrate_long = exrate.melt(id_vars=['Country Name'], var_name='Year', value_name='Exchange Rate (per US$)')
exrate_long = exrate_long.dropna()
exrate_long['Year'] = exrate_long['Year'].astype(int)
exrate_long['Country Name'] = exrate_long['Country Name'].astype(str)

#reclean exchange rate data
def process_exrate_data():
    exrate = pd.read_csv("data/exchange_rate.csv", header = 4)
    exrate = exrate.drop(exrate.columns[[1, 2, 3]], axis=1)
    exrate_long = exrate.melt(id_vars=['Country Name'], var_name='Year', value_name='Exchange Rate (per US$)')
    exrate_long = exrate_long.dropna()
    exrate_long['Year'] = exrate_long['Year'].astype(int)
    exrate_long['Country Name'] = exrate_long['Country Name'].astype(str)
    
    return exrate_long


fta = pd.read_csv("data/cleaned data/adjusted_fta_data.csv")
fta_sg = fta[(fta['Country'] == 'SGP') | (fta['Partner Country'] == 'SGP')]
fta_sg['Country Code'] = fta_sg.apply(
    lambda row: row['Country'] if row['Country'] != 'SGP' else row['Partner Country'],
    axis=1
)
fta_sg = fta_sg.drop(columns=["Country", "Country Code"])
countries = pd.read_csv("data/COW-country-codes.csv")
fta_sg = fta_sg.merge(countries, left_on='Partner Country', right_on='StateAbb', how='inner')


#reclean FTA
def process_FTA_data():
    fta = pd.read_csv("data/cleaned data/adjusted_fta_data.csv")
    fta_sg = fta[(fta['Country'] == 'SGP') | (fta['Partner Country'] == 'SGP')]
    fta_sg['Country Code'] = fta_sg.apply(
        lambda row: row['Country'] if row['Country'] != 'SGP' else row['Partner Country'],
        axis=1
    )
    fta_sg = fta_sg.drop(columns=["Country", "Country Code"])
    countries = pd.read_csv("data/COW-country-codes.csv")
    fta_sg = fta_sg.merge(countries, left_on='Partner Country', right_on='StateAbb', how='inner')
    
    return fta_sg

#%%

# testing data merger

trade_data = process_trade_data()
unga_data = process_unga_data()
gdp_data = process_gdp_data()
exrate_data = process_exrate_data()
fta_data = process_FTA_data()

merged_data = pd.merge(unga_data, trade_data, how='left', left_on=['year', 'Partner'], right_on=['Year', 'Partner'])
merged_data = pd.merge(merged_data, gdp_data, how='left', left_on=['Partner', 'year'], right_on=['Country Name', 'Year'])
merged_data = pd.merge(merged_data, exrate_data, how='left', left_on=['Partner', 'year'], right_on=['Country Name', 'Year'])
merged_data = pd.merge(merged_data, fta_data, how='left', left_on=['Partner', 'year'], right_on=['StateNme', 'Year'])
#%%
#todo:
    # take out exchange rate from the data
    # change benchmark inputs based on chunk 1
    #check XGBooost and prediction plots

# reorganised updated script by grace
# part2: XGBoost

import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Assuming process_trade_data and process_unga_data are already defined

def prepare_data_for_xgboost():
    # Load trade and UNGA data
    trade_data = process_trade_data()
    unga_data = process_unga_data()
    gdp_data = process_gdp_data()
    exrate_data = process_exrate_data()
    fta_data = process_FTA_data()

    # Merge the trade data with the UNGA voting data
    merged_data = pd.merge(unga_data, trade_data, how='left', left_on=['year', 'Partner'], right_on=['Year', 'Partner'])

    # Merge the GDP data with the merged data (based on Partner and Year)
    merged_data = pd.merge(merged_data, gdp_data, how='left', left_on=['Partner', 'year'], right_on=['Country Name', 'Year'])
    
    merged_data = pd.merge(merged_data, exrate_data, how='left', left_on=['Partner', 'year'], right_on=['Country Name', 'Year'])
    
    merged_data = pd.merge(merged_data, fta_data, how='left', left_on=['Partner', 'year'], right_on=['StateNme', 'Year'])

    # Drop any columns that won't be useful for modeling
    merged_data = merged_data.drop(columns=['Country1', 'Country2', 'Year_x', 'Year_y', 'Country Name_x', 'Country Name_y'])  # Drop 'Country Name' after merging

    # Handle missing values if any
    merged_data = merged_data.dropna()

    # Feature Engineering: Create lag features for Trade_Value
    merged_data["Trade_Value_Lag1"] = merged_data.groupby(["Partner", "Indicator Type"])["Trade_Value"].shift(1)
    merged_data["Trade_Value_Lag2"] = merged_data.groupby(["Partner", "Indicator Type"])["Trade_Value"].shift(2)
    merged_data["Trade_Value_Lag3"] = merged_data.groupby(["Partner", "Indicator Type"])["Trade_Value"].shift(3)

    # Drop rows with NaN values (due to lagging)
    merged_data = merged_data.dropna()

    # Define features (X) and target (y)
    X = merged_data[["Trade_Value_Lag1", "Trade_Value_Lag2", "Trade_Value_Lag3", "IdealPointDistance", "agree", "GDP", 'Exchange Rate (per US$)', 'Adjusted_value']]
    y = merged_data["Trade_Value"]

    # Return features and target
    return X, y

# Prepare data
X, y = prepare_data_for_xgboost()

# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Train the XGBoost model
model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Print the evaluation metrics
print(f"Mean Absolute Error: {mae}")
print(f"R-squared (R²): {r2}")
print(f"Mean Squared Error (MSE): {mse}")

#%%

#extra
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Trade Volume")
plt.ylabel("Predicted Trade Volume")
plt.title("Predicted vs Actual Trade Volume")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # identity line
plt.grid(True)
plt.tight_layout()
plt.show()



