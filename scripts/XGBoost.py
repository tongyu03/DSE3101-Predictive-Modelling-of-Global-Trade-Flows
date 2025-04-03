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

#%%
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

    # Merge the trade data with the UNGA voting data
    # Merge on Partner and Year. Note that 'Partner' from trade_data matches either 'Country1' or 'Country2' in unga_data.
    merged_data = pd.merge(unga_data, trade_data, how='left', left_on=['year', 'Partner'], right_on=['Year', 'Partner'])

    # Drop any columns that won't be useful for modeling
    merged_data = merged_data.drop(columns=['Country1', 'Country2', 'Year'])

    # Handle missing values if any
    merged_data = merged_data.dropna()

    # Feature Engineering: Create lag features for Trade_Value
    merged_data["Trade_Value_Lag1"] = merged_data.groupby(["Partner", "Indicator Type"])["Trade_Value"].shift(1)
    merged_data["Trade_Value_Lag2"] = merged_data.groupby(["Partner", "Indicator Type"])["Trade_Value"].shift(2)
    merged_data["Trade_Value_Lag3"] = merged_data.groupby(["Partner", "Indicator Type"])["Trade_Value"].shift(3)

    # Drop rows with NaN values (due to lagging)
    merged_data = merged_data.dropna()

    # Define features (X) and target (y)
    X = merged_data[["Trade_Value_Lag1", "Trade_Value_Lag2", "Trade_Value_Lag3", "IdealPointDistance", "agree"]]
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
##XGBoost

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
merged_df = merged_df.drop(columns=['name', 'GDP', 'year'])
print(merged_df.head())

#integrate exchange_rate_script_1.py
exchange_rate = pd.read_csv("data/exchange_rate.csv")
merged_df = pd.merge(merged_df, exchange_rate, how='left', left_on=['Partner','Year'], right_on=['Data Source', 'Year'])
merged_df = merged_df.drop(columns=['Data Source', 'Country Code'])
merged_df.rename(columns={"World Development Indicators": "Country Code"}, inplace=True)
merged_df.rename(columns={"Value": "Exchange Rate"}, inplace=True)
print(merged_df.head())

#integrate FTA data
fta = pd.read_csv("data/adjusted_fta_data.csv")
fta_sg = fta[(fta['Country'] == 'SGP') | (fta['Partner Country'] == 'SGP')]
fta_sg['Country Code'] = fta_sg.apply(
    lambda row: row['Country'] if row['Country'] != 'SGP' else row['Partner Country'],
    axis=1
)
merged_df = pd.merge(merged_df, fta_sg, how='left', left_on=['Year', 'Country Code'], right_on=['Year', 'Country Code'])
print(merged_df.head())

"""
XGBOOST MODEL
"""
# time-series training
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Sort by Year for time-series modeling
trade_summary = merged_df.sort_values(by=["Year"])

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
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"R-squared (R²): {r2}")

from sklearn.metrics import mean_squared_error

print("Mean Trade Value:", y.mean())
print("Median Trade Value:", y.median())
print("Min Trade Value:", y.min())
print("Max Trade Value:", y.max())
print("Variance:", y.var())
print("MSE:", mean_squared_error(y_test, y_pred))

#important datasets to be considered
#relations between countries of interests that indirectly impact Singapore
unga_others = unga[(unga['Country1'] != 'Singapore') & (unga['Country2'] != 'Singapore')]
print(unga_others.head())

