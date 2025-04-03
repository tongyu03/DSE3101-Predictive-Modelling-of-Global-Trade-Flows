# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 22:58:50 2025

@author: grace
"""

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

