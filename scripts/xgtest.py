from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from sklearn.model_selection import TimeSeriesSplit
import warnings

def process_unga_data():
    unga = pd.read_csv("data/cleaned data/unga_voting_2.csv")

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

def prepare_data_for_regression(log_transform=True):
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

    # Lag features for Trade_Value
    merged_data["Trade_Volume_Lag1"] = merged_data.groupby(["Partner"])['Trade Volume'].shift(1)
    merged_data["Trade_Volume_Lag2"] = merged_data.groupby(["Partner"])['Trade Volume'].shift(2)
    merged_data["Trade_Volume_Lag3"] = merged_data.groupby(["Partner"])['Trade Volume'].shift(3)

    # Lag features for GDP and Exchange Rate
    merged_data["GDP_Lag1"] = merged_data.groupby(["Partner"])['GDP'].shift(1)
    merged_data["GDP_Lag2"] = merged_data.groupby(["Partner"])['GDP'].shift(2)
    merged_data["GDP_Lag3"] = merged_data.groupby(["Partner"])['GDP'].shift(3)

    merged_data["ExRate_Lag1"] = merged_data.groupby(["Partner"])['Exchange Rate (per US$)'].shift(1)
    merged_data["ExRate_Lag2"] = merged_data.groupby(["Partner"])['Exchange Rate (per US$)'].shift(2)
    merged_data["ExRate_Lag3"] = merged_data.groupby(["Partner"])['Exchange Rate (per US$)'].shift(3)

    merged_data = merged_data.dropna()

    if log_transform:
        # Log-transform lag features and target
        merged_data["log_Trade_Volume_Lag1"] = np.log(merged_data["Trade_Volume_Lag1"])
        merged_data["log_Trade_Volume_Lag2"] = np.log(merged_data["Trade_Volume_Lag2"])
        merged_data["log_Trade_Volume_Lag3"] = np.log(merged_data["Trade_Volume_Lag3"])

        merged_data["log_GDP_Lag1"] = np.log(merged_data["GDP_Lag1"])
        merged_data["log_GDP_Lag2"] = np.log(merged_data["GDP_Lag2"])
        merged_data["log_GDP_Lag3"] = np.log(merged_data["GDP_Lag3"])

        merged_data["log_ExRate_Lag1"] = np.log(merged_data["ExRate_Lag1"])
        merged_data["log_ExRate_Lag2"] = np.log(merged_data["ExRate_Lag2"])
        merged_data["log_ExRate_Lag3"] = np.log(merged_data["ExRate_Lag3"])

    return merged_data


def run_xgboost_model():
    # Prepare data
    merged_data = prepare_data_for_regression()
    X = merged_data.drop(columns=['Trade Volume', 'Country1', 'Country2', 'Partner'])
    y = merged_data['Trade Volume']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize XGBoost Regressor
    model = XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                         max_depth=5, alpha=10, n_estimators=1000)

    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("RMSE: ", rmse)
    print("R-squared: ", r2)
    print("MAE: ", mae)

    # Feature importance plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=model.feature_importances_, y=X.columns)
    plt.title('Feature Importance')
    plt.show()

run_xgboost_model()
