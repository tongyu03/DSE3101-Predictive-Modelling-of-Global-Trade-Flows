"""
@author: Shannen
"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import ast

#cleaning
def process_trade_data():
    # Load the CSV file
    sgp_trade_raw = "data/raw data/en_SGP_AllYears_WITS_Trade_Summary.CSV"
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
    unga = pd.read_csv("data/cleaned data/unga_voting_2.csv")

    # Split the 'CountryPair' column into 'Country1' and 'Country2'
    unga['CountryPair'] = unga['CountryPair'].apply(lambda x: ast.literal_eval(x))
    unga[['Country1', 'Country2']] = pd.DataFrame(unga['CountryPair'].tolist(), index=unga.index)

    # Filter data for years 1989 to 2021
    unga = unga[(unga['year'] >= 1989) & (unga['year'] <= 2021)]

    # Filter the data for Singapore-related pairs
    unga_sg = unga[(unga['Country1'] == 'Singapore') | (unga['Country2'] == 'Singapore')]

    # Create 'Partner' column to identify the non-Singapore country
    unga_sg = unga_sg.copy()  # Create an explicit copy
    unga_sg.loc[:, 'Partner'] = unga_sg.apply(lambda row: row['Country1'] if row['Country1'] != 'Singapore' else row['Country2'], axis=1)

    # Select relevant columns
    unga_sg = unga_sg[['agree', 'year', 'IdealPointDistance', 'Country1', 'Country2', 'Partner']]

    # Ensure 'year' is of integer type
    unga_sg['year'] = unga_sg['year'].astype(int)

    return unga_sg

process_unga_data()

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

process_gdp_data()

#dummy chunk
exrate = pd.read_csv("data/raw data/exchange_rate.csv", header = 4)
exrate = exrate.drop(exrate.columns[[1, 2, 3]], axis=1)
exrate_long = exrate.melt(id_vars=['Country Name'], var_name='Year', value_name='Exchange Rate (per US$)')
exrate_long = exrate_long.dropna()
exrate_long['Year'] = exrate_long['Year'].astype(int)
exrate_long['Country Name'] = exrate_long['Country Name'].astype(str)

#reclean exchange rate data
def process_exrate_data():
    exrate = pd.read_csv("data/raw data/exchange_rate.csv", header = 4)
    exrate = exrate.drop(exrate.columns[[1, 2, 3]], axis=1)
    exrate_long = exrate.melt(id_vars=['Country Name'], var_name='Year', value_name='Exchange Rate (per US$)')
    exrate_long = exrate_long.dropna()
    exrate_long['Year'] = exrate_long['Year'].astype(int)
    exrate_long['Country Name'] = exrate_long['Country Name'].astype(str)

    return exrate_long


fta = pd.read_csv("data/cleaned data/adjusted_fta_data_2.csv")
fta_sg = fta[(fta['Country'] == 'SGP') | (fta['Partner Country'] == 'SGP')]
fta_sg = fta_sg.copy()  # Create an explicit copy
fta_sg.loc[:, 'Country Code'] = fta_sg.apply(
    lambda row: row['Country'] if row['Country'] != 'SGP' else row['Partner Country'],
    axis=1
)
fta_sg = fta_sg.drop(columns=["Country", "Country Code"])
countries = pd.read_csv("data/raw data/COW-country-codes.csv")
fta_sg = fta_sg.merge(countries, left_on='Partner Country', right_on='StateAbb', how='inner')


#reclean FTA
def process_FTA_data():
    fta = pd.read_csv("data/cleaned data/adjusted_fta_data_2.csv")
    fta_sg = fta[(fta['Country'] == 'SGP') | (fta['Partner Country'] == 'SGP')]
    fta_sg.loc[:, 'Country Code'] = fta_sg.apply(
        lambda row: row['Country'] if row['Country'] != 'SGP' else row['Partner Country'],
        axis=1
    )
    fta_sg = fta_sg.drop(columns=["Country", "Country Code"])
    countries = pd.read_csv("data/raw data/COW-country-codes.csv")
    fta_sg = fta_sg.merge(countries, left_on='Partner Country', right_on='StateAbb', how='inner')

    return fta_sg

#%%

#same as XGBoost

def prepare_data_for_regression():
    trade_data = process_trade_data()
    unga_data = process_unga_data()
    gdp_data = process_gdp_data()
    exrate_data = process_exrate_data()
    fta_data = process_FTA_data()
    trade_data = trade_data.rename(columns={"Year": "year"})
    gdp_data = gdp_data.rename(columns={"Year": "year"})
    exrate_data = exrate_data.rename(columns={"Year": "year"})
    fta_data = fta_data.rename(columns={"Year": "year"})
    # Merge datasets
    merged_data = pd.merge(unga_data, trade_data, how='left', left_on=['year', 'Partner'], right_on=['year', 'Partner'])
    merged_data = pd.merge(merged_data, gdp_data, how='left', left_on=['Partner', 'year'], right_on=['Country Name', 'year'])
    merged_data = pd.merge(merged_data, exrate_data, how='left', left_on=['Partner', 'year'], right_on=['Country Name', 'year'])
    merged_data = pd.merge(merged_data, fta_data, how='left', left_on=['Partner', 'year'], right_on=['StateNme', 'year'])

    # Drop redundant columns
    cols_to_drop = ['Country1', 'Country2', 'Year_x', 'Year_y', 'Country Name_x', 'Country Name_y']
    existing_cols_to_drop = [col for col in cols_to_drop if col in merged_data.columns]
    merged_data = merged_data.drop(columns=existing_cols_to_drop)
    merged_data = merged_data.dropna()

    # Lag features for Trade_Value
    merged_data["Trade_Value_Lag1"] = merged_data.groupby(["Partner", "Indicator Type"])['Trade_Value'].shift(1)
    merged_data["Trade_Value_Lag2"] = merged_data.groupby(["Partner", "Indicator Type"])['Trade_Value'].shift(2)
    merged_data["Trade_Value_Lag3"] = merged_data.groupby(["Partner", "Indicator Type"])['Trade_Value'].shift(3)

    # Drop NaN values
    merged_data = merged_data.dropna()

    # Define features (X) and target (y)
    X = merged_data[["Trade_Value_Lag1", "Trade_Value_Lag2", "Trade_Value_Lag3", "IdealPointDistance", "agree", "GDP", 'Exchange Rate (per US$)', 'Adjusted_value']]
    y = merged_data["Trade_Value"]

    return X, y

# Prepare data
X, y = prepare_data_for_regression()

# Split data 70-20-10
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42, shuffle=False)  # 0.33 * 0.3 = 0.1 for test set

# Train multiple linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Make predictions on validation set
y_val_pred = lin_reg.predict(X_val)

# Evaluate model performance on validation set
mae_val = mean_absolute_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)
mse_val = mean_squared_error(y_val, y_val_pred)

print("Validation Set Performance:")
print(f"Mean Absolute Error (MAE): {mae_val}")
print(f"R-squared (R²): {r2_val}")
print(f"Mean Squared Error (MSE): {mse_val}")

# Make predictions on test set
y_test_pred = lin_reg.predict(X_test)

# Evaluate model performance on test set
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print("\nTest Set Performance:")
print(f"Mean Absolute Error (MAE): {mae_test}")
print(f"R-squared (R²): {r2_test}")
print(f"Mean Squared Error (MSE): {mse_test}")

# Display key statistics of trade values
print("\nTrade Value Statistics:")
print(f"Mean Trade Value: {y.mean()}")
print(f"Median Trade Value: {y.median()}")
print(f"Min Trade Value: {y.min()}")
print(f"Max Trade Value: {y.max()}")
print(f"Variance: {y.var()}")

# Display model coefficients
coefficients = pd.DataFrame({"Feature": X.columns, "Coefficient": lin_reg.coef_})
print("\nModel Coefficients:")
print(coefficients)
