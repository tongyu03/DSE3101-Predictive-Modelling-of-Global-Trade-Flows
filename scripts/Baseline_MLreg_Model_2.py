"""

@author: Shannen
Updated trade data

"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import ast
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge, Lasso


#new trade data

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
    trade_data = pd.read_csv("data/cleaned data/cleaned_yearly_trade_data.csv")
    unga_data = process_unga_data()
    gdp_data = process_gdp_data()
    exrate_data = process_exrate_data()
    fta_data = process_FTA_data()

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

    # Drop redundant columns
    cols_to_drop = ['Country1', 'Country2', 'Year_x', 'Year_y', 'Country Name_x', 'Country Name_y']
    existing_cols_to_drop = [col for col in cols_to_drop if col in merged_data.columns]
    merged_data = merged_data.drop(columns=existing_cols_to_drop)
    merged_data = merged_data.dropna()

    # Lag features for Trade_Value
    merged_data["Trade_Volume_Lag1"] = merged_data.groupby(["Partner"])['Trade Volume'].shift(1)
    merged_data["Trade_Volume_Lag2"] = merged_data.groupby(["Partner"])['Trade Volume'].shift(2)
    merged_data["Trade_Volume_Lag3"] = merged_data.groupby(["Partner"])['Trade Volume'].shift(3)

    merged_data = merged_data.dropna()

    if log_transform:
        # Log-transform lag features and target
        merged_data["log_Trade_Volume_Lag1"] = np.log(merged_data["Trade_Volume_Lag1"])
        merged_data["log_Trade_Volume_Lag2"] = np.log(merged_data["Trade_Volume_Lag2"])
        merged_data["log_Trade_Volume_Lag3"] = np.log(merged_data["Trade_Volume_Lag3"])

        # Log-transform target (Trade_Value)
        X = merged_data[["log_Trade_Volume_Lag1", "log_Trade_Volume_Lag2", "log_Trade_Volume_Lag3",
                         "IdealPointDistance", "agree", "GDP", 'Exchange Rate (per US$)', 'Adjusted_value']]
        y = np.log(merged_data["Trade Volume"])

    else:
        X = merged_data[["Trade_Volume_Lag1", "Trade_Volume_Lag2", "Trade_Volume_Lag3",
                         "IdealPointDistance", "agree", "GDP", 'Exchange Rate (per US$)', 'Adjusted_value']]
        y = merged_data["Trade Volume"]

    return X, y

X, y = prepare_data_for_regression()
print(f"Number of datapoints: {len(X)}")


# TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

r2_scores = []
mae_scores = []
mse_scores = []
coefficients_list = []

# Initialize lists before the loop
all_y_test = []
all_y_pred = []
aic_values = []
bic_values = []

n_params = X.shape[1]

def calculate_aic_bic(y_true, y_pred, n_params):
    log_likelihood = -0.5 * np.sum((y_true - y_pred) ** 2)
    n = len(y_true)
    aic = 2 * n_params - 2 * log_likelihood
    bic = np.log(n) * n_params - 2 * log_likelihood
    
    return aic, bic

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2_scores.append(r2_score(y_test, y_pred))
    mae_scores.append(mean_absolute_error(y_test, y_pred))
    mse_scores.append(mean_squared_error(y_test, y_pred))
    coefficients_list.append(model.coef_)

    # Collect for plotting
    all_y_test.extend(y_test)
    all_y_pred.extend(y_pred)
    #Calc Aic and BIC
    aic, bic = calculate_aic_bic(y_test, y_pred, n_params)
    aic_values.append(aic)
    bic_values.append(bic)
average_aic = np.mean(aic_values)
average_bic = np.mean(bic_values)
print(f"Average AIC: {average_aic}")
print(f"Average BIC: {average_bic}")

# Output average metrics
print(f"\nK-Fold Cross-Validation Results (k=5)")
print(f"Average R²: {np.mean(r2_scores):.4f}")
print(f"Average MAE: {np.mean(mae_scores):.2f}")
print(f"Average MSE: {np.mean(mse_scores):.2f}")

# Trade stats
print("\nTrade Value Statistics:")
print(f"Mean Trade Volume: {y.mean():.2f}")
print(f"Median Trade Volume: {y.median():.2f}")
print(f"Min Trade Volume: {y.min():.2f}")
print(f"Max Trade Volume: {y.max():.2f}")
print(f"Variance: {y.var():.2f}")

# Average Coefficients across folds
avg_coefficients = np.mean(coefficients_list, axis=0)
coefficients_df = pd.DataFrame({
    "Feature": X.columns,
    "Average Coefficient": avg_coefficients
})
print("\nAverage Model Coefficients (across folds):")
print(coefficients_df)

plt.figure(figsize=(8, 6))
plt.scatter(all_y_test, all_y_pred, alpha=0.6)
plt.plot([min(all_y_test), max(all_y_test)],
         [min(all_y_test), max(all_y_test)],
         color='red', linestyle='--')

plt.xlabel("Actual Trade Volume")
plt.ylabel("Predicted Trade Volume")
plt.title("Actual vs Predicted Trade Value (Linear Regression, K-Fold CV)")
plt.grid(True)
plt.tight_layout()
plt.show()

#residuals plot
residuals = np.array(all_y_test) - np.array(all_y_pred)

plt.figure(figsize=(8, 6))
plt.scatter(all_y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Trade Volume")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residuals vs Predicted (Linear Regression)")
plt.grid(True)
plt.tight_layout()
plt.show()

#check autocorrelation
print(y.corr(y.shift(1)))
print(y.corr(y.shift(2)))
print(y.corr(y.shift(3)))

# Initialize Ridge and Lasso models
ridge_model = Ridge(alpha=1.0)
lasso_model = Lasso(alpha=0.1)

r2_scores_ridge = []
r2_scores_lasso = []

#Overly high R^2
#Cross-validation for Ridge and Lasso
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Ridge regression
    ridge_model.fit(X_train, y_train)
    y_pred_ridge = ridge_model.predict(X_test)
    r2_scores_ridge.append(r2_score(y_test, y_pred_ridge))

    # Lasso regression
    lasso_model.fit(X_train, y_train)
    y_pred_lasso = lasso_model.predict(X_test)
    r2_scores_lasso.append(r2_score(y_test, y_pred_lasso))

# Output average R² for Ridge and Lasso
print(f"Average R² for Ridge Regression: {np.mean(r2_scores_ridge):.4f}")
print(f"Average R² for Lasso Regression: {np.mean(r2_scores_lasso):.4f}")