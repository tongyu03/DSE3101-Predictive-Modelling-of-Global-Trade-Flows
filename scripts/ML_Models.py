"""
@author: shannen, grace
Test all model approaches
"""
# run this script from top to bottom
# load packages

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import ast
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
import seaborn as sns
import joblib

#%%

# data processing functions
def process_unga_data():
    unga = pd.read_csv("data/cleaned data/unga_voting_4.csv")
    unga['CountryPair'] = unga['CountryPair'].apply(lambda x: ast.literal_eval(x))
    unga[['Country1', 'Country2']] = pd.DataFrame(unga['CountryPair'].tolist(), index=unga.index)
    unga = unga[(unga['year'] >= 1989) & (unga['year'] <= 2023)]
    unga_sg = unga[(unga['Country1'] == 'Singapore') | (unga['Country2'] == 'Singapore')]
    unga_sg['Partner'] = unga_sg.apply(lambda row: row['Country1'] if row['Country1'] != 'Singapore' else row['Country2'], axis=1)
    unga_sg = unga_sg[['agree', 'year', 'IdealPointDistance', 'Country1', 'Country2', 'Partner']]
    unga_sg['year'] = unga_sg['year'].astype(int)
    return unga_sg

def process_gdp_data():
    gdp = pd.read_csv("data/raw data/GDP.csv", header = 2)
    gdp = gdp.drop(gdp.columns[[1, 2, 3]], axis=1)
    gdp_long = gdp.melt(id_vars=['Country Name'], var_name='Year', value_name='GDP')
    gdp_long = gdp_long.dropna(subset=['GDP'])
    countries = pd.read_csv("data/raw data/COW-country-codes.csv")
    gdp_long['Country Name'] = gdp_long['Country Name'].replace('United States', 'United States of America')
    gdp_long['Country Name'] = gdp_long['Country Name'].replace('Korea, Rep.', 'South Korea')
    gdp_long = countries.merge(gdp_long, left_on='StateNme', right_on='Country Name', how='inner')
    gdp_long = gdp_long.drop(gdp_long.columns[[0, 1, 2]], axis=1)
    gdp_long['Year'] = gdp_long['Year'].astype(int)
    gdp_long['Country Name'] = gdp_long['Country Name'].astype(str)
    gdp_long = gdp_long[gdp_long['Year'] >= 2013]
    return gdp_long

def process_exrate_data():
    exrate = pd.read_csv("data/raw data/exchange_rate.csv", header = 4)
    exrate = exrate.drop(exrate.columns[[1, 2, 3]], axis=1)
    exrate_long = exrate.melt(id_vars=['Country Name'], var_name='Year', value_name='Exchange Rate (per US$)')
    exrate_long = exrate_long.dropna()
    exrate_long['Year'] = exrate_long['Year'].astype(int)
    exrate_long['Country Name'] = exrate_long['Country Name'].astype(str)
    exrate_long['Country Name'] = exrate_long['Country Name'].replace('United States', 'United States of America')
    exrate_long['Country Name'] = exrate_long['Country Name'].replace('Korea, Rep.', 'South Korea')
    return exrate_long

def process_FTA_data():
    fta = pd.read_csv("data/cleaned data/adjusted_fta_data_2.csv")
    fta_sg = fta[(fta['Partner Country'].isin(['SG', 'SGP']))]
    fta_sg['Country Code'] = fta_sg.apply(
        lambda row: row['Country'] if row['Country'] not in ['SG', 'SGP'] else row['Partner Country'],
        axis=1
    )
    fta_sg = fta_sg.drop(columns=["Country", "Partner Country"])
    iso3_to_country = {
        'CHN': 'China', 'HKG': 'Hong Kong', 'JPN': 'Japan', 'KOR': 'South Korea',
        'MYS': 'Malaysia', 'SAU': 'Saudi Arabia', 'THA': 'Thailand', 'USA': 'United States of America',
        'IDN': 'Indonesia', 'ARE': 'United Arab Emirates', 'IND': 'India', 'PHL': 'Philippines',
        'VNM': 'Vietnam', 'AUS': 'Australia', 'TWN': 'Taiwan', 'DEU': 'Germany',
    }
    fta_sg['Country'] = fta_sg['Country Code'].replace(iso3_to_country)
    fta_sg = fta_sg.sort_values(by='Year')
    return fta_sg

#%%



# merge datasets and prepare for linear regression

def prepare_data_for_regression(log_transform=True, add_interactions=True):
    trade_data = pd.read_csv("data/cleaned data/10 years Trade Product Data.csv")
    trade_data['Country'] = trade_data['Country'].replace('USA', 'United States of America')
    trade_data['Country'] = trade_data['Country'].replace('Rep. of Korea', 'South Korea')
    sg_gdp = pd.read_csv("data/cleaned data/singapore_gdp.csv").iloc[:-1]
    # Create lagged GDP column
    sg_gdp['SG_GDP_Lag1'] = sg_gdp['Singapore_GDP'].shift(1)
    unga_data = process_unga_data()
    gdp_data = process_gdp_data()
    exrate_data = process_exrate_data()
    fta_data = process_FTA_data()

    # Clean up column names and ensure consistent naming
    trade_data.columns = trade_data.columns.str.strip()
    gdp_data.columns = gdp_data.columns.str.strip()
    exrate_data.columns = exrate_data.columns.str.strip()
    fta_data.columns = fta_data.columns.str.strip()
    sg_gdp.columns = sg_gdp.columns.str.strip()


    trade_data = trade_data.rename(columns={"Year": "year", "Country": "Partner"})
    gdp_data = gdp_data.rename(columns={"Year": "year"})
    exrate_data = exrate_data.rename(columns={"Year": "year"})
    fta_data = fta_data.rename(columns={"Year": "year"})
    sg_gdp = sg_gdp.rename(columns={"Year": "year"}) #sgp gdp code
    sg_gdp['year'] = sg_gdp['year'].astype(int)


    # Merge datasets
    merged_data = pd.merge(unga_data, trade_data, how='left', left_on=['year', 'Partner'], right_on=['year', 'Partner'])
    
    #checks
    merged_data = merged_data[merged_data['year'] >= 2013]
    merged_data = merged_data.dropna()
    merged_data = merged_data.drop_duplicates()
    
    merged_data = pd.merge(merged_data, gdp_data, how='left', left_on=['Partner', 'year'], right_on=['Country Name', 'year'])
    merged_data = merged_data.drop_duplicates()
    
    merged_data = pd.merge(merged_data, exrate_data, how='left', left_on=['Partner', 'year'], right_on=['Country Name', 'year'])
    merged_data = pd.merge(merged_data, fta_data, how='left', left_on=['Partner', 'year'], right_on=['Country', 'year'])
    merged_data = pd.merge(merged_data, sg_gdp, how='left', on='year') #sgp gdp code


    # Process HS code
    if "HS Code" in merged_data.columns:
        merged_data["HS Code"] = merged_data["HS Code"].astype(str)
        merged_data["HS_Section"] = merged_data["HS Code"].str[:2]
        merged_data["HS_Section"] = merged_data["HS_Section"].fillna('00')
    else:
        print("'HS_Code' column is missing from merged data.")
        merged_data["HS_Section"] = '00'

    # One-hot encode HS sections
    ohe = OneHotEncoder(sparse_output=False, drop='first')
    hs_section_encoded = ohe.fit_transform(merged_data[['HS_Section']])
    hs_section_df = pd.DataFrame(hs_section_encoded, columns=ohe.get_feature_names_out(['HS_Section']))
    hs_section_df.index = merged_data.index
    merged_data = pd.concat([merged_data, hs_section_df], axis=1)
    hs_cols = list(hs_section_df.columns)

    # Create lag features
    merged_data["Import_Lag1"] = merged_data.groupby(["Partner", "Product"])['Imports'].shift(1)
    merged_data["Import_Lag2"] = merged_data.groupby(["Partner", "Product"])['Imports'].shift(2)
    merged_data["Import_Lag3"] = merged_data.groupby(["Partner", "Product"])['Imports'].shift(3)
    merged_data["GDP_Lag1"] = merged_data.groupby(["Partner"])['GDP'].shift(1)
    merged_data["SG_Lag1"] = merged_data.groupby(["Partner"])['Singapore_GDP'].shift(1) #sgp gdp code

    # Add GDP growth rate as a feature
    # merged_data["GDP_Growth"] = merged_data.groupby(["Partner"])['GDP'].pct_change()

    # Add exchange rate change as a feature
    merged_data["ExRate_Change"] = merged_data.groupby(["Partner"])['Exchange Rate (per US$)'].pct_change()

    # dd time-based features
    # merged_data["Time_Since_FTA"] = merged_data["year"] - merged_data.groupby("Partner")["Adjusted_value"].cumsum()
    # merged_data["Time_Since_FTA"] = merged_data["Time_Since_FTA"].fillna(0)

    # Add interaction terms
    # if add_interactions:
    #     merged_data["GDP_x_FTA"] = merged_data["GDP_Lag1"] * merged_data["Adjusted_value"]
    #     merged_data["GDP_x_IdealPoint"] = merged_data["GDP_Lag1"] * merged_data["IdealPointDistance"]

    # Remove rows with missing values
    merged_data = merged_data.dropna()
    merged_data = merged_data.drop_duplicates()


    if log_transform:
        # Apply log transformations to relevant columns
        merged_data["log_Import_Lag1"] = np.log(merged_data["Import_Lag1"].clip(lower=1))
        merged_data["log_Import_Lag2"] = np.log(merged_data["Import_Lag2"].clip(lower=1))
        merged_data["log_Import_Lag3"] = np.log(merged_data["Import_Lag3"].clip(lower=1))
        merged_data["log_GDP_Lag1"] = np.log(merged_data["GDP_Lag1"].clip(lower=1))
        merged_data["log_SG_GDP_Lag1"] = np.log(merged_data["SG_GDP_Lag1"].clip(lower=1)) #sgp gdp code

        #features for model
        #grace: removed GDP_Growth
        #grace: removed Time_Since_FTA
        feature_cols = [
                        "log_Import_Lag1", 
                        "IdealPointDistance", "log_GDP_Lag1", 'log_SG_GDP_Lag1',
                        'Exchange Rate (per US$)', 'Adjusted_value'] #added sgp gdp code

        # if add_interactions:
        #     feature_cols.extend(["GDP_x_FTA", "GDP_x_IdealPoint"])

        # Add HS code columns
        #feature_cols.extend(hs_cols)

        X = merged_data[feature_cols]
        y = np.log(merged_data["Imports"].clip(lower=1))
    else:
        #grace: removed GDP_Growth
        #grace: removed Time_Since_FTA
        feature_cols = [
                        "Import_Lag1",
                        "IdealPointDistance", "GDP_Lag1", 'SG_GDP_Lag1',
                        'Exchange Rate (per US$)', 'Adjusted_value'] #added sgp gdp code

        # if add_interactions:
        #     feature_cols.extend(["GDP_x_FTA", "GDP_x_IdealPoint"])

        #feature_cols.extend(hs_cols)
        X = merged_data[feature_cols]
        y = merged_data["Imports"]
        
        # Remove rows with missing values
        merged_data = merged_data.dropna()
        merged_data = merged_data.drop_duplicates()

    return X, y, merged_data

#%%


# model evaluation

def calculate_aic_bic(y_true, y_pred, n_params):
    log_likelihood = -0.5 * np.sum((y_true - y_pred) ** 2)
    n = len(y_true)
    aic = 2 * n_params - 2 * log_likelihood
    bic = np.log(n) * n_params - 2 * log_likelihood
    return aic, bic

def evaluate_model(model, X, y, cv_split, model_name="Model"):
    r2_scores = []
    mae_scores = []
    mse_scores = []
    aic_values = []
    bic_values = []
    all_y_test = []
    all_y_pred = []

    n_params = X.shape[1] if hasattr(model, 'coef_') else X.shape[1] + 1  # Approximation for non-linear models

    for train_index, test_index in cv_split.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2_scores.append(r2_score(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        mse_scores.append(mean_squared_error(y_test, y_pred))

        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

        aic, bic = calculate_aic_bic(y_test, y_pred, n_params)
        aic_values.append(aic)
        bic_values.append(bic)

    # Summary statistics
    print(f"\n{model_name} Results (TimeSeriesSplit, n_splits=5)")
    print(f"Average R²: {np.mean(r2_scores):.4f}")
    print(f"Average MAE: {np.mean(mae_scores):.2f}")
    print(f"Average MSE: {np.mean(mse_scores):.2f}")
    print(f"Average AIC: {np.mean(aic_values):.2f}")
    print(f"Average BIC: {np.mean(bic_values):.2f}")

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(all_y_test, all_y_pred, alpha=0.6)
    plt.plot([min(all_y_test), max(all_y_test)],
             [min(all_y_test), max(all_y_test)],
             color='red', linestyle='--')
    plt.xlabel("Actual Trade Volume")
    plt.ylabel("Predicted Trade Volume")
    plt.title(f"Actual vs Predicted Trade Value ({model_name})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Create residuals plot
    residuals = np.array(all_y_test) - np.array(all_y_pred)
    plt.figure(figsize=(8, 6))
    plt.scatter(all_y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Predicted Trade Volume")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title(f"Residuals vs Predicted ({model_name})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {
        'r2': np.mean(r2_scores),
        'mae': np.mean(mae_scores),
        'mse': np.mean(mse_scores),
        'aic': np.mean(aic_values),
        'bic': np.mean(bic_values),
    }

#%%

# Fine tuning

def tune_model_hyperparameters(X, y):
    param_grid_ridge = {'ridge__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    param_grid_lasso = {'lasso__alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}
    param_grid_rf = {
        'rf__n_estimators': [50, 75, 100],
        'rf__max_depth': [5, 10, 15],
        'rf__min_samples_split': [4, 6, 8]
    }
    param_grid_gb = {
        'gb__n_estimators': [50, 75, 100],
        'gb__learning_rate': [0.05, 0.1, 0.15],
        'gb__max_depth': [2, 3, 4, 5],
        'gb__subsample': [0.8, 1.0]  # Add subsample for regularization
    }

    ridge_pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
    lasso_pipe = Pipeline([("scaler", StandardScaler()), ("lasso", Lasso())])
    rf_pipe = Pipeline([("rf", RandomForestRegressor(random_state=42))])
    gb_pipe = Pipeline([("gb", GradientBoostingRegressor(random_state=42))])

    grid_ridge = GridSearchCV(ridge_pipe, param_grid_ridge, cv=5, scoring='r2')
    grid_lasso = GridSearchCV(lasso_pipe, param_grid_lasso, cv=5, scoring='r2')
    grid_rf = GridSearchCV(rf_pipe, param_grid_rf, cv=5, scoring='r2')
    grid_gb = GridSearchCV(gb_pipe, param_grid_gb, cv=5, scoring='r2')

    print("Tuning Ridge Regression...")
    grid_ridge.fit(X, y)
    print(f"Best Ridge parameters: {grid_ridge.best_params_}, Best score: {grid_ridge.best_score_:.4f}")

    print("Tuning Lasso Regression...")
    grid_lasso.fit(X, y)
    print(f"Best Lasso parameters: {grid_lasso.best_params_}, Best score: {grid_lasso.best_score_:.4f}")

    print("Tuning Random Forest...")
    grid_rf.fit(X, y)
    print(f"Best Random Forest parameters: {grid_rf.best_params_}, Best score: {grid_rf.best_score_:.4f}")

    print("Tuning Gradient Boosting...")
    grid_gb.fit(X, y)
    print(f"Best Gradient Boosting parameters: {grid_gb.best_params_}, Best score: {grid_gb.best_score_:.4f}")

    return {
        'ridge': grid_ridge.best_estimator_,
        'lasso': grid_lasso.best_estimator_,
        'rf': grid_rf.best_estimator_,
        'gb': grid_gb.best_estimator_
    }


def evaluate_model_by_year(model, X, y, merged_data):
    print("\n Year-by-Year Evaluation (True Time-based Split):")
    unique_years = sorted(merged_data['year'].unique())

    for test_year in unique_years[-5:]:
        train_mask = merged_data['year'] < test_year
        test_mask = merged_data['year'] == test_year

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"Year {test_year} - R²: {r2_score(y_test, y_pred):.4f}")


def analyze_feature_importance(X, y, model, model_name):
    model.fit(X, y)

    if hasattr(model, 'named_steps') and 'gb' in model.named_steps:
        importance = model.named_steps['gb'].feature_importances_
    elif hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
    else:
        print(f"Cannot extract feature importance from {model_name}")
        return None

    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
    plt.title(f'Top 15 Features Importance - {model_name}')
    plt.tight_layout()
    plt.show()

    return feature_importance


def compare_models(models_dict, results_dict):
    model_names = list(results_dict.keys())
    r2_values = [results_dict[name]['r2'] for name in model_names]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, r2_values)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Baseline R² (0.5)')
    plt.ylabel('R² Score')
    plt.title('Model Performance Comparison')
    plt.ylim(0, 1)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.4f}', ha='center', va='bottom')

    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    X, y, merged_data = prepare_data_for_regression(log_transform=True, add_interactions=True)
    print(f"Number of datapoints: {len(X)}")
    print(f"Features included: {X.columns.tolist()}")

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    best_models = tune_model_hyperparameters(X_scaled, y)

    results = {
        'Linear Regression': evaluate_model(LinearRegression(), X_scaled, y, TimeSeriesSplit(n_splits=5), "Linear Regression"),
        'Ridge Regression (tuned)': evaluate_model(best_models['ridge'], X_scaled, y, TimeSeriesSplit(n_splits=5), "Ridge (tuned)"),
        'Lasso Regression (tuned)': evaluate_model(best_models['lasso'], X_scaled, y, TimeSeriesSplit(n_splits=5), "Lasso (tuned)"),
        'Random Forest (tuned)': evaluate_model(best_models['rf'], X_scaled, y, TimeSeriesSplit(n_splits=5), "Random Forest (tuned)"),
        'Gradient Boosting (tuned)': evaluate_model(best_models['gb'], X_scaled, y, TimeSeriesSplit(n_splits=5), "Gradient Boosting (tuned)")
    }

    best_model = best_models['gb']
    best_model_name = "Gradient Boosting (tuned)"

    evaluate_model_by_year(best_model, X_scaled, y, merged_data)

    feature_importance = analyze_feature_importance(X_scaled, y, best_model, best_model_name)

    joblib.dump(best_model, "best_model.pkl")
    print("Model saved to 'best_model.pkl'.")
    joblib.dump(scaler, "scaler.pkl")
    print("Scaler saved to 'scaler.pkl'.")

    print("\nModel Comparison Summary:")
    for name, result in results.items():
        print(f"{name}: R² = {result['r2']:.4f}, AIC = {result['aic']:.2f}, BIC = {result['bic']:.2f}")

    compare_models(best_models, results)

    return best_model, results[best_model_name], feature_importance


if __name__ == "__main__":
    best_model, best_performance, top_features = main()
    print(f"\nBest Model achieved R² of {best_performance['r2']:.4f}")
    print("Top 10 most important features:")
    print(top_features.head(10))



#%%

# handover to frontend: predictions

def get_merged_data():
    _, _, merged_data = prepare_data_for_regression()
    return merged_data

merged_data = get_merged_data()
print(merged_data)

#CHECKING USA
#usa_in_country1 = merged_data['Country1'].str.contains("United States of America", na=False)
#usa_in_country2 = merged_data['Country2'].str.contains("United States of America", na=False)
#usa_rows = merged_data[usa_in_country1 | usa_in_country2]
#print(usa_rows)

def predict_2024_with_correction():
    import joblib
    from sklearn.linear_model import LinearRegression
    import numpy as np
    import pandas as pd

    # Load model and scaler
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")

    # Get merged data
    _, _, merged_data = prepare_data_for_regression(log_transform=True, add_interactions=True)

    # Use only one HS_Section per Partner
    data_2023 = merged_data[merged_data['year'] == 2023].copy()
    data_2023 = data_2023.drop_duplicates(subset=["Partner", "HS_Section"])
    actual_2023_totals = (
        data_2023.groupby("Partner")["Imports"]
        .sum()
        .reset_index()
        .rename(columns={"Imports": "Actual_Import_2023"})
    )

    # Create input for 2024
    predict_data = data_2023.copy()
    predict_data['year'] = 2024

    # Manually set lag values and features
    predict_data["log_Import_Lag1"] = np.log(data_2023["Imports"].clip(lower=1))
    predict_data["log_GDP_Lag1"] = data_2023["log_GDP_Lag1"]
    predict_data["log_SG_GDP_Lag1"] = data_2023["log_SG_GDP_Lag1"]
    predict_data["Exchange Rate (per US$)"] = data_2023["Exchange Rate (per US$)"]
    predict_data["Adjusted_value"] = data_2023["Adjusted_value"]
    predict_data["IdealPointDistance"] = data_2023["IdealPointDistance"]

    feature_cols = [
        "log_Import_Lag1", 
        "IdealPointDistance", "log_GDP_Lag1", "log_SG_GDP_Lag1",
        "Exchange Rate (per US$)", "Adjusted_value"
    ]

    X_2024 = predict_data[feature_cols]
    X_2024_scaled = pd.DataFrame(scaler.transform(X_2024), columns=X_2024.columns)

    # Make raw predictions in log space and convert to raw
    log_preds = model.predict(X_2024_scaled)
    raw_preds = np.exp(log_preds)

    results = predict_data[["Partner", "Product"]].copy()
    results["Raw_Predicted_Import_2024"] = raw_preds

    # Sum predicted imports per partner
    pred_2024_totals = (
        results.groupby("Partner")["Raw_Predicted_Import_2024"]
        .sum()
        .reset_index()
    )

    # Merge with actual 2023
    comparison_df = pd.merge(pred_2024_totals, actual_2023_totals, on="Partner", how="outer")

    # Step 1: Compute log values
    comparison_df["log_Actual_Import_2023"] = np.log(comparison_df["Actual_Import_2023"].clip(lower=1))
    comparison_df["log_Raw_Predicted_Import_2024"] = np.log(comparison_df["Raw_Predicted_Import_2024"].clip(lower=1))

    # Step 2: Fit residual correction model in log space
    X_log = comparison_df[["log_Raw_Predicted_Import_2024"]]
    y_log = comparison_df["log_Actual_Import_2023"] - comparison_df["log_Raw_Predicted_Import_2024"]
    correction_model = LinearRegression().fit(X_log, y_log)

    # Step 3: Apply correction in log space
    log_adjustment = correction_model.predict(X_log)
    comparison_df["log_Corrected_Predicted_Import_2024"] = comparison_df["log_Raw_Predicted_Import_2024"] + log_adjustment
    comparison_df["Corrected_Predicted_Import_2024"] = np.exp(comparison_df["log_Corrected_Predicted_Import_2024"])

    # % Change Calculations
    comparison_df["% Change (Raw)"] = (
        (comparison_df["Raw_Predicted_Import_2024"] - comparison_df["Actual_Import_2023"])
        / comparison_df["Actual_Import_2023"]
        * 100
    )
    comparison_df["% Change (Corrected)"] = (
        (comparison_df["Corrected_Predicted_Import_2024"] - comparison_df["Actual_Import_2023"])
        / comparison_df["Actual_Import_2023"]
        * 100
    )

    print("\n2024 Raw vs Corrected Predictions (Top 10 by Corrected Volume):")
    print(
        comparison_df.sort_values("Corrected_Predicted_Import_2024", ascending=False)
        .head(10)
        .round(2)
    )

    return results, comparison_df



predictions_2024 = predict_2024_with_correction()

