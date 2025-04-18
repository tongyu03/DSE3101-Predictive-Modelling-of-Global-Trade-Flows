
"""
@author: shannen
Exports
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
import seaborn as sns
import joblib
from sklearn.inspection import permutation_importance

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
    trade_data = pd.read_csv("data/cleaned data/Concatenated_Trade_Data.csv")
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
    merged_data["Export_Lag1"] = merged_data.groupby(["Partner"])['Exports'].shift(1)
    merged_data["Export_Lag2"] = merged_data.groupby(["Partner"])['Exports'].shift(2)
    merged_data["Export_Lag3"] = merged_data.groupby(["Partner"])['Exports'].shift(3)
    merged_data["GDP_Lag1"] = merged_data.groupby(["Partner"])['GDP'].shift(1)
    #merged_data["SG_Lag1"] = merged_data.groupby(["Partner"])['Singapore_GDP'].shift(1) #sgp gdp code

    # Add GDP growth rate as a feature
    # merged_data["GDP_Growth"] = merged_data.groupby(["Partner"])['GDP'].pct_change()

    # Add exchange rate change as a feature
    merged_data["ExRate_Change"] = merged_data.groupby(["Partner"])['Exchange Rate (per US$)'].pct_change()

    # dd time-based features
    # merged_data["Time_Since_FTA"] = merged_data["year"] - merged_data.groupby("Partner")["Adjusted_value"].cumsum()
    # merged_data["Time_Since_FTA"] = merged_data["Time_Since_FTA"].fillna(0)

    # Add interaction terms
    if add_interactions:
        merged_data["GDP_x_FTA"] = merged_data["GDP"] * merged_data["Adjusted_value"]
        merged_data["GDP_x_IdealPoint"] = merged_data["GDP"] * merged_data["IdealPointDistance"]

    # Remove rows with missing values
    merged_data = merged_data.dropna()
    merged_data = merged_data.drop_duplicates()

    if log_transform:
        # Apply log transformations to relevant columns
        merged_data["log_Export_Lag1"] = np.log(merged_data["Export_Lag1"].clip(lower=1))
        merged_data["log_Export_Lag2"] = np.log(merged_data["Export_Lag2"].clip(lower=1))
        merged_data["log_Export_Lag3"] = np.log(merged_data["Export_Lag3"].clip(lower=1))
        merged_data["log_GDP_Lag1"] = np.log(merged_data["GDP_Lag1"].clip(lower=1))
        merged_data["log_SG_GDP_Lag1"] = np.log(merged_data["SG_GDP_Lag1"].clip(lower=1)) #sgp gdp code

        #features for model
        #grace: removed GDP_Growth
        #grace: removed Time_Since_FTA
        feature_cols = ["year","log_Export_Lag1", "log_Export_Lag2", "log_Export_Lag3",
                        "IdealPointDistance", "log_GDP_Lag1", 'log_SG_GDP_Lag1',
                        'Exchange Rate (per US$)', 'Adjusted_value',
                        'ExRate_Change'] #added sgp gdp code

        if add_interactions:
            feature_cols.extend(["GDP_x_FTA", "GDP_x_IdealPoint"])

        # Add HS code columns
        feature_cols.extend(hs_cols)

        X = merged_data[feature_cols]
        y = np.log(merged_data["Exports"].clip(lower=1))
    else:
        #grace: removed GDP_Growth
        #grace: removed Time_Since_FTA
        feature_cols = ["Export_Lag1", "Export_Lag2", "Export_Lag3",
                        "IdealPointDistance", "GDP_Lag1", 'SG_GDP_Lag1',
                        'Exchange Rate (per US$)', 'Adjusted_value',
                        'ExRate_Change'] #added sgp gdp code

        if add_interactions:
            feature_cols.extend(["GDP_x_FTA", "GDP_x_IdealPoint"])

        feature_cols.extend(hs_cols)
        X = merged_data[feature_cols]
        y = merged_data["Exports"]

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

#fine tuning

# grid search CV
from sklearn.pipeline import Pipeline

def tune_model_hyperparameters(X, y, cv_split):
    # Define parameter grids for each model using pipeline names
    param_grid_ridge = {
        'ridge__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
    }

    param_grid_lasso = {
        'lasso__alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
    }

    param_grid_rf = {
        'rf__n_estimators': [50, 75, 100],
        'rf__max_depth': [5, 10, 15],
        'rf__min_samples_split': [4, 6, 8]
    }

    param_grid_gb = {
        'gb__n_estimators': [50, 75, 100],
        'gb__learning_rate': [0.05, 0.1, 0.15],
        'gb__max_depth': [2, 3, 4, 5]
    }

    # Define pipelines
    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge())
    ])

    lasso_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", Lasso())
    ])

    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),  # Not always needed for RF, but keeps consistent
        ("rf", RandomForestRegressor(random_state=42))
    ])

    gb_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("gb", GradientBoostingRegressor(random_state=42))
    ])

    # Set up GridSearchCV
    grid_ridge = GridSearchCV(ridge_pipe, param_grid_ridge, cv=cv_split, scoring='r2')
    grid_lasso = GridSearchCV(lasso_pipe, param_grid_lasso, cv=cv_split, scoring='r2')
    grid_rf = GridSearchCV(rf_pipe, param_grid_rf, cv=cv_split, scoring='r2')
    grid_gb = GridSearchCV(gb_pipe, param_grid_gb, cv=cv_split, scoring='r2')

    # Fit and report
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

# Feature importance
def analyze_feature_importance(X, y, model, model_name):
    model.fit(X, y)

    if hasattr(model, 'feature_importances_'):  # tree-based models
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):  # linear models
        importance = np.abs(model.coef_)
    else:
        print(f"Cannot extract feature importance from {model_name}")
        return

    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    # Plot top 15 features
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
    plt.title(f'Top 15 Features Importance - {model_name}')
    plt.tight_layout()
    plt.show()

    return feature_importance

# Polynomial features
def add_polynomial_features(X, degree=2):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(X.columns)
    X_poly_df = pd.DataFrame(X_poly, columns=feature_names)

    return X_poly_df

# Compare all models
def compare_models(models_dict, results_dict):
    model_names = list(results_dict.keys())
    r2_values = [results_dict[name]['r2'] for name in model_names]

    # Plot comparison
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, r2_values)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Baseline R² (0.5)')
    plt.ylabel('R² Score')
    plt.title('Model Performance Comparison')
    plt.ylim(0, 1)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')

    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # Prepare data with enhanced features
    X, y, merged_data = prepare_data_for_regression(log_transform=True, add_interactions=True)
    print(f"Number of datapoints: {len(X)}")
    print(f"Features included: {X.columns.tolist()}")

    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    # Evaluate base linear model (as benchmark)
    linear_model = LinearRegression()
    linear_results = evaluate_model(linear_model, X, y, tscv, "Linear Regression")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    # Hyperparameter tuning
    best_models = tune_model_hyperparameters(X_scaled, y, tscv)

    # Evaluate tuned models on scaled data
    results = {
        'Linear Regression': linear_results,
        'Ridge Regression (tuned)': evaluate_model(best_models['ridge'], X_scaled, y, tscv, "Ridge (tuned)"),
        'Lasso Regression (tuned)': evaluate_model(best_models['lasso'], X_scaled, y, tscv, "Lasso (tuned)"),
        'Random Forest (tuned)': evaluate_model(best_models['rf'], X_scaled, y, tscv, "Random Forest (tuned)"),
        'Gradient Boosting (tuned)': evaluate_model(best_models['gb'], X_scaled, y, tscv, "Gradient Boosting (tuned)")
    }

    # Feature importance for best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
    best_model = best_models.get(best_model_name.split()[0].lower(), linear_model)

    feature_importance = analyze_feature_importance(X_scaled, y, best_model, best_model_name)

    # Save best model and scaler
    joblib.dump(best_model, "best_model.pkl")
    print("Model saved to 'best_model.pkl'.")

    joblib.dump(scaler, "scaler.pkl")
    print("Scaler saved to 'scaler.pkl'.")

    # Compare all models
    print("\nModel Comparison Summary:")
    for name, result in results.items():
        print(f"{name}: R² = {result['r2']:.4f}, AIC = {result['aic']:.2f}, BIC = {result['bic']:.2f}")

    compare_models(best_models, results)

    # Return best model and performance metrics
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

X, _, _ = prepare_data_for_regression(log_transform=True, add_interactions=True)
X.columns.to_series().to_csv("feature_columns.csv", index=False)

def predict_export_value(year_or_range):
    import numbers
    import matplotlib.pyplot as plt

    # Load model & scaler
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_cols = pd.read_csv("feature_columns.csv").squeeze().tolist()

    # Get base data (latest available year for forecasting)
    merged_data = get_merged_data()
    latest_year = merged_data['year'].max()
    base_input = merged_data[merged_data['year'] == latest_year].copy()

    if base_input.empty:
        print(f"No data available for latest year ({latest_year})")
        return None

    # Determine prediction years
    if isinstance(year_or_range, numbers.Integral):
        years = [year_or_range]
    else:
        years = sorted(list(year_or_range))  # ensure chronological order

    if min(years) <= latest_year:
        raise ValueError(f"Prediction year(s) must start after {latest_year} (latest data year)")

    results = []
    current_input = base_input.copy()

    for target_year in years:
        input_df = current_input[expected_cols].copy()
        input_scaled = scaler.transform(input_df)

        # Predict log(exports)
        prediction_log = model.predict(input_scaled)

        # Cap log predictions
        prediction_log = np.clip(prediction_log, 18, 24)  # log bounds ≈ [65B, 2.6T]

        # Diagnostic plot
        plt.figure(figsize=(8, 5))
        plt.plot(prediction_log, marker='o')
        plt.axhline(np.log(1e11), color='red', linestyle='--', label='Upper Bound ($100B)')
        plt.axhline(np.log(1e6), color='green', linestyle='--', label='Lower Bound ($1M)')
        plt.title(f'Log Predictions for {target_year}')
        plt.ylabel('Log(Export Value)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Convert back to level values and clip
        prediction_actual = np.exp(prediction_log)
        prediction_actual = np.clip(prediction_actual, 1e6, 1e11)

        # Construct detailed product-level results
        result = current_input[['Partner', 'HS Code', 'HS_Section']].copy()
        result['Predicted Exports'] = prediction_actual
        result['Target Year'] = target_year

        # Aggregated country-level result
        agg = result.groupby('Partner', as_index=False)['Predicted Exports'].sum()
        agg['HS Code'] = 'All Products'
        agg['HS_Section'] = 'All'
        agg['Target Year'] = target_year
        agg = agg[result.columns]

        results.append(pd.concat([result, agg], ignore_index=True))

        # Roll forward for next iteration using current prediction
        next_input = current_input.copy()
        next_input['Export_Lag3'] = current_input['Export_Lag2']
        next_input['Export_Lag2'] = current_input['Export_Lag1']
        next_input['Export_Lag1'] = prediction_actual

        next_input['log_Export_Lag3'] = current_input['log_Export_Lag2']
        next_input['log_Export_Lag2'] = current_input['log_Export_Lag1']
        next_input['log_Export_Lag1'] = np.log(next_input['Export_Lag1'].clip(lower=1))

        current_input = next_input  # update for next year

    final_df = pd.concat(results).reset_index(drop=True)
    return final_df

predict_export_value(2024)
predict_export_value(range(2024, 2028))


# Get predictions from 2024 to 2027
#predictions = predict_export_value(2024)

# Save to CSV
#predictions.to_csv("predicted_exports_2024.csv", index=False)