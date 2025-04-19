# DSE3101-Predictive-Modelling-of-Global-Trade-Flows
Predictive modeling project for DSE3101, focusing on global trade flows using machine learning techniques. Includes data preprocessing, feature engineering, model development (e.g., time series, regression), and performance evaluation to analyze and forecast international trade volumes.


## Deployment
To deploy our dashboard, please complete the following instructions:
1) Ensure you have Docker installed on your laptop
2) Run the following command on your Command Prompt:
docker run -p 8000:8000 lowevie/trade-dashboard:latest
3) Open your browser and go to:
http://localhost:8000


## Description of key files and folders

### app.py
Shiny Script which contains our final trade dashboard. Instructions to run and view are as stated above

### ML_Models.py
Trained machine learning models for predicting import volume. Run sequentially to load and display results.

### ML_Exports.py
Trained machine learning models for predicting export volume. Usage is similar to ML_Models.py.

## Key Feature Engineering Scripts
These scripts contain pre-processed datasets used as inputs for the final models:

### unga_voting_script_2.py - Geopolitical Distance
Processed UN General Assembly voting data to measure political alignment via Ideal Point Distance. Extracted bilateral data focused on Singapore, retaining agreement scores and distance metrics.

### er_script_1.py - Exchange rate
Transformed exchange rate data into long format, standardized country names (e.g., "Korea, Rep." → "South Korea"), and removed missing values. Produced a clean annual time series of exchange rates (per USD) by country.

### fta_script_1.py - Free Trade Agreement and Tariff Dataset 
Reshaped GDP data into long format, standardized country names, merged with country codes, and filtered for years from 2013 onwards to align with trade data.

### gdp_script_1.py - Gross Domestic Product (GDP)
Filtered for Singapore only, transposed data to set years as index, renamed GDP column to "Singapore_GDP", and removed the first row used for renaming.

### sg_gdp.py - Singapore GDP
Filtered for Singapore, transposed to set years as index, renamed GDP to "Singapore_GDP", and removed the initial row used for renaming.

### add_trade_clean.py - Historical Import and Export data 
Extracted UN Comtrade trade data (2013–2023), reshaped into long format, and cleaned to ensure compatibility for merging into the final models.

## Model Validation

We used time series cross-validation to evaluate model performance while respecting the temporal order of the data. Unlike traditional k-fold cross-validation, this method preserves the time-dependent structure by training on past data and validating on future data.

Specifically, we used a rolling-window approach, where each training set included data up to a certain year, and the following years were used for validation. This mimics real-world forecasting and avoids lookahead bias.
Performance metrics such as RMSE and R² were averaged across validation windows to assess model stability over time.