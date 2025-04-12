import pandas as pd
import numpy as np
import ast
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Mapping from ISO3 to country names
iso3_to_country = {
    'CHN': 'China',
    'HKG': 'Hong Kong',
    'JPN': 'Japan',
    'KOR': 'South Korea',
    'MYS': 'Malaysia',
    'SAU': 'Saudi Arabia',
    'THA': 'Thailand',
    'USA': 'USA',
    'IDN': 'Indonesia'
}

# Process UNGA voting data
def process_unga_data():
    unga = pd.read_csv("data/cleaned data/unga_voting_3.csv")
    unga['CountryPair'] = unga['CountryPair'].apply(lambda x: ast.literal_eval(x))
    unga[['Country1', 'Country2']] = pd.DataFrame(unga['CountryPair'].tolist(), index=unga.index)
    unga['Country1'] = unga['Country1'].replace("United States of America", "USA")
    unga['Country2'] = unga['Country2'].replace("United States of America", "USA")
    unga = unga[(unga['year'] >= 1989) & (unga['year'] <= 2021)]
    unga_sg = unga[(unga['Country1'] == 'Singapore') | (unga['Country2'] == 'Singapore')]
    unga_sg['Partner'] = unga_sg.apply(lambda row: row['Country1'] if row['Country1'] != 'Singapore' else row['Country2'], axis=1)
    unga_sg = unga_sg[['agree', 'year', 'IdealPointDistance', 'Country1', 'Country2', 'Partner']]
    unga_sg['year'] = unga_sg['year'].astype(int)
    return unga_sg

# Process GDP data
def process_gdp_data():
    gdp = pd.read_csv("data/raw data/GDP.csv", header = 2)
    gdp = gdp.drop(gdp.columns[[1, 2, 3]], axis=1)
    gdp_long = gdp.melt(id_vars=['Country Name'], var_name='Year', value_name='GDP')
    gdp_long = gdp_long.dropna(subset=['GDP'])
    gdp_long['Country Name'] = gdp_long['Country Name'].replace({
        'United States': 'United States of America'
    })
    countries = pd.read_csv("data/raw data/COW-country-codes.csv")
    gdp_long = countries.merge(gdp_long, left_on='StateNme', right_on='Country Name', how='inner')
    gdp_long = gdp_long.drop(gdp_long.columns[[0, 1, 2]], axis=1)
    gdp_long['Year'] = gdp_long['Year'].astype(int)
    gdp_long['Country Name'] = gdp_long['Country Name'].astype(str)
    return gdp_long

# Process Exchange Rate data
def process_exrate_data():
    exrate = pd.read_csv("data/raw data/exchange_rate.csv", header = 4)
    exrate = exrate.drop(exrate.columns[[1, 2, 3]], axis=1)
    exrate_long = exrate.melt(id_vars=['Country Name'], var_name='Year', value_name='Exchange Rate (per US$)')
    exrate_long = exrate_long.dropna()
    exrate_long['Year'] = exrate_long['Year'].astype(int)
    exrate_long['Country Name'] = exrate_long['Country Name'].astype(str)
    exrate_long['Country Name'] = exrate_long['Country Name'].replace({
        "Hong Kong SAR, China": "Hong Kong",
        "Korea, Rep.": "South Korea",
        "United States": "USA"
    })
    return exrate_long

# Process FTA data
def process_FTA_data():
    fta = pd.read_csv("data/cleaned data/adjusted_fta_data_2.csv")
    fta_sg = fta[(fta['Partner Country'].isin(['SG', 'SGP']))]
    fta_sg['Country Code'] = fta_sg.apply(lambda row: row['Country'] if row['Country'] not in ['SG', 'SGP'] else row['Partner Country'], axis=1)
    fta_sg = fta_sg.drop(columns=["Country", "Partner Country"])
    fta_sg['Country'] = fta_sg['Country Code'].replace(iso3_to_country)
    fta_sg = fta_sg.sort_values(by='Year')
    return fta_sg

# Merge all data
def merge_all_data():
    trade_data_geo = pd.read_csv("data/cleaned data/10 years Trade Product Data.csv")
    unga_data_geo = process_unga_data()
    gdp_data_geo = process_gdp_data()
    exrate_data_geo = process_exrate_data()
    fta_data_geo = process_FTA_data()

    trade_data_geo.columns = trade_data_geo.columns.str.strip()
    gdp_data_geo.columns = gdp_data_geo.columns.str.strip()
    exrate_data_geo.columns = exrate_data_geo.columns.str.strip()
    fta_data_geo.columns = fta_data_geo.columns.str.strip()

    # Rename columns for merging consistency
    trade_data_geo = trade_data_geo.rename(columns={"Year": "year"})
    trade_data_geo["Country"] = trade_data_geo["Country"].replace({
        "China, Hong Kong SAR": "Hong Kong",
        "Rep. of Korea": "South Korea"
    })
    trade_data_geo = trade_data_geo.rename(columns={"Country": "Partner"})
    gdp_data_geo = gdp_data_geo.rename(columns={"Year": "year"})
    gdp_data_geo['Country Name'] = gdp_data_geo['Country Name'].replace({
        "United States of America": "USA"
    })
    exrate_data_geo = exrate_data_geo.rename(columns={"Year": "year"})
    fta_data_geo = fta_data_geo.rename(columns={"Year": "year"})

    # Merge data
    merged_data_geo = pd.merge(unga_data_geo, trade_data_geo, how='left', left_on=['year', 'Partner'], right_on=['year', 'Partner'])
    merged_data_geo = pd.merge(merged_data_geo, gdp_data_geo, how='left', left_on=['Partner', 'year'], right_on=['Country Name', 'year'])
    merged_data_geo = pd.merge(merged_data_geo, exrate_data_geo, how='left', left_on=['Partner', 'year'], right_on=['Country Name', 'year'])
    merged_data_geo = pd.merge(merged_data_geo, fta_data_geo, how='left', left_on=['Partner', 'year'], right_on=['Country', 'year'])

    # Lag features for GDP
    merged_data_geo["GDP_Lag1"] = merged_data_geo.groupby(["Partner"])['GDP'].shift(1)

    # Drop empty data
    merged_data_geo = merged_data_geo.dropna()

    # Log exchange rate
    merged_data_geo['Exchange Rate (per US$)_scaled'] = np.log1p(merged_data_geo['Exchange Rate (per US$)'])

    # Group and aggregate data
    Geopol_df = merged_data_geo.groupby(["Country", "year"]).agg({
        "Imports": "sum",
        "Exports": "sum",
        "IdealPointDistance": "median",
        "GDP_Lag1": "median",
        "Exchange Rate (per US$)_scaled": "median",
        "Adjusted_value": "median",
        "FTA_binary": "median"
    }).reset_index()

    return Geopol_df

# Define and train the model
def train_model(Geopol_df):
    # Prepare training data
    X = Geopol_df[['IdealPointDistance', 'GDP_Lag1', 'Exchange Rate (per US$)_scaled', 'Adjusted_value', 'FTA_binary']]
    y = Geopol_df['Imports']  # Assuming you're predicting imports as an example target

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define model pipeline with scaling and Ridge regression
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=1.0))
    ])

    # Train the model
    pipeline.fit(X_train, y_train)
    
    return pipeline, X.columns

# Get the coefficients
def get_model_coefficients(pipeline, feature_names):
    model = pipeline.named_steps['reg']
    coefficients = model.coef_
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })
    return coef_df




import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

# Assuming you have a trained Ridge model pipeline

# Define the features and their respective model coefficients
features = ['IdealPointDistance', 'GDP_Lag1', 'Exchange Rate (per US$)_scaled', 'Adjusted_value', 'FTA_binary']

# Assuming your model pipeline has already been trained and is available
# For example:
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('reg', Ridge(alpha=1.0))
])

# Model training (you must replace this part with your trained pipeline)
# pipeline.fit(X_train, y_train)  # Where X_train and y_train are your training data

# Define a function to calculate the geopolitical score for a given country and year
def calculate_geopolitical_score(country_list, year_list, pipeline, data):
    result = []
    
    for country in country_list:
        for year in year_list:
            country_data = data[(data['Country'] == country) & (data['year'] == year)].copy()

            if country_data.empty:
                result.append({"Country": country, "year": year, "Geopolitical_Score": "No data"})
                continue

            X_country = country_data[features]
            X_country_scaled = pipeline.named_steps['scaler'].transform(X_country)

            geo_score = np.dot(X_country_scaled, pipeline.named_steps['reg'].coef_) + pipeline.named_steps['reg'].intercept_
            country_data['Geopolitical_Score'] = geo_score
            result.append({"Country": country, "year": year, "Geopolitical_Score": geo_score[0]})

    return pd.DataFrame(result)

# Example usage for a country and a specific year
country = 'China'  # Replace with any country of interest
year = 2020  # Replace with the year of interest

# Assume you have your merged dataset named 'Geopol_df'
# Geopol_df = merge_all_data()  # This should be your actual data

# Calculate the geopolitical score
geopolitical_score_df = calculate_geopolitical_score(country, year, pipeline, Geopol_df)

print(geopolitical_score_df)
