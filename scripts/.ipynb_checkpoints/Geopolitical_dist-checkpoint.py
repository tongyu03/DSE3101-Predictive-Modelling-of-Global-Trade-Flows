import pandas as pd
import numpy as np
import ast

def process_unga_data():
    unga = pd.read_csv("data/cleaned data/unga_voting_3.csv")
    unga['CountryPair'] = unga['CountryPair'].apply(lambda x: ast.literal_eval(x))
    unga[['Country1', 'Country2']] = pd.DataFrame(unga['CountryPair'].tolist(), index=unga.index)
    unga = unga[(unga['year'] >= 1989) & (unga['year'] <= 2021)]
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
    gdp_long = countries.merge(gdp_long, left_on='StateNme', right_on='Country Name', how='inner')
    gdp_long = gdp_long.drop(gdp_long.columns[[0, 1, 2]], axis=1)
    gdp_long['Year'] = gdp_long['Year'].astype(int)
    gdp_long['Country Name'] = gdp_long['Country Name'].astype(str)
    return gdp_long

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
    fta_sg = fta[
        (fta['Partner Country'].isin(['SG', 'SGP']))
    ]
    fta_sg['Country Code'] = fta_sg.apply(
        lambda row: row['Country'] if row['Country'] not in ['SG', 'SGP'] else row['Partner Country'],
        axis=1
    )
    fta_sg = fta_sg.drop(columns=["Country", "Partner Country"])
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

trade_data_geo = pd.read_csv("data/cleaned data/10 years Trade Product Data.csv")
unga_data_geo = process_unga_data()
gdp_data_geo = process_gdp_data()
exrate_data_geo = process_exrate_data()
fta_data_geo = process_FTA_data()

trade_data_geo.columns = trade_data_geo.columns.str.strip()
gdp_data_geo.columns = gdp_data_geo.columns.str.strip()
exrate_data_geo.columns = exrate_data_geo.columns.str.strip()
fta_data_geo.columns = fta_data_geo.columns.str.strip()
# Rename columns to ensure consistent merging
trade_data_geo = trade_data_geo.rename(columns={"Year": "year"})
trade_data_geo = trade_data_geo.rename(columns={"Country": "Partner"})
gdp_data_geo = gdp_data_geo.rename(columns={"Year": "year"})
exrate_data_geo = exrate_data_geo.rename(columns={"Year": "year"})
fta_data_geo = fta_data_geo.rename(columns={"Year": "year"})

merged_data_geo = pd.merge(unga_data_geo, trade_data_geo, how='left', left_on=['year', 'Partner'], right_on=['year', 'Partner'])
merged_data_geo = pd.merge(merged_data_geo, gdp_data_geo, how='left', left_on=['Partner', 'year'], right_on=['Country Name', 'year'])
merged_data_geo = pd.merge(merged_data_geo, exrate_data_geo, how='left', left_on=['Partner', 'year'], right_on=['Country Name', 'year'])
merged_data_geo = pd.merge(merged_data_geo, fta_data_geo, how='left', left_on=['Partner', 'year'], right_on=['Country', 'year'])

#print(merged_data_geo.columns)
#print(merged_data_geo.head())

if "HS Code" in merged_data_geo.columns:
    merged_data_geo["HS Code"] = merged_data_geo["HS Code"].astype(str)  # Ensure it's a string
    merged_data_geo["HS_Section"] = merged_data_geo["HS Code"].str[:2]  # first 2 digits of HS code

    # Check for any NaN or invalid values
    merged_data_geo["HS_Section"] = merged_data_geo["HS_Section"].fillna('00')  # Fill NaN with '00' or any placeholder you prefer
else:
    print("'HS_Code' column is missing from merged data.")
    merged_data_geo["HS_Section"] = '00'  # Handle the case where HS_Code is missing

# Lag features for GDP
merged_data_geo["GDP_Lag1"] = merged_data_geo.groupby(["Partner"])['GDP'].shift(1)

# drop empty data
merged_data_geo = merged_data_geo.dropna()


Geopol_df = merged_data_geo[["Country", "year", "IdealPointDistance", "GDP_Lag1",
                 'Exchange Rate (per US$)', 'Adjusted_value','Imports', 'Exports']]

# Define a geopolitical score

def get_geopolitical_data(country, year):
    # Filter the DataFrame for the specified country
    country_data = Geopol_df[Geopol_df['Country'] == country and Geopol_df['year'] == year].copy()
    
    # Calculate the geopolitical score
    country_data['Geopolitical_Score'] = (
        country_data['IdealPointDistance'] +
        country_data['GDP_Lag1'] +
        country_data['Exchange Rate (per US$)'] +
        country_data['Adjusted_value'] 
    )
    
    return country_data[['Country', 'year', 'Geopolitical_Score']]
    

