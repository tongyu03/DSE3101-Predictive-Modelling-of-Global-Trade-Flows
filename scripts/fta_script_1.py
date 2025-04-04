import pandas as pd
import numpy as np

# Read files
fta_data = pd.read_csv("data/raw data/Free Trade Agreement Dataset.csv")
tariff_weights_2016 = pd.read_csv("data/raw data/tariff_weights_2016.csv")

# Merge with tariff weights
fta_data = pd.merge(fta_data, tariff_weights_2016[['Country', 'tariff_weight']], 
                    left_on='Country', right_on='Country', how='left')

fta_data.rename(columns={'tariff_weight': 'tariff_weight_2016_Country'}, inplace=True)

fta_data = pd.merge(fta_data, tariff_weights_2016[['Country', 'tariff_weight']], 
                    left_on='Partner Country', right_on='Country', how='left')

fta_data.rename(columns={'tariff_weight': 'tariff_weight_2016_Partner'}, inplace=True)

fta_data = fta_data.fillna(0)

# Formula for Adjusted_value
fta_data['Adjusted_value'] = np.where(
    fta_data['Year'] < 2017,  
    (fta_data['FTA Weight']  - 
     (fta_data['Tariff %']/100 * fta_data['Weight in Exports']/100 * fta_data['tariff_weight_2016_Country']) -
     (fta_data['Tariff %']/100 * fta_data['Weight in Exports']/100 * fta_data['tariff_weight_2016_Partner'])),  # If True (Year < 2017)
    fta_data['FTA Weight']  - (fta_data['Tariff %']/100 * fta_data['Weight in Exports']/100)  # If False (Year >= 2017), no tariff weight applied
)

# Ensure Adjusted_value is non-negative
fta_data['Adjusted_value'] = fta_data['Adjusted_value'].clip(lower=0)

# Cleaning
fta_data.rename(columns={'Country_x': 'Country'}, inplace=True)

# Select only the relevant columns: Country, Partner Country, Adjusted_value
final_data = fta_data[['Year', 'Country', 'Partner Country', 'Adjusted_value']]

# Create a duplicate where Country and Partner Country are swapped
swapped_data = final_data.copy()
swapped_data.rename(columns={'Country': 'Partner Country', 'Partner Country': 'Country'}, inplace=True)

# Concatenate the original data with the swapped data
final_data = pd.concat([final_data, swapped_data], ignore_index=True)

# Save the final data to a CSV
final_data.to_csv("data/cleaned data/adjusted_fta_data_2.csv", index=False)

unique_combinations = final_data[['Country', 'Partner Country']].drop_duplicates()
print(unique_combinations)