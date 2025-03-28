import pandas as pd
import numpy as np
#author: junlu
# Read files
fta_data = pd.read_csv("data/Free Trade Agreement Dataset.csv")
tariff_weights_2016 = pd.read_csv("data/tariff_weights_2016.csv")

fta_data = pd.merge(fta_data, tariff_weights_2016[['Country', 'tariff_weight']], 
                    left_on='Country', right_on='Country', how='left')


fta_data.rename(columns={'tariff_weight': 'tariff_weight_2016_Country'}, inplace=True)
fta_data = pd.merge(fta_data, tariff_weights_2016[['Country', 'tariff_weight']], 
                    left_on='Partner Country', right_on='Country', how='left')

fta_data.rename(columns={'tariff_weight': 'tariff_weight_2016_Partner'}, inplace=True)

fta_data = fta_data.fillna(0)

#formula
fta_data['Adjusted_value'] = np.where(
    fta_data['Year'] < 2017,  
    (fta_data['FTA Weight'] +fta_data['Other Agreements'] - 
     (fta_data['Tariff %']/100 * fta_data['Weight in Exports']/100 * fta_data['tariff_weight_2016_Country']) -
     (fta_data['Tariff %']/100 * fta_data['Weight in Exports']/100 * fta_data['tariff_weight_2016_Partner'])),  # If True (Year < 2017)
    fta_data['FTA Weight'] + fta_data['Other Agreements'] - (fta_data['Tariff %']/100 * fta_data['Weight in Exports']/100)  # If False (Year >= 2017), no tariff weight applied
)

#Cleaning
fta_data.rename(columns={'Country_x': 'Country'}, inplace=True)

# Select only the relevant columns: Country, Partner Country, Adjusted_value
final_data = fta_data[['Year', 'Country', 'Partner Country', 'Adjusted_value']]

final_data.to_csv("data/adjusted_fta_data.csv", index=False)