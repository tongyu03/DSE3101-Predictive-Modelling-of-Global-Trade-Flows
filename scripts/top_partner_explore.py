"""
Exploratory for top partner countries 

@author: junlu
"""

import numpy as np
import pandas as pd

import_data = pd.read_csv('data/raw data/Import_data_Singapore.csv', skiprows = 10)
export_data = pd.read_csv('data/raw data/Export_data_Singapore.csv', skiprows = 10)

import_data = import_data.melt(id_vars=["Data Series"], var_name="Dates", value_name="Imports")
import_data = import_data.rename(columns={"Data Series": "Country"})
export_data = export_data.melt(id_vars=["Data Series"], var_name="Dates", value_name="Exports")
export_data = export_data.rename(columns={"Data Series": "Country"})
data = pd.merge(import_data, export_data, on = ("Country", "Dates"), how = "left")
data["Country"] = data["Country"].str.strip()  # Remove extra spaces
data["Country"] = data["Country"].replace("Korea, Rep Of", "South Korea")
data["Trade Volume"] = data["Imports"] + data["Exports"]
data = data.sort_values(by = "Trade Volume", ascending = False)
data = data.dropna()

data["Dates"] = pd.to_datetime(data["Dates"].str.strip(), format="%Y %b")  
data["Year"] = data["Dates"].dt.year  # Extract year
data["Month"] = data["Dates"].dt.month  # Extract month (numeric)
data = data.drop(columns=["Dates"])  # Drop the 'Dates' column
data = data[["Country", "Year", "Month", "Imports", "Exports", "Trade Volume"]]  # Reorder columns
data = data[(data["Year"] <= 2024)]
data = data.groupby(['Country','Year'])[['Imports', 'Exports', 'Trade Volume']].sum().reset_index()
unwanted_entries = ['Total All Markets', 'Asia', 'Europe', 'America', 'Oceania', 'Africa', 'Other Markets Africa', 
                    'Other Markets Europe', 'Other Markets Asia', 'Other Markets America', 'Other Markets Oceania', 'Panama', 'New Caledonia']  # Add other unwanted entries if needed
partners = ['Thailand', 'Malaysia', 'United States', 
            'China', 'Hong Kong', 'Japan', "South Korea", 'Indonesia', 'Taiwan', 'India', 'Saudi Arabia']
data = data[data['Year']>=2013]
data = data[~data['Country'].isin(unwanted_entries)]
total_trade_vol = data.groupby(['Country'])[['Trade Volume']].sum().reset_index()
total_trade_vol = total_trade_vol.sort_values(by='Trade Volume', ascending=False)
partners
partners_trade= total_trade_vol[total_trade_vol['Country'].isin(partners)]
partners_trade_vol = partners_trade.groupby(['Country'])[['Trade Volume']].sum().reset_index()

proportion = partners_trade_vol['Trade Volume'].sum() / total_trade_vol['Trade Volume'].sum()*100
print(partners_trade_vol)
print(partners_trade_vol.count())
print(total_trade_vol.head(15))
print(proportion)