# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 22:28:07 2025

@author: grace
"""

import pandas as pd
import matplotlib.pyplot as plt

exchange_rate_raw = "data/raw data/key trade partners' exchange rates.csv"

exchange_rate_raw = pd.read_csv(exchange_rate_raw)

# Remove the first 3 rows and reset the index
exchange_rate_cleaned = exchange_rate_raw.iloc[3:].reset_index(drop=True)

# Display the cleaned dataframe
print(exchange_rate_cleaned)

# Define the country codes for filtering
country_codes = ["CHN", "HKG", "JPN", "KOR", "MYS", "SAU", "THA", "USA", "SGP"]

# Filter the dataset based on the country codes
filtered_data = exchange_rate_cleaned[exchange_rate_cleaned.iloc[:, 1].isin(country_codes)]

# Drop columns 3 and 4 (zero-based index, meaning these correspond to column positions 2 and 3)
filtered_data = filtered_data.drop(filtered_data.columns[[2, 3]], axis=1)

# Generate column names from 1960 to 2023
year_columns = list(range(1960, 2024))

# Rename columns from the 3rd column onwards
filtered_data.columns = list(filtered_data.columns[:2]) + year_columns

# @author: shannen
#pivot table
filtered_data = filtered_data.melt(id_vars=["Data Source", "World Development Indicators"], var_name="Year", value_name="Value")
filtered_data["Year"] = filtered_data["Year"].astype(int)
print(filtered_data.head())
filtered_data.to_csv('data/exchange_rate.csv', index=False)