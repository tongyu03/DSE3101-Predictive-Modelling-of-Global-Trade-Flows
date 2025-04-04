# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 22:03:53 2025

@author: grace
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
sgp_trade_raw = "data/raw data/en_SGP_AllYears_WITS_Trade_Summary.CSV"

sgp_trade_cleaned = pd.read_csv(sgp_trade_raw)

# convert to long format
sgp_trade_cleaned = sgp_trade_cleaned.melt(id_vars=["Reporter", "Partner", "Product categories", "Indicator Type", "Indicator"], 
                  var_name="Year", value_name="Trade_Value")

sgp_trade_cleaned["Year"] = pd.to_numeric(sgp_trade_cleaned["Year"], errors='coerce')

sgp_trade_cleaned = sgp_trade_cleaned.dropna()

sgp_trade_cleaned = sgp_trade_cleaned[~sgp_trade_cleaned.apply(lambda row: row.astype(str).str.contains("\.\.\.").any(), axis=1)]

# Define the indicators to keep
relevant_indicators = [
    "Import(US$ Mil)", 
    "Export(US$ Mil)", 
    "Imports (in US$ Mil)", 
    "Exports (in US$ Mil)", 
    "Trade (US$ Mil)-Top 5 Export Partner", 
    "Trade (US$ Mil)-Top 5 Import Partner"
]

sgp_trade_cleaned = sgp_trade_cleaned[sgp_trade_cleaned["Indicator"].isin(relevant_indicators)]

sgp_trade_cleaned = sgp_trade_cleaned[sgp_trade_cleaned["Product categories"] == "All Products"]

sgp_trade_cleaned = sgp_trade_cleaned[sgp_trade_cleaned["Partner"] != "World"]
#print(sgp_trade_cleaned.head())

trade_summary = sgp_trade_cleaned.groupby(["Year", "Partner", "Indicator Type"])["Trade_Value"].sum().reset_index()


# Plot trade volume over time
plt.figure(figsize=(12, 6))

# Loop through unique partners and indicator types
for partner in trade_summary["Partner"].unique():
    for indicator_type in trade_summary["Indicator Type"].unique():
        subset = trade_summary[(trade_summary["Partner"] == partner) & (trade_summary["Indicator Type"] == indicator_type)]
        plt.plot(subset["Year"], subset["Trade_Value"], label=f"{partner} - {indicator_type}")

# Customize the plot
plt.xlabel("Year")
plt.ylabel("Trade Volume (US$ Mil)")
plt.title("Trade Volume Over Time by Partner and Indicator Type")
plt.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.grid(True)

# Show the plot
plt.show()


#%%




















