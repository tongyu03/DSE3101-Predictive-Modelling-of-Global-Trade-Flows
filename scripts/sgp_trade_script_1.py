# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 22:03:53 2025

@author: grace
"""

import pandas as pd

# Load the CSV file
sgp_trade_raw = "en_SGP_AllYears_WITS_Trade_Summary.CSV"

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

print(sgp_trade_cleaned.head())