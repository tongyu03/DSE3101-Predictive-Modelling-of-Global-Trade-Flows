import numpy as np
import pandas as pd

import_data = pd.read_csv('data/Import_data_Singapore.csv', skiprows = 10)
export_data = pd.read_csv('../data/Export_data_Singapore.csv', skiprows = 10)

import_data = import_data.melt(id_vars=["Data Series"], var_name="Dates", value_name="Imports")
import_data = import_data.rename(columns={"Data Series": "Country"})
export_data = export_data.melt(id_vars=["Data Series"], var_name="Dates", value_name="Exports")
export_data = export_data.rename(columns={"Data Series": "Country"})
data = pd.merge(import_data, export_data, on = ("Country", "Dates"), how = "left")
data = data[data["Country"].isin(["    China",
                                "    Hong Kong",
                                "    Japan",
                                "    Korea, Rep Of",
                                "    Malaysia",
                                "    Saudi Arabia",
                                "    Thailand",
                                "    United States"
        ])]
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

data.to_csv('cleaned_monthly_trade_data.csv', index = False)




