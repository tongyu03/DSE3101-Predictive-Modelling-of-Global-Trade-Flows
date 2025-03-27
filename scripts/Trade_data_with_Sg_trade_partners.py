import numpy as np
import pandas as pd

import_data = pd.read_csv('Import data_Sinapore.csv', skiprows = 10)
export_data = pd.read_csv('Export data_Singapore.csv', skiprows = 10)
import_data = import_data.melt(id_vars=["Data Series"], var_name="Dates", value_name="Imports")
import_data = import_data.rename(columns={"Data Series": "Country"})
#print(import_data.head(10))
export_data = export_data.melt(id_vars=["Data Series"], var_name="Dates", value_name="Exports")
export_data = export_data.rename(columns={"Data Series": "Country"})
#print(export_data.head(10))
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
data["Dates"] = data["Dates"].str.strip()  # Remove extra spaces
data["Dates"] = pd.to_datetime(data["Dates"], format="%Y %b")
data["Trade Volume"] = data["Imports"] + data["Exports"]
data = data.sort_values(by = "Trade Volume", ascending = False)
data = data.dropna()
#print(data)
data.to_csv('data.csv', index = False)

