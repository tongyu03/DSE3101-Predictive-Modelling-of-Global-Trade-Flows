import pandas as pd

gdp_data = pd.read_csv("data/raw data/GDP.csv", header = 2)
print(gdp_data.head(5))
# Find the row where the country is Singapore
singapore_gdp_data = gdp_data[gdp_data["Country Name"] == "Singapore"].iloc[:, 4:].transpose()  # Adjusting to get GDP columns for years

# Rename columns to correspond with the years in your data
singapore_gdp_data.columns = singapore_gdp_data.iloc[0]  # First row should be the years
singapore_gdp_data.columns = ["Singapore_GDP"]
singapore_gdp_data = singapore_gdp_data[1:]  # Remove the first row, which was the year labels

singapore_gdp_data.to_csv("data/cleaned data/singapore_gdp.csv", header=True)
