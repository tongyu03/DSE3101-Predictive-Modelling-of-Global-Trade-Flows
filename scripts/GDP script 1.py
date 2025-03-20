import pandas as pd

# Load the GDP data
gdp_df = pd.read_csv("data/GDP.csv", header=4)

# Clean the GDP data
#Change the data to long format
gdp_long = gdp_df.melt(id_vars=["Country Code", "Indicator Name", "Indicator Code"], 
                       var_name="Year", value_name="GDP")

gdp_long["Year"] = gdp_long["Year"].astype(int)

gdp_long = gdp_long.drop(columns=["Indicator Name", "Indicator Code"])

# Filter countries of interest
countries_to_keep = ["CHN", "HKG", "JPN", "KOR", "MYS", "SAU", "THA", "USA", "SGP"]
gdp_long = gdp_long[gdp_long["Country Code"].isin(countries_to_keep)]

# Reference USA
usa_gdp = gdp_long[gdp_long["Country Code"] == "USA"][["Year", "GDP"]].rename(columns={"GDP": "USA_GDP"})
gdp_long = gdp_long.merge(usa_gdp, on="Year", how="left")
gdp_long["GDP_normalized"] = gdp_long["GDP"] / gdp_long["USA_GDP"]
gdp_long = gdp_long.drop(columns=["USA_GDP"])

gdp_long = gdp_long.dropna()  
# Save the cleaned and processed GDP data to a new CSV
gdp_long.to_csv("data/Processed_GDP.csv", index=False)


print(gdp_long.head())

