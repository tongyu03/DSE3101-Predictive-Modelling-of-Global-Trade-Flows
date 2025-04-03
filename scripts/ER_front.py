import pandas as pd

exchange_df = pd.read_csv("data/cleaned data/exchange_rate.csv")

exchange_df["Data Source"] = exchange_df["Data Source"].replace({
    "Hong Kong SAR, China": "Hong Kong",
    "Korea, Rep.": "South Korea"
})
exchange_df = exchange_df[(exchange_df["Year"] >= 2003)]

# Filter for Singapore's exchange rate (SGD per USD)
sgd_exchange = exchange_df[exchange_df["Data Source"] == "Singapore"].copy()
sgd_exchange = sgd_exchange.rename(columns={"Value": "SGD_per_USD"})
exchange_df = exchange_df.merge(sgd_exchange[["Year", "SGD_per_USD"]], on="Year", how="left")
# Convert LCU per USD to LCU per SGD
exchange_df["LCU_per_SGD"] = (exchange_df["Value"] / exchange_df["SGD_per_USD"]).round(2)
exchange_df = exchange_df.drop(columns=["Value", "SGD_per_USD", "World Development Indicators"])
exchange_df = exchange_df.rename(columns={"LCU_per_SGD": "ER"})

currency_mapping = {
    "China": "CNY",
    "Hong Kong": "HKD",
    "Japan": "JPY",
    "South Korea": "KRW",
    "Malaysia": "MYR",
    "Saudi Arabia": "SAR",
    "Singapore": "SGD",
    "Thailand": "THB",
    "United States": "USD"
}

exchange_df["Currency"] = exchange_df["Data Source"].map(currency_mapping)

exchange_df.to_csv('ER_sg.csv')
