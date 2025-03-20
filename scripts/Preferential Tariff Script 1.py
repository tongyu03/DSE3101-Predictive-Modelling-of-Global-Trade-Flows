import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv("../DSE3101-Predictive-Modelling-of-Global-Trade-Flows/Data/Preferential Tariffs.csv")

# Strip any leading/trailing spaces from column names
df.columns = df.columns.str.strip()

#target countries
df_singapore = df[df['partneriso3'] == 'SGP']
reporter_countries = ["HKG", "MYS", "SAU", "JPN", "CHN", "USA", "THA", "KOR"]


df_filtered = df_singapore[df_singapore['reporteriso3'].isin(reporter_countries)]


df_filtered['mfn'] = pd.to_numeric(df_filtered['mfn'], errors='coerce')
df_filtered['prf'] = pd.to_numeric(df_filtered['prf'], errors='coerce')
df_filtered['PTA'] = pd.to_numeric(df_filtered['PTA'], errors='coerce')

df_filtered['log_mfn'] = np.log(df_filtered['mfn'] + 1)  # Add 1 to avoid log(0)
df_filtered['log_prf'] = np.log(df_filtered['prf'] + 1)

grouped = df_filtered.groupby('reporteriso3').agg(
    # Count the occurrences for each country
    count=('reporteriso3', 'size'),
    
    # Sum the MFN and PRF values for each country
    total_mfn=('log_mfn', 'sum'),
    total_prf=('log_prf', 'sum'),
    
    # Apply custom aggregation for PTA (if PTA > 0, then 1, else 0)
    total_pta=('PTA', lambda x: ((x > 0).sum() > 0).astype(int)),  # count occurrences where PTA > 0
    
    # Keep the first entry for reporter_wbgroup and partner_wbgroup
    reporter_wbgroup=('reporter_wbgroup', 'first'),
    partner_wbgroup=('partner_wbgroup', 'first')
)

# Calculate the average MFN and PRF by dividing by count
grouped['avg_mfn'] = grouped['total_mfn'] / grouped['count']
grouped['avg_prf'] = grouped['total_prf'] / grouped['count']

# Reset index to make 'reporteriso3' a column again
result = grouped.reset_index()

# Now, display the result with 'reporteriso3' and 'partneriso3' included
print(result)

# Optionally, save this aggregated data to a new CSV file
result.to_csv("/Users/junlu/dse rstudio/DSE3101-Predictive-Modelling-of-Global-Trade-Flows/aggregated_trade_data_log.csv", index=False)
