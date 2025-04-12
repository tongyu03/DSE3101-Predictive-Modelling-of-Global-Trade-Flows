import pandas as pd
import numpy as np
# Load raw data
import_data = pd.read_csv('DSE3101-Predictive-Modelling-of-Global-Trade-Flows/data/raw data/Add_Import_Data.csv')
export_data = pd.read_csv('DSE3101-Predictive-Modelling-of-Global-Trade-Flows/data/raw data/Add_Export_Data.csv')
original = pd.read_csv('DSE3101-Predictive-Modelling-of-Global-Trade-Flows/data/cleaned data/10 years Trade Product Data.csv')
# Strip any leading/trailing spaces from column names
import_data.columns = import_data.columns.str.strip()
export_data.columns = export_data.columns.str.strip()

# Reset index and shift the import data (to align data as required)
import_data = import_data.reset_index(drop=True)
import_data = import_data.shift(-1)

# Select relevant columns for imports and exports
import_clean = import_data[['refYear', 'reporterDesc', 'cmdCode', 'cmdDesc', 'flowDesc', 'partnerDesc', 'partnerISO', 'primaryValue']]
export_clean = export_data[['refYear', 'reporterDesc', 'cmdCode', 'cmdDesc', 'flowDesc', 'partnerDesc', 'partnerISO', 'primaryValue']]

# Group by relevant keys and aggregate the primary values
import_agg = import_clean.groupby(['refYear', 'cmdDesc', 'cmdCode', 'partnerDesc', 'partnerISO']).agg({'primaryValue': 'sum'}).reset_index()
export_agg = export_clean.groupby(['refYear', 'cmdDesc', 'cmdCode', 'partnerDesc', 'partnerISO']).agg({'primaryValue': 'sum'}).reset_index()

# Rename 'primaryValue' columns for clarity
import_agg = import_agg.rename(columns={'primaryValue': 'importValue'})
export_agg = export_agg.rename(columns={'primaryValue': 'exportValue'})

# Convert 'cmdDesc' and merge keys columns to string type to ensure proper merging
import_agg['cmdDesc'] = import_agg['cmdDesc'].astype(str)
export_agg['cmdDesc'] = export_agg['cmdDesc'].astype(str)
import_agg['refYear'] = import_agg['refYear'].astype(int)
export_agg['refYear'] = export_agg['refYear'].astype(int)
import_agg['cmdCode'] = import_agg['cmdCode'].astype(int)
export_agg['cmdCode'] = export_agg['cmdCode'].astype(int)

merge_keys = ['refYear', 'cmdDesc', 'cmdCode', 'partnerDesc', 'partnerISO']

# Ensure the merge keys are of the same type in both dataframes
for col in merge_keys:
    import_agg[col] = import_agg[col].astype(str)
    export_agg[col] = export_agg[col].astype(str)


# Merge the import and export data using outer join
trade_data = pd.merge(import_agg, export_agg, on=merge_keys, how='outer')

# Fill missing import or export values with NaN, and keep the other as it is
trade_data['importValue'] = trade_data['importValue'].fillna(np.nan)
trade_data['exportValue'] = trade_data['exportValue'].fillna(np.nan)

# Calculate total trade volume correctly (use NaN where importValue or exportValue is missing)
trade_data['totalTradeValue'] = trade_data['importValue'] + trade_data['exportValue']

# Rename columns for better clarity
trade_data = trade_data.rename(columns={
    'refYear': 'Year',
    'cmdDesc': 'Product',
    'cmdCode': 'HS Code',
    'partnerDesc': 'Country',
    'partnerISO': 'Country ISO',
    'importValue': 'Imports',
    'exportValue': 'Exports',
    'totalTradeValue': 'Trade Volume'
})

# Display final merged data
print(trade_data.head(10))

# Save the cleaned and merged data to a CSV file
trade_data.to_csv('DSE3101-Predictive-Modelling-of-Global-Trade-Flows/data/cleaned data/Add 10 years Trade Product Data.csv', index=False)
concatenated_data = pd.concat([original, trade_data], ignore_index=True)
concatenated_data['Year'] = pd.to_numeric(concatenated_data['Year'], errors='coerce')
concatenated_data = concatenated_data.sort_values(by='Year', ascending=True)
concatenated_data.to_csv('DSE3101-Predictive-Modelling-of-Global-Trade-Flows/data/cleaned data/Concatenated_Trade_Data.csv', index=False)
