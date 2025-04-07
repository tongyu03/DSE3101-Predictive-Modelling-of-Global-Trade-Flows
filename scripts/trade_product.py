"""
Exploratory and Cleaning for Time Series Trading Data between 2013 and 2023 for 
Singapore and its Top 10 Trading Partners and top 10 Industries
Columns: Country, Year, Product, HS Code, Import, Export, Trade Volume
@author: junlu
"""

import pandas as pd


# Define the file path
file_path = 'data/raw data/Import_Export_by_Industry_SG.csv'
hs_file_path = 'data/raw data/Harmonic Code and Products.csv'

data = pd.read_csv(file_path)
hs_code = pd.read_csv(hs_file_path)

exports = data[data['TradeFlowName'] == 'Export']
imports = data[data['TradeFlowName'] == 'Import']

exports_grouped = exports.groupby(['ReporterName', 'PartnerName', 'ProductCode', 'Year'])['TradeValue in 1000 USD'].sum().reset_index()
imports_grouped = imports.groupby(['ReporterName', 'PartnerName', 'ProductCode', 'Year'])['TradeValue in 1000 USD'].sum().reset_index()
trade_data = pd.merge(exports_grouped, imports_grouped, on=['ReporterName', 'PartnerName', 'ProductCode', 'Year'], how='outer', suffixes=('_Export', '_Import'))
trade_data['Total_Trade'] = trade_data['TradeValue in 1000 USD_Export'].fillna(0) + trade_data['TradeValue in 1000 USD_Import'].fillna(0)
trade_data = trade_data.sort_values(by='Year', ascending=False)
trade_data = pd.merge(trade_data, hs_code, left_on='ProductCode', right_on='HS Code', how='left')
#to get top industries
trade_data_by_product = trade_data.groupby(['HS Code','Product descriptions'])['Total_Trade'].sum().reset_index()
top_10_industries = trade_data_by_product.sort_values(by='Total_Trade', ascending=False).head(10)


print(top_10_industries)

print(trade_data)

trade_data.to_csv('data/cleaned data/trade_product_data.csv', index=False)
# Export the top 10 industries to a CSV file

import_data = pd.read_csv('data/raw data/Import Top Industries Trade Data.csv')
export_data = pd.read_csv('data/raw data/Export Top Industries Trade Data.csv')
import_data.columns = import_data.columns.str.strip()
export_data.columns = export_data.columns.str.strip()

import_clean = import_data[['refYear', 'reporterDesc', 'cmdCode', 'cmdDesc', 'flowDesc', 'partnerDesc', 'partnerISO', 'primaryValue']]
export_clean = export_data[['refYear', 'reporterDesc', 'cmdCode', 'cmdDesc', 'flowDesc', 'partnerDesc', 'partnerISO', 'primaryValue']]

import_agg = import_clean.groupby(['refYear', 'cmdDesc', 'cmdCode', 'partnerDesc', 'partnerISO']).agg({'primaryValue': 'sum'}).reset_index()
export_agg = export_clean.groupby(['refYear', 'cmdDesc', 'cmdCode', 'partnerDesc','partnerISO']).agg({'primaryValue': 'sum'}).reset_index()

#rename columns
import_agg = import_agg.rename(columns={'primaryValue': 'importValue'})
export_agg = export_agg.rename(columns={'primaryValue': 'exportValue'})

trade_data = pd.merge(import_agg, export_agg, on=['refYear', 'cmdDesc', 'cmdCode', 'partnerDesc', 'partnerISO'], how='outer')

trade_data['totalTradeValue'] = trade_data['importValue'] + trade_data['exportValue']
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

print(trade_data)
trade_data.to_csv('data/cleaned data/10 years Trade Product Data.csv', index=False)

