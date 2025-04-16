import pandas as pd

#import trade product data
trade_pdt_df = pd.read_csv("data/cleaned data/10 years Trade Product Data.csv")
trade_pdt_df['Country'] = trade_pdt_df['Country'].replace({
    'China, Hong Kong SAR': 'Hong Kong',
    'Rep. of Korea': 'South Korea'
})
trade_pdt_df = trade_pdt_df[~trade_pdt_df['Country'].isin(['Hong Kong'])] # drop hongkong
trade_pdt_df = trade_pdt_df.drop(columns=['Country ISO'])
trade_pdt_df['Country'] = trade_pdt_df['Country'].replace({
    'USA': 'United States'
})

# Rename Products
industry_rename_map = {
    'Chemical products n.e.c.': 'Other Chemicals',
    'Electrical machinery and equipment and parts thereof; sound recorders and reproducers; television image and sound recorders and reproducers, parts and accessories of such articles': 'Electrical Equipment & AV Gear',
    'Essential oils and resinoids; perfumery, cosmetic or toilet preparations': 'Cosmetics & Fragrances',
    'Metal; miscellaneous products of base metal': 'Base Metal Products',
    'Mineral fuels, mineral oils and products of their distillation; bituminous substances; mineral waxes': 'Mineral Fuels & Oils',
    'Natural, cultured pearls; precious, semi-precious stones; precious metals, metals clad with precious metal, and articles thereof; imitation jewellery; coin': 'Jewellery & Precious Metals',
    'Nuclear reactors, boilers, machinery and mechanical appliances; parts thereof': 'Machinery & Boilers',
    'Optical, photographic, cinematographic, measuring, checking, medical or surgical instruments and apparatus; parts and accessories': 'Optical & Medical Instruments',
    'Organic chemicals': 'Organic Chemicals',
    'Plastics and articles thereof': 'Plastics',
    'Machinery and mechanical appliances, boilers, nuclear reactors; parts thereof': 'Machinery & Boilers'  # same as above
}
trade_pdt_df["Product"] = trade_pdt_df["Product"].replace(industry_rename_map)

# Create new 'All Products Category'
all_products_df = trade_pdt_df.groupby(['Year', 'Country'], as_index=False)[['Imports', 'Exports', 'Trade Volume']].sum()
all_products_df['Product'] = 'All Products'
all_products_df['HS Code'] = 99 
all_products_df = all_products_df[trade_pdt_df.columns]
trade_pdt_df = pd.concat([trade_pdt_df, all_products_df], ignore_index=True)
trade_pdt_df = trade_pdt_df.sort_values(by='Year').reset_index(drop=True)

# trade_pdt_df.to_csv("10_years_trade_frontend.csv", index=False)

#%%

#import predicted imports data
predicted_imports = pd.read_csv("data/cleaned data/predicted_imports_2024_2026.csv")

# import predicted exports data
predicted_exports = pd.read_csv("data/cleaned data/predicted export.csv")
predicted_exports = predicted_exports[predicted_exports['Partner'].isin([
    "China", "Indonesia", "Japan", "Malaysia", "Saudi Arabia", 
    "South Korea", "Thailand", "United States of America"
])]

# Merge the predicted_imports and predicted_exports datasets on common columns
predicted_df = pd.merge(predicted_imports, predicted_exports, 
    on=['Partner', 'HS Code', 'HS_Section', 'Target Year'], how='inner'  
)
predicted_df['Partner'] = predicted_df['Partner'].replace({
    'United States of America': 'United States'
})

hs_code_to_product = {
    38: "Other Chemicals",
    85: "Electrical Equipment & AV Gear",
    33: "Cosmetics & Fragrances",
    83: "Base Metal Products",
    27: "Mineral Fuels & Oils",
    71: "Jewellery & Precious Metals",
    84: "Machinery & Boilers",
    90: "Optical & Medical Instruments",
    29: "Organic Chemicals",
    39: "Plastics",
    99: "All Products"
}
# Create product column
predicted_df['HS Code'] = predicted_df['HS Code'].replace('All Products', 99)
predicted_df['HS Code'] = pd.to_numeric(predicted_df['HS Code'], errors='coerce')
predicted_df['Product'] = predicted_df['HS Code'].map(hs_code_to_product)
predicted_df = predicted_df.drop(columns=['HS_Section'])
predicted_df = predicted_df.rename(columns={
    'Partner': 'Country',
    'Target Year': 'Year',
    'Predicted Imports': 'Imports',
    'Predicted Exports': 'Exports'
})

# Create Trade Volume column
predicted_df['Trade Volume'] = predicted_df['Imports'] + predicted_df['Exports']
predicted_df = predicted_df[trade_pdt_df.columns]

# concatenate trade_pdt_df with predicted
combined_df = pd.concat([trade_pdt_df, predicted_df], ignore_index=True)

combined_df.to_csv("data/cleaned data/trade_with_predicted.csv", index=False)


