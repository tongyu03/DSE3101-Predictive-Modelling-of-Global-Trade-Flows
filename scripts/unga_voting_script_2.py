import pandas as pd

agree_sc = pd.read_csv("data/raw data/AgreementScoresAll_Jun2024.csv")
cow = pd.read_csv("data/raw data/COW-country-codes.csv")

agree_sc = agree_sc[["ccode1", "ccode2", "agree", "year", "IdealPointAll.x", "IdealPointAll.y", "IdealPointDistance"]]

cow_dict = {row['CCode']: row['StateNme'] for index, row in cow.iterrows()}
#no data for Hong Kong, China

#left join
result = []

for _, row in agree_sc.iterrows():
    ccode1 = row['ccode1']
    country1 = cow_dict.get(ccode1, None)
    ccode2 = row['ccode2']
    country2 = cow_dict.get(ccode2, None)
    result.append(list(row) + [country1, country2])

columns = agree_sc.columns.tolist() + ['Country1', 'Country2']
df = pd.DataFrame(result, columns=columns)

#filter target countries
target_countries = ["China", "Hong Kong", "Japan", "Korea", "Malaysia", "Saudi Arabia", "Thailand", "United States of America", "Singapore", "Indonesia"]
df = df[(df["Country1"].isin(target_countries)) & (df["Country2"].isin(target_countries))]

#print("Unique Years in Filtered Data:", df["year"].unique())
#1946 - 2023

#impute IdealPointDistance values
df.loc[df["IdealPointDistance"].isna(), "IdealPointDistance"] = (
        df["IdealPointAll.x"] - df["IdealPointAll.y"]
).abs()

#print(df[df.duplicated()])
#0

#check for na values
#print(df.isna().sum())
#ccode1                  0
#ccode2                  0
#agree                   0
#year                    0
#IdealPointAll.x         6
#IdealPointAll.y         6
#IdealPointDistance      6
#Country1                0
#Country2                0

#remove NA
df = df.dropna()
#total entries
print(len(df))
#1818

print(df)

df['CountryPair'] = df.apply(
    lambda row: tuple(sorted([row['Country1'], row['Country2']])), axis=1
)

# Group by
df = df.groupby(['year', 'CountryPair']).agg({
    'agree': 'mean',
    'IdealPointAll.x': 'mean',
    'IdealPointAll.y': 'mean',
    'IdealPointDistance': 'mean'
}).reset_index()

df.to_csv('data/cleaned data/unga_voting.csv', index=False)