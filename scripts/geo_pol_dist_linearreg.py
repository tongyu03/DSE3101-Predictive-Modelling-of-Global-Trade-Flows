import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

geopol_df = pd.read_csv("data\cleaned data\geopolitical_data.csv")

X = geopol_df.iloc[:, 4:]
X["GDP_Lag1"] = np.log10(X["GDP_Lag1"])
X['Exchange Rate (per US$)'] = np.log1p(X['Exchange Rate (per US$)'])

import numpy as np
import pandas as pd

# Compute trade volume (log10)
y = np.log10(geopol_df["Imports"] + geopol_df["Exports"])

#print(X.head())
#print(y)

model = LinearRegression()
model.fit(X, y)
weights = model.coef_

print(weights)

