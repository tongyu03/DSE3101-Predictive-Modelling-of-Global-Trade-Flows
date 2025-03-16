#backend team

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import shap
import streamlit as st
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load data
#df = pd.read_csv("raw_data.csv")

# Handle missing values
#df.fillna(df.median(), inplace=True)

# Normalize numerical features
scaler = StandardScaler()
#df[['GDP', 'Exchange_Rate', 'Tariff']] = scaler.fit_transform(df[['GDP', 'Exchange_Rate', 'Tariff']])

