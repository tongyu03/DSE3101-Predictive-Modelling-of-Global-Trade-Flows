#backend team

import pandas as pd
#import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import shap
import streamlit as st
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load data
pt_est = pd.read_csv("/Users/shannenigans/Desktop/R files/data/IdealpointestimatesAll_Jun2024.csv")

print(pt_est.head())
pt_est = pt_est["Countryname", ""]

