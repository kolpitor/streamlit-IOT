import os
os.system('pip3 install pdpbox==0.2.1')

from pdpbox.pdp import pdp_isolate, pdp_plot
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from numpy import mean
import streamlit as st

"""
# IOT
"""


max_depth_input = st.slider("Max depth", 1, 100, 5)
colsample_bytree_input = st.slider("Colsample bytree", 0.0, 1.0, 0.5)
learning_rate_input = st.slider("Learning rate", 0.0, 1.0, 0.2)
alpha_input = st.slider("Alpha", 1, 100, 10)
n_estimators_input = st.slider("n estimators", 1, 100, 20)
city_input = st.selectbox(
     'Which city do you want to predict rain ?',
     ("Canberra",
    "Albury",
    "Penrith",
    "Sydney",
    "MountGinini",
    "Bendigo",
    "Brisbane",
    "Portland"), index=0)


df = pd.read_csv("city_temperature.csv")

def mergeStateToCountry():
    df.loc[df['State'].notna(), 'Country'] = df['State']
    df = df.loc[:, ~df.columns.str.contains('State')]

i = 0

for region in df["Region"].unique():
    df["Region"] = df["Region"].replace(region, str(i))
    i += 1
    
i = 0

for country in df["Country"].unique():
    df["Country"] = df["Country"].replace(country, str(i))
    i += 1
    
i = 0

for state in df["State"].unique():
    df["State"] = df["State"].replace(state, str(i))
    i += 1
    
i = 0

for city in df["City"].unique():
    df["City"] = df["City"].replace(city, str(i))
    i += 1

df = df.astype({"Region": "int"})
df = df.astype({"Country": "int"})
df = df.astype({"State": "int"})
df = df.astype({"City": "int"})

target = 'AvgTemperature'
# Here Y would be our target
Y = df[target]
# Here X would contain the other column
#X = df.loc[:, df.columns != target]
X = df[['Month', 'Day', 'Year']]

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.25, random_state=42)

y_pred = [Y_train.mean()] * len(Y_train)

st.write('Baseline MAE: %f' % (round(mean_absolute_error(Y_train, y_pred), 5)))

lm = make_pipeline(StandardScaler(), LinearRegression(),)

lm.fit(X_train, Y_train)

st.write('Linear Regression Training MAE: %f' % (round(mean_absolute_error(Y_train, lm.predict(X_train)), 5)))
st.write('Linear Regression Test MAE: %f' % (round(mean_absolute_error(Y_val, lm.predict(X_val)), 5)))

forestModel = make_pipeline(
    SelectKBest(k="all"), 
    StandardScaler(), 
    RandomForestRegressor(
        n_estimators=100,
        max_depth=50,
        random_state=77,
        n_jobs=-1))

forestModel.fit (X_train, Y_train)

st.write('Random Forest Regressor Model Training MAE: %f' % (mean_absolute_error(Y_train, forestModel.predict(X_train))))
st.write('Random Forest Regressor Model Test MAE: %f' % (mean_absolute_error(Y_val, forestModel.predict(X_val))))
