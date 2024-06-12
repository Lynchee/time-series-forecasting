import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import os

st.title("UCovid-19PMFit")
st.write("  The researcher is interested in applying the concepts of NLP and transfer learning to improve the ability to forecast the time series of the COVID-19 pandemic spread. The proposed approach involves designing a concept of learning through a General domain of time series patterns from multiple countries, totaling over 200 countries. This concept is referred to as the Universal Covid-19 Time Series Pattern Model Fine-tuning for Covid-19 Time Series Forecasting (UCovid-19PMFit). Subsequently, this model is fine-tuned to learn and forecast the pandemic spread in countries of interest. The 15 countries for which the model is fine-tuned to forecast the pandemic spread are Thailand, Malaysia, Japan, India, Vietnam, Norway, United Kingdom, Italy, Spain, France, Canada, Mexico, Cuba, Brazil, and Argentina.")

countryList = ['Thailand', 'UK']


selectedOption = st.selectbox("Select a country:", countryList)

st.write('-----------')

st.write('Number of *****')

colD1, colD2, colD3, colD4, colD5, colD6 = st.columns(6)
with colD1:
    numD1 = st.number_input("D1", value=None,
                            placeholder="", label_visibility="visible", key='D1')

with colD2:
    numD2 = st.number_input("D2", value=None,
                            placeholder="", label_visibility="visible", key='D2')

with colD3:
    numD3 = st.number_input("D3", value=None,
                            placeholder="", label_visibility="visible", key='D3')

with colD4:
    numD4 = st.number_input("D4", value=None,
                            placeholder="", label_visibility="visible", key='D4')

with colD5:
    numD5 = st.number_input("D5", value=None,
                            placeholder="", label_visibility="visible", key='D5')

with colD6:
    numD6 = st.number_input("D6", value=None,
                            placeholder="", label_visibility="visible", key='D6')


colD7, colD8, colD9, colD10, colD11, colD12 = st.columns(6)
with colD7:
    numD7 = st.number_input("D7", value=None,
                            placeholder="", label_visibility="visible", key='D7')

with colD8:
    numD8 = st.number_input("D8", value=None,
                            placeholder="", label_visibility="visible", key='D8')

with colD9:
    numD9 = st.number_input("D9", value=None,
                            placeholder="", label_visibility="visible", key='D9')

with colD10:
    numD10 = st.number_input("D10", value=None,
                             placeholder="", label_visibility="visible", key='D10')

with colD11:
    numD11 = st.number_input("D11", value=None,
                             placeholder="", label_visibility="visible", key='D11')

with colD12:
    numD12 = st.number_input("D12", value=None,
                             placeholder="", label_visibility="visible", key='D12')

colD13, colD14, _, _, _, _ = st.columns(6)
with colD13:
    numD13 = st.number_input("D13", value=None,
                             placeholder="", label_visibility="visible", key='D13')
with colD14:
    numD13 = st.number_input("D14", value=None,
                             placeholder="", label_visibility="visible", key='D14')


# ---------------------------- Preproces data --------------------------------
df_Confirmed = pd.read_csv(
    'UCovid-19PMFit/data/Preparing_time_series_covid19_confirmed_global.csv')
df_Deaths = pd.read_csv(
    'UCovid-19PMFit/data/Preparing_time_series_covid19_deaths_global.csv')

Country = 'Thailand'
df_Confirmed = df_Confirmed[df_Confirmed['Country/Region']
                            == Country].drop(columns="Country/Region")
df_Deaths = df_Deaths[df_Deaths['Country/Region']
                      == Country].drop(columns="Country/Region")

df_Confirmed = df_Confirmed.T
df_Confirmed.columns = ["Confirmed"]
df_Confirmed_train = df_Confirmed.iloc[:-30]
df_Confirmed_test = df_Confirmed.iloc[-45:]

df_Deaths = df_Deaths.T
df_Deaths.columns = ["Deaths"]
df_Deaths_train = df_Deaths.iloc[:-30]
df_Deaths_test = df_Deaths.iloc[-45:]

data = pd.concat([df_Confirmed_train, df_Deaths_train], axis=1)


def MinMaxScale(df):
    df_norm = pd.DataFrame([])
    min = []
    max = []
    MinMaxScaleColumnName = []
    for c in df.columns:
        new_df = (df[[c]]-df[[c]].min())/(df[[c]].max()-df[[c]].min())
        min.append(df[[c]].min())
        max.append(df[[c]].max())
        MinMaxScaleColumnName.append(c)
        df_norm = pd.concat([df_norm, new_df], axis=1)
    return df_norm, min, max


data, minScale, maxScale = MinMaxScale(data)

data_test = pd.concat([df_Confirmed_test, df_Deaths_test], axis=1)


def MinMaxScaleForTest(df, minScale, maxScale):
    df_norm = pd.DataFrame([])
    min = []
    max = []
    MinMaxScaleColumnName = []
    for i, c in enumerate(df.columns):
        new_df = (df[[c]]-minScale[i][c])/(maxScale[i][c]-minScale[i][c])
        df_norm = pd.concat([df_norm, new_df], axis=1)

    return df_norm


data_test = MinMaxScaleForTest(data_test, minScale, maxScale)

sequence_length = 14


def generate_data(X, sequence_length=14, step=1):
    X_local = []
    y_local = []
    for start in range(0, len(X) - sequence_length-1, step):
        end = start + sequence_length
        X_local.append(X[start:end])
        y_local.append(X[end][0])
    return np.array(X_local), np.array(y_local)


X_sequence, y = generate_data(data.loc[:, ["Deaths"]].values)
X_sequence_test, y_test = generate_data(data_test.loc[:, ["Deaths"]].values)


X_sequence, y = shuffle(X_sequence, y, random_state=0)
X_sequence.shape, y.shape

DataX_train, y_train = X_sequence, y
DataX_valid, y_valid = X_sequence_test, y_test

print(DataX_train.shape, DataX_valid.shape)
print(y_train.shape, y_test.shape)


# ------------------------------------- Train model -----------------------------------
st.title("Trains model")


st.title("Predicts model")


# --------------------------------------- Results -------------------------------------
