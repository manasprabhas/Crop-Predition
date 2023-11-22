import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import time
import pandas as pd

st.title("Crop Prediction :rice: :corn:")
navigation = st.sidebar.radio(label="Select Action", options=["Prediction"])


crops = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
         'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
         'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya',
         'pigeonpeas', 'pomegranate', 'rice', 'watermelon']

soli_data = ['Red Sandy Loam Soil', 'Clay Loam Soil', 'Saline Coastal Alluvium Soil',
             'Non Calcareous Red Soil', 'Non Calcareous Brown Soil', 'Calcareous Black Soil',
             'Red Loamy Soil', 'Black Soil', 'Red Loamy(New Delta) Soil',
             'Alluvium(Old Delta) Soil', 'Coastal Alluvium Soil',
             'Deep Red Soil', 'Saline Coastal Soil', 'Alluvium Soil',
             'Deep Red Loam Soil', 'Latteritic Soil']


if navigation == "Prediction":
    st.header("\nEnter Values to Predict the Optimal Crop")

    file_path = data.csv
    df = pd.read_csv(file_path)
    df = df.sample(frac=1).reset_index(drop=True)

    X = df.drop("label", axis=1)
    y = df.label

    ordinal_enc = OrdinalEncoder()
    y = ordinal_enc.fit_transform(y.values.reshape(-1, 1))

    num_attributes = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    cat_attributes = ["soil"]

    num_pipeline = Pipeline([
        ("std_scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder())
    ])
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attributes),
        ("cat", cat_pipeline, cat_attributes)])

    X = full_pipeline.fit_transform(X)
    clf = GaussianNB()
    clf.fit(X, y.ravel())
    cf = DecisionTreeClassifier()
    cf.fit(X, y.ravel())

    n = st.number_input("Nitrogen Ratio", min_value=0.00)
    p = st.number_input("Phosphorous Ratio", min_value=0.00)
    k = st.number_input("Potassium Ratio", min_value=0.00)
    temperature = st.number_input("Temperature(Celcius)", min_value=0.00)
    humidity = st.number_input("Humidity", min_value=0.00)
    ph = st.number_input("pH of the Soil", min_value=1.000000, max_value=14.000000)
    rainfall = st.number_input("Rainfall", min_value=0.00)
    soil_type = st.selectbox('Soil Type', soli_data)
    var = soli_data.index(soil_type)
    inputs = np.array([[n, p, k, temperature, humidity, ph, rainfall, var]])

    prediction = clf.predict(inputs)
    pred = cf.predict(inputs)
    index = int(prediction[0])
    ind = int(pred[0])
    crop = crops[index]

    if st.button("Predict DT"):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.05)
            progress.progress(i + 1)
        st.success("Predicted crop : " + crop)
    elif st.button("Predict NB"):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.05)
            progress.progress(i + 1)
        st.success("Predicted crop : " + crop)