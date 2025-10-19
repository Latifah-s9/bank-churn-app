import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("🏦 Bank Customer Churn Prediction App")
st.write("Enter customer details below to predict if they are likely to leave the bank.")

credit_score = st.slider("Credit Score", 300, 850, 600)
age = st.slider("Age", 18, 92, 35)
balance = st.number_input("Balance", min_value=0.0, step=100.0)
num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_card = st.selectbox("Has Credit Card?", [0, 1])
is_active = st.selectbox("Is Active Member?", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, step=500.0)
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Female", "Male"])

geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0
gender_male = 1 if gender == "Male" else 0

features = np.array([[credit_score, age, balance, num_products, has_card,
                      is_active, estimated_salary, geo_germany, geo_spain, gender_male]])
import os

model_path = os.path.join(os.path.dirname(__file__), 'churn_model.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

if st.button("Predict"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.error("⚠️ This customer is likely to LEAVE the bank.")
    else:
        st.success("✅ This customer is likely to STAY with the bank .")
