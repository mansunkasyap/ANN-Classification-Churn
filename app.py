import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

model = load_model("model.h5")

#Load the encoder and scaler
with open(r"C:\Users\HP\Personal\GEN_AI\ANN_Project\onehot_encoder_geo.pkl", 'rb') as file:
    label_encoder_geo = pickle.load(file)
    print("Geo encoder Loaded..")

with open(r"C:\Users\HP\Personal\GEN_AI\ANN_Project\label_encoder_gender.pkl", 'rb') as file:
    label_encoder_gender = pickle.load(file)
    print("Gender encoder Loaded..")

with open(r"C:\Users\HP\Personal\GEN_AI\ANN_Project\scaler.pkl", 'rb') as file:
    scaler = pickle.load(file)
    print("Scaler Loaded....")


## Streamlit App
st.title("Customer Churn Prediction")

st.set_page_config(page_title="Customer Input Form", layout="centered")

st.title("🏦 Customer Information")
st.write("Enter customer details below:")

with st.form("customer_form"):

    credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=600)

    geography = st.selectbox("Geography", label_encoder_geo.categories_[0])

    gender = st.selectbox("Gender", label_encoder_gender.classes_)

    age = st.number_input("Age", min_value=18, max_value=100, value=40)

    tenure = st.slider("Tenure (Years)", 0, 10, 3)

    balance = st.number_input("Account Balance", min_value=0.0, value=60000.0)

    num_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=1)

    has_cr_card = st.selectbox("Has Credit Card?", [0, 1])

    is_active = st.selectbox("Is Active Member?", [0, 1])

    estimated_salary = st.number_input( "Estimated Salary", min_value=0.0, value=50000.0)

    submitted = st.form_submit_button("Submit", use_container_width=True)



if submitted:

    # Step 1: Create input dictionary
    input_data = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": label_encoder_gender.transform([gender])[0],
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active,
        "EstimatedSalary": estimated_salary
    }

    st.success("Customer data submitted successfully!")

    st.subheader("Customer Input Data")
    st.json(input_data)

    # Step 2: Convert dictionary to DataFrame
    input_df = pd.DataFrame([input_data])

    # Step 3: One Hot Encode Geography
    geography_ohe = label_encoder_geo.transform(
        [[geography]]
    )

    if hasattr(geography_ohe, "toarray"):
        geography_ohe = geography_ohe.toarray()

    geography_ohe_df = pd.DataFrame(
        geography_ohe,
        columns=label_encoder_geo.get_feature_names_out(
            ["Geography"]
        )
    )

    # Step 4: Remove original Geography column
    input_df = input_df.drop("Geography", axis=1)

    # Step 5: Concatenate encoded Geography columns
    input_df = pd.concat(
        [input_df.reset_index(drop=True), geography_ohe_df],
        axis=1
    )

    # Step 6: Arrange columns in training order
    input_df = input_df[scaler.feature_names_in_]

    st.subheader("Processed Input Data")
    st.dataframe(input_df)

    # Step 7: Scale input
    input_scaler_data = scaler.transform(input_df)

    # Step 8: Predict churn
    prediction = model.predict(input_scaler_data)

    probability = float(prediction[0][0])

    # Step 9: Display prediction
    st.subheader("Prediction Result")

    st.write(f"Churn Probability: {probability:.2%}")

    if probability > 0.5:
        st.error("Customer is likely to churn!")
    else:
        st.success("Customer is likely to stay with us!")
