import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

##load the trained model
model = tf.keras.models.load_model('model.keras',compile=False)

##load the encoder and scaler
with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('sclaer.pkl','rb') as file:
    scaler_geo = pickle.load(file)

##Streamlit app

st.title('Customer Churn Prediction')

# Input fields
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender",label_encoder_gender.classes_)
age = st.slider("Age", 18, 100, 40)
tenure = st.slider("Tenure", 0, 10, 3)
balance = st.number_input("Balance", value=60000.0)
num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active = st.selectbox("Is Active Member", [0, 1])
salary = st.number_input("Estimated Salary", value=50000.0)

#prepare intput data

##example input
input_data = pd.DataFrame( {
    'CreditScore' : [credit_score],
    
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance' : [balance],
    'NumOfProducts':[num_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active],
    'EstimatedSalary':[salary]
})


geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)


##combine one-hot encoder columns

input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

##scale the input data
input_data_scaled = scaler_geo.transform(input_data)


#predict the churn
prediciton = model.predict(input_data_scaled)
prediciton_proba = prediciton[0][0]


if st.button("Predict Churn"):
    prediction = model.predict(input_data_scaled)
    prediction_proba = float(prediction[0][0])  # Convert to native Python float

    st.write(f"Churn Probability: {prediction_proba:.2%}")
    st.progress(prediction_proba)  # Must be a float between 0.0 and 1.0

    if prediction_proba > 0.5:
        st.error("⚠️ The customer is likely to churn.")
    else:
        st.success("✅ The customer is not likely to churn.")


