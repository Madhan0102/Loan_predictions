import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load(r'loan_eligibility_model.pkl')

# Define the input fields for the user
st.title("Loan Eligibility Prediction")

# Collect input data from the user
gender = st.selectbox('Gender', ['Male', 'Female'])
married = st.selectbox('Married', ['No', 'Yes'])
dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
self_employed = st.selectbox('Self Employed', ['No', 'Yes'])
applicant_income = st.number_input('Applicant Income', min_value=0)
coapplicant_income = st.number_input('Coapplicant Income', min_value=0)
loan_amount = st.number_input('Loan Amount', min_value=0)
loan_amount_term = st.number_input('Loan Amount Term', min_value=0)
credit_history = st.selectbox('Credit History', [0.0, 1.0])
property_area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])

# Create a mapping for categorical variables to numerical format
gender_map = {'Male': 1, 'Female': 0}
married_map = {'Yes': 1, 'No': 0}
education_map = {'Graduate': 1, 'Not Graduate': 0}
self_employed_map = {'Yes': 1, 'No': 0}
property_area_map = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}
dependents_map = {'0': 0, '1': 1, '2': 2, '3+': 3}

# Map the inputs to numerical values
input_data = [
    gender_map[gender],
    married_map[married],
    dependents_map[dependents],
    education_map[education],
    self_employed_map[self_employed],
    applicant_income,
    coapplicant_income,
    loan_amount,
    loan_amount_term,
    credit_history,
    property_area_map[property_area]
]

# Reshape the data as required by the model
input_data_reshaped = np.asarray(input_data).reshape(1, -1)

# Predict using the loaded model
prediction = model.predict(input_data_reshaped)

# Display the prediction result
if prediction[0] == 0:
    st.write('The person is **not eligible** for a loan.')
else:
    st.write('The person is **eligible** for a loan.')

# Plot the prediction as a pie chart
labels = 'Eligible', 'Not Eligible'
sizes = [prediction[0], 1 - prediction[0]]
colors = ['lightgreen', 'lightcoral']
explode = (0.1, 0)  # explode the eligible slice

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

st.pyplot(fig1)
