import streamlit as st
import pandas as pd
import pickle as pkl

model = pkl.load(open('model.pkl', 'rb'))
df = pkl.load(open('df.pkl', 'rb'))

st.set_page_config(page_title="DiabetesT", page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)

st.title('Diabetes Diagnosis App')
st.write('This app predicts the probability of a patient having diabetes based on their health metrics.')

age = st.number_input('age')
pregnant = st.number_input('how many times have you been pregnant')
glucose = st.number_input('your plasma glucose concentration a 2 hours in an oral glucose tolerance test')
bp = st.number_input('your diastolic blood pressure (mm Hg)')
skin = st.number_input('your triceps skin fold thickness (mm)')
insulin = st.number_input('your 2-Hour serum insulin value (mu U/ml) ')
bmi = st.number_input('body mass index (weight in kg/(height in m)^2)')
dpf = st.number_input('diabetes Pedigree Function')

data = {"Pregnancies": pregnant,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age}

features = pd.DataFrame(data, index=[0])

pred = model.predict(features)
pred_proba = model.predict_proba(features)

if st.button('Predict'):
    if pred == 1:
        st.subheader('You have diabetes')
        st.subheader(("Possibility of having diabetes: ", pred_proba[0][0]*100, "%"))
    else:
        st.subheader('You do not have diabetes')
        st.subheader(("Possibility of having diabetes: ", pred_proba[0][1]*100, "%"))


st.write("")
st.write("")
st.write("")
st.caption("**test results do not express the exact truth. results are just a machine learning predictions. please consult your doctor for more accurate outcomes.")
