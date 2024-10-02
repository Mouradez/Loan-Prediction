import streamlit as st
import pandas as pd
from model import loaded_model
import numpy as np

    
app_mode = st.sidebar.selectbox('Select Page',['Home','Prediction'])

if app_mode=='Home':
    st.title('Loan Prediction')
    st.markdown("""This machine learning project aims to develop a predictive model that determines whether a customer will be approved for a loan by a bank. The model is trained using a loan dataset that includes various features related to the customer's demographic and financial information.
""")
    st.image('images/loan_image.jpeg',width=700)
    st.header('Dataset :')
    st.markdown("""The dataset used for this project is the "Loan Dataset" from Kaggle. It contains the following features:

Loan_ID: Unique identifier for each loan application.
                
Gender: Gender of the applicant (Male, Female).
                
Married: Marital status of the applicant (Yes, No).
                
Dependents: Number of dependents (0, 1, 2, 3+).
                
Education: Education level of the applicant (Graduate, Not Graduate).
                
Self_Employed: Employment status (Yes, No).
                
ApplicantIncome: Monthly income of the applicant.
                
CoapplicantIncome: Monthly income of the co-applicant.
                
LoanAmount: Amount of loan requested.
                
Loan_Amount_Term: Term of the loan in months.
                
Credit_History: Credit history of the applicant (1 for good credit, 0 for bad credit).
                
Property_Area: Location of the property (Urban, Rural, Semi-Urban).
                
Loan_Status: Target variable indicating whether the loan was approved (Y for Yes, N for No).
""")
    data = pd.read_csv('data/loan_data_set.csv')
    st.write(data.head())
    st.markdown("Applicant Income VS Loan Amount")
    st.bar_chart(data[['ApplicantIncome','LoanAmount']].head(10))
    
elif app_mode == 'Prediction':
    st.subheader('Prediction')
    st.markdown('Sir/Mme, You need to fill all necessary informations in order to get a reply to your loan request !')
    st.image("images/loan_prediction.jpg",width=700)
    st.sidebar.header('Informations about the client :')
    gender_dict = {"Male": 1, "Female": 2}
    feature_dict = {"No": 1, "Yes": 2}
    edu = {'Graduate': 1, 'Not Graduate': 2}
    prop = {'Rural': 1, 'Urban': 2, 'Semiurban': 3}

    ApplicantIncome = st.sidebar.slider('ApplicantIncome', 0, 10000, 0)
    CoapplicantIncome = st.sidebar.slider('CoapplicantIncome', 0, 10000, 0)
    LoanAmount = st.sidebar.slider('LoanAmount in K$', 9.0, 700.0, 200.0)
    Loan_Amount_Term = st.sidebar.selectbox('Loan_Amount_Term', (12.0, 36.0, 60.0, 84.0, 120.0, 180.0, 240.0, 300.0, 360.0))
    Credit_History = st.sidebar.radio('Credit_History', (0.0, 1.0))
    Gender = st.sidebar.radio('Gender', tuple(gender_dict.keys()))
    Married = st.sidebar.radio('Married', tuple(feature_dict.keys()))
    Self_Employed = st.sidebar.radio('Self Employed', tuple(feature_dict.keys()))
    Dependents = st.sidebar.radio('Dependents', options=['0', '1', '2', '3+'])
    Education = st.sidebar.radio('Education', tuple(edu.keys()))
    Property_Area = st.sidebar.radio('Property_Area', tuple(prop.keys()))

    Gender_encoded = gender_dict[Gender]
    Married_encoded = feature_dict[Married]
    Education_encoded = edu[Education]
    Self_Employed_encoded = feature_dict[Self_Employed]
    Property_Area_encoded = prop[Property_Area]

    Dependents_encoded = {'0': 0, '1': 1, '2': 2, '3+': 3}[Dependents]

    feature_list = [
        ApplicantIncome,
        CoapplicantIncome,
        LoanAmount,
        Loan_Amount_Term,
        Credit_History,
        Gender_encoded,
        Married_encoded,
        Dependents_encoded,
        Education_encoded,
        Self_Employed_encoded,
        Property_Area_encoded
    ]

    single_sample = np.array(feature_list).reshape(1, -1)
    if st.button("Predict"):
        prediction = loaded_model.predict(single_sample)
        if prediction[0] == 0:
            st.error('According to our Calculations, you will not get the loan from Bank')
        elif prediction[0] == 1:
            st.success('Congratulations!! you will get the loan from Bank')