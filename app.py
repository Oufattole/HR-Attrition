from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('catboost')

def predict(model, input_df):
    predictions_df = predict_model(estimator = model, data = input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app is created to predict if an employee will leave in 6 months')
    st.sidebar.success('https://www.pycaret.org')
    

    st.title("HR Attrition")

    if add_selectbox == 'Online':

        Age = st.slider("Age", min_value = 18, max_value = 100, step = 1)
        BusinessTravel = st.select_slider("BusinessTravel", options = ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
        DailyRate = st.slider("DailyRate", min_value = 100, max_value = 1500, step = 1)
        Department = st.select_slider("Department", options = ['Sales', 'Research & Development', 'Human Resources'])
        DistanceFromHome = st.slider("DistanceFromHome", min_value = 1, max_value = 30, step = 1)
        Education = st.select_slider("Education", options = [1, 2, 3, 4, 5])
        EducationField = st.select_slider("EducationField", options = ['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources'])
        EnvironmentSatisfaction = st.select_slider("EnvironmentSatisfaction", options = [1, 2, 3, 4])
        Gender = st.select_slider("Gender", options = ["Female", "Male"])
        # HourlyRate =
        # JobInvolvement =
        # JobLevel =
        # JobRole =
        # JobSatisfaction =
        # MaritalStatus =
        # MonthlyIncome =
        # MonthlyRate =
        # NumCompaniesWorked =
        # Over18 =
        # OverTime =
        # PercentSalaryHike =
        # PerformanceRating =
        # RelationshipSatisfaction =
        # StandardHours =
        # StockOptionLevel =
        # TotalWorkingYears =
        # TrainingTimesLastYear =
        # WorkLifeBalance =
        # YearsAtCompany =
        # YearsInCurrentRole =
        # YearsSinceLastPromotion =
        # YearsWithCurrManager =

        output=""

        input_dict = {'Age' : Age}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model = model, input_df = input_df)
            output = '$' + str(output)

        st.success('The output is {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()