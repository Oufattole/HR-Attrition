from pycaret.classification import *
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import shap
import lime 
from lime import lime_tabular

model = load_model('gbc_model')
prep = joblib.load("prep_pipe.pkl")

train_data = pd.read_csv("data.csv") 

def get_preprocessed_data(model, input_df):
    """
    Get prediction related columns i.e. Label (the prediction) and the probabilty of the prediction

    Preprocesses the data using the preprocessing steps used during training
    """
    predictions_df = predict_model(estimator = model, data = input_df)
    return predictions_df


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot}</body>"
    components.html(shap_html, height=height)

def run():

    # Expand the layout
    st.set_page_config(layout = "wide")

    # Select whether or not to do single row or multiple row predictions
    add_selectbox = st.sidebar.selectbox("Single or Multi predict?", ("Single", "Multi"))
    
    # Title
    st.title("HR Attrition")
    st.subheader("Using this app you can identify whether an employee(s) will potentially leave in the next six months and also why.")

    # Initialise columns
    col1, col2, col3, col4 = st.beta_columns(4)

    # Create individual app elements to provide input for single row prediction
    if add_selectbox == "Single":

        with col1:
            Age = st.number_input("Age", min_value = 18, max_value = 100, step = 1)
            BusinessTravel = st.select_slider("BusinessTravel", options = ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])
            DailyRate = st.number_input("DailyRate", min_value = 100, max_value = 1500, step = 1)
            Department = st.select_slider("Department", options = ['Sales', 'Research & Development', 'Human Resources'])
            DistanceFromHome = st.number_input("DistanceFromHome", min_value = 1, max_value = 30, step = 1)
            Education = st.select_slider("Education", options = [1, 2, 3, 4, 5])
            EducationField = st.select_slider("EducationField", options = ['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources'])
            EnvironmentSatisfaction = st.select_slider("EnvironmentSatisfaction", options = [1, 2, 3, 4])
            
        with col2:
            Gender = st.select_slider("Gender", options = ["Female", "Male"])
            HourlyRate = st.number_input("HourlyRate", min_value = 1, max_value = 100, step = 1)
            JobInvolvement = st.select_slider("JobInvolvement", options = [1, 2, 3, 4])
            JobLevel = st.select_slider("JobLevel", options = [1, 2, 3, 4, 5])
            JobRole = st.select_slider("JobRole", options = ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'])
            JobSatisfaction = st.select_slider("JobSatisfaction", options = [1, 2, 3, 4])
            MaritalStatus = st.select_slider("MaritalStatus", options = ['Single', 'Married', 'Divorced'])
            MonthlyIncome = st.number_input("MonthlyIncome", min_value = 1000, max_value = 20000, step = 1)
            
        with col3:
            MonthlyRate = st.number_input("MonthlyRate", min_value = 2000, max_value = 27000, step = 1)
            NumCompaniesWorked = st.slider("NumCompaniesWorked", min_value = 0, max_value = 9, step = 1)
            Over18 = st.select_slider("Over18", options = ["N", "Y"])
            OverTime = st.select_slider("OverTime", options = ["N", "Y"])
            PercentSalaryHike = st.number_input("PercentSalaryHike", min_value = 10, max_value = 25, step = 1)
            PerformanceRating = st.number_input("PerformanceRating", min_value = 3, max_value = 4, step = 1)
            RelationshipSatisfaction = st.select_slider("RelationshipSatisfaction", options = [1, 2, 3, 4])
            # StandardHours =
            StockOptionLevel = st.select_slider("StockOptionLevel", options = [0, 1, 2, 3])
            
        with col4:
            TotalWorkingYears = st.slider("TotalWorkingYears", min_value = 0, max_value = 40, step = 1)
            TrainingTimesLastYear = st.slider("TrainingTimesLastYear", min_value = 0, max_value = 6, step = 1)
            WorkLifeBalance = st.slider("WorkLifeBalance", min_value = 0, max_value = 4, step = 1)
            YearsAtCompany = st.slider("YearsAtCompany", min_value = 0, max_value = 40, step = 1)
            YearsInCurrentRole = st.slider("YearsInCurrentRole", min_value = 0, max_value = 18, step = 1)
            YearsSinceLastPromotion = st.slider("YearsSinceLastPromotion", min_value = 0, max_value = 15, step = 1)
            YearsWithCurrManager = st.slider("YearsWithCurrManager", min_value = 0, max_value = 18, step = 1)

        output = ""

        input_dict = {"Age" : Age, 
                    "BusinessTravel" :  BusinessTravel ,
                    "DailyRate" :  DailyRate ,
                    "Department" :  Department ,
                    "DistanceFromHome" :  DistanceFromHome ,
                    "Education" :  Education ,
                    "EducationField" :  EducationField ,
                    "EnvironmentSatisfaction" :  EnvironmentSatisfaction ,
                    "Gender" :  Gender ,
                    "HourlyRate" :  HourlyRate ,
                    "JobInvolvement" :  JobInvolvement ,
                    "JobLevel" :  JobLevel ,
                    "JobRole" :  JobRole ,
                    "JobSatisfaction" :  JobSatisfaction ,
                    "MaritalStatus" :  MaritalStatus ,
                    "MonthlyIncome" :  MonthlyIncome ,
                    "MonthlyRate" :  MonthlyRate ,
                    "NumCompaniesWorked" :  NumCompaniesWorked ,
                    "Over18" :  Over18 ,
                    "OverTime" :  OverTime ,
                    "PercentSalaryHike" :  PercentSalaryHike ,
                    "PerformanceRating" :  PerformanceRating ,
                    "RelationshipSatisfaction" :  RelationshipSatisfaction ,
                    "StockOptionLevel" :  StockOptionLevel ,
                    "TotalWorkingYears" :  TotalWorkingYears ,
                    "TrainingTimesLastYear" :  TrainingTimesLastYear ,
                    "WorkLifeBalance" :  WorkLifeBalance ,
                    "YearsAtCompany" :  YearsAtCompany ,
                    "YearsInCurrentRole" :  YearsInCurrentRole ,
                    "YearsSinceLastPromotion" :  YearsSinceLastPromotion ,
                    "YearsWithCurrManager" :  YearsWithCurrManager}

        # Create row of data based upon the element inputs
        input_df = pd.DataFrame([input_dict])
 
        # Make single prediction for data input using the model
        if st.button("Predict"):

            prediction_df = get_preprocessed_data(model = model, input_df = input_df)
            prediction_df.rename(columns = {"Label" : "Attrition"}, inplace = True)
            prediction_df.drop(columns = "Score", inplace = True)

            output = prediction_df['Attrition'][0]

            # Output of the prediction
            st.success('{} this person will leave.'.format(output))

            # Add columns that need to be ignored
            #prediction_df['EmployeeNumber'] = 0
            #prediction_df['StandardHours'] = 1
            #prediction_df['EmployeeCount'] = 1

            test_shap = prep.transform(prediction_df)

            lime_explainer = lime_tabular.LimeTabularExplainer(
                training_data = train_data.to_numpy(),
                feature_names = train_data.columns,
                class_names = ['No', 'Yes'],
                mode = 'classification', 
            )
            print(lime_explainer)
            print(test_shap)
            print(test_shap.to_numpy()[0])

            lime_exp = lime_explainer.explain_instance(
                data_row = test_shap.to_numpy()[0],
                predict_fn = model['trained_model'].predict_proba
            )
            st.pyplot(lime_exp.as_pyplot_figure())
            st_shap(lime_exp.as_html(), height = 2000)

    # Batch prediction (multiple people to predict)
    if add_selectbox == "Multi":

        # Upload rows to predict
        file_upload = st.file_uploader("Upload csv file for predictions", type = ["csv"])

        # Make prediction and output data
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator = model, data = data)
            st.write(predictions)

if __name__ == '__main__':
    run()