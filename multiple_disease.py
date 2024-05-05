import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load diabetes model and scaler
db = pickle.load(open('diabetes_c.pkl', 'rb'))
mmsd = pickle.load(open('scalerdiab.pkl', 'rb'))

# Load heart disease model and scaler
hb = pickle.load(open('heart_c.pkl', 'rb'))
mmsh = pickle.load(open('scalerheart.pkl', 'rb'))

# Load Parkinson's disease model and scaler
pb = pickle.load(open('parkinson_c.pkl', 'rb'))
mmsp = pickle.load(open('scalerpark.pkl', 'rb'))

# Load breast cancer model and scaler
cb = pickle.load(open('breast_c.pkl', 'rb'))
mms = pickle.load(open('scaler.pkl', 'rb'))

# Load lung cancer model
lung_cancer_model = pickle.load(open("lung_cancer_model.sav", "rb"))

# Function to predict diabetes
def predict_diabetes(pregnancies, glucose, blood_pressure, insulin, bmi, dpf, age):
    features = np.array([pregnancies, glucose, blood_pressure, insulin, bmi, dpf, age]).reshape(1, -1)
    features_scaled = mmsd.transform(features)
    prediction = db.predict(features_scaled)
    return prediction[0]

# Function to predict heart disease
def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    features = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
    features_scaled = mmsh.transform(features)
    prediction = hb.predict(features_scaled)
    return prediction[0]

# Function to predict Parkinson's disease
def predict_parkinsons_disease(fo, fhi, flo, PPQ, DDP, Shimmer, APQ, spread1, spread2, D2, PPE):
    features = np.array([fo, fhi, flo, PPQ, DDP, Shimmer, APQ, spread1, spread2, D2, PPE]).reshape(1, -1)
    features_scaled = mmsp.transform(features)
    prediction = pb.predict(features_scaled)
    return prediction[0]

# Function to predict breast cancer
def predict_breast_cancer(texture_mean, perimeter_mean, compactness_mean, texture_se, perimeter_se, smoothness_se, compactness_se, texture_worst, perimeter_worst, area_worst, compactness_worst):
    features = np.array([texture_mean, perimeter_mean, compactness_mean, texture_se, perimeter_se, smoothness_se, compactness_se, texture_worst, perimeter_worst, area_worst, compactness_worst]).reshape(1, -1)
    features_scaled = mms.transform(features)
    prediction = cb.predict(features_scaled)
    return prediction[0]

# Streamlit app
def main():
    st.sidebar.title("Multiple Disease Prediction System")
    
    # Sidebar navigation
    selected = st.sidebar.selectbox(
        "Select Disease",
        ["Diabetes Prediction", "Heart Disease Prediction", "Parkinson's Disease Prediction", "Breast Cancer Prediction", "Lung Cancer Prediction"]
    )
    
    if selected == "Diabetes Prediction":
        st.title("Diabetes Prediction")
        
        # User input for diabetes prediction
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1)
        glucose = st.number_input("Glucose Level")
        blood_pressure = st.number_input("Blood Pressure Value")
        insulin = st.number_input("Insulin Level")
        bmi = st.number_input("BMI Value")
        dpf = st.number_input("Diabetes Pedigree Function Value")
        age = st.number_input("Age of the Person")
        
        # Predict button for diabetes
        if st.button("Diabetes Test Result"):
            prediction = predict_diabetes(pregnancies, glucose, blood_pressure, insulin, bmi, dpf, age)
            if prediction == 0:
                st.success("You have no diabetes.")
            else:
                st.error("You have diabetes.")
    
    elif selected == "Heart Disease Prediction":
        st.title("Heart Disease Prediction")
        
        # User input for heart disease prediction
        age = st.number_input("Age")
        sex = st.number_input("Sex (1=male; 0=female)")
        cp = st.number_input("Chest Pain Types")
        trestbps = st.number_input("Resting Blood Pressure")
        chol = st.number_input("Serum Cholestoral in mg/dl")
        fbs = st.number_input("Fasting Blood Sugar > 120 mg/dl")
        restecg = st.number_input("Resting Electrocardiographic Results")
        thalach = st.number_input("Maximum Heart Rate Achieved")
        exang = st.number_input("Exercise Induced Angina")
        oldpeak = st.number_input("ST Depression induced by Exercise")
        slope = st.number_input("Slope of the peak exercise ST Segment")
        ca = st.number_input("Major vessels colored by Flourosopy")
        thal = st.number_input("Thalassemia (0 = normal; 1 = fixed defect; 2 = reversable defect)")
        
        # Predict button for heart disease
        if st.button("Heart Disease Test Result"):
            prediction = predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
            if prediction == 0:
                st.success("You have no heart problems.")
            else:
                st.error("You have heart problems.")
    
    elif selected == "Parkinson's Disease Prediction":
        st.title("Parkinson's Disease Prediction")
        
        # User input for Parkinson's disease prediction
        fo = st.number_input("MDVP:Fo")
        fhi = st.number_input("MDVP:Fhi")
        flo = st.number_input("MDVP:Flo")
        PPQ = st.number_input("MDVP:PPQ")
        DDP = st.number_input("Jitter:DDP")
        Shimmer = st.number_input("MDVP:Shimmer")
        APQ = st.number_input("MDVP:APQ")
        spread1 = st.number_input("spread1")
        spread2 = st.number_input("spread2")
        D2 = st.number_input("D2")
        PPE = st.number_input("PPE")
        
        # Predict button for Parkinson's disease
        if st.button("Parkinson's Disease Test Result"):
            prediction = predict_parkinsons_disease(fo, fhi, flo, PPQ, DDP, Shimmer, APQ, spread1, spread2, D2, PPE)
            if prediction == 0:
                st.success("You have no Parkinson's disease.")
            else:
                st.error("You have Parkinson's disease.")
    
    elif selected == "Breast Cancer Prediction":
        st.title("Breast Cancer Prediction")
        
        # User input for breast cancer prediction
        texture_mean = st.number_input("Texture Mean")
        perimeter_mean = st.number_input("Perimeter Mean")
        compactness_mean = st.number_input("Compactness Mean")
        texture_se = st.number_input("Texture Error")
        perimeter_se = st.number_input("Perimeter Error")
        smoothness_se = st.number_input("Smoothness Error")
        compactness_se = st.number_input("Compactness Error")
        texture_worst = st.number_input("Texture Worst")
        perimeter_worst = st.number_input("Perimeter Worst")
        area_worst = st.number_input("Area Worst")
        compactness_worst = st.number_input("Compactness Worst")
        
        # Predict button for breast cancer
        if st.button("Breast Cancer Test Result"):
            prediction = predict_breast_cancer(texture_mean, perimeter_mean, compactness_mean, texture_se, perimeter_se, smoothness_se, compactness_se, texture_worst, perimeter_worst, area_worst, compactness_worst)
            if prediction == 0:
                st.success("No Breast Cancer detected.")
            else:
                st.error("Breast Cancer detected.")
    
    elif selected == "Lung Cancer Prediction":
        st.title("Lung Cancer Prediction")
        
        # User input for lung cancer prediction
        gender = st.number_input("Gender (1=Male; 0=Female)")
        age = st.number_input("Age")
        smoking = st.number_input("Smoking")
        yellow_fingers = st.number_input("Yellow Fingers")
        anxiety = st.number_input("Anxiety")
        peer_pressure = st.number_input("Peer Pressure")
        chronic_disease = st.number_input("Chronic Disease")
        fatigue = st.number_input("Fatigue")
        allergy = st.number_input("Allergy")
        wheezing = st.number_input("Wheezing")
        alcohol_consuming = st.number_input("Alcohol Consuming")
        coughing = st.number_input("Coughing")
        shortness_of_breath = st.number_input("Shortness of Breath")
        swallowing_difficulty = st.number_input("Swallowing Difficulty")
        chest_pain = st.number_input("Chest Pain")
        
        # Predict button for lung cancer
        if st.button("Lung Cancer Test Result"):
            prediction = lung_cancer_model.predict([[gender, age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue, allergy, wheezing, alcohol_consuming, coughing, shortness_of_breath, swallowing_difficulty, chest_pain]])
            if prediction[0] == 0:
                st.success("No Lung Cancer detected.")
            else:
                st.error("Lung Cancer detected.")

if __name__ == "__main__":
    main()
