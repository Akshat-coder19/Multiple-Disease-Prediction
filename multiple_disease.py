import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing


#loading the models

db = pickle.load(open('diabetes_c.pkl', 'rb'))
mmsd = pickle.load(open('scalerdiab.pkl', 'rb'))

hb = pickle.load(open('heart_c.pkl', 'rb'))
mmsh = pickle.load(open('scalerheart.pkl', 'rb'))

pb = pickle.load(open('parkinson_c.pkl', 'rb'))
mmsp = pickle.load(open('scalerpark.pkl', 'rb'))

cb = pickle.load(open('breast_c.pkl', 'rb'))
mms = pickle.load(open('scaler.pkl', 'rb'))

lung_cancer = pickle.load(open("lung_cancer_model.sav", "rb"))








#sidebar for navigation

with st.sidebar:
    
    selected = option_menu("Multiple Disease Prediction System ", 
                           
                           ["Diabetes Prediction",
                            "Heart Disese Prediction",
                            "Parkinsons Disease Prediction",
                            "Lung Cancer Prediction",
                
                            "Breast Cancer Prediction"],
                           
                           icons = ["activity", "heart-fill", "people-fill", 
                                    "lungs", "gender-female"],
                           
                           default_index = 0)





#Diabetes Prediction Page:

if(selected == "Diabetes Prediction"):
    
    #page title
    st.title("Diabetes Prediction ")
    
    

# getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input("Number of Pregnancies")
        
    with col2:
        Glucose = st.text_input("Glucose Level")
    
    with col3:
        BloodPressure = st.text_input("Blood Pressure Value")
    
    with col1:
        Insulin = st.text_input("Insulin Level")
    
    with col2:
        BMI = st.text_input("BMI Value")
    
    with col3:
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")
    
    with col1:
        Age = st.text_input("Age of the Person")


# code for Prediction
    diabetes_diagnosis = " "
    
    # creating a button for Prediction
    
    if st.button("Diabetes Test Result"):
    	def predict():
        	features = [Pregnancies, Glucose, BloodPressure, Insulin, BMI, DiabetesPedigreeFunction, Age]
        	final_features = [np.array(features)]
        	final_features = mmsd.transform(final_features)
        	df = pd.DataFrame(final_features)
        	prediction = db.predict(df)

        	if prediction[0] == 0:
            		return "You have no diabetes"
        	else:
            		return "You have diabetes"

    	diabetes_diagnosis = predict()

        
    st.success(diabetes_diagnosis)



















#Heart Disease Prediction Page:

if(selected == "Heart Disese Prediction"):
    
    #page title
    st.title("Heart Disease Prediction ")
    
    
    
# getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age")
        
    with col2:
        sex = st.number_input("1=male;0=female")
        
    with col3:
        cp = st.number_input("Chest Pain Types")
        
    with col1:
        trestbps = st.number_input("Resting Blood Pressure")
        
    with col2:
        chol = st.number_input("Serum Cholestoral in mg/dl")
        
    with col3:
        fbs = st.number_input("Fasting Blood Sugar > 120 mg/dl")
        
    with col1:
        restecg = st.number_input("Resting Electrocardiographic Results")
        
    with col2:
        thalach = st.number_input("Maximum Heart Rate Achieved")
        
    with col3:
        exang = st.number_input("Exercise Induced Angina")
        
    with col1:
        oldpeak = st.number_input("ST Depression induced by Exercise")
        
    with col2:
        slope = st.number_input("Slope of the peak exercise ST Segment")
        
    with col3:
        ca = st.number_input("Major vessels colored by Flourosopy")
        
    with col1:
        thal = st.number_input("thalassemia: 0 = normal; 1 = fixed defect; 2 = reversable defect")
        
        
     
     
    # code for Prediction
    heart_diagnosis = " "
    
    # creating a button for Prediction
    
    if st.button("Heart Prediction Test Result"):
    	def predict():
        	features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        	final_features = [np.array(features)]
        	final_features = mmsh.transform(final_features)
        	df = pd.DataFrame(final_features)
        	prediction = hb.predict(df)

        	if prediction[0] == 0:
            		return "You have no heart problems"
        	else:
            		return "You have heart problems"

    	heart_diagnosis = predict()

        
    st.success(heart_diagnosis)
        











#Parkinsons Disease Prediction Page:

if(selected == "Parkinsons Disease Prediction"):
    
    #page title
    st.title("Parkinsons Disease Prediction ")



# getting the input data from the user

    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input("MDVP:(Fo)")
        
    with col2:
        fhi = st.text_input("MDVP:(Fhi)")
        
    with col3:
        flo = st.text_input("MDVP:(Flo)")
        
        
    with col4:
        PPQ = st.text_input("MDVP:(PPQ)")
        
    with col5:
        DDP = st.text_input("Jitter:(DDP)")
        
    with col1:
        Shimmer = st.text_input("MDVP:(Shimmer)")
        
        

        
    with col2:
        APQ = st.text_input("MDVP:(APQ)")
        
   
      
    with col3:
        spread1 = st.text_input("spread1")
        
    with col4:
        spread2 = st.text_input("spread2")
        
    with col5:
        D2 = st.text_input("D2")
        
    with col1:
        PPE = st.text_input("PPE")
        
    
    
    # code for Prediction
    parkinsons_diagnosis = " "
    
    # creating a button for Prediction    
    if st.button("Parkinson Prediction Test Result"):
    	def predict():
        	features = [fo, fhi, flo, PPQ, DDP, Shimmer, APQ, spread1, spread2, D2, PPE]
        	final_features = [np.array(features)]
        	final_features = mmsp.transform(final_features)
        	df = pd.DataFrame(final_features)
        	prediction = pb.predict(df)

        	if prediction[0] == 0:
            		return "You have no Parkinson's disease"
        	else:
            		return "You have Parkinson's disease"

    	parkinsons_diagnosis = predict()

        
    st.success(parkinsons_diagnosis)  
















#Breast Cancer Prediction Page:

if(selected == "Breast Cancer Prediction"):
    
    #page title
    st.title("Breast Cancer Prediction")



# getting the input data from the user

    col1, col2, col3, col4, col5 = st.columns(5)
        
    with col1:
        texture_mean = st.number_input("texture_mean")
        
    with col2:
        perimeter_mean = st.number_input("perimeter_mean")
        
        
    with col3:
        compactness_mean = st.number_input("compactness_mean")
        
    with col4:
        texture_se = st.number_input("texture_error")
        
    with col5:
        perimeter_se = st.number_input("perimeter_error")
        
    with col1:
       smoothness_se = st.number_input("smoothness_error")
        
    with col2:
       compactness_se = st.number_input("symmetry_error")
    with col3:
        texture_worst = st.number_input("texture_worst")
    
    with col4:
        perimeter_worst = st.number_input("perimeter_worst")
        
    with col5:
       area_worst = st.number_input("area_worst")
        
        
    with col1:
        compactness_worst = st.number_input("compactness_worst")
        
        
    
    #code for Prediction
    breast_cancer_check = " "

    if st.button("Breast Cancer Test Result"):
    	def predict():
        	features = [texture_mean, perimeter_mean, compactness_mean, texture_se, perimeter_se,smoothness_se, compactness_se, texture_worst, perimeter_worst, area_worst, compactness_worst]
        	final_features = [np.array(features)]
        	final_features = mms.transform(final_features)
        	df = pd.DataFrame(final_features)
        	prediction = cb.predict(df)

        	if prediction[0] == 0:
            		return "Breast cancer"
        	else:
            		return "No Breast Cancer"

    	breast_cancer_check = predict()

    
    st.success(breast_cancer_check)

#Lung Cancer Prediction Page:

if(selected == "Lung Cancer Prediction"):
    
    #page title
    st.title("Lung Cancer Prediction ")



# getting the input data from the user
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        GENDER = st.number_input("1=Male;0=Female")
        
    with col2:
        AGE = st.number_input("AGE")
    
    with col3:
        SMOKING = st.number_input("SMOKING")
    
    with col4:
        YELLOW_FINGERS = st.number_input("YELLOW_FINGERS")
    
    with col1:
        ANXIETY = st.number_input("ANXIETY")
    
    with col2:
        PEER_PRESSURE = st.number_input("PEER_PRESSURE")
    
    with col3:
        CHRONIC_DISEASE = st.number_input("CHRONIC DISEASE")
    
    with col4:
        FATIGUE = st.number_input("FATIGUE")
    
    with col1:
        ALLERGY = st.number_input("ALLERGY")
    
    with col2:
        WHEEZING = st.number_input("WHEEZING")
    
    with col3:
        ALCOHOL_CONSUMING = st.number_input("ALCOHOL CONSUMING")
    
    with col4:
        COUGHING = st.number_input("COUGHING")
    
    with col1:
        SHORTNESS_OF_BREATH = st.number_input("SHORTNESS OF BREATH")
    
    with col2:
        SWALLOWING_DIFFICULTY = st.number_input("SWALLOWING DIFFICULTY")
    
    with col3:
        CHEST_PAIN = st.number_input("CHEST PAIN")
    


    


# code for Prediction
    lung_cancer_result = " "
    
    # creating a button for Prediction
    
    if st.button("Lung Cancer Test Result"):
        lung_cancer_report = lung_cancer.predict([[GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN]])
        
        if (lung_cancer_report[0] == 0):
          lung_cancer_result = "Hurrah! You have no Lung Cancer."
        else:
          lung_cancer_result = "Sorry! You have Lung Cancer."
        
    st.success(lung_cancer_result)
        
    












