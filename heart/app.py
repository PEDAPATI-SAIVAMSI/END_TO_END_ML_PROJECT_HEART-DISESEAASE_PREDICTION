import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pycaret.classification import setup, compare_models, finalize_model, predict_model
from pycaret.regression import setup as setup_regression, compare_models as compare_models_regression, finalize_model as finalize_model_regression, predict_model as predict_model_regression

# Load the data from GitHub
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/PEDAPATI-SAIVAMSI/END_TO_END_ML_PROJECT_HEART-DISESEAASE_PREDICTION/main/heart/heart.csv'
    data = pd.read_csv(url, on_bad_lines='skip')  # Skips bad lines
    return data

# Preprocess the data
def preprocess_data(df):
    label_encoders = {}

    # Check for missing values in 'Sex' column
    st.write("Missing values in 'Sex':", df['Sex'].isnull().sum())
    
    # If missing values, fill them (replace 'M' with appropriate handling if needed)
    df['Sex'].fillna('M', inplace=True)
    
    # Log unique values in 'Sex' column
    st.write("Unique values in 'Sex':", df['Sex'].unique())

    # Encoding 'Sex' column
    le_sex = LabelEncoder()
    le_sex.fit(['M', 'F'])
    df['Sex'] = le_sex.transform(df['Sex'])
    label_encoders['Sex'] = le_sex

    # Encoding 'ChestPainType' column
    le_cp = LabelEncoder()
    le_cp.fit(['TA', 'ATA', 'NAP', 'ASY'])
    df['ChestPainType'] = le_cp.transform(df['ChestPainType'])
    label_encoders['ChestPainType'] = le_cp

    # Encoding 'RestingECG' column
    le_recg = LabelEncoder()
    le_recg.fit(['Normal', 'ST', 'LVH'])
    df['RestingECG'] = le_recg.transform(df['RestingECG'])
    label_encoders['RestingECG'] = le_recg

    # Encoding 'ExerciseAngina' column
    le_ea = LabelEncoder()
    le_ea.fit(['Y', 'N'])
    df['ExerciseAngina'] = le_ea.transform(df['ExerciseAngina'])
    label_encoders['ExerciseAngina'] = le_ea

    # Encoding 'ST_Slope' column
    le_sts = LabelEncoder()
    le_sts.fit(['Up', 'Flat', 'Down'])
    df['ST_Slope'] = le_sts.transform(df['ST_Slope'])
    label_encoders['ST_Slope'] = le_sts

    return df, label_encoders

# Streamlit app
st.title('Heart Disease Prediction and Model Comparison')

# Load and display data
data = load_data()
st.write('Dataset:', data.head())

# Preprocess the data
data, label_encoders = preprocess_data(data)

# Split the dataset
X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection
model_type = st.selectbox('Choose a model comparison method', ['PyCaret Classification', 'PyCaret Regression'])

if model_type == 'PyCaret Classification':
    # PyCaret Classification
    st.write("Setting up PyCaret for Classification...")
    clf_setup = setup(data, target='HeartDisease', session_id=42, silent=True, verbose=False)
    best_model = compare_models()
    finalized_model = finalize_model(best_model)
    
    # Make predictions
    predictions = predict_model(finalized_model, data=X_test)
    accuracy = (predictions['Label'] == y_test).mean()
    st.write(f'Best Classification Model Accuracy: {accuracy * 100:.2f}%')

elif model_type == 'PyCaret Regression':
    # If you want to use PyCaret for regression instead
    st.write("Setting up PyCaret for Regression...")
    reg_setup = setup_regression(data, target='HeartDisease', session_id=42, silent=True, verbose=False)
    best_model_reg = compare_models_regression()
    finalized_model_reg = finalize_model_reg(best_model_reg)
    
    # Make predictions
    predictions_reg = predict_model_reg(finalized_model_reg, data=X_test)
    st.write(f'Best Regression Model Predictions: {predictions_reg.head()}')

# Prediction section
st.header('Predict Heart Disease')

# User input for prediction
age = st.slider('Age', min_value=20, max_value=100, value=50)
sex = st.selectbox('Sex', options=['M', 'F'])
chest_pain = st.selectbox('Chest Pain Type', options=['TA', 'ATA', 'NAP', 'ASY'])
resting_bp = st.slider('Resting BP', min_value=80, max_value=200, value=120)
cholesterol = st.slider('Cholesterol', min_value=100, max_value=400, value=200)
fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])
resting_ecg = st.selectbox('Resting ECG', options=['Normal', 'ST', 'LVH'])
max_hr = st.slider('Max HR', min_value=60, max_value=220, value=150)
exercise_angina = st.selectbox('Exercise-induced Angina', options=['Y', 'N'])
oldpeak = st.slider('Oldpeak', min_value=0.0, max_value=6.0, value=1.0)
st_slope = st.selectbox('ST Slope', options=['Up', 'Flat', 'Down'])

# Convert inputs to a dataframe
input_data = pd.DataFrame({
    'Age': [age],
    'Sex': [label_encoders['Sex'].transform([sex])[0]],
    'ChestPainType': [label_encoders['ChestPainType'].transform([chest_pain])[0]],
    'RestingBP': [resting_bp],
    'Cholesterol': [cholesterol],
    'FastingBS': [fasting_bs],
    'RestingECG': [label_encoders['RestingECG'].transform([resting_ecg])[0]],
    'MaxHR': [max_hr],
    'ExerciseAngina': [label_encoders['ExerciseAngina'].transform([exercise_angina])[0]],
    'Oldpeak': [oldpeak],
    'ST_Slope': [label_encoders['ST_Slope'].transform([st_slope])[0]]
})

# Make prediction
if model_type == 'PyCaret Classification':
    prediction = predict_model(finalized_model, data=input_data)
    st.subheader('Prediction')
    if prediction['Label'][0] == 1:
        st.write('The model predicts that the patient **has heart disease**.')
    else:
        st.write('The model predicts that the patient **does not have heart disease**.')
    st.write(f'Probability of having heart disease: {prediction["Score"][0] * 100:.2f}%')

elif model_type == 'PyCaret Regression':
    prediction_reg = predict_model_reg(finalized_model_reg, data=input_data)
    st.subheader('Regression Prediction')
    st.write(f'Predicted Value: {prediction_reg["Label"][0]}')
