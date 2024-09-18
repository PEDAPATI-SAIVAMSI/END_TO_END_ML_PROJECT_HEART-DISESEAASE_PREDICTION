import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the data from GitHub
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/PEDAPATI-SAIVAMSI/END_TO_END_ML_PROJECT_HEART-DISESEAASE_PREDICTION/main/heart/heart.csv'
    data = pd.read_csv(url, on_bad_lines='skip')  # Skips bad lines
    return data

# Exploratory Data Analysis (EDA)
def eda(data):
    """Exploratory Data Analysis (EDA)"""
    st.subheader("Exploratory Data Analysis (EDA)")

    # Summary statistics
    st.write("Dataset Summary:", data.describe())

    # Visualize distribution of target variable
    st.write("Heart Disease Distribution")
    plt.figure(figsize=(6, 4))
    sns.countplot(x='HeartDisease', data=data, palette='Set2')
    st.pyplot(plt)

    # Correlation matrix - only for numerical data
    st.write("Correlation Matrix")
    numerical_data = data.select_dtypes(include=['int64', 'float64'])  # Select only numeric columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    st.pyplot(plt)

    # Check for missing values
    st.write("Missing Values:", data.isnull().sum())

# Preprocess the data
def preprocess_data(df):
    label_encoders = {}

    # Check for missing values in 'Sex' column and handle
    st.write("Missing values in 'Sex':", df['Sex'].isnull().sum())
    df['Sex'].fillna('M', inplace=True)

    # Encoding categorical columns
    le_sex = LabelEncoder()
    le_sex.fit(['M', 'F'])
    df['Sex'] = le_sex.transform(df['Sex'])
    label_encoders['Sex'] = le_sex

    le_cp = LabelEncoder()
    le_cp.fit(['TA', 'ATA', 'NAP', 'ASY'])
    df['ChestPainType'] = le_cp.transform(df['ChestPainType'])
    label_encoders['ChestPainType'] = le_cp

    le_recg = LabelEncoder()
    le_recg.fit(['Normal', 'ST', 'LVH'])
    df['RestingECG'] = le_recg.transform(df['RestingECG'])
    label_encoders['RestingECG'] = le_recg

    le_ea = LabelEncoder()
    le_ea.fit(['Y', 'N'])
    df['ExerciseAngina'] = le_ea.transform(df['ExerciseAngina'])
    label_encoders['ExerciseAngina'] = le_ea

    le_sts = LabelEncoder()
    le_sts.fit(['Up', 'Flat', 'Down'])
    df['ST_Slope'] = le_sts.transform(df['ST_Slope'])
    label_encoders['ST_Slope'] = le_sts

    return df, label_encoders

# Train the model
def train_model(X_train, y_train, model_type):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if model_type == 'SVM':
        model = SVC(kernel='linear', probability=True)
    elif model_type == 'Random Forest Classifier':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'Random Forest Regressor':
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train_scaled, y_train)
    return model, scaler

# Evaluate model performance
def evaluate_model(y_test, y_pred, model_choice):
    if model_choice != 'Random Forest Regressor':
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1 Score: {f1:.2f}")
    else:
        st.write(f"Predictions: {y_pred}")

# Streamlit app
st.title('Heart Disease Prediction')

# Load and display data
data = load_data()
st.write('Dataset:', data.head())

# Exploratory Data Analysis (EDA)
eda(data)

# Data Preprocessing
data, label_encoders = preprocess_data(data)

# Split the dataset
X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection
model_choice = st.selectbox('Choose a model', ['SVM', 'Random Forest Classifier', 'Random Forest Regressor'])

# Train the model
model, scaler = train_model(X_train, y_train, model_choice)

# Make predictions
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)

# Evaluate the model
evaluate_model(y_test, y_pred, model_choice)

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

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_data_scaled)

# Display the prediction result
st.subheader('Prediction')
if model_choice != 'Random Forest Regressor':
    prediction_proba = model.predict_proba(input_data_scaled)
    if prediction[0] == 1:
        st.write('The model predicts that the patient **has heart disease**.')
    else:
        st.write('The model predicts that the patient **does not have heart disease**.')
    st.write(f'Probability of having heart disease: {prediction_proba[0][1] * 100:.2f}%')
