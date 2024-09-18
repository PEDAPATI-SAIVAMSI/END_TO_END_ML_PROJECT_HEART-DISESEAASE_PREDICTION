import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, confusion_matrix

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

# Evaluate the model
def evaluate_model(y_test, y_pred, model_choice):
    if model_choice == 'Random Forest Regressor':
        mse = mean_squared_error(y_test, y_pred)
        st.write(f'Mean Squared Error: {mse}')
    else:
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f'Accuracy: {accuracy * 100:.2f}%')
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))
        
        # Display confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(plt)

# Streamlit app
st.title('Heart Disease Prediction')

# Load and display data
data = load_data()
st.write('Dataset:', data.head())

# Data visualization
st.subheader("Data Distribution")
plt.figure(figsize=(10, 6))
sns.countplot(x='HeartDisease', data=data, palette='Set2')
st.pyplot(plt)

# Preprocess the data
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

if model_choice != 'Random Forest Regressor':
    prediction_proba = model.predict_proba(input_data_scaled)
    st.subheader('Prediction')
    if prediction[0] == 1:
        st.write('The model predicts that the patient **has heart disease**.')
    else:
        st.write('The model predicts that the patient **does not have heart disease**.')
    st.write(f'Probability of having heart disease: {prediction_proba[0][1] * 100:.2f}%')
