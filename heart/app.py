import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/PEDAPATI-SAIVAMSI/END_TO_END_ML_PROJECT_HEART-DISESEAASE_PREDICTION/main/heart/heart.csv'
    data = pd.read_csv(url)
    return data


# Exploratory Data Analysis (EDA)
def eda(data):
    st.subheader("Exploratory Data Analysis (EDA)")
    st.write("Dataset Head:", data.head())
    st.write("Dataset Description:", data.describe())
    st.write("Missing Values:", data.isnull().sum())

    st.write("Distribution of Heart Disease")
    plt.figure(figsize=(6, 4))
    sns.countplot(x='HeartDisease', data=data, palette='Set2')
    st.pyplot(plt)

    st.write("Correlation Heatmap")
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.select_dtypes(include=['float64', 'int64']).corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)

# Preprocess the data
def preprocess_data(df):
    label_encoders = {}
    
    # Encode categorical columns
    categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    
    # Scale numerical columns
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, scaler, label_encoders

# Model training
def train_model(X_train, y_train, model_choice):
    if model_choice == 'Logistic Regression':
        model = LogisticRegression()
    elif model_choice == 'Random Forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    model.fit(X_train, y_train)
    return model

# App Layout
st.title('Heart Disease Prediction')

# Load Data
data = load_data()
st.write("Dataset loaded successfully!")

# Exploratory Data Analysis
eda(data)

# Preprocess Data
X, y, scaler, label_encoders = preprocess_data(data)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection
model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest"])

# Train the Model
model = train_model(X_train, y_train, model_choice)

# Model Predictions
y_pred = model.predict(X_test)

# Evaluate the Model
st.write(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.write("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
st.pyplot(plt)

# User input for predictions
st.subheader('Predict Heart Disease')
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

# Transform user input
try:
    user_data = pd.DataFrame({
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

    user_data_scaled = scaler.transform(user_data)

    # Make prediction
    prediction = model.predict(user_data_scaled)
    if prediction[0] == 1:
        st.write("The model predicts that the patient **has heart disease**.")
    else:
        st.write("The model predicts that the patient **does not have heart disease**.")

except KeyError as e:
    st.error(f"Error in encoding input values: {e}")

