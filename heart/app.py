import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Collect and Understand Data
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/PEDAPATI-SAIVAMSI/END_TO_END_ML_PROJECT_HEART-DISESEAASE_PREDICTION/main/heart/heart.csv'
    data = pd.read_csv(url, on_bad_lines='skip')  # Skips bad lines
    return data

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

    # Correlation matrix
    st.write("Correlation Matrix")
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    st.pyplot(plt)

    # Check for missing values
    st.write("Missing Values:", data.isnull().sum())

# 2. Data Preprocessing
def preprocess_data(df):
    label_encoders = {}

    # Handle missing values
    df['Sex'].fillna('M', inplace=True)
    
    # Encoding categorical variables
    le_sex = LabelEncoder()
    le_cp = LabelEncoder()
    le_recg = LabelEncoder()
    le_ea = LabelEncoder()
    le_sts = LabelEncoder()

    df['Sex'] = le_sex.fit_transform(df['Sex'])
    df['ChestPainType'] = le_cp.fit_transform(df['ChestPainType'])
    df['RestingECG'] = le_recg.fit_transform(df['RestingECG'])
    df['ExerciseAngina'] = le_ea.fit_transform(df['ExerciseAngina'])
    df['ST_Slope'] = le_sts.fit_transform(df['ST_Slope'])

    label_encoders = {
        'Sex': le_sex, 'ChestPainType': le_cp, 'RestingECG': le_recg, 
        'ExerciseAngina': le_ea, 'ST_Slope': le_sts
    }

    return df, label_encoders

# 3. Select and Train Models
def train_model(X_train, y_train, model_type):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if model_type == 'SVM':
        model = SVC(kernel='linear', probability=True)
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    elif model_type == 'Random Forest Classifier':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        param_grid = {'n_estimators': [100, 200], 'max_depth': [5, 10, None]}

    # Hyperparameter tuning with GridSearchCV
    grid = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    
    return grid.best_estimator_, scaler

# 4. Evaluate Models
def evaluate_model(y_test, y_pred, model_choice):
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f'Accuracy: {accuracy * 100:.2f}%')
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    
    # Confusion matrix
    st.write("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)

# 5. Streamlit App
st.title('Heart Disease Prediction - End-to-End ML Project')

# Collect and Understand Data
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
model_choice = st.selectbox('Choose a model', ['SVM', 'Random Forest Classifier'])

# Model Training
model, scaler = train_model(X_train, y_train, model_choice)

# Model Prediction
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)

# Evaluate the Model
evaluate_model(y_test, y_pred, model_choice)

# 6. Optimize and Tune
st.subheader("Model Optimization")
st.write("Best Hyperparameters:")
st.write(model.get_params())

# 7. Predict using User Input
st.header('Predict Heart Disease')
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

# Convert inputs to DataFrame
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

# Scale and Predict
input_data_scaled = scaler.transform(input_data)
prediction = model.predict(input_data_scaled)

# Display Prediction
st.subheader('Prediction Result')
if prediction[0] == 1:
    st.write('The model predicts that the patient **has heart disease**.')
else:
    st.write('The model predicts that the patient **does not have heart disease**.')
