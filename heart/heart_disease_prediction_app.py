import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# UI color styling
st.markdown(
    '''
    <style>
    body {
        background-color: #F0F2F6;
    }
    .stApp {
        color: #003366;
        background-color: #DCE3F3;
    }
    .sidebar .sidebar-content {
        background-color: #4A90E2;
    }
    h1 {
        color: #003366;
    }
    .st-bb {
        background-color: #4A90E2;
        color: white;
    }
    </style>
    ''', 
    unsafe_allow_html=True
)

# Header for the app
st.title("Heart Disease Prediction")
st.markdown("### Predict whether a patient has heart disease based on their data.")

# Dataset loading function
@st.cache_data
def load_heart_data():
    csv_path = "https://raw.githubusercontent.com/PEDAPATI-SAIVAMSI/END_TO_END_ML_PROJECT_HEART-DISESEAASE_PREDICTION/main/heart/heart.csv"
    return pd.read_csv(csv_path)

# Load the dataset
data = load_heart_data()

# Select features and target
X = data.drop("HeartDisease", axis=1)
y = data["HeartDisease"]

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
categorical_cols = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_cols),
    ("cat", Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop='first', handle_unknown="ignore")),
    ]), categorical_cols),
])

# Define the parameter grid
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_features': [2, 4, 6, 8],
}

# Build pipeline and GridSearchCV
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# User Input Form
st.subheader('Predict Heart Disease')

# Create input fields for user data
age = st.slider('Age', min_value=20, max_value=100, value=50)
sex = st.selectbox('Sex', options=['M', 'F'])
chest_pain = st.selectbox('Chest Pain Type', options=data['ChestPainType'].unique())
resting_bp = st.slider('Resting BP', min_value=80, max_value=200, value=120)
cholesterol = st.slider('Cholesterol', min_value=100, max_value=400, value=200)
fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])
resting_ecg = st.selectbox('Resting ECG', options=data['RestingECG'].unique())
max_hr = st.slider('Max HR', min_value=60, max_value=220, value=150)
exercise_angina = st.selectbox('Exercise-induced Angina', options=['Y', 'N'])
oldpeak = st.slider('Oldpeak', min_value=0.0, max_value=6.0, value=1.0)
st_slope = st.selectbox('ST Slope', options=data['ST_Slope'].unique())

# Create input dataframe
user_data = pd.DataFrame({
    'Age': [age], 
    'RestingBP': [resting_bp], 
    'Cholesterol': [cholesterol],
    'MaxHR': [max_hr], 
    'Oldpeak': [oldpeak], 
    'Sex': [sex], 
    'ChestPainType': [chest_pain],
    'FastingBS': [fasting_bs], 
    'RestingECG': [resting_ecg], 
    'ExerciseAngina': [exercise_angina], 
    'ST_Slope': [st_slope]
})

# Predict using the pipeline
if st.button("Predict"):
    prediction = grid_search.best_estimator_.predict(user_data)
    prediction_prob = grid_search.best_estimator_.predict_proba(user_data)

    # Display the prediction result
    st.subheader("Prediction")
    if prediction[0] == 1:
        st.write("**The model predicts that you have heart disease.**")
    else:
        st.write("**The model predicts that you do not have heart disease.**")

    # Display prediction probabilities
    st.subheader("Prediction Probabilities")
    st.write(f"Probability of having heart disease: {prediction_prob[0][1] * 100:.2f}%")
    st.write(f"Probability of not having heart disease: {prediction_prob[0][0] * 100:.2f}%")

# Add some styling
st.markdown("""<style>
    .css-1v0mbdj {
        background-color: #f5f5f5;
    }
    .css-1v0mbdj h1 {
        color: #2c3e50;
    }
    .css-1v0mbdj h3 {
        color: #3498db;
    }
</style>""", unsafe_allow_html=True)
