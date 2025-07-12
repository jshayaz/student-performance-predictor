import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("student_performance_model_v2.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Engineering Student Performance Predictor")
st.write("This app predicts whether a student is likely to **Pass** or **Fail**, based on academic and study-related inputs.")

# Input fields
GPA = st.number_input("GPA (0 - 10 scale)", min_value=0.0, max_value=10.0, value=6.5)
internal_marks = st.number_input("Internal Marks (0 - 30)", min_value=0, max_value=30, value=20)
lab_marks = st.number_input("Lab Marks (0 - 20)", min_value=0, max_value=20, value=15)
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=80)
assignment_submission = st.selectbox("Assignment Submission", ["Yes", "No"])
backlogs = st.number_input("Number of Backlogs", min_value=0, max_value=10, value=0)
external_tuition = st.selectbox("External Tuition", ["Yes", "No"])
mentorship = st.selectbox("Faculty Mentorship Received", ["Yes", "No"])

# Prepare input DataFrame
user_input = pd.DataFrame({
    "GPA": [GPA],
    "internal_marks": [internal_marks],
    "lab_marks": [lab_marks],
    "attendance": [attendance],
    "assignment_submission": [assignment_submission],
    "backlogs": [backlogs],
    "external_tuition": [external_tuition],
    "mentorship": [mentorship]
})

# One-hot encode categorical features to match training
user_input_encoded = pd.get_dummies(user_input)

# Load reference feature names
reference_features = joblib.load("reference_columns.pkl")

# Add missing columns (if any) with 0, and ensure order
for col in reference_features:
    if col not in user_input_encoded.columns:
        user_input_encoded[col] = 0

user_input_encoded = user_input_encoded[reference_features]

# Scale numeric fields
numeric_cols = ['GPA', 'internal_marks', 'lab_marks', 'attendance', 'backlogs']
user_input_encoded[numeric_cols] = scaler.transform(user_input_encoded[numeric_cols])

# Predict
if st.button("Predict"):
    prediction = model.predict(user_input_encoded)[0]

    if prediction == 1:
        st.success("✅ The student is likely to **Pass**.")
    else:
        st.error("❌ The student is likely to **Fail**.")

        # Give helpful suggestions
        st.markdown("### 🔍 Suggestions to Improve:")
        st.markdown("- Increase GPA and internal/lab marks")
        st.markdown("- Attend classes regularly")
        st.markdown("- Complete all assignments on time")
        st.markdown("- Clear backlogs and seek faculty mentorship")
