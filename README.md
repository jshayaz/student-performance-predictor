# 🎓 Student Performance Predictor (Engineering Students)

This project predicts whether an engineering student is likely to **Pass** or **Fail** in a subject based on various academic and study-related inputs using a machine learning model.

## 📌 Features

- ✅ Predicts student performance using Random Forest Classifier
- ✅ UI built using Streamlit
- ✅ Balanced dataset with `RandomOverSampler`
- ✅ Supports model retraining and saving
- ✅ Focused on key study-related features
- ✅ Visual feedback when a student is predicted to **Fail**

## 🧠 Model Inputs (Features Used)

| Feature                 | Description                              |
|------------------------|------------------------------------------|
| GPA                    | Grade Point Average                      |
| Internal Marks         | Marks from internal assessments          |
| Lab Marks              | Marks from lab exams or lab evaluations  |
| Attendance (%)         | Percentage of class attendance           |
| Assignment Submission  | Whether the student submits assignments  |
| Backlogs               | Number of active backlogs                |
| External Tuition       | If the student attends external tuitions |
| Mentorship             | If the student is under a mentor program |

## 🖥 Technologies Used

- Python
- Pandas
- Scikit-learn
- Imbalanced-learn
- Matplotlib & Seaborn
- Streamlit (for UI)
- Joblib (for saving model)

