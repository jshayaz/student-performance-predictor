import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import joblib
import numpy as np

# Load your dataset
df = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\individual project\student-performance-predictor\data\student_performance_cleaned.csv")
# Ensure required columns are present
required_columns = ['GPA', 'internal_marks', 'lab_marks', 'attendance', 'assignment_submission',
                    'backlogs', 'external_tuition', 'mentorship', 'performance']

if not all(col in df.columns for col in required_columns):
    raise ValueError("Dataset is missing required columns.")

# One-hot encode categorical columns
categorical_cols = ['assignment_submission', 'external_tuition', 'mentorship']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Separate features and target
X = df_encoded.drop("performance", axis=1)
y = df_encoded["performance"]

# Save column order for app prediction
joblib.dump(X.columns.tolist(), "reference_columns.pkl")

# Scale numeric features
numeric_features = ['GPA', 'internal_marks', 'lab_marks', 'attendance', 'backlogs']
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Balance data with oversampling
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Evaluate
y_pred = model.predict(X_test)
print("📊 Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, "student_performance_model_v2.pkl")
print("✅ Model saved as student_performance_model_v2.pkl")
