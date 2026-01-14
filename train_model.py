# train_model.py
# --------------------------------------------------
# Patient Survival Prediction - Training Script
# Refactored directly from original Jupyter Notebook
# --------------------------------------------------

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, roc_curve, auc



# --------------------------------------------------
# 1. Load Data
# --------------------------------------------------


DATA_PATH = "/content/drive/MyDrive/Survival-Patient/Survival.csv"

df = pd.read_csv(DATA_PATH)

print("✅ Data loaded successfully")
print(f"Dataset shape: {df.shape}")
# --------------------------------------------------
# 2. Column Selection (EXACT from notebook)
# --------------------------------------------------

cols = [
    'Treated_with_drugs',
    'Patient_Age',
    'Patient_Body_Mass_Index',
    'Patient_Smoker',
    'Patient_Rural_Urban',
    'Patient_mental_condition',
    'A', 'B', 'C', 'D', 'E', 'F', 'Z',
    'Number_of_prev_cond',
    'Survived_1_year'
]

df = df[cols]


# --------------------------------------------------
# 3. Data Cleaning (EXACT notebook logic)
# --------------------------------------------------

# Remove "Cannot say" responses
df = df[df.Patient_Smoker != 'Cannot say']

# Fill Number_of_prev_cond with median
df['Number_of_prev_cond'] = df['Number_of_prev_cond'].fillna(
    df['Number_of_prev_cond'].median()
)

# Fill remaining missing values with mode
for column in df.columns:
    df[column] = df[column].fillna(df[column].mode()[0])

print("Data cleaning completed")


# --------------------------------------------------
# 4. One-Hot Encoding (same as notebook)
# --------------------------------------------------

categorical_cols = [
    'Treated_with_drugs',
    'Patient_Smoker',
    'Patient_Rural_Urban',
    'Patient_mental_condition'
]

df_encoded = pd.get_dummies(df, columns=categorical_cols)


# --------------------------------------------------
# 5. Feature / Target Split
# --------------------------------------------------

X = df_encoded.drop(columns=['Survived_1_year'])
y = df_encoded['Survived_1_year']

# Save feature names (CRITICAL for Streamlit)
feature_names = X.columns.tolist()

print("Total features:", len(feature_names))


# --------------------------------------------------
# 6. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=45
)


# --------------------------------------------------
# 7. Model Training (same hyperparameters)
# --------------------------------------------------

model = GradientBoostingClassifier(
    max_depth=4,
    n_estimators=150,
    learning_rate=0.1,
    random_state=45
)

model.fit(X_train, y_train)

print("Model training completed")


# --------------------------------------------------
# 8. Model Evaluation
# --------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Performance")
print("Accuracy:", round(accuracy, 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# --------------------------------------------------
# 8.1 Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# --------------------------------------------------
# 8.2 ROC Curve & AUC
# --------------------------------------------------

y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print("\nROC AUC Score:", round(roc_auc, 4))

# --------------------------------------------------
# 9. Save Model Artifacts
# --------------------------------------------------


joblib.dump(model, "/content/drive/MyDrive/Survival-Patient/gradient_boosting.pkl")
joblib.dump(feature_names, "/content/drive/MyDrive/Survival-Patient/feature_names.pkl")

print("\n💾 Model and feature schema saved successfully")
print("Saved files:")
print("- models/gradient_boosting.pkl")
print("- models/feature_names.pkl")

