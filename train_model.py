# train_model.py
# --------------------------------------------------
# Patient Survival Prediction - Training Script
# Refactored directly from original Jupyter Notebook
# --------------------------------------------------



import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Survival.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------------------------------
# 2. Load Data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("✅ Data loaded successfully")
print(f"Dataset shape: {df.shape}")

# --------------------------------------------------
# 3. Column Selection
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
# 4. Data Cleaning
# --------------------------------------------------

df = df[df.Patient_Smoker != 'Cannot say']

df['Number_of_prev_cond'] = df['Number_of_prev_cond'].fillna(
    df['Number_of_prev_cond'].median()
)

for column in df.columns:
    df[column] = df[column].fillna(df[column].mode()[0])

print("✅ Data cleaning completed")

# --------------------------------------------------
# 5. One-Hot Encoding
# --------------------------------------------------

categorical_cols = [
    'Treated_with_drugs',
    'Patient_Smoker',
    'Patient_Rural_Urban',
    'Patient_mental_condition'
]

df_encoded = pd.get_dummies(df, columns=categorical_cols)

# --------------------------------------------------
# 6. Feature / Target Split
# --------------------------------------------------

X = df_encoded.drop(columns=['Survived_1_year'])
y = df_encoded['Survived_1_year']

feature_names = X.columns.tolist()

print("Total features:", len(feature_names))

# --------------------------------------------------
# 7. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=45
)

# --------------------------------------------------
# 8. Model Training
# --------------------------------------------------

model = GradientBoostingClassifier(
    max_depth=4,
    n_estimators=150,
    learning_rate=0.1,
    random_state=45
)

model.fit(X_train, y_train)

print("✅ Model training completed")

# --------------------------------------------------
# 9. Evaluation
# --------------------------------------------------

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Performance")
print("Accuracy:", round(accuracy, 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print("\nROC AUC Score:", round(roc_auc, 4))

# --------------------------------------------------
# 10. Save Artifacts
# --------------------------------------------------

joblib.dump(model, os.path.join(MODEL_DIR, "gradient_boosting.pkl"))
joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))

print("\n💾 Model and feature schema saved successfully")
