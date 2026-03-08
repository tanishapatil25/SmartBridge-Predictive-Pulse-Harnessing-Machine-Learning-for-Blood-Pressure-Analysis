import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('patient_data.csv')

# Clean columns 
df.columns = df.columns.str.strip()

# Gender is named 'C' in the dataset
df.rename(columns={'C': 'Gender'}, inplace=True)

# Clean string values
for col in df.select_dtypes(['object']).columns:
    df[col] = df[col].astype(str).str.strip()

# Map Gender
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Map Severity
severity_map = {'Mild': 0, 'Moderate': 1, 'Sever': 2}
df['Severity'] = df['Severity'].map(severity_map)

# Map Diagnosed (Whendiagnoused)
diag_map = {'<1 Year': 0, '1 - 5 Years': 1, '>5 Years': 2, '1-5 Year': 1, '>5 Year': 2}
df['Whendiagnoused'] = df['Whendiagnoused'].map(diag_map)

# Extract numeric values from Age, Systolic, Diastolic
def extract_mean(valStr):
    if pd.isna(valStr): return valStr
    valStr = str(valStr).replace('+', '')
    parts = valStr.split('-')
    if len(parts) == 2:
        return (int(float(parts[0].strip())) + int(float(parts[1].strip()))) / 2
    else:
        return int(float(parts[0].strip()))

df['Age'] = df['Age'].apply(extract_mean)
df['Systolic'] = df['Systolic'].apply(extract_mean)
df['Diastolic'] = df['Diastolic'].apply(extract_mean)

# Drop any rows with NaN in the required columns
required_cols = ['Age', 'Gender', 'Systolic', 'Diastolic', 'Severity', 'Whendiagnoused', 'Stages']
df = df.dropna(subset=required_cols)

X = df[['Age', 'Gender', 'Systolic', 'Diastolic', 'Severity', 'Whendiagnoused']]
y = df['Stages']

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model
joblib.dump(model, "model.pkl")
print("Model created successfully from actual data.")
