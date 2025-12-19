import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

print("1. Loading Real Dataset (7,043 rows)...")
df = pd.read_csv('churn_data.csv')

# ---------------------------------------------------------
# NEW STEP: Cleaning the "Dirty" Data
# The real dataset has some blank spaces " " in TotalCharges. 
# We force them to be numbers.
# ---------------------------------------------------------
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True) # Remove the 11 rows that had missing values

# Select columns
df = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'Churn']]

print("2. Encoding Data...")
le = LabelEncoder()
df['Contract'] = le.fit_transform(df['Contract'])
joblib.dump(le, 'contract_encoder.pkl')

# Define X and y
X = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract']]
y = df['Churn']

print(f"3. Training Model on {len(df)} customers...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Check Accuracy
accuracy = model.score(X_test, y_test) * 100
print(f"📊 Model Accuracy: {accuracy:.2f}%")

joblib.dump(model, 'churn_model.pkl')
print("✅ Success! The 'Brain' has been upgraded with real data.")