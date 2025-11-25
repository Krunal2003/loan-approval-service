# train_model.py - Run this first to create your model
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Create synthetic loan dataset
np.random.seed(42)
n_samples = 1000

data = {
    'income': np.random.randint(20000, 150000, n_samples),
    'age': np.random.randint(18, 70, n_samples),
    'loan_amount': np.random.randint(5000, 50000, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples),
    'employment_years': np.random.randint(0, 40, n_samples)
}

df = pd.DataFrame(data)

# Create target: approved if credit_score > 600 and income > loan_amount/2
df['approved'] = ((df['credit_score'] > 600) & 
                  (df['income'] > df['loan_amount'] / 2)).astype(int)

# Add some noise
noise = np.random.random(n_samples) > 0.9
df.loc[noise, 'approved'] = 1 - df.loc[noise, 'approved']

# Prepare data
X = df[['income', 'age', 'loan_amount', 'credit_score', 'employment_years']]
y = df['approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print(f"Training Accuracy: {train_score:.3f}")
print(f"Testing Accuracy: {test_score:.3f}")

# Save model and scaler
joblib.dump(model, 'loan_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\n✅ Model and scaler saved successfully!")
print("Files created: loan_model.pkl, scaler.pkl")