import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

print("--- Loading Student Performance Dataset ---")
df = pd.read_csv('student_data.csv')
print(df.head())

# Splitting Features and Target
X = df[['Study_Hours', 'Attendance_Rate', 'Prev_Score']]
y = df['Final_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Training Random Forest Regressor ---")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")

# Save the trained model artifact
joblib.dump(model, 'student_model.pkl')
print("\nModel successfully trained and saved as 'student_model.pkl'!")
