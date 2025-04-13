# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

# Load the dataset
data = pd.read_csv('data.csv')

# Features and Target
X = data[['Study Hours', 'Practice Test Score', 'Previous Exam Score']]
y = data['Final Exam Score']

# Split the data into training and testing sets (optional for checking accuracy)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Save the trained model
joblib.dump(model, 'model/exam_score_predictor.joblib')

print("✅ Model trained and saved successfully!")
