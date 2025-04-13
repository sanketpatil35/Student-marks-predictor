# app.py

from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model/exam_score_predictor.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        study_hours = float(request.form['study_hours'])
        practice_score = float(request.form['practice_score'])
        previous_score = float(request.form['previous_score'])

        # Prepare input for prediction
        input_features = np.array([[study_hours, practice_score, previous_score]])

        # Predict the final score
        prediction = model.predict(input_features)[0]

        return render_template('index.html', prediction_text=f"Predicted Final Exam Score: {round(prediction, 2)}")
    except Exception as e:
        return f"Something went wrong: {e}"

if __name__ == '__main__':
    app.run(debug=True)
