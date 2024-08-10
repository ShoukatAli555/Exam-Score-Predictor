import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load('Student_Score_Predictor')  # Replace 'your_model.pkl' with your model file path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    study_hours = int(request.form['study_hours'])
    prediction = model.predict([[study_hours]])[0]  # Remove unnecessary indexing
    prediction = round(prediction, 2)  # Round the prediction to two decimal places
    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
