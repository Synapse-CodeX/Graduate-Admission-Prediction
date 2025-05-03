

from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("admission_model.h5")
scaler = joblib.load("scaler.save")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        gre = float(request.form['gre'])
        toefl = float(request.form['toefl'])
        rating = float(request.form['rating'])
        sop = float(request.form['sop'])
        lor = float(request.form['lor'])
        cgpa = float(request.form['cgpa'])
        research = int(request.form['research'])

        features = np.array([[gre, toefl, rating, sop, lor, cgpa, research]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0][0]

        return render_template('index.html', prediction_text=f'Predicted Chance of Admit: {prediction:.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == '__main__':
    app.run(debug=True)
