from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('model/best_anemia_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/check')
def check():
    return render_template('check.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form in check.html
    Gender = float(request.form['gender'])
    Hemoglobin = float(request.form['hemoglobin'])
    MCH = float(request.form['mch'])
    MCHC = float(request.form['mchc'])
    MCV = float(request.form['mcv'])

    # Prepare the input array directly
    features_values = np.array([[Gender, Hemoglobin, MCH, MCHC, MCV]])

    # Predict the result
    prediction = model.predict(features_values)[0]
    if prediction == 0:
        result = "You don't have any Anemic Disease"
    else:
        result = "You have anemic disease"

    text = f"Hence, based on calculation: {result}"
    return render_template('predict.html', prediction_text=text)

if __name__ == '__main__':
    app.run(debug=True)
