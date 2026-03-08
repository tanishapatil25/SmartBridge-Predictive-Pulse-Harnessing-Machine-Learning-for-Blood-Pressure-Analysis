from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        Age = int(request.form['Age'])
        Gender = 1 if request.form['Gender'].lower() == 'male' else 0
        Systolic = int(request.form['Systolic'])
        Diastolic = int(request.form['Diastolic'])

        severity_map = {'Mild': 0, 'Moderate': 1, 'Sever': 2}
        Severity = severity_map.get(request.form['Severity'], 0)

        diag_map = {'<1 Year': 0, '1-5 Year': 1, '>5 Year': 2}
        Diagnosed = diag_map.get(request.form['Whendiagnoused'], 0)

        input_data = np.array([[Age, Gender, Systolic, Diastolic, Severity, Diagnosed]])
        prediction = model.predict(input_data)[0]

        return render_template('result.html', prediction=f"Predicted Stage: {prediction}")
    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
