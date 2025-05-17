from flask import Flask, render_template, request, jsonify
import joblib
import os
import numpy as np

app = Flask(__name__)

# Paths
model_path = '/Users/anand/Desktop/Machine Learning/11.Projects/01.Algerian_Forest_Fires/App/Model_file/final_model.pkl'

# Load the pipeline model (includes scaler + classifier)
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
            # Extract inputs from the form
            temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            DC = float(request.form.get('DC'))
            ISI = float(request.form.get('ISI'))
            BUI = float(request.form.get('BUI'))

            # Prepare input for prediction
            input_data = np.array([[temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI]])

            # Predict directly (scaling is inside pipeline)
            prediction = model.predict(input_data)

            return render_template('index.html', prediction=prediction[0])

        except Exception as e:
            return jsonify({"error": str(e)})

    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == '__main__':
    app.run(debug=True)
