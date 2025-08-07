from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

# Load the preprocessing pipeline and KMeans model
preprocessor = joblib.load('preprocessing')
model = joblib.load('Clust_Univ.pkl')

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        input_data = {
            'Operating Airline': request.form['airline'],
            'Terminal': request.form['terminal'],
            'Month': int(request.form['month']),
            'Year': int(request.form['year']),
            'Passenger Count': float(request.form['passenger_count'])
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Preprocess and predict
        transformed = preprocessor.transform(input_df)
        prediction = model.predict(transformed)[0]

        return render_template('result.html', prediction=int(prediction))

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
