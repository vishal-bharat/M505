from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('FineTech_app_ML_model.joblib')

# Define feature names (adjust these based on your model's expected inputs)
feature_names = ['age', 'income', 'app_opens', 'time_spent', 'features_used']

@app.route('/')
def home():
    """Render the homepage with the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission, make prediction, and display result."""
    try:
        # Extract features from the form
        features = [float(request.form[name]) for name in feature_names]
        
        # Make prediction
        prediction = model.predict([features])[0]
        
        # Check if model supports probability estimates
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba([features])[0][1]  # Probability of class 1
            result = f"This customer is {'likely' if prediction == 1 else 'not likely'} to subscribe with a probability of {probability:.2f}."
        else:
            result = f"Prediction: {prediction}"
        
        return render_template('result.html', result=result)
    except ValueError:
        result = "Please enter valid numeric values for all fields."
        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)