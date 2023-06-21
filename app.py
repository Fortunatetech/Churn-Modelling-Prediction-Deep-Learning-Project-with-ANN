from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained model
import pickle

# Load the trained model using pickle
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)


# Create a Flask app
app = Flask(__name__)

# Define a route for model prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.json['data']
    
    # Preprocess the input data (scaling, encoding, etc.)
    preprocessed_data = preprocess_data(data)
    
    # Make predictions using the preprocessed data
    predictions = model.predict(preprocessed_data)
    
    # Convert predictions to a human-readable format
    results = postprocess_predictions(predictions)
    
    # Return the results as a JSON response
    return jsonify(results)

# Define a function to preprocess the input data
def preprocess_data(data):
     # Create a DataFrame from the input data
    df = pd.DataFrame(data)

    label_encoder = LabelEncoder()
    # Encode categorical variables using LabelEncoder
    df['Geography'] = label_encoder.transform(df['Geography'])
    df['Gender'] = label_encoder.transform(df['Gender'])
    
    # Perform feature scaling on numerical features
    scaler = StandardScaler()
    df_scaled = scaler.transform(df)
    
    # Return the preprocessed data
    return df_scaled

# Define a function to postprocess the predictions
def postprocess_predictions(predictions):
     # Convert the predictions to binary labels (0 or 1)
    binary_predictions = [1 if pred >= 0.5 else 0 for pred in predictions]
    
    # Convert binary labels to human-readable format (churn or not churn)
    results = ['Churn' if pred == 1 else 'Not Churn' for pred in binary_predictions]
    
    # Return the results
    return results


# Run the Flask app
if __name__ == '__main__':
    app.run()