from flask import Flask, request, jsonify
import pandas as pd
from pycaret.classification import load_model, predict_model

# Load the model
model = load_model('final_iris_model')

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get data posted as json
    data = pd.DataFrame([data])  # Convert json to pandas DataFrame
    predictions = predict_model(model, data=data)  # Make predictions
    return jsonify(predictions.to_dict(orient='records'))  # Return predictions as json

if __name__ == '__main__':
    app.run(port=12345)
