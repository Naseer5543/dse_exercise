import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the saved model from the file
loaded_model = joblib.load('lgbr_cars.model')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['data']
        input_data = np.array(data).reshape(1, -1)
        predictions = loaded_model.predict(input_data)
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
