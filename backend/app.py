from flask import Flask, request, jsonify, render_template
import pickle
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)
# Load the model
Pkl_Filename = "saki.pkl"
with open(Pkl_Filename, 'rb') as file:
    model = pickle.load(file)


# API route for predictions
@app.route('/getPrediction', methods=['POST'])
def get_prediction():
    try:
        # Get JSON data from the request
        data = request.json
        features = np.array(data['features']).reshape(1, -1)

        # Make prediction
        pred = model.predict(features)[0]

        # Return prediction as JSON
        print(pred)
        return jsonify({'pred': round(pred, 3)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
