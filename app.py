from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load models
model = joblib.load('race_model.pkl')
constructor_map = joblib.load('constructor_map.pkl')

@app.route('/predict', methods=['GET'])
def predict():
    driver = request.args.get('driver')
    grid = int(request.args.get('grid'))
    gap = float(request.args.get('gap'))
    constructor = request.args.get('constructor')
    champ = int(request.args.get('champ'))
    
    constructor_num = constructor_map.get(constructor, -1)
    
    features = np.array([[grid, gap, constructor_num, champ]])
    podium_prob = model.predict_proba(features)[0][1]
    podium_pred = int(model.predict(features)[0])
    
    return jsonify({
        'driver': driver,
        'podium': podium_pred,
        'probability': round(float(podium_prob), 3)
    })

if __name__ == '__main__':
    app.run(debug=True)