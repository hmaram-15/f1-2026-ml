from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load race prediction model
model = joblib.load('race_model.pkl')
constructor_map = joblib.load('constructor_map.pkl')

# Load tire degradation model
tire_model = joblib.load('tire_model.pkl')
tire_poly = joblib.load('tire_poly.pkl')
race_map = joblib.load('race_map.pkl')
compound_map = joblib.load('compound_map.pkl')


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


@app.route('/predict_tire', methods=['GET'])
def predict_tire():
    tyre_life = int(request.args.get('tyre_life'))
    compound = request.args.get('compound')
    race = request.args.get('race')
    stint = int(request.args.get('stint', 1))
    
    compound_num = compound_map.get(compound.upper(), -1)
    race_num = race_map.get(race, -1)
    
    if compound_num == -1 or race_num == -1:
        return jsonify({'error': 'Unknown compound or race'}), 400
    
    features = np.array([[tyre_life, compound_num, race_num, stint]])
    features_poly = tire_poly.transform(features)
    delta_pred = tire_model.predict(features_poly)[0]
    
    return jsonify({
        'compound': compound,
        'race': race,
        'tyre_life': tyre_life,
        'stint': stint,
        'predicted_delta_seconds': round(float(delta_pred), 3)
    })


@app.route('/races', methods=['GET'])
def races():
    return jsonify(list(race_map.keys()))


@app.route('/compounds', methods=['GET'])
def compounds():
    return jsonify(list(compound_map.keys()))


if __name__ == '__main__':
    app.run(debug=True)