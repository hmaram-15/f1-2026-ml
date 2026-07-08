from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
from dotenv import load_dotenv
from google import genai
import requests

load_dotenv()
gemini_api_key = os.getenv('GEMINI_API_KEY')
gemini_client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None

# Store conversations in memory, keyed by session_id
conversations = {}

JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"

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
valid_combinations = joblib.load('valid_combinations.pkl')

# Realistic max TyreLife per compound, based on 95th percentile across 2026 races
MAX_TYRE_LIFE = {
    'SOFT': 33,
    'MEDIUM': 35,
    'HARD': 37,
    'INTERMEDIATE': 3,
    'WET': 0,
}


def fetch_race_context():
    """Fetch current championship standings and latest race results from Jolpica"""
    try:
        # Find the latest completed round
        rounds = get_completed_rounds_for_chat()
        if not rounds:
            return "(No 2026 race data available yet)"
        
        latest_round = max(rounds)
        
        # Get standings after the latest round
        standings_url = f"{JOLPICA_BASE}/2026/{latest_round}/driverStandings.json"
        standings_resp = requests.get(standings_url, timeout=5)
        standings_data = standings_resp.json()
        
        # Get results from the latest round
        results_url = f"{JOLPICA_BASE}/2026/{latest_round}/results.json"
        results_resp = requests.get(results_url, timeout=5)
        results_data = results_resp.json()
        
        standings_list = standings_data.get('MRData', {}).get('StandingsTable', {}).get('StandingsLists', [])
        results_list = results_data.get('MRData', {}).get('RaceTable', {}).get('Races', [])
        
        top5_standings = standings_list[0].get('DriverStandings', [])[:5] if standings_list else []
        top3_results = results_list[0].get('Results', [])[:3] if results_list else []
        
        context = f"""
Current 2026 F1 Championship (after Round {latest_round}):
Top 5 Driver Standings: {top5_standings}
Latest Race (Round {latest_round}) Top 3: {top3_results}
"""
        return context
    except Exception as e:
        return f"(Live data unavailable: {e})"


def get_completed_rounds_for_chat():
    """Ask Jolpica how many 2026 races have results so far."""
    rounds = set()
    offset = 0
    limit = 100
    
    while True:
        url = f"{JOLPICA_BASE}/2026/results.json?limit={limit}&offset={offset}"
        response = requests.get(url)
        data = response.json()
        
        races = data['MRData']['RaceTable']['Races']
        for r in races:
            rounds.add(int(r['round']))
        
        total = int(data['MRData']['total'])
        offset += limit
        
        if offset >= total:
            break
    
    return sorted(rounds)


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
    compound = request.args.get('compound', '').upper()
    race = request.args.get('race')
    stint = int(request.args.get('stint', 1))
    
    compound_num = compound_map.get(compound, -1)
    race_num = race_map.get(race, -1)
    
    if compound_num == -1 or race_num == -1:
        return jsonify({'error': 'Unknown compound or race'}), 400

    if (race_num, compound_num) not in valid_combinations:
        return jsonify({'error': f'{compound} has no reliable training data for {race} — try a different compound or race'}), 400

    max_life = MAX_TYRE_LIFE.get(compound, 0)
    if max_life == 0:
        return jsonify({'error': f'{compound} tires have no reliable 2026 data yet'}), 400
    
    if tyre_life > max_life:
        return jsonify({
            'error': f'{compound} tires rarely exceed {max_life} laps in 2026 data — prediction would be unreliable extrapolation',
            'max_reliable_tyre_life': max_life
        }), 400
    
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


@app.route('/chat', methods=['POST'])
def chat():
    if not gemini_api_key:
        return jsonify({'error': 'Gemini API key not configured'}), 500
    
    data = request.get_json()
    user_message = data.get('message')
    session_id = data.get('session_id', 'default')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    if session_id not in conversations:
        conversations[session_id] = []
    
    race_context = fetch_race_context()
    
    system_context = f"""You are an F1 2026 season analyst. Use this live data to answer questions accurately:

{race_context}

Answer the user's question about F1 2026 based on this data and your general F1 knowledge."""
    
    conversations[session_id].append({
        'role': 'user',
        'parts': [{'text': f"{system_context}\n\nUser question: {user_message}"}]
    })
    
    try:
        response = gemini_client.models.generate_content(
        model='gemini-2.5-flash',
        contents=conversations[session_id]
)
        
        conversations[session_id].append({
            'role': 'model',
            'parts': [{'text': response.text}]
        })
        
        return jsonify({
            'response': response.text,
            'context_used': race_context[:200]
        })
    except Exception as e:
        return jsonify({'error': f'Gemini API error: {str(e)}'}), 500


@app.route('/races', methods=['GET'])
def races():
    return jsonify(list(race_map.keys()))


@app.route('/compounds', methods=['GET'])
def compounds():
    return jsonify(list(compound_map.keys()))


if __name__ == '__main__':
    app.run(debug=True)