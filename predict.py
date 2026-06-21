import joblib
import numpy as np

# Load saved model and preprocessors
model      = joblib.load('tire_model.pkl')
poly       = joblib.load('tire_poly.pkl')
race_map   = joblib.load('race_map.pkl')
compound_map = joblib.load('compound_map.pkl')

print('Available circuits:', list(race_map.keys()))
print('Available compounds:', list(compound_map.keys()))

def predict_lap_delta(circuit, compound, tyre_life):
    if circuit not in race_map:
        print(f'Circuit {circuit} not in training data')
        return None
    if compound not in compound_map:
        print(f'Compound {compound} not in training data')
        return None

    race_num     = race_map[circuit]
    compound_num = compound_map[compound]

    X = np.array([[tyre_life, compound_num, race_num]])
    X_poly = poly.transform(X)
    delta = model.predict(X_poly)[0]

    print(f'{circuit} | {compound} | Tyre life {tyre_life} laps → +{delta:.3f}s above fastest lap')
    return delta

# Test predictions
predict_lap_delta('Monaco', 'MEDIUM', 10)
predict_lap_delta('Monaco', 'MEDIUM', 30)
predict_lap_delta('Monaco', 'HARD',   20)
predict_lap_delta('Australia', 'HARD', 15)
predict_lap_delta('Australia', 'HARD', 40)