from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import requests
import pandas as pd

JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
ROUNDS = None  # set dynamically at runtime


def time_to_seconds(time_str):
    if time_str is None:
        return None
    if ':' not in time_str:
        return None
    minutes, seconds = time_str.split(':')
    return int(minutes) * 60 + float(seconds)

def get_completed_rounds():
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

def get_qualifying(round_num):
    url = f"{JOLPICA_BASE}/2026/{round_num}/qualifying.json"
    response = requests.get(url)
    data = response.json()
    
    results = data['MRData']['RaceTable']['Races'][0]['QualifyingResults']
    
    rows = []
    for r in results:
        q1 = time_to_seconds(r.get('Q1'))
        q2 = time_to_seconds(r.get('Q2'))
        q3 = time_to_seconds(r.get('Q3'))
        
        times = [t for t in [q1, q2, q3] if t is not None]
        best_time = min(times)
        
        rows.append({
            'Round': round_num,
            'Driver': r['Driver']['code'],
            'Constructor': r['Constructor']['name'],
            'GridPosition': int(r['position']),
            'BestQualiTime': best_time,
        })
    
    df = pd.DataFrame(rows)
    pole_time = df['BestQualiTime'].min()
    df['GapToPole'] = df['BestQualiTime'] - pole_time
    
    return df


def get_standings(round_num):
    if round_num == 1:
        return None
    
    prior_round = round_num - 1
    url = f"{JOLPICA_BASE}/2026/{prior_round}/driverStandings.json"
    response = requests.get(url)
    data = response.json()
    
    standings_list = data['MRData']['StandingsTable']['StandingsLists']
    
    if len(standings_list) == 0:
        return None
    
    standings = standings_list[0]['DriverStandings']
    
    rows = []
    for s in standings:
        if 'position' not in s:
            print(f"Round {round_num} — bad entry:", s)
            continue
        rows.append({
            'Round': round_num,
            'Driver': s['Driver']['code'],
            'ChampPosition': int(s['position']),
        })
    
    return pd.DataFrame(rows)


def get_race_results(round_num):
    url = f"{JOLPICA_BASE}/2026/{round_num}/results.json"
    response = requests.get(url)
    data = response.json()
    
    results = data['MRData']['RaceTable']['Races'][0]['Results']
    
    rows = []
    for r in results:
        rows.append({
            'Round': round_num,
            'Driver': r['Driver']['code'],
            'FinishPosition': int(r['position']),
        })
    
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ROUNDS = get_completed_rounds()
    print(f"Completed rounds detected: {ROUNDS}")
    
    all_data = []
    
    for round_num in ROUNDS:
        print(f"Processing round {round_num}...")
        
        quali_df = get_qualifying(round_num)
        race_df = get_race_results(round_num)
        standings_df = get_standings(round_num)
        
        if standings_df is None:
            print(f"Skipping round {round_num} — no standings available")
            continue
        
        merged = pd.merge(quali_df, race_df, on=['Round', 'Driver'])
        merged = pd.merge(merged, standings_df, on=['Round', 'Driver'])
        
        all_data.append(merged)
    
    final_df = pd.concat(all_data, ignore_index=True)
    
    constructor_map = {name: i for i, name in enumerate(final_df['Constructor'].unique())}
    final_df['ConstructorNum'] = final_df['Constructor'].map(constructor_map)
    
    final_df['Podium'] = (final_df['FinishPosition'] <= 3).astype(int)
    final_df['Winner'] = (final_df['FinishPosition'] == 1).astype(int)
    
    print(final_df['Podium'].value_counts())
    print(final_df['Winner'].value_counts())
    print(f"\nTotal rows: {len(final_df)}")

    # Define features and target
    features = ['GridPosition', 'GapToPole', 'ConstructorNum', 'ChampPosition']
    X = final_df[features].values
    y = final_df['Podium'].values
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost
    model = XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(model, 'race_model.pkl')
    joblib.dump(constructor_map, 'constructor_map.pkl')
    print("\nModel saved to race_model.pkl")