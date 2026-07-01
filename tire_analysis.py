import fastf1
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


fastf1.Cache.enable_cache('cache')


compound_colors = {
    'SOFT':         '#E8002D',
    'MEDIUM':       '#FFF200',
    'HARD':         '#FFFFFF',
    'INTERMEDIATE': '#39B54A',
    'WET':          '#0067FF',
}

def plot_driver_degradation(laps, driver, race_name):
    driver_laps = laps.pick_drivers(driver)
    driver_laps = driver_laps[driver_laps['TrackStatus'] == '1']
    driver_laps = driver_laps[driver_laps['TyreLife'] > 1]
    driver_laps = driver_laps.copy()
    driver_laps['LapTimeSeconds'] = driver_laps['LapTime'].dt.total_seconds()
    driver_laps = driver_laps.dropna(subset=['LapTimeSeconds'])

    plt.figure(figsize=(14, 7))
    plt.gca().set_facecolor('#0A0A0A')
    plt.gcf().set_facecolor('#141414')

    for compound, color in compound_colors.items():
        stint = driver_laps[driver_laps['Compound'] == compound]
        if len(stint) < 3:
            continue

        mean = stint['LapTimeSeconds'].mean()
        std  = stint['LapTimeSeconds'].std()
        stint = stint[abs(stint['LapTimeSeconds'] - mean) < 2 * std]

        plt.scatter(stint['TyreLife'], stint['LapTimeSeconds'],
                    color=color, label=compound, zorder=3,
                    edgecolors='gray', linewidths=0.4)

        X = stint['TyreLife'].values.reshape(-1, 1)
        y = stint['LapTimeSeconds'].values

        poly    = PolynomialFeatures(degree=2)
        X_poly  = poly.fit_transform(X)
        model   = LinearRegression()
        model.fit(X_poly, y)

        x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred  = model.predict(poly.transform(x_range))
        plt.plot(x_range, y_pred, color=color, linewidth=2)

    plt.xlabel('Tyre Life (laps)', color='white')
    plt.ylabel('Lap Time (seconds)', color='white')
    plt.title(f'{driver} - Tyre Degradation by Compound · {race_name}', color='white')
    plt.tick_params(colors='white')
    plt.legend(facecolor='#1C1C1C', labelcolor='white')
    plt.grid(True, alpha=0.2, color='gray')
    plt.tight_layout()
    plt.show()

# Races to load
# Auto-load all completed 2026 races
schedule = fastf1.get_event_schedule(2026, include_testing=False)
completed = schedule[schedule['EventDate'] < pd.Timestamp.now()]
print(completed['EventName'].tolist())
RACES = [(2026, row['EventName'], '') for _, row in completed.iterrows()]

# Load all sessions and collect laps
all_laps = []

for year, race, _ in RACES:
    print(f'Loading {race} {year}...')
    try:
        s = fastf1.get_session(year, race, 'R')
        s.load(telemetry=False)

        if s.laps is None or len(s.laps) == 0:
            print(f'Skipping {race} — no lap data')
            continue
        
        laps = s.laps.copy()
        winner_laps = s.results['Laps'].max()
        finishers = s.results[
            (s.results['Laps'] >= winner_laps - 1) & 
            (s.results['Status'] != 'Retired')
        ]['DriverNumber'].values
        laps = laps[laps['DriverNumber'].isin(finishers)]
        
        laps['Race'] = race
        laps['Year'] = year
        all_laps.append(laps)
    except Exception as e:
        print(f'Skipping {race} — {e}')

combined = pd.concat(all_laps, ignore_index=True)
print("Races loaded:", combined['Race'].unique())

# Apply filters
combined = combined[combined['TrackStatus'] == '1']
combined = combined[combined['TyreLife'] > 1]
combined = combined.copy()
combined['LapTimeSeconds'] = combined['LapTime'].dt.total_seconds()
combined = combined.dropna(subset=['LapTimeSeconds'])

# Remove outliers per race per compound
def filter_outliers(df):
    filtered = []
    for (race, compound), group in df.groupby(['Race', 'Compound']):
        mean = group['LapTimeSeconds'].mean()
        std  = group['LapTimeSeconds'].std()
        filtered.append(group[abs(group['LapTimeSeconds'] - mean) < 2 * std])
    return pd.concat(filtered)

combined = filter_outliers(combined)
# Add stint number per driver
combined['Stint'] = combined.groupby(['Race', 'Driver'])['TyreLife'].transform(
    lambda x: (x.diff() < 0).cumsum() + 1
)

print(combined[['Race', 'Driver', 'Compound', 'TyreLife', 'Stint']].head(30))

print(f'\nTotal clean laps: {len(combined)}')
print(combined[['Race', 'Driver', 'Compound', 'TyreLife', 'LapTimeSeconds']].head(20))

# Encode compound as number
compound_map = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2, 'INTERMEDIATE': 3, 'WET': 4}
combined['CompoundNum'] = combined['Compound'].map(compound_map)

# Encode circuit as number
race_map = {race: i for i, race in enumerate(combined['Race'].unique())}
combined['RaceNum'] = combined['Race'].map(race_map)

# Normalize lap times per race — predict delta from fastest lap
combined['FastestLap'] = combined.groupby('Race')['LapTimeSeconds'].transform('min')
combined['LapTimeDelta'] = combined['LapTimeSeconds'] - combined['FastestLap']

# Now predict delta instead of raw lap time
X = combined[['TyreLife', 'CompoundNum', 'RaceNum', 'Stint']].values
y = combined['LapTimeDelta'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polynomial features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly  = poly.transform(X_test)

# Fit model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Evaluate
y_pred = model.predict(X_test_poly)
mae    = mean_absolute_error(y_test, y_pred)

print(f'\nModel MAE: {mae:.3f} seconds delta')
print(f'Meaning degradation predictions are off by {mae:.3f}s on average')

import joblib

# Save the trained model and preprocessors
joblib.dump(model, 'tire_model.pkl')
joblib.dump(poly, 'tire_poly.pkl')
joblib.dump(race_map, 'race_map.pkl')
joblib.dump(compound_map, 'compound_map.pkl')

print('Model saved to tire_model.pkl')