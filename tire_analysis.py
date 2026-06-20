import fastf1
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

fastf1.Cache.enable_cache('cache')

session = fastf1.get_session(2026, 'Monaco', 'R')
session.load()

laps = session.laps

# Get all Antonelli laps without quicklaps filter
ant_laps = laps.pick_drivers('LIN')
ant_laps = ant_laps[ant_laps['TrackStatus'] == '1']
ant_laps = ant_laps[ant_laps['TyreLife'] > 1]
ant_laps = ant_laps.copy()
ant_laps['LapTimeSeconds'] = ant_laps['LapTime'].dt.total_seconds()

# Remove NaT lap times
ant_laps = ant_laps.dropna(subset=['LapTimeSeconds'])

# Compound colors
compound_colors = {
    'SOFT':         '#E8002D',
    'MEDIUM':       '#FFF200',
    'HARD':         '#FFFFFF',
    'INTERMEDIATE': '#39B54A',
    'WET':          '#0067FF',
}

plt.figure(figsize=(14, 7))

# Plot each compound separately with its own regression curve
for compound, color in compound_colors.items():
    stint = ant_laps[ant_laps['Compound'] == compound]
    if len(stint) < 3:
        continue

    # Filter outliers per compound
    mean = stint['LapTimeSeconds'].mean()
    std  = stint['LapTimeSeconds'].std()
    stint = stint[abs(stint['LapTimeSeconds'] - mean) < 2 * std]

    # Scatter
    plt.scatter(stint['TyreLife'], stint['LapTimeSeconds'],
                color=color, label=compound, zorder=3,
                edgecolors='gray', linewidths=0.4)

    # Polynomial regression per compound
    X = stint['TyreLife'].values.reshape(-1, 1)
    y = stint['LapTimeSeconds'].values

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred  = model.predict(poly.transform(x_range))

    plt.plot(x_range, y_pred, color=color, linewidth=2)

plt.xlabel('Tyre Life (laps)')
plt.ylabel('Lap Time (seconds)')
plt.title('LIN - Tyre Degradation by Compound · Monaco 2026', color='white')
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().set_facecolor('#0A0A0A')
plt.gcf().set_facecolor('#141414')
plt.tick_params(colors='white')
plt.xlabel('Tyre Life (laps)', color='white')
plt.ylabel('Lap Time (seconds)', color='white')
plt.title('ANT - Tyre Degradation by Compound · Monaco 2026', color='white')
plt.legend(facecolor='#1C1C1C', labelcolor='white')
plt.grid(True, alpha=0.2, color='gray')

plt.tight_layout()
plt.show()