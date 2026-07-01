# F1 2026 ML Pipeline

Machine learning models for F1 2026 race analysis, served via a Flask API on Render.

**Live API:** https://f1-2026-ml.onrender.com

---

## Models

### Model 01 — Tire Degradation (tire_analysis.py)
Predicts lap time delta from circuit fastest lap based on tyre life, compound, and stint number.

- **Type:** Polynomial regression (degree 2)
- **Data:** FastF1 lap telemetry — 8 races, 5,746 clean laps
- **MAE:** 1.130 seconds delta
- **Features:** TyreLife, CompoundNum, RaceNum, Stint

**Data filtering methodology:**
- `TrackStatus == '1'` — green flag laps only, removes Safety Car and VSC laps
- `TyreLife > 1` — removes cold out laps
- 2 standard deviation filter per race/compound — removes outliers from incidents or mechanical issues
- Finisher filter (`winner_laps - 1`) — excludes DNFs and heavily lapped drivers whose stints don't represent real tyre behaviour

**Key insight:** Lap times are normalized as deltas from each circuit's fastest lap rather than raw times. This removes circuit-to-circuit variation and dropped MAE from ~5.5s to ~1.1s.

---

### Model 02 — Podium Prediction (race_model.py)
XGBoost binary classifier predicting whether a driver will finish in the top 3.

- **Type:** XGBoost classification
- **Data:** Jolpica F1 API — 7 rounds (Rounds 2–8), 153 training rows
- **Accuracy:** 87% overall, 60% podium precision/recall
- **Features:** GridPosition, GapToPole, ConstructorNum, ChampPosition

**Data leakage prevention:** Championship standings are always fetched from the round prior to the one being predicted, ensuring the model only sees information available at prediction time.

**Why 2026 only:** The 2026 regulation cycle is a full reset — constructor performance hierarchies from 2023–2025 are irrelevant and would actively mislead the model.

---

## API

The Flask API (`app.py`) is deployed on Render and exposes a single prediction endpoint:

```
GET /predict?driver=RUS&grid=1&gap=0.0&constructor=Mercedes&champ=3
```

Response:

```json
{
  "driver": "RUS",
  "podium": 1,
  "probability": 0.648
}
```

---

## Tech Stack

- Python 3.13
- FastF1 — lap telemetry and session data
- pandas — data processing and merging
- scikit-learn — polynomial features and linear regression
- XGBoost — podium classification
- Flask + flask-cors — REST API
- joblib — model serialization
- Render — API hosting

---

## Project Structure

```
tire_analysis.py     # Model 01 — tire degradation
race_model.py        # Model 02 — podium prediction
app.py               # Flask API
predict.py           # Local prediction utility
requirements.txt
```

---

*2026 season data only. Tire model auto-loads completed races; race model round range is updated per race.*