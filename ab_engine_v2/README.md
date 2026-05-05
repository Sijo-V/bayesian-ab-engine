# Bayesian A/B Test Significance Engine v2

End-to-end Bayesian A/B testing framework with a Flask backend and
a pure HTML/CSS/JS dashboard — no Streamlit required.

---

## Quick start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the server
```bash
python app.py
```

### 3. Open the dashboard
```
http://localhost:5050
```

That's it. Configure settings in the sidebar, click **Run Analysis**.

---

## Project structure

```
ab_engine_v2/
├── app.py                          # Flask server + API routes
├── data/
│   ├── simulate_events.py          # Synthetic experiment generator
│   └── loader.py                   # CSV / DataFrame ingestion
├── engine/
│   ├── bayesian_model.py           # PyMC Beta-Binomial model (NUTS)
│   ├── posterior_analysis.py       # P(B>A), HDI, ROPE, revenue impact
│   ├── novelty_detector.py         # Rolling cohort lift + slope regression
│   └── validator.py                # Sample ratio mismatch check
├── dashboard/
│   └── templates/
│       └── index.html              # Full dashboard (HTML + Chart.js)
├── tests/
│   └── test_model.py               # pytest unit tests
├── notebooks/
│   └── exploration.ipynb           # Model walkthrough
└── requirements.txt
```

---

## API routes

| Route | Method | Description |
|---|---|---|
| `GET  /` | GET | Serves the HTML dashboard |
| `POST /api/run` | POST | Runs full analysis on simulated data |
| `POST /api/upload` | POST | Runs full analysis on uploaded CSV |

### POST /api/run — body

```json
{
  "n_users": 10000,
  "control_cvr": 0.05,
  "treatment_cvr": 0.065,
  "novelty_boost": 0.04,
  "experiment_days": 21,
  "prob_threshold": 0.90,
  "min_lift": 0.003,
  "rope_bound": 0.002,
  "annual_gmv": 10000000
}
```

### POST /api/upload — multipart form

- `file`: CSV file
- `config`: JSON string with model settings (same keys as above)

---

## Using your own data

Upload a CSV with these columns:

| Column | Type | Example |
|---|---|---|
| `user_id` | string | `abc-123` |
| `variant` | string | `control` or `treatment` |
| `timestamp` | datetime | `2024-01-05 14:32:00` |
| `session_day` | int | `4` |
| `converted` | int | `1` or `0` |

---

## How the engine works

**Bayesian model (PyMC)**
Places a `Beta(1,1)` prior on each variant's true CVR, updates with observed
conversions via Binomial likelihood, samples posterior with NUTS (4 chains).

**Novelty detector**
Computes daily lift (treatment CVR − control CVR) per experiment day, fits a
linear regression over time, flags a statistically significant negative slope
as a novelty effect.

**Go / No-go logic**
Four conditions: P(B>A) ≥ threshold · HDI lower ≥ MDE · No SRM · Low novelty.
All four pass → GO. Novelty risk → HOLD. Low probability → NO GO.

---

## Run tests

```bash
pytest tests/ -v
```
