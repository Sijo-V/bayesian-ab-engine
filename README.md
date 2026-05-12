# 🧪 Bayesian A/B Test Significance Engine

> An end-to-end Bayesian experimentation platform that replaces p-value testing with direct probability statements, detects novelty effects automatically, and outputs a clear **GO / HOLD / NO GO** decision with revenue impact — all through a browser-based dashboard.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat&logo=flask&logoColor=white)
![PyMC](https://img.shields.io/badge/PyMC-5.x-0D6EFD?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 📌 What This Project Does

Most A/B test tools give you a p-value and leave you to interpret it. This engine does three things differently:

| Feature | What it means |
|---|---|
| **Bayesian inference** | Instead of a yes/no p-value, you get a direct probability: *"94.2% chance treatment beats control"* |
| **Novelty effect detection** | Automatically flags when early conversion lifts are caused by user curiosity, not genuine improvement |
| **Go / No-go decision engine** | Evaluates 4 conditions and outputs a verdict + dollar revenue impact range |

---

## 🖥️ Dashboard Preview

```
┌─────────────────────────────────────────────────────────────┐
│  🧪 Bayesian A/B Test Significance Engine                    │
├──────────────────┬──────────────────┬────────────┬──────────┤
│  P(Treatment     │  Absolute lift   │  Relative  │ Novelty  │
│  wins): 94.2%    │  +1.8pp          │  lift:+36% │ Low      │
├──────────────────┴──────────────────┴────────────┴──────────┤
│  [Posterior Distribution Chart]  [Novelty Timeline Chart]   │
├─────────────────────────────────────────────────────────────┤
│  Decision: ✅ GO — All conditions met. Safe to ship.        │
│  ✅ P(B>A) = 94.2%  ✅ HDI: 0.6%–3.1%  ✅ No SRM  ✅ Low novelty │
│  Revenue impact: $60,000 – $310,000 / year                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Project Architecture

```
bayesian-ab-engine/
│
├── app.py                          # Flask server — API routes & entry point
│
├── data/
│   ├── simulate_events.py          # Synthetic experiment data generator
│   └── loader.py                   # CSV ingestion + column validation
│
├── engine/
│   ├── bayesian_model.py           # PyMC Beta-Binomial model (NUTS sampler)
│   ├── posterior_analysis.py       # P(B>A), HDI, ROPE, revenue impact
│   ├── novelty_detector.py         # Rolling cohort lift + slope regression
│   └── validator.py                # Sample ratio mismatch (chi-squared)
│
├── dashboard/
│   └── templates/
│       └── index.html              # Full dashboard — HTML + Chart.js
│
├── notebooks/
│   └── exploration.ipynb           # Step-by-step model walkthrough
│
├── tests/
│   └── test_model.py               # pytest unit tests for all engine modules
│
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

The engine runs through **5 stages** every time you click Run Analysis:

```
Raw Data  →  Validate  →  Bayesian Model  →  Novelty Check  →  Decision
```

**1. Data input**
Upload a real experiment CSV or use the built-in simulator. The simulator generates synthetic users with a deliberate novelty effect baked in — so the detector always has something real to find.

**2. Validation**
A chi-squared test checks for Sample Ratio Mismatch (SRM) — whether users were fairly split 50/50 between control and treatment. If the split is uneven, results cannot be trusted and a warning is raised before any analysis runs.

**3. Bayesian inference**
A `Beta(1,1)` flat prior is placed on each variant's true conversion rate. A `Binomial` likelihood updates the prior with observed conversions. PyMC's NUTS sampler runs 4 chains × 2,000 draws = **8,000 posterior samples**. From these, the engine computes:
- `P(B > A)` — probability treatment beats control
- `HDI 95%` — credible interval on the lift
- `ROPE overlap` — how much of the posterior is in the "doesn't matter commercially" zone
- Revenue impact — HDI bounds × annual GMV baseline

**4. Novelty effect detection**
Daily lift (treatment CVR − control CVR) is computed per experiment day. A linear regression fits a trend line through the daily values. A statistically significant **negative slope** = lift was high early and is decaying = novelty effect flagged. The novelty score (0–1) feeds directly into the verdict.

**5. Go / No-go decision**
Four conditions are evaluated:

| Condition | Description |
|---|---|
| `P(B > A) ≥ threshold` | Statistical probability treatment wins (default 90%) |
| `HDI lower ≥ MDE` | Minimum detectable effect cleared (default 0.3pp) |
| `No SRM` | Assignment was fair and unbiased |
| `Novelty score < 0.5` | Lift is stable, not inflating from novelty |

- All four pass → **GO**
- High novelty risk → **HOLD — extend experiment**
- Low probability → **NO GO — insufficient evidence**
- SRM detected → **NO GO — experiment broken**

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or newer
- Git

### 1. Clone the repository
```bash
git clone https://github.com/Sijo-V/bayesian-ab-engine.git
cd bayesian-ab-engine
```

### 2. Create and activate a virtual environment

**Mac / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
> ⏳ PyMC is a large package — first install takes 3–8 minutes. This is normal.

### 4. Start the server
```bash
python app.py
```

### 5. Open the dashboard
```
http://localhost:5050
```

Click **Run Analysis** in the sidebar to run your first analysis on simulated data.

---

## 📂 Using Your Own Data

Upload a CSV file via the **Upload CSV** tab in the dashboard sidebar. The file must have these exact columns:

| Column | Type | Description |
|---|---|---|
| `user_id` | string | Unique identifier per user |
| `variant` | string | Exactly `"control"` or `"treatment"` |
| `timestamp` | datetime | When the visit occurred |
| `session_day` | integer | Days since experiment start (0-indexed) |
| `converted` | integer | `1` = converted, `0` = did not convert |

### Preparing the Kaggle A/B Testing Dataset

Download `ab_data.csv` from [kaggle.com/datasets/zhangluyuan/ab-testing](https://www.kaggle.com/datasets/zhangluyuan/ab-testing), then run the included prep script:

```bash
python prep.py
```

This renames the `group` column to `variant`, derives `session_day` from the timestamp, and removes duplicate user IDs. Upload the resulting `ab_data_clean.csv` to the dashboard.

---

## 🧪 Running the Tests

```bash
pytest tests/ -v
```

The test suite covers:
- Data simulator output shape and balance
- Loader column validation and error handling
- SRM detection on balanced and imbalanced splits
- Model input integrity checks
- Novelty detector on experiments with and without novelty effects

---

## 📓 Jupyter Notebook Walkthrough

```bash
pip install jupyter
jupyter notebook notebooks/exploration.ipynb
```

The notebook walks through every engine module step by step — simulating data, running the Bayesian model, plotting the posterior, and visualising the novelty effect timeline — with charts and explanations at each stage.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.10+ | All backend logic |
| Bayesian model | PyMC 5 | NUTS MCMC posterior sampling |
| Posterior analysis | ArviZ | HDI computation, trace diagnostics |
| Statistics | SciPy | Chi-squared SRM test, Gaussian KDE |
| Data | pandas, NumPy | DataFrame operations, array maths |
| Web server | Flask | REST API, serves dashboard HTML |
| Dashboard | HTML + Chart.js | Posterior chart, novelty timeline, decision panel |
| PDF export | ReportLab | Stakeholder-ready analysis report |
| Simulation | Faker | Realistic synthetic user IDs |
| Testing | pytest | Unit tests for all engine modules |

---

## 📊 API Reference

The Flask server exposes two endpoints:

### `POST /api/run`
Runs the full analysis pipeline on **simulated data**.

**Request body (JSON):**
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

### `POST /api/upload`
Runs the full analysis pipeline on an **uploaded CSV file**.

**Form data:**
- `file` — the CSV file
- `config` — JSON string with the same model settings as above

**Both endpoints return:**
```json
{
  "ok": true,
  "prob_b_beats_a": 0.942,
  "lift_mean": 0.018,
  "lift_hdi_lower": 0.006,
  "lift_hdi_upper": 0.031,
  "novelty_risk": "Low",
  "verdict": "GO",
  "revenue_low": 60000,
  "revenue_high": 310000,
  "kde_x": [...],
  "kde_y": [...],
  "daily_lifts": [...]
}
```

---

## 🧠 Key Concepts

**Why Bayesian over a t-test?**
A t-test gives a p-value — you can only reject or fail to reject the null hypothesis. Bayesian inference gives `P(B > A)` — the direct probability treatment is better. This is the question product teams actually want answered.

**What is ROPE?**
Region Of Practical Equivalence. A zone (e.g. ±0.2pp) where the lift is too small to matter commercially. If most of the posterior falls inside ROPE, the variants are practically equivalent — even if technically different. This prevents shipping changes with negligible business impact.

**What is a novelty effect?**
When users click a new design simply because it is new. Conversion lifts in the first few days decay to baseline by week 2–3. This engine fits a trend line to daily lift values and flags a statistically significant negative slope as a novelty effect — preventing false positive shipping decisions.

**What is SRM?**
Sample Ratio Mismatch. When users are not split evenly between variants (e.g. 60/40 instead of 50/50). Caused by assignment bugs, bot traffic, or caching issues. Makes results statistically invalid regardless of the model output.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 👤 Author

**Sijo V**
[github.com/Sijo-V](https://github.com/Sijo-V)

---

*Built as a portfolio project demonstrating Bayesian statistical modelling, full-stack Python development, and real-world experimentation analysis.*
