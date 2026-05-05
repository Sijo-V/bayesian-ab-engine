"""
app.py  —  Flask API server for the Bayesian A/B Test Engine
Run:  python app.py
Then open:  http://localhost:5050
"""

import sys, json, math, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask, request, jsonify, send_from_directory

import numpy as np
from data.simulate_events import simulate_experiment
from data.loader import load_from_dataframe
from engine.validator import validate_experiment
from engine.bayesian_model import prepare_inputs, run_bayesian_ab_test
from engine.posterior_analysis import analyse_posterior
from engine.novelty_detector import detect_novelty_effect

app = Flask(__name__, static_folder="dashboard/static", template_folder="dashboard/templates")

# ── Serve the dashboard ───────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("dashboard/templates", "index.html")

# ── /api/simulate  POST ───────────────────────────────────────────────────────
@app.route("/api/simulate", methods=["POST"])
def simulate():
    try:
        cfg = request.get_json(force=True) or {}
        df = simulate_experiment(
            n_users              = int(cfg.get("n_users", 10000)),
            true_control_cvr     = float(cfg.get("control_cvr", 0.05)),
            true_treatment_cvr   = float(cfg.get("treatment_cvr", 0.065)),
            novelty_boost        = float(cfg.get("novelty_boost", 0.04)),
            novelty_decay_days   = int(cfg.get("novelty_decay_days", 5)),
            experiment_days      = int(cfg.get("experiment_days", 21)),
        )
        return jsonify({
            "ok": True,
            "rows": len(df),
            "preview": df.head(10).to_dict(orient="records"),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


# ── /api/run  POST  (main analysis endpoint) ──────────────────────────────────
@app.route("/api/run", methods=["POST"])
def run_analysis():
    try:
        cfg = request.get_json(force=True) or {}

        # 1. Generate / load data
        df = simulate_experiment(
            n_users              = int(cfg.get("n_users", 10000)),
            true_control_cvr     = float(cfg.get("control_cvr", 0.05)),
            true_treatment_cvr   = float(cfg.get("treatment_cvr", 0.065)),
            novelty_boost        = float(cfg.get("novelty_boost", 0.04)),
            novelty_decay_days   = int(cfg.get("novelty_decay_days", 5)),
            experiment_days      = int(cfg.get("experiment_days", 21)),
        )

        # 2. Validate
        val = validate_experiment(df)

        # 3. Bayesian model
        inputs = prepare_inputs(df)
        trace  = run_bayesian_ab_test(
            inputs,
            draws=int(cfg.get("draws", 1500)),
            tune=int(cfg.get("tune", 800)),
            chains=int(cfg.get("chains", 2)),
        )

        # 4. Posterior analysis
        rope_bound = float(cfg.get("rope_bound", 0.002))
        annual_gmv = float(cfg.get("annual_gmv", 10_000_000))
        summary = analyse_posterior(
            trace,
            rope_lower=-rope_bound,
            rope_upper=rope_bound,
            annual_baseline_gmv=annual_gmv,
        )

        # 5. Novelty detection
        novelty = detect_novelty_effect(df)

        # 6. Decision logic
        prob_thresh = float(cfg.get("prob_threshold", 0.90))
        min_lift    = float(cfg.get("min_lift", 0.003))

        checks = {
            "prob_beats_a": summary.prob_b_beats_a >= prob_thresh,
            "hdi_positive":  summary.lift_hdi_lower >= min_lift,
            "no_srm":        not val.srm_detected,
            "low_novelty":   novelty.novelty_score < 0.5,
        }
        if all(checks.values()):
            verdict = "GO"
        elif checks["prob_beats_a"] and checks["hdi_positive"] and not checks["no_srm"]:
            verdict = "NO GO — SRM present"
        elif checks["prob_beats_a"] and checks["hdi_positive"] and not checks["low_novelty"]:
            verdict = "HOLD — extend experiment"
        elif not checks["prob_beats_a"]:
            verdict = "NO GO — insufficient evidence"
        else:
            verdict = "INCONCLUSIVE"

        # 7. Build posterior KDE for chart
        lift_samples = trace.posterior["lift"].values.flatten()
        kde_x = list(np.linspace(float(lift_samples.min()), float(lift_samples.max()), 120))
        from scipy.stats import gaussian_kde
        kde    = gaussian_kde(lift_samples, bw_method=0.15)
        kde_y  = list(float(v) for v in kde(kde_x))
        kde_x  = [round(v * 100, 4) for v in kde_x]   # convert to %

        return jsonify({
            "ok": True,
            # experiment summary
            "control_n":       val.control_n,
            "treatment_n":     val.treatment_n,
            "control_cvr":     round(float(df[df.variant=="control"]["converted"].mean()), 5),
            "treatment_cvr":   round(float(df[df.variant=="treatment"]["converted"].mean()), 5),
            # posterior
            "prob_b_beats_a":  round(summary.prob_b_beats_a, 4),
            "lift_mean":       round(summary.lift_mean, 5),
            "lift_hdi_lower":  round(summary.lift_hdi_lower, 5),
            "lift_hdi_upper":  round(summary.lift_hdi_upper, 5),
            "rel_lift_mean":   round(summary.rel_lift_mean, 4),
            "rope_overlap":    round(summary.rope_overlap, 4),
            "revenue_low":     round(summary.revenue_low,  0) if summary.revenue_low  else None,
            "revenue_high":    round(summary.revenue_high, 0) if summary.revenue_high else None,
            # novelty
            "novelty_score":   novelty.novelty_score,
            "novelty_risk":    novelty.risk_level,
            "novelty_slope":   novelty.slope,
            "novelty_p":       novelty.slope_p_value,
            "novelty_rec":     novelty.recommendation,
            "daily_lifts":     novelty.daily_lifts,
            # validation
            "srm_detected":    val.srm_detected,
            "srm_p":           round(val.srm_p_value, 4),
            "val_warnings":    val.warnings,
            # decision
            "verdict":         verdict,
            "checks":          checks,
            # chart data
            "kde_x":  kde_x,
            "kde_y":  kde_y,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


# ── /api/upload  POST  (CSV upload) ──────────────────────────────────────────
@app.route("/api/upload", methods=["POST"])
def upload_csv():
    try:
        import pandas as pd, io
        file = request.files.get("file")
        if not file:
            return jsonify({"ok": False, "error": "No file uploaded"}), 400
        df = pd.read_csv(file, parse_dates=["timestamp"])
        df = load_from_dataframe(df)

        cfg  = json.loads(request.form.get("config", "{}"))
        rope_bound = float(cfg.get("rope_bound", 0.002))
        annual_gmv = float(cfg.get("annual_gmv", 10_000_000))
        prob_thresh = float(cfg.get("prob_threshold", 0.90))
        min_lift    = float(cfg.get("min_lift", 0.003))

        val     = validate_experiment(df)
        inputs  = prepare_inputs(df)
        trace   = run_bayesian_ab_test(inputs, draws=1500, tune=800, chains=2)
        summary = analyse_posterior(trace, rope_lower=-rope_bound, rope_upper=rope_bound,
                                     annual_baseline_gmv=annual_gmv)
        novelty = detect_novelty_effect(df)

        checks = {
            "prob_beats_a": summary.prob_b_beats_a >= prob_thresh,
            "hdi_positive":  summary.lift_hdi_lower >= min_lift,
            "no_srm":        not val.srm_detected,
            "low_novelty":   novelty.novelty_score < 0.5,
        }
        if all(checks.values()):
            verdict = "GO"
        elif checks["prob_beats_a"] and checks["hdi_positive"] and not checks["no_srm"]:
            verdict = "NO GO — SRM present"
        elif checks["prob_beats_a"] and checks["hdi_positive"] and not checks["low_novelty"]:
            verdict = "HOLD — extend experiment"
        elif not checks["prob_beats_a"]:
            verdict = "NO GO — insufficient evidence"
        else:
            verdict = "INCONCLUSIVE"

        lift_samples = trace.posterior["lift"].values.flatten()
        from scipy.stats import gaussian_kde
        kde_x = list(np.linspace(float(lift_samples.min()), float(lift_samples.max()), 120))
        kde   = gaussian_kde(lift_samples, bw_method=0.15)
        kde_y = [float(v) for v in kde(kde_x)]
        kde_x = [round(v * 100, 4) for v in kde_x]

        return jsonify({
            "ok": True,
            "control_n": val.control_n, "treatment_n": val.treatment_n,
            "control_cvr": round(float(df[df.variant=="control"]["converted"].mean()), 5),
            "treatment_cvr": round(float(df[df.variant=="treatment"]["converted"].mean()), 5),
            "prob_b_beats_a": round(summary.prob_b_beats_a, 4),
            "lift_mean": round(summary.lift_mean, 5),
            "lift_hdi_lower": round(summary.lift_hdi_lower, 5),
            "lift_hdi_upper": round(summary.lift_hdi_upper, 5),
            "rel_lift_mean": round(summary.rel_lift_mean, 4),
            "rope_overlap": round(summary.rope_overlap, 4),
            "revenue_low": round(summary.revenue_low, 0) if summary.revenue_low else None,
            "revenue_high": round(summary.revenue_high, 0) if summary.revenue_high else None,
            "novelty_score": novelty.novelty_score,
            "novelty_risk": novelty.risk_level,
            "novelty_slope": novelty.slope,
            "novelty_p": novelty.slope_p_value,
            "novelty_rec": novelty.recommendation,
            "daily_lifts": novelty.daily_lifts,
            "srm_detected": val.srm_detected,
            "srm_p": round(val.srm_p_value, 4),
            "val_warnings": val.warnings,
            "verdict": verdict,
            "checks": checks,
            "kde_x": kde_x, "kde_y": kde_y,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    print("\n🧪  Bayesian A/B Engine  →  http://localhost:5050\n")
    app.run(debug=True, port=5050)
