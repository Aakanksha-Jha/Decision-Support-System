"""
live_predict.py
---------------
Live-updated ML prediction. Rebuilds the current month's rainfall features from
live Open-Meteo data (observed month-to-date + forecast), keeps all other
features from the latest dataset row, and re-scores the saved XGBoost model.

ASSUMPTIONS (verify with src/validate_live_scaling.py before relying on it):
  1. mean_rain ~ monthly rainfall total in METRES (mm / 1000). Inferred from the
     seasonal pattern in the dataset, not documented anywhere.
  2. sum_rain = mean_rain x pixel_count (pixel_count is constant per circle in the data).
  3. max_rain = mean_rain x that circle's median historical max/mean ratio.
  4. Non-rain features and 'impact' are the latest known values unless
     data/current_impact.csv (object_id, impact) is provided.
Circle coordinates are placeholders until replaced with real centroids.
"""
import calendar
import os
import sys
import datetime as dt

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config  # noqa: E402
import live_rain  # noqa: E402
from predict_impact import categorise, demand_from_probability  # noqa: E402

MM_TO_FEATURE = 0.001
LIVE_PATH = os.path.join(config.OUTPUT_DIR, "predictions_live.csv")
IMPACT_OVERRIDE = os.path.join(config.DATA_DIR, "current_impact.csv")


def month_projection(centroids, today=None):
    """Projected total rain (mm) for the current calendar month, per circle."""
    today = today or dt.date.today()
    dim = calendar.monthrange(today.year, today.month)[1]
    past = today.day - 1
    remaining = dim - today.day + 1                      # incl. today
    fdays = max(1, min(16, remaining))
    rows = []
    for i in range(0, len(centroids), live_rain.CHUNK):
        part = centroids.iloc[i:i + live_rain.CHUNK]
        res = live_rain._fetch_chunk(part["latitude"].tolist(), part["longitude"].tolist(),
                                     past_days=max(past, 0), forecast_days=fdays)
        for cid, r in zip(part[config.ID_COLUMN], res):
            vals = [v or 0.0 for v in r["daily"]["precipitation_sum"]]
            have = len(vals)
            total = sum(vals)
            missing = max(dim - have, 0)                 # days beyond forecast horizon
            total += missing * (total / max(have, 1))
            rows.append({config.ID_COLUMN: cid, "projected_month_mm": total})
    return pd.DataFrame(rows)


def run(centroids=None, today=None):
    centroids = centroids if centroids is not None else pd.read_csv(config.CENTROIDS_PATH)
    model = joblib.load(config.MODEL_PATH)
    df = pd.read_csv(config.RAW_DATASET_PATH)
    df[config.TIME_COLUMN] = pd.to_datetime(df[config.TIME_COLUMN], format="%Y_%m")
    ratio = (df["max_rain"] / df["mean_rain"]).groupby(df[config.ID_COLUMN]).median()
    npx = (df["sum_rain"] / df["mean_rain"]).groupby(df[config.ID_COLUMN]).median()
    latest = df[df[config.TIME_COLUMN] == df[config.TIME_COLUMN].max()].copy()

    proj = month_projection(centroids, today)
    latest = latest.merge(proj, on=config.ID_COLUMN, how="left")
    latest["mean_rain"] = latest["projected_month_mm"] * MM_TO_FEATURE
    latest["max_rain"] = latest["mean_rain"] * latest[config.ID_COLUMN].map(ratio)
    latest["sum_rain"] = latest["mean_rain"] * latest[config.ID_COLUMN].map(npx)
    if os.path.exists(IMPACT_OVERRIDE):
        imp = pd.read_csv(IMPACT_OVERRIDE).set_index(config.ID_COLUMN)["impact"]
        latest["impact"] = latest[config.ID_COLUMN].map(imp).fillna(latest["impact"])

    prob = model.predict_proba(latest[config.FEATURE_COLUMNS])[:, 1]
    out = latest[[config.ID_COLUMN, "sum_population", "projected_month_mm"]].copy()
    out["predicted_probability"] = prob
    out["impact_category"] = out["predicted_probability"].apply(categorise)
    out = pd.concat([out, pd.DataFrame([demand_from_probability(p, s) for p, s in zip(prob, out["sum_population"])])], axis=1)
    out = out.sort_values(["predicted_probability", "predicted_affected_population"], ascending=False).reset_index(drop=True)
    out["priority_rank"] = out.index + 1
    out["updated_at"] = pd.Timestamp.now(tz="Asia/Kolkata").strftime("%d %b %Y, %H:%M IST")
    return out


if __name__ == "__main__":
    o = run()
    o.to_csv(LIVE_PATH, index=False)
    print(o["impact_category"].value_counts().to_string())
    print(o[[config.ID_COLUMN, "predicted_probability", "impact_category", "priority_rank"]].head(10).to_string(index=False))
