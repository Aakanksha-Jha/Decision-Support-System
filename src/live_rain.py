"""
live_rain.py
------------
Fetches LIVE observed + forecast daily rainfall for every Revenue Circle
centroid from the free Open-Meteo API (no key needed) and flags circles using
the IMD heavy-rainfall category (>= 64.5 mm in 24 h).

This is a real-time WATCH layer. It does not change the ML model's score.
NOTE: circle coordinates in data/circle_centroids.csv are placeholders until
replaced with real Revenue Circle centroids, so live values are only as
accurate as those coordinates.
"""
import json
import os
import sys
import urllib.parse
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config  # noqa: E402

API = "https://api.open-meteo.com/v1/forecast"
HEAVY_MM = 64.5      # IMD "heavy rain" lower bound, mm per 24 h
PAST_DAYS = 7
FORECAST_DAYS = 3
CHUNK = 40


def _fetch_chunk(lats, lons, past_days=PAST_DAYS, forecast_days=FORECAST_DAYS):
    q = urllib.parse.urlencode({
        "latitude": ",".join(f"{v:.4f}" for v in lats),
        "longitude": ",".join(f"{v:.4f}" for v in lons),
        "daily": "precipitation_sum",
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "Asia/Kolkata",
    })
    with urllib.request.urlopen(f"{API}?{q}", timeout=30) as r:
        data = json.loads(r.read().decode())
    return data if isinstance(data, list) else [data]


def fetch_live_rain(centroids: pd.DataFrame) -> pd.DataFrame:
    """Return one row per circle: rain_past_3d, rain_past_7d, rain_next_3d_max, heavy_rain_alert."""
    rows = []
    for i in range(0, len(centroids), CHUNK):
        part = centroids.iloc[i:i + CHUNK]
        results = _fetch_chunk(part["latitude"].tolist(), part["longitude"].tolist())
        for cid, res in zip(part[config.ID_COLUMN], results):
            rain = [v or 0.0 for v in res["daily"]["precipitation_sum"]]
            past, fut = rain[:PAST_DAYS], rain[PAST_DAYS:]
            rows.append({
                config.ID_COLUMN: cid,
                "rain_past_3d_mm": round(sum(past[-3:]), 1),
                "rain_past_7d_mm": round(sum(past), 1),
                "rain_next_3d_max_mm": round(max(fut) if fut else 0.0, 1),
                "heavy_rain_alert": bool((max(fut) if fut else 0) >= HEAVY_MM or max(past[-3:]) >= HEAVY_MM),
            })
    out = pd.DataFrame(rows)
    out["fetched_at"] = pd.Timestamp.now(tz="Asia/Kolkata").strftime("%d %b %Y, %H:%M IST")
    return out


if __name__ == "__main__":
    c = pd.read_csv(config.CENTROIDS_PATH)
    df = fetch_live_rain(c)
    print(df.sort_values("rain_next_3d_max_mm", ascending=False).head(10).to_string(index=False))
