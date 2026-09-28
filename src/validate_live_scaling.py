"""
Checks the live-rain unit assumption. Fetches ARCHIVED Open-Meteo rainfall for
Jan-Apr 2026 at the circle centroids, converts to metres, and compares the
all-circle monthly average with the dataset's mean_rain.

Read the ratio (dataset / archive): close to 1.0 means the assumption
mean_rain = monthly mm / 1000 is reasonable. Far from 1 means do NOT trust
live_predict.py yet. Per-circle correlation will be weak while centroids are placeholders.
Run:  python src/validate_live_scaling.py
"""
import json, os, sys, urllib.parse, urllib.request
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import config, live_rain  # noqa: E402

c = pd.read_csv(config.CENTROIDS_PATH)
df = pd.read_csv(config.RAW_DATASET_PATH)
df["m"] = pd.to_datetime(df[config.TIME_COLUMN], format="%Y_%m").dt.strftime("%Y-%m")
vals = {}
for i in range(0, len(c), live_rain.CHUNK):
    p = c.iloc[i:i + live_rain.CHUNK]
    q = urllib.parse.urlencode({"latitude": ",".join(map(str, p.latitude)), "longitude": ",".join(map(str, p.longitude)),
        "start_date": "2026-01-01", "end_date": "2026-04-30", "daily": "precipitation_sum", "timezone": "Asia/Kolkata"})
    with urllib.request.urlopen("https://archive-api.open-meteo.com/v1/archive?" + q, timeout=60) as r:
        res = json.loads(r.read().decode()); res = res if isinstance(res, list) else [res]
    for cid, r_ in zip(p[config.ID_COLUMN], res):
        s = pd.Series(r_["daily"]["precipitation_sum"], index=pd.to_datetime(r_["daily"]["time"])).fillna(0)
        for m, v in s.groupby(s.index.strftime("%Y-%m")).sum().items():
            vals[(cid, m)] = v / 1000.0
a = pd.Series(vals).groupby(level=1).mean()
d = df[df.m.isin(a.index)].groupby("m").mean_rain.mean()
print(pd.DataFrame({"archive_m": a, "dataset_mean_rain": d, "ratio": d / a}).round(3))
