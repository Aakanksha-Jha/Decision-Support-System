"""
Command-line lookup: give a Revenue Circle ID, get its predicted flood severity.

Usage (from the project root, after running src/pipeline_run.py):
    python src/check_circle.py 18-308-00153
    python src/check_circle.py            # lists the 10 highest-priority circles
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config  # noqa: E402


def main():
    if not os.path.exists(config.PREDICTIONS_PATH):
        sys.exit("No predictions found. Run: python src/pipeline_run.py")
    pred = pd.read_csv(config.PREDICTIONS_PATH)
    pred[config.ID_COLUMN] = pred[config.ID_COLUMN].astype(str)

    if len(sys.argv) < 2:
        print("Top 10 highest-priority circles (pass an ID to check one):\n")
        top = pred.sort_values("priority_rank").head(10)
        print(top[[config.ID_COLUMN, "impact_category", "predicted_probability", "priority_rank"]].to_string(index=False))
        return

    cid = sys.argv[1].strip()
    hit = pred[pred[config.ID_COLUMN] == cid]
    if hit.empty:
        sys.exit(f"Circle '{cid}' not found. Run without arguments to see valid examples.")
    r = hit.iloc[0]
    print(f"Revenue Circle : {cid}")
    print(f"Month scored   : {r['timeperiod']} (prediction is for the following month)")
    print(f"Severity       : {r['impact_category'].upper()}")
    print(f"Probability    : {r['predicted_probability'] * 100:.1f}%")
    print(f"Priority rank  : {int(r['priority_rank'])} of {len(pred)}")
    if r["impact_category"] != "Low":
        print(f"Est. affected  : {r['predicted_affected_population']:,.0f} people")
        print(f"Needs          : {int(r['boats_needed'])} boats, {int(r['food_units_needed'])} food units, {int(r['medical_teams_needed'])} medical teams")
    print("Why            :")
    for x in str(r["top_reasons"]).split(";"):
        print("   -", x.strip())


if __name__ == "__main__":
    main()
