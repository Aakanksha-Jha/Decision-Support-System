# Roadmap parameters — beyond the pilot model

The pilot model (`src/config.py::FEATURE_COLUMNS`) uses ~20 validated
features: rainfall, satellite vegetation/built-up indices, population,
household, land-use, infrastructure, terrain, drainage and past-impact
variables, trained on the Assam Revenue Circle dataset.

These six parameters are **proposed additions for a general-purpose Flood
DSS**, not yet implemented or trained on real data. They are documented here
so they can be added once a data source is confirmed for each.

| Parameter | Why it matters | Candidate real-world source |
|---|---|---|
| River water level / discharge | Flags a rising river before rainfall alone would | CWC / India-WRIS gauge data |
| Soil moisture | A saturated catchment floods faster from the same rainfall | Satellite soil-moisture products (e.g. SMAP) |
| Historical flood frequency | How often a circle has flooded before is a stable, low-noise risk signal | Multi-year flood-impact records (state disaster agency archives) |
| Relief-shelter capacity | How many people nearby shelters can actually hold | State disaster agency shelter registries |
| Livestock population | Livestock loss is a major rural impact and changes what relief to send | Livestock census |
| Network connectivity | Poor mobile coverage areas need offline or in-person outreach planning | Telecom operator coverage maps |

## How to add one

1. Source and clean the data at the same Revenue-Circle x month grain as
   `data/baseline_modeling_dataset_with_metadata.csv`.
2. Merge it into that dataset on `object_id` + `timeperiod`.
3. Add the column name to `FEATURE_COLUMNS` in `src/config.py`.
4. Retrain via `python src/train_model.py` (or `pipeline_run.py`) and re-run
   `python src/explain_shap.py` to confirm the new feature's SHAP contribution
   is stable and directionally sensible before trusting it in production.

None of these six are wired into the pilot model or the live-rescoring
scripts (`src/live_rain.py`, `src/live_predict.py`) yet.
