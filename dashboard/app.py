"""
dashboard/app.py
-----------------
Streamlit dashboard for the Assam Flood DSS. Visualises:

  - Predicted flood-impact probability / category per Revenue Circle
  - Why each circle was flagged (SHAP top reasons)
  - The relief base -> circle allocation plan and unmet demand
  - An interactive map (Folium) of circles, bases and allocation lines

This reads the CSVs already produced by src/pipeline_run.py — it does not
retrain or re-optimise anything itself, so it opens instantly. If the
outputs don't exist yet, it tells the user to run the pipeline first.

Run:
    python src/pipeline_run.py      # once, to generate the CSVs
    streamlit run dashboard/app.py
"""

import os
import sys

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config  # noqa: E402
import live_rain  # noqa: E402
import live_predict  # noqa: E402

st.set_page_config(page_title="Assam Flood DSS", layout="wide")


@st.cache_data
def load_outputs():
    missing = [
        p
        for p in [
            config.PREDICTIONS_PATH,
            config.ALLOCATION_SUMMARY_PATH,
            config.ALLOCATION_PATH,
            config.CENTROIDS_PATH,
            config.BASES_PATH,
        ]
        if not os.path.exists(p)
    ]
    if missing:
        return None

    predictions = pd.read_csv(config.PREDICTIONS_PATH)
    summary = pd.read_csv(config.ALLOCATION_SUMMARY_PATH)
    allocation = pd.read_csv(config.ALLOCATION_PATH)
    centroids = pd.read_csv(config.CENTROIDS_PATH)
    bases = pd.read_csv(config.BASES_PATH)
    return predictions, summary, allocation, centroids, bases


def category_color(category: str) -> str:
    return {"High": "red", "Medium": "orange", "Low": "green"}.get(category, "gray")


def build_map(predictions, allocation, centroids, bases) -> folium.Map:
    merged = predictions.merge(centroids, on=config.ID_COLUMN, how="left")

    center_lat = merged["latitude"].mean()
    center_lon = merged["longitude"].mean()
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles="CartoDB positron")

    # Relief bases
    for _, base in bases.iterrows():
        folium.Marker(
            location=[base["latitude"], base["longitude"]],
            tooltip=f"Relief Base: {base['base_name']}",
            popup=(
                f"<b>{base['base_name']}</b><br>"
                f"Boats: {base['boat_capacity']}<br>"
                f"Food units: {base['food_capacity']}<br>"
                f"Medical teams: {base['medical_teams']}"
            ),
            icon=folium.Icon(color="blue", icon="home"),
        ).add_to(fmap)

    # Revenue circles, coloured by predicted impact category
    for _, row in merged.iterrows():
        if pd.isna(row["latitude"]):
            continue
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5 + 6 * row["predicted_probability"],
            color=category_color(row["impact_category"]),
            fill=True,
            fill_opacity=0.75,
            tooltip=(
                f"{row[config.ID_COLUMN]} — {row['impact_category']} "
                f"({row['predicted_probability']:.2f})"
            ),
            popup=row.get("top_reasons", ""),
        ).add_to(fmap)

    # Allocation lines: base -> circle for the largest resource flow only,
    # to keep the map readable.
    if not allocation.empty:
        boats_alloc = allocation[allocation["resource"] == "boats"]
        base_lookup = bases.set_index("base_id")[["latitude", "longitude"]]
        circle_lookup = centroids.set_index(config.ID_COLUMN)[["latitude", "longitude"]]
        for _, row in boats_alloc.iterrows():
            if row["base_id"] not in base_lookup.index or row[config.ID_COLUMN] not in circle_lookup.index:
                continue
            b = base_lookup.loc[row["base_id"]]
            c = circle_lookup.loc[row[config.ID_COLUMN]]
            folium.PolyLine(
                locations=[[b["latitude"], b["longitude"]], [c["latitude"], c["longitude"]]],
                color="blue",
                weight=1,
                opacity=0.4,
            ).add_to(fmap)

    return fmap


def circle_lookup(predictions, summary):
    """Pick a Revenue Circle -> see its predicted severity, rank, reasons and relief needs."""
    st.subheader("Check a Revenue Circle")
    st.caption(
        "Select or type a Revenue Circle ID to see its predicted flood severity for the coming month. "
        "(A lightweight, no-install version of this lookup is also published as a standalone web page — "
        "see the project README for the link.)"
    )
    ids = sorted(predictions[config.ID_COLUMN].astype(str).tolist())
    default = predictions.sort_values("priority_rank")[config.ID_COLUMN].iloc[0]
    chosen = st.selectbox("Revenue Circle ID", ids, index=ids.index(str(default)))
    row = predictions[predictions[config.ID_COLUMN].astype(str) == chosen].iloc[0]

    cat = row["impact_category"]
    icon = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}.get(cat, "⚪")
    a, b, c = st.columns(3)
    a.metric("Predicted severity", f"{icon} {cat}")
    b.metric("Flood-impact probability", f"{row['predicted_probability'] * 100:.1f}%")
    b.caption("Cut-offs: Low < 33%, Medium 33-66%, High >= 66%")
    c.metric("Priority rank", f"{int(row['priority_rank'])} of {len(predictions)}")

    if cat == "Low":
        st.success("Low predicted impact next month. No relief resources are being allocated to this circle.")
    else:
        d, e, f, g = st.columns(4)
        d.metric("Est. people affected", f"{row['predicted_affected_population']:,.0f}")
        e.metric("Boats needed", int(row["boats_needed"]))
        f.metric("Food units needed", int(row["food_units_needed"]))
        g.metric("Medical teams needed", int(row["medical_teams_needed"]))
        st.caption("Resource needs use a simple assumed conversion from probability to affected people (see config.py).")

    st.markdown("**Why this rating (top drivers, from SHAP):**")
    for reason in str(row["top_reasons"]).split(";"):
        st.write("- " + reason.strip())

    circ = summary[summary[config.ID_COLUMN].astype(str) == chosen]
    if not circ.empty:
        st.markdown("**Relief fulfilment for this circle:**")
        st.dataframe(
            circ[["resource", "required", "allocated", "unmet", "fulfilment_pct"]],
            use_container_width=True, hide_index=True,
        )


@st.cache_data(ttl=3600, show_spinner="Fetching live rainfall...")
def load_live_rain(centroids):
    return live_rain.fetch_live_rain(centroids)


def live_watch(predictions, centroids):
    st.subheader("Live rainfall watch (real-time)")
    st.caption(
        "Observed rain (last 7 days) and forecast (next 3 days) at each circle's location, refreshed hourly. "
        "Heavy-rain alert = 64.5 mm or more in 24 h (IMD 'heavy rain' category). "
        "This is a watch layer alongside the model; it does not change the ML score."
    )
    if st.button("Refresh now"):
        load_live_rain.clear()
    try:
        live = load_live_rain(centroids)
    except Exception as e:  # offline / API down
        st.info(f"Live rainfall is unavailable right now ({type(e).__name__}). The model results below still work.")
        return
    m = predictions[[config.ID_COLUMN, "impact_category", "priority_rank"]].merge(live, on=config.ID_COLUMN)
    m["combined_flag"] = m.apply(
        lambda r: "Watch closely" if r["heavy_rain_alert"] and r["impact_category"] != "Low"
        else ("Heavy rain" if r["heavy_rain_alert"] else ""), axis=1)
    st.caption(f"Last updated: {live['fetched_at'].iloc[0]}")
    a, b = st.columns(2)
    a.metric("Circles with heavy-rain alert", int(m["heavy_rain_alert"].sum()))
    b.metric("Alert AND Medium/High model risk", int((m["combined_flag"] == "Watch closely").sum()))
    st.dataframe(
        m.sort_values(["heavy_rain_alert", "rain_next_3d_max_mm"], ascending=False)
        .drop(columns=["fetched_at"]), use_container_width=True, hide_index=True)


@st.cache_data(ttl=3600, show_spinner="Re-scoring with live rainfall...")
def load_live_prediction(centroids):
    return live_predict.run(centroids)


def live_prediction(predictions, centroids):
    st.subheader("Live-updated ML prediction")
    st.caption(
        "Re-scores the trained model using this month's rainfall so far plus the forecast. "
        "EXPERIMENTAL: assumes the rainfall features are monthly totals in metres (unverified), "
        "uses placeholder circle locations, and keeps other inputs at their last known values."
    )
    if st.button("Re-score now"):
        load_live_prediction.clear()
    try:
        live = load_live_prediction(centroids)
    except Exception as e:
        st.info(f"Live prediction unavailable right now ({type(e).__name__}).")
        return
    base = predictions[[config.ID_COLUMN, "impact_category"]].rename(columns={"impact_category": "monthly_category"})
    m = live.merge(base, on=config.ID_COLUMN)
    m["changed"] = m["impact_category"] != m["monthly_category"]
    st.caption(f"Last updated: {live['updated_at'].iloc[0]}")
    a, b, c = st.columns(3)
    a.metric("High (live)", int((m["impact_category"] == "High").sum()))
    b.metric("Medium (live)", int((m["impact_category"] == "Medium").sum()))
    c.metric("Circles whose category changed", int(m["changed"].sum()))
    st.dataframe(
        m[[config.ID_COLUMN, "impact_category", "predicted_probability", "priority_rank",
           "projected_month_mm", "monthly_category", "predicted_affected_population"]],
        use_container_width=True, hide_index=True)


def main():
    st.title("Assam Flood — Explainable Impact Assessment & Relief Allocation")
    st.caption(
        "Stage 1 (ML + SHAP) predicts next-month flood impact per Revenue Circle. "
        "Stage 2 (Integer Linear Programming) allocates boats, food and medical "
        "teams from relief bases to the circles that need them most."
    )

    data = load_outputs()
    if data is None:
        st.warning(
            "No pipeline outputs found yet. Run `python src/pipeline_run.py` from the "
            "project root first, then reload this page."
        )
        return

    predictions, summary, allocation, centroids, bases = data

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Revenue circles scored", len(predictions))
    col2.metric("High-impact circles", int((predictions["impact_category"] == "High").sum()))
    col3.metric("Medium-impact circles", int((predictions["impact_category"] == "Medium").sum()))
    required_total = summary["required"].sum()
    fulfilled_total = summary["allocated"].sum()
    col4.metric(
        "Overall demand fulfilled",
        f"{100 * fulfilled_total / required_total:.1f}%" if required_total > 0 else "N/A",
    )

    circle_lookup(predictions, summary)

    live_watch(predictions, centroids)

    live_prediction(predictions, centroids)

    st.subheader("Flood impact & relief map")
    fmap = build_map(predictions, allocation, centroids, bases)
    st_folium(fmap, width=None, height=520)
    st.caption(
        "Circle size/colour = predicted flood-impact probability (green = low, "
        "orange = medium, red = high). Blue markers = relief bases. Thin blue "
        "lines = planned boat allocations."
    )

    st.subheader("Relief priority ranking")
    display_cols = [
        config.ID_COLUMN,
        "predicted_probability",
        "impact_category",
        "priority_rank",
        "predicted_affected_population",
        "boats_needed",
        "food_units_needed",
        "medical_teams_needed",
        "top_reasons",
    ]
    st.dataframe(
        predictions[display_cols].sort_values("priority_rank"),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Relief allocation — demand fulfilment by circle")
    pivot = summary.pivot_table(
        index=[config.ID_COLUMN, "impact_category", "priority_rank"],
        columns="resource",
        values="fulfilment_pct",
    ).reset_index().sort_values("priority_rank")
    st.dataframe(pivot, use_container_width=True, hide_index=True)

    st.subheader("Detailed base → circle allocation")
    st.dataframe(allocation.sort_values(["object_id", "resource"]), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
