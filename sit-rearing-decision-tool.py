#!/usr/bin/env python3
"""
SIRC Tray Monitor — NH3 decision-support demonstration
------------------------------------------------------
Data policy
  * Measured data are shown ONLY when loaded from CSV files in ./data/ :
        data/traces.csv       columns: species, treatment, rep, day, nh3_ppm
        data/calibration.csv  columns: species, treatment, peak_nh3_ppm, ph
  * If those files are absent, the app shows SIMULATED example data scaled to the
    ranges reported in the proposal (Figs. 2 and 4) and labels it as simulated.
  * No C. hominivorax data are shown until measured C. hominivorax data exist.
  * All decision thresholds are PLACEHOLDERS (see THRESHOLDS below); production
    thresholds will replace them (Objective 3).

Run:  streamlit run nws_sirc_dashboard.py
Deps: pip install streamlit plotly numpy pandas
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="SIRC Tray Monitor", layout="wide")

DATA_DIR = Path(__file__).parent / "data"

# ── Values reported in the proposal ──────────────────────────────────────────
A_CAL, B_CAL = 2.60, 2.85          # pH = A + B·log10(peak NH3 ppm); Fig. 4
CAL_R2, CAL_P, CAL_N = 0.69, 0.006, 9
CAL_LOO_ERR = 0.42                 # ± pH units, leave-one-out
CAL_RANGE = (10.0, 40.0)           # ppm range of calibration data (Fig. 4A)
NIOSH_REL = 25.0                   # ppm, 8-h TWA

# ── PLACEHOLDER decision thresholds (to be replaced by Objective 3 values) ───
THRESHOLDS = {
    "mix_ph_max": 7.0,             # acidify if mix-time pH above this
    "egg_load_max_pct": 100.0,     # reduce if egg load above production baseline (%)
    "moisture_range": (60.0, 75.0),# flag if outside (recorded, not a decision rule yet)
    "slope_cue_ppm_per_day": 6.0,  # apply suppressant if 24-h NH3 slope exceeds this
}

TREAT_COLORS = {"FMA": "#E8754F", "AFMA": "#3B7DD8", "RMA": "#2E9E6B"}
TREAT_NAMES = {"FMA": "Untreated (FMA)", "AFMA": "Acidified (AFMA)",
               "RMA": "Conditioned (RMA)"}


def nh3_to_ph(nh3):
    return A_CAL + B_CAL * np.log10(np.clip(nh3, 0.1, None))


# ══════════════════════════════════════════════════════════════════════════════
# DATA: measured if available, otherwise clearly-labelled simulation
# ══════════════════════════════════════════════════════════════════════════════
def _simulate_traces():
    """Monotonic rise matching the ranges in Fig. 2 (illustration only)."""
    rng = np.random.default_rng(7)
    # species: (days, {treatment: (approx. final ppm, n reps)})
    spec = {
        "Lucilia cuprina": (6, {"FMA": (22, 3), "AFMA": (17, 3), "RMA": (12, 3)}),
        "Cochliomyia macellaria": (14, {"FMA": (38, 6), "AFMA": (22, 6)}),
    }
    rows = []
    for sp, (ndays, treats) in spec.items():
        days = np.arange(0, ndays + 0.01, 1 / 24)
        onset = 0.4 * ndays
        for tr, (final, n) in treats.items():
            for r in range(1, n + 1):
                f = final * (1 + rng.normal(0, 0.15))
                y = f / (1 + np.exp(-(days - 0.75 * ndays) / (0.12 * ndays)))
                y = np.where(days < onset * 0.5, 0.0, y)
                y = np.maximum(y + rng.normal(0, 0.04 * f, len(days)), 0)
                rows += [dict(species=sp, treatment=tr, rep=r, day=d, nh3_ppm=v)
                         for d, v in zip(days, y)]
    return pd.DataFrame(rows)


@st.cache_data
def load_data():
    tr_path, cal_path = DATA_DIR / "traces.csv", DATA_DIR / "calibration.csv"
    traces = pd.read_csv(tr_path) if tr_path.exists() else None
    calib = pd.read_csv(cal_path) if cal_path.exists() else None
    return traces, calib


traces, calib = load_data()
traces_measured = traces is not None
if not traces_measured:
    traces = _simulate_traces()

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title("SIRC Tray Monitor")
st.caption("Decision-support demonstration · Mikaelyan Lab, NC State University")

status = []
status.append("Emission traces: **measured**" if traces_measured
              else "Emission traces: **simulated example** (scaled to proposal Fig. 2)")
status.append(f"Calibration points: **measured (n = {len(calib)})**" if calib is not None
              else "Calibration points: **not loaded** (fitted equation only)")
status.append("All decision thresholds are **placeholders**.")
st.warning("  \n".join(status))

tab1, tab2, tab3, tab4 = st.tabs([
    "Emission traces", "NH₃ → pH calibration",
    "Before seeding (mix-time check)", "After seeding (tray slope cue)",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EMISSION TRACES (species × treatment)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    sp = st.selectbox("Species", sorted(traces.species.unique()))
    spdf = traces[traces.species == sp]
    fig = go.Figure()
    for tr in [t for t in TREAT_COLORS if t in spdf.treatment.unique()]:
        tdf = spdf[spdf.treatment == tr]
        n = tdf.rep.nunique()
        g = tdf.groupby("day")["nh3_ppm"]
        m, se = g.mean(), g.std() / np.sqrt(n)
        col = TREAT_COLORS[tr]
        fig.add_trace(go.Scatter(
            x=np.r_[m.index, m.index[::-1]], y=np.r_[m + se, (m - se)[::-1]],
            fill="toself", fillcolor=col, opacity=0.18, line=dict(width=0),
            showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=m.index, y=m, mode="lines",
                                 line=dict(color=col, width=2.5),
                                 name=f"{TREAT_NAMES[tr]} (n = {n})"))
    fig.add_hline(y=NIOSH_REL, line_dash="dash", line_color="grey",
                  annotation_text="NIOSH REL 25 ppm (8-h TWA)",
                  annotation_position="top left")
    fig.update_layout(xaxis_title="Day post-seeding",
                      yaxis_title="Headspace NH₃ (ppm, hourly mean)",
                      legend=dict(orientation="h", y=1.08), height=430)
    st.plotly_chart(fig, width="stretch")
    st.caption("Mean ± SEM across replicate chambers."
               + ("" if traces_measured else
                  " **Simulated example**: replace with data/traces.csv."))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    c1, c2 = st.columns([3, 1])
    c1.latex(r"\mathrm{pH}=2.60+2.85\,\log_{10}(\mathrm{peak\ NH_3,\ ppm})")
    c2.metric("R² (n = 9)", f"{CAL_R2}")
    c2.metric("LOO error", f"±{CAL_LOO_ERR} pH")

    x = np.linspace(*CAL_RANGE, 200)
    y = nh3_to_ph(x)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=np.r_[x, x[::-1]], y=np.r_[y + CAL_LOO_ERR, (y - CAL_LOO_ERR)[::-1]],
        fill="toself", fillcolor="rgba(120,120,120,0.15)", line=dict(width=0),
        name=f"±{CAL_LOO_ERR} pH (LOO)", hoverinfo="skip"))
    fig2.add_trace(go.Scatter(x=x, y=y, mode="lines",
                              line=dict(color="black", width=2), name="Calibration fit"))
    if calib is not None:
        for tr in calib.treatment.unique():
            d = calib[calib.treatment == tr]
            fig2.add_trace(go.Scatter(
                x=d.peak_nh3_ppm, y=d.ph, mode="markers",
                marker=dict(size=10, color=TREAT_COLORS.get(tr, "#666")),
                name=TREAT_NAMES.get(tr, tr)))
    fig2.update_layout(xaxis_title="Peak headspace NH₃ (ppm, log scale)",
                       xaxis_type="log", yaxis_title="Substrate pH at day 6",
                       legend=dict(orientation="h", y=1.08), height=430)
    st.plotly_chart(fig2, width="stretch")

    st.markdown("**Estimate pH from a sensor reading**")
    nh3_in = st.number_input("Peak NH₃ (ppm)", 1.0, 500.0, 25.0, 1.0)
    st.metric("Estimated substrate pH", f"{nh3_to_ph(nh3_in):.2f}",
              delta=f"±{CAL_LOO_ERR}", delta_color="off")
    if not CAL_RANGE[0] <= nh3_in <= CAL_RANGE[1]:
        st.error(f"Outside the calibrated range ({CAL_RANGE[0]:.0f}–{CAL_RANGE[1]:.0f} ppm); "
                 "estimate is an extrapolation and should not be used.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BEFORE SEEDING: proceed / acidify / reduce egg load
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.caption("Operator enters readings taken on the mixed diet before eggs are seeded.")
    c1, c2, c3 = st.columns(3)
    mix_ph = c1.number_input("Mix-time diet pH", 4.0, 9.0, 6.8, 0.1)
    moisture = c2.number_input("Diet moisture (%)", 40.0, 90.0, 68.0, 0.5)
    egg_load = c3.number_input("Egg load (% of production baseline)", 25.0, 300.0,
                               100.0, 5.0)

    actions = []
    if mix_ph > THRESHOLDS["mix_ph_max"]:
        actions.append(f"**Acidify** the batch (pH {mix_ph:.1f} > "
                       f"{THRESHOLDS['mix_ph_max']:.1f}), then re-read pH.")
    if egg_load > THRESHOLDS["egg_load_max_pct"]:
        actions.append(f"**Reduce egg load** to ≤ {THRESHOLDS['egg_load_max_pct']:.0f}% "
                       "of baseline (higher density shifts final pH alkaline; Fig. 3A).")
    lo, hi = THRESHOLDS["moisture_range"]
    if not lo <= moisture <= hi:
        st.info(f"Moisture {moisture:.1f}% is outside {lo:.0f}–{hi:.0f}%. "
                "Recorded for analysis; not yet a decision rule.")

    if actions:
        st.error("  \n".join(actions))
    else:
        st.success("**Proceed** — seed the tray.")
    st.caption("Thresholds are placeholders; production values come from Objective 3.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — AFTER SEEDING: apply suppressant / leave it
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.caption("The 24-h NH₃ slope on each instrumented tray decides whether that "
               "tray gets permanganate or Yucca extract.")
    sp = st.selectbox("Example trace from species", sorted(traces.species.unique()),
                      key="sp4")
    sdf = traces[traces.species == sp]
    opts = sdf[["treatment", "rep"]].drop_duplicates().itertuples(index=False)
    labels = {f"{t.treatment} rep {t.rep}": (t.treatment, t.rep) for t in opts}
    pick = st.selectbox("Tray", list(labels), key="tray4")
    tr, rep = labels[pick]
    d = sdf[(sdf.treatment == tr) & (sdf.rep == rep)].sort_values("day")
    max_day = float(d.day.max())
    now = st.slider("Current time (day post-seeding)", 1.0, max_day,
                    min(4.0, max_day), 0.25)

    seen = d[d.day <= now]
    cue = THRESHOLDS["slope_cue_ppm_per_day"]

    def slope_at(t):
        w = seen[(seen.day <= t) & (seen.day >= t - 1)]
        return np.polyfit(w.day, w.nh3_ppm, 1)[0] if len(w) > 2 else 0.0

    # Evaluate the cue at every logged point so far; once it fires it stays on.
    times = seen.day.values[seen.day.values >= 1]
    slopes = np.array([slope_at(t) for t in times]) if len(times) else np.array([])
    slope = slopes[-1] if len(slopes) else 0.0
    fired = np.where(slopes > cue)[0]
    cue_day = float(times[fired[0]]) if fired.size else None
    current = float(seen.nh3_ppm.iloc[-1])

    c1, c2 = st.columns(2)
    c1.metric("Current NH₃", f"{current:.1f} ppm")
    c2.metric("24-h slope", f"{slope:.1f} ppm/day")
    if cue_day is not None:
        st.error(f"**Apply suppressant to this tray** — slope exceeded {cue:.0f} ppm/day "
                 f"on day {cue_day:.2f}.")
    elif current > NIOSH_REL:
        st.error(f"**Apply suppressant to this tray** — NH₃ above the "
                 f"{NIOSH_REL:.0f} ppm REL.")
    else:
        st.success("**Leave it** — no suppressant needed on this tray.")

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=seen.day, y=seen.nh3_ppm, mode="lines",
                              line=dict(color=TREAT_COLORS.get(tr, "#264653"), width=2.5),
                              name="Logged NH₃"))
    fig4.add_hline(y=NIOSH_REL, line_dash="dash", line_color="grey",
                   annotation_text="NIOSH REL 25 ppm", annotation_position="top left")
    if cue_day is not None:
        y_cue = float(seen.loc[seen.day == cue_day, "nh3_ppm"].iloc[0])
        fig4.add_trace(go.Scatter(x=[cue_day], y=[y_cue], mode="markers+text",
                                  marker=dict(symbol="triangle-down", size=14,
                                              color="#B22222"),
                                  text=["  Slope cue: apply suppressant"],
                                  textposition="middle left",
                                  textfont=dict(color="#B22222")))
    fig4.update_layout(xaxis_title="Day post-seeding", yaxis_title="NH₃ (ppm)",
                       xaxis_range=[0, max_day],
                       yaxis_range=[0, max(d.nh3_ppm.max() * 1.15, 30)],
                       height=340, showlegend=False)
    st.plotly_chart(fig4, width="stretch")
    st.caption("Slope cue is a placeholder" + ("." if traces_measured else
               "; trace is a simulated example."))

st.divider()
st.caption("Demonstration build. Thresholds are placeholders pending C. hominivorax "
           "data from USDA-ARS Pacora.")
