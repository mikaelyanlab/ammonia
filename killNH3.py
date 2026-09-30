"""Dummy Pacora tray app for the AFRI A1713 narrative.

Mix: operator enters diet pH, moisture, egg load → proceed / acidify / reduce eggs.
After seed: dummy NH3 traces → apply suppressant or leave it.
Not production software. Thresholds are placeholders.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="NWS tray advisor (demo)",
    page_icon="🪰",
    layout="wide",
)

# Interim placeholders — replace with Obj. 1 thresholds
PH_ACIDIFY = 7.2
EGG_REDUCE = 1.15  # fraction of baseline
SLOPE_SUPPRESS = 0.35  # ppm / h over last 12 h
NIOSH_REL = 25.0
BASELINE_EGGS = 8000


def mix_recommendation(pH: float, moisture: float, eggs: int) -> tuple[str, str]:
    reasons = []
    action = "Proceed"
    if pH >= PH_ACIDIFY:
        action = "Acidify the mix"
        reasons.append(f"Launch pH {pH:.2f} is at or above {PH_ACIDIFY}.")
    if eggs / BASELINE_EGGS >= EGG_REDUCE:
        if action == "Proceed":
            action = "Reduce egg load"
        else:
            action = "Acidify the mix and reduce egg load"
        reasons.append(
            f"Egg load {eggs:,} is {eggs / BASELINE_EGGS:.2f}× the {BASELINE_EGGS:,} baseline."
        )
    if moisture < 68 or moisture > 82:
        reasons.append(f"Moisture {moisture:.0f}% is outside the 68–82% working window.")
    if not reasons:
        reasons.append("Launch pH, moisture, and egg load are inside the interim window.")
    return action, " ".join(reasons)


def dummy_trace(tray_id: str, kind: str, hours: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    t = hours
    if kind == "low":
        base = 0.15 * t + 0.8 * np.log1p(t / 8)
        noise = rng.normal(0, 0.25, size=t.size)
        return np.clip(base + noise, 0, 18)
    if kind == "rising":
        base = 0.08 * t + 0.0022 * t**2
        noise = rng.normal(0, 0.35, size=t.size)
        return np.clip(base + noise, 0, 48)
    base = 0.12 * t + 0.0008 * t**2
    noise = rng.normal(0, 0.3, size=t.size)
    return np.clip(base + noise, 0, 28)


@st.cache_data
def make_trays() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    hours = np.arange(0, 96 + 1, 2)
    rows = []
    specs = [
        ("T-104", "low", 7.05, 74, 7800),
        ("T-111", "rising", 7.45, 76, 9600),
        ("T-118", "mid", 7.15, 71, 8200),
        ("T-122", "rising", 7.38, 79, 9100),
        ("T-130", "low", 6.92, 73, 7600),
    ]
    for tray_id, kind, pH, moist, eggs in specs:
        y = dummy_trace(tray_id, kind, hours, rng)
        for h, ppm in zip(hours, y):
            rows.append(
                {
                    "tray": tray_id,
                    "kind": kind,
                    "launch_pH": pH,
                    "moisture": moist,
                    "eggs": eggs,
                    "hour": int(h),
                    "NH3_ppm": float(ppm),
                }
            )
    return pd.DataFrame(rows)


def slope_ppm_per_h(sub: pd.DataFrame, window_h: int = 12) -> float:
    tmax = sub["hour"].max()
    tail = sub[sub["hour"] >= tmax - window_h]
    if len(tail) < 2:
        return 0.0
    x = tail["hour"].to_numpy(float)
    y = tail["NH3_ppm"].to_numpy(float)
    return float(np.polyfit(x, y, 1)[0])


df = make_trays()

st.title("NWS sterile-fly tray advisor")
st.caption(
    "Demo only — dummy traces and placeholder thresholds. "
    "Pacora month-six product described in the AFRI A1713 narrative."
)

tab_mix, tab_run = st.tabs(["Before seed  ·  mix", "After seed  ·  running trays"])

with tab_mix:
    st.subheader("Read the mixed diet, then ask for a start recommendation")
    c1, c2, c3 = st.columns(3)
    with c1:
        pH = st.number_input("Diet pH at mix", min_value=5.0, max_value=9.0, value=7.35, step=0.05)
    with c2:
        moisture = st.slider("Diet moisture (%)", 55, 90, 75)
    with c3:
        eggs = st.number_input("Planned eggs per tray", min_value=1000, max_value=20000, value=9200, step=100)

    action, reason = mix_recommendation(pH, moisture, eggs)
    color = {"Proceed": "green"}.get(action, "orange")
    st.markdown(f"### Recommendation: :{color}[{action}]")
    st.write(reason)
    st.caption(
        f"Interim rules used here: acidify if pH ≥ {PH_ACIDIFY}; "
        f"reduce eggs if load ≥ {EGG_REDUCE:.2f}× {BASELINE_EGGS:,} eggs/tray. "
        "Replace after Objective 1."
    )

with tab_run:
    st.subheader("Ammonia on seeded trial trays")
    trays = df["tray"].unique().tolist()
    selected = st.multiselect("Trays", trays, default=trays)
    view = df[df["tray"].isin(selected)]

    fig = go.Figure()
    for tray, sub in view.groupby("tray"):
        fig.add_trace(
            go.Scatter(x=sub["hour"], y=sub["NH3_ppm"], mode="lines", name=tray)
        )
    fig.add_hline(
        y=NIOSH_REL,
        line_dash="dash",
        annotation_text="NIOSH REL 25 ppm",
        annotation_position="top left",
    )
    fig.update_layout(
        xaxis_title="Hours after seed",
        yaxis_title="Headspace NH₃ (ppm)",
        height=380,
        margin=dict(l=40, r=20, t=20, b=40),
        legend_title="Tray",
    )
    st.plotly_chart(fig, use_container_width=True)

    rows = []
    for tray, sub in df.groupby("tray"):
        sl = slope_ppm_per_h(sub)
        last = float(sub.sort_values("hour")["NH3_ppm"].iloc[-1])
        cue = "Apply suppressant on this tray" if sl >= SLOPE_SUPPRESS else "Leave it"
        rows.append(
            {
                "Tray": tray,
                "Launch pH": sub["launch_pH"].iloc[0],
                "Eggs": int(sub["eggs"].iloc[0]),
                "NH₃ now (ppm)": round(last, 1),
                "Slope, last 12 h (ppm/h)": round(sl, 3),
                "Cue": cue,
            }
        )
    table = pd.DataFrame(rows)
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption(
        f"Dummy rule: apply permanganate or Yucca on a tray when 12-h slope ≥ {SLOPE_SUPPRESS} ppm/h. "
        "Sensors write the trace; the operator does not type NH₃."
    )
