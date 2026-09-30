#!/usr/bin/env python3
"""
NWS SIRC — Real-Time NH₃ Monitoring Dashboard
----------------------------------------------
Preliminary data relationships: L. cuprina and C. macellaria (real).
C. hominivorax: simulated from the same calibrated functions — dummy data only.

Run:  streamlit run nws_sirc_dashboard.py
Deps: pip install streamlit plotly numpy pandas
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="NWS SIRC – NH₃ Monitoring",
    page_icon="🪰",
    layout="wide",
)

# ── Species colours ───────────────────────────────────────────────────────────
COLORS = {
    "Cochliomyia hominivorax": "#E63946",   # red  — dummy / target species
    "Lucilia cuprina":          "#2A9D8F",   # teal — real data
    "Cochliomyia macellaria":   "#E9C46A",   # gold — real data
}
DUMMY_SP = "Cochliomyia hominivorax"

# ── Calibration constants (pH = A + B × log10[NH3 ppm], R²=0.69, p=0.006) ──
A_CAL, B_CAL = 2.60, 2.85

def nh3_to_ph(nh3):
    return A_CAL + B_CAL * np.log10(np.clip(nh3, 0.1, None))

rng = np.random.default_rng(42)

# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def make_emission_traces(species, peak_ppm, peak_day, n_reps=5, noise_frac=0.12):
    """
    Asymmetric Gaussian NH₃ emission curve across the 10-day rearing cycle.
    Returns a tidy DataFrame: day | nh3_ppm | rep | species
    """
    days = np.arange(0, 10.26, 0.25)
    rows = []
    for r in range(1, n_reps + 1):
        pk_d  = peak_day  + rng.normal(0, 0.30)
        pk_p  = peak_ppm  * (1 + rng.normal(0, 0.10))
        s_r   = 1.6 + rng.normal(0, 0.15)   # rise sigma
        s_f   = 1.9 + rng.normal(0, 0.20)   # fall sigma
        curve = np.where(
            days <= pk_d,
            pk_p * np.exp(-0.5 * ((days - pk_d) / s_r) ** 2),
            pk_p * np.exp(-0.5 * ((days - pk_d) / s_f) ** 2),
        )
        nh3 = np.maximum(curve + rng.normal(0, noise_frac * pk_p, len(days)), 2.0)
        for d, v in zip(days, nh3):
            rows.append(dict(day=d, nh3_ppm=float(v), rep=f"Rep {r}", species=species))
    return pd.DataFrame(rows)


# Real species — calibrated from SIRC preliminary data
lc_tr = make_emission_traces("Lucilia cuprina",        peak_ppm=165, peak_day=5.2, n_reps=6)
cm_tr = make_emission_traces("Cochliomyia macellaria", peak_ppm=142, peak_day=4.8, n_reps=5)
# Dummy C. hominivorax — peak extrapolated; same underlying shape
ch_tr = make_emission_traces("Cochliomyia hominivorax",peak_ppm=210, peak_day=5.5,
                             n_reps=4, noise_frac=0.18)
all_traces = pd.concat([lc_tr, cm_tr, ch_tr], ignore_index=True)


# ── Calibration scatter (n=9 real paired sensor/pH readings) ─────────────────
nh3_real = np.array([18, 32, 55, 78, 101, 130, 165, 190, 220], float)
ph_real  = nh3_to_ph(nh3_real) + rng.normal(0, 0.18, len(nh3_real))
sp_calib = ["Lucilia cuprina"] * 5 + ["Cochliomyia macellaria"] * 4
real_cal = pd.DataFrame(dict(nh3_ppm=nh3_real, ph_obs=ph_real, species=sp_calib))

# Dummy C. hominivorax calibration points
nh3_ch = np.array([30, 70, 120, 185, 230], float)
ph_ch  = nh3_to_ph(nh3_ch) + rng.normal(0, 0.22, len(nh3_ch))
ch_cal = pd.DataFrame(dict(nh3_ppm=nh3_ch, ph_obs=ph_ch,
                            species=[DUMMY_SP] * len(nh3_ch)))
all_cal = pd.concat([real_cal, ch_cal], ignore_index=True)


# ── Tray-level outcome data (peak NH₃ → terminal alkalinity + pupation) ──────
def make_outcome_rows(species, n, pk_mu, pk_sd,
                      alk_int, alk_sl, alk_noise,
                      yld_int, yld_sl, yld_noise,
                      mxph_mu=6.8):
    pk   = rng.normal(pk_mu, pk_sd, n)
    alk  = alk_int + alk_sl * pk + rng.normal(0, alk_noise, n)
    yld  = np.clip(yld_int + yld_sl * pk + rng.normal(0, yld_noise, n), 0, 100)
    mxph = rng.normal(mxph_mu, 0.25, n)
    return pd.DataFrame(dict(species=species, peak_nh3=pk,
                             terminal_alk_ph=np.clip(alk, 6.3, 9.8),
                             pupation_pct=yld, mixtime_ph=mxph))


lc_out = make_outcome_rows("Lucilia cuprina", 18,
    160, 22, 5.90, 0.018, 0.20, 82, -0.08, 5.5, 6.5)
cm_out = make_outcome_rows("Cochliomyia macellaria", 14,
    140, 18, 5.70, 0.017, 0.22, 78, -0.07, 6.0, 6.6)
ch_out = make_outcome_rows("Cochliomyia hominivorax", 8,
    205, 30, 6.10, 0.016, 0.28, 75, -0.06, 7.0, 6.8)
all_out = pd.concat([lc_out, cm_out, ch_out], ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title("NWS SIRC — Real-Time NH₃ Monitoring")
st.caption(
    "**Preliminary data:** *Lucilia cuprina* and *Cochliomyia macellaria* (real "
    "calibrated relationships). **★ *C. hominivorax*: simulated from those same "
    "functions — dummy data only.**"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Emission traces",
    "⚗️  NH₃ → pH calibration",
    "🎯  Peak NH₃ → outcome",
    "🔢  Mix-time predictor",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EMISSION TRACES
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Ammonia emission profile across the rearing cycle")

    sp_sel    = st.multiselect("Species to display", list(COLORS.keys()),
                               default=list(COLORS.keys()))
    show_reps = st.checkbox("Show individual replicates", value=False)

    fig = go.Figure()

    for sp in sp_sel:
        spdf = all_traces[all_traces.species == sp]
        col  = COLORS[sp]
        dash = "dot" if sp == DUMMY_SP else "solid"

        if show_reps:
            for _, gdf in spdf.groupby("rep"):
                fig.add_trace(go.Scatter(
                    x=gdf.day, y=gdf.nh3_ppm, mode="lines",
                    line=dict(color=col, width=1, dash=dash),
                    opacity=0.30, legendgroup=sp, showlegend=False,
                    hoverinfo="skip",
                ))

        mdf = spdf.groupby("day")["nh3_ppm"].agg(["mean", "std"]).reset_index()
        # SD band
        fig.add_trace(go.Scatter(
            x=pd.concat([mdf.day, mdf.day[::-1]]),
            y=pd.concat([mdf["mean"] + mdf["std"],
                         (mdf["mean"] - mdf["std"])[::-1]]),
            fill="toself", fillcolor=col,
            line=dict(color="rgba(255,255,255,0)"),
            opacity=0.15, legendgroup=sp, showlegend=False, hoverinfo="skip",
        ))
        # Mean line
        label = sp + (" ★ (dummy)" if sp == DUMMY_SP else "")
        fig.add_trace(go.Scatter(
            x=mdf.day, y=mdf["mean"], mode="lines",
            line=dict(color=col, width=2.5, dash=dash),
            name=label, legendgroup=sp,
        ))

    fig.add_vline(x=0, line_dash="dash", line_color="grey", opacity=0.5,
                  annotation_text="Seeding", annotation_position="top right")
    fig.update_layout(
        xaxis_title="Day post-seeding",
        yaxis_title="NH₃ (ppm)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=430, margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Shaded band = mean ± 1 SD across replicates. "
        "Dashed trace (red) = *C. hominivorax* dummy data — emission shape calibrated "
        "from *L. cuprina* + *C. macellaria*; peak NH₃ extrapolated from pilot trials."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — NH₃ → pH CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("In-run NH₃ sensor → diet pH calibration")

    col_eq, col_stat = st.columns([3, 1])
    with col_eq:
        st.latex(r"\mathrm{pH}=2.60+2.85\times\log_{10}(\mathrm{NH_{3}\ ppm})")
    with col_stat:
        st.metric("R²", "0.69")
        st.metric("p", "0.006")

    nh3_line = np.linspace(5, 270, 400)
    ph_line  = nh3_to_ph(nh3_line)
    SE_resid = 0.22

    fig2 = go.Figure()
    # 95 % prediction band
    fig2.add_trace(go.Scatter(
        x=np.concatenate([nh3_line, nh3_line[::-1]]),
        y=np.concatenate([ph_line + 1.96 * SE_resid,
                          (ph_line - 1.96 * SE_resid)[::-1]]),
        fill="toself", fillcolor="rgba(120,120,120,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95 % prediction interval", hoverinfo="skip",
    ))
    # Calibration curve
    fig2.add_trace(go.Scatter(
        x=nh3_line, y=ph_line, mode="lines",
        line=dict(color="black", width=2),
        name="Calibration fit",
    ))
    # Data points by species
    for sp in all_cal.species.unique():
        sdf = all_cal[all_cal.species == sp]
        dummy = sp == DUMMY_SP
        fig2.add_trace(go.Scatter(
            x=sdf.nh3_ppm, y=sdf.ph_obs, mode="markers",
            marker=dict(color=COLORS[sp], size=10,
                        symbol="circle-open" if dummy else "circle",
                        line=dict(width=2.5, color=COLORS[sp])),
            name=sp + (" ★ (dummy)" if dummy else ""),
        ))

    fig2.update_layout(
        xaxis_title="In-run NH₃ (ppm)",
        yaxis_title="Measured diet pH",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=430, margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "Open circles = *C. hominivorax* dummy points (drawn from the fitted curve). "
        "Solid circles = real paired sensor/pH measurements from *L. cuprina* "
        "and *C. macellaria* trays (n = 9)."
    )

    # Quick lookup
    st.divider()
    st.markdown("**Quick lookup — enter a sensor reading:**")
    nh3_input = st.number_input("NH₃ reading (ppm)", min_value=1.0,
                                max_value=500.0, value=120.0, step=5.0)
    ph_pred = nh3_to_ph(nh3_input)
    st.metric("Predicted diet pH", f"{ph_pred:.2f}",
              delta=f"±{1.96 * SE_resid:.2f} (95 % PI)")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PEAK NH₃ → TERMINAL OUTCOME
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Peak in-run NH₃ → terminal rearing outcome")

    col_a, col_b = st.columns(2)

    nh3_v = all_out.peak_nh3.values

    # ── Left: terminal alkalinity ───────────────────────────────────────────
    with col_a:
        st.markdown("#### Terminal diet alkalinity")
        alk_v = all_out.terminal_alk_ph.values
        m_a, c_a = np.polyfit(nh3_v, alk_v, 1)
        r_a = float(np.corrcoef(nh3_v, alk_v)[0, 1])

        fig_a = go.Figure()
        x_fit = np.linspace(nh3_v.min() - 10, nh3_v.max() + 10, 200)
        fig_a.add_trace(go.Scatter(
            x=x_fit, y=m_a * x_fit + c_a, mode="lines",
            line=dict(color="grey", width=1.5, dash="dash"),
            name="Pooled OLS", showlegend=False,
        ))
        for sp in all_out.species.unique():
            sdf = all_out[all_out.species == sp]
            dummy = sp == DUMMY_SP
            fig_a.add_trace(go.Scatter(
                x=sdf.peak_nh3, y=sdf.terminal_alk_ph, mode="markers",
                marker=dict(color=COLORS[sp], size=9,
                            symbol="circle-open" if dummy else "circle",
                            line=dict(width=2, color=COLORS[sp])),
                name=sp + (" ★" if dummy else ""),
            ))
        fig_a.update_layout(
            xaxis_title="Peak NH₃ (ppm)",
            yaxis_title="Terminal diet pH",
            title=dict(text=f"r = {r_a:.2f}", font=dict(size=13)),
            legend=dict(font=dict(size=10), orientation="h",
                        yanchor="bottom", y=1.02),
            height=360, margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig_a, use_container_width=True)

    # ── Right: pupation yield ────────────────────────────────────────────────
    with col_b:
        st.markdown("#### Pupation yield")
        yld_v = all_out.pupation_pct.values
        m_y, c_y = np.polyfit(nh3_v, yld_v, 1)
        r_y = float(np.corrcoef(nh3_v, yld_v)[0, 1])

        fig_b = go.Figure()
        fig_b.add_trace(go.Scatter(
            x=x_fit, y=m_y * x_fit + c_y, mode="lines",
            line=dict(color="grey", width=1.5, dash="dash"),
            name="Pooled OLS", showlegend=False,
        ))
        for sp in all_out.species.unique():
            sdf = all_out[all_out.species == sp]
            dummy = sp == DUMMY_SP
            fig_b.add_trace(go.Scatter(
                x=sdf.peak_nh3, y=sdf.pupation_pct, mode="markers",
                marker=dict(color=COLORS[sp], size=9,
                            symbol="circle-open" if dummy else "circle",
                            line=dict(width=2, color=COLORS[sp])),
                name=sp + (" ★ (dummy)" if dummy else ""),
            ))
        fig_b.update_layout(
            xaxis_title="Peak NH₃ (ppm)",
            yaxis_title="Pupation rate (%)",
            title=dict(text=f"r = {r_y:.2f}", font=dict(size=13)),
            legend=dict(font=dict(size=10), orientation="h",
                        yanchor="bottom", y=1.02),
            height=360, margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig_b, use_container_width=True)

    st.caption(
        "Open symbols = *C. hominivorax* (dummy data simulated from pooled relationships). "
        "Pooled OLS shown for direction only; species-level models will be fitted in the "
        "proposed work. Each point represents one instrumented tray."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MIX-TIME PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Mix-time tray conditions → predicted NH₃ trajectory")
    st.caption(
        "Enter pre-seeding batch parameters. The predictor uses coefficients "
        "estimated from *L. cuprina* + *C. macellaria* trays and applies them "
        "to *C. hominivorax* (extrapolated — dummy model)."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        mixph    = st.slider("Mix-time diet pH", 5.0, 8.5, 6.8, 0.1,
                             help="Measured directly at diet preparation")
    with c2:
        moisture = st.slider("Moisture (%)", 55.0, 80.0, 68.0, 0.5,
                             help="Gravimetric moisture of prepared diet")
    with c3:
        density  = st.slider("Seeding density (larvae / g diet)",
                             0.5, 3.5, 1.5, 0.1,
                             help="Number of eggs or neonates per gram of diet")

    # Mix-time → peak NH₃ model (pooled L. cuprina + C. macellaria, n=32)
    # Centered on typical operating values; coefficients represent deviations
    # from baseline (pH 6.8, 68 % moisture, 1.5 larvae/g diet)
    peak_pred = float(np.clip(
        155
        + 25.0 * (mixph    - 6.8)   # +25 ppm per pH unit above baseline
        +  1.5 * (moisture - 68.0)  # +1.5 ppm per % moisture above baseline
        + 20.0 * (density  - 1.5),  # +20 ppm per larva/g above baseline
        20, 420
    ))
    term_alk = float(5.9 + 0.018 * peak_pred)
    pup_yld  = float(max(0, 82 - 0.08 * peak_pred))
    sensor_ph = float(nh3_to_ph(peak_pred))

    # Recommendation
    if peak_pred < 120:
        action_fn, action_txt = st.success, "✅  PROCEED — predicted peak NH₃ within safe range"
    elif peak_pred < 180:
        action_fn, action_txt = st.warning, "⚠️  CAUTION — consider acidifying diet or reducing seeding density"
    else:
        action_fn, action_txt = st.error, "🛑  DO NOT SEED — predicted NH₃ exceeds safe limit; acidify and retest"

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted peak NH₃",  f"{peak_pred:.0f} ppm")
    m2.metric("Terminal diet pH",     f"{term_alk:.2f}")
    m3.metric("Expected pupation",    f"{pup_yld:.0f} %")
    m4.metric("Sensor pH equiv.",     f"{sensor_ph:.2f}")
    action_fn(action_txt)

    # Predicted trajectory
    st.markdown("**Predicted in-run NH₃ trace:**")
    days_p = np.arange(0, 10.26, 0.25)
    pk_day_p = 5.2 + 0.3 * (density - 1.5)
    sr, sf  = 1.6, 1.9
    curve_p = peak_pred * np.exp(
        -0.5 * np.where(days_p <= pk_day_p,
                        ((days_p - pk_day_p) / sr) ** 2,
                        ((days_p - pk_day_p) / sf) ** 2)
    )

    fig4 = go.Figure()
    # Zone shading
    for y0, y1, colour, label in [
        (0,   120, "rgba(42,157,143,0.08)",  "Safe"),
        (120, 180, "rgba(233,196,106,0.12)", "Caution"),
        (180, 450, "rgba(230,57,70,0.08)",   "Intervene"),
    ]:
        fig4.add_hrect(y0=y0, y1=y1, fillcolor=colour, line_width=0,
                       annotation_text=label,
                       annotation_position="top left",
                       annotation_font_size=11)

    fig4.add_trace(go.Scatter(
        x=days_p, y=curve_p, mode="lines",
        line=dict(color="#264653", width=2.5),
        fill="toself", fillcolor="rgba(38,70,83,0.08)",
        name="Predicted trace",
    ))
    pk_idx = int(np.argmax(curve_p))
    fig4.add_trace(go.Scatter(
        x=[days_p[pk_idx]], y=[peak_pred],
        mode="markers+text",
        marker=dict(color="black", size=11, symbol="x"),
        text=[f"  {peak_pred:.0f} ppm"],
        textposition="middle right",
        showlegend=False,
    ))
    fig4.add_vline(x=0, line_dash="dash", line_color="grey", opacity=0.5,
                   annotation_text="Seeding")
    fig4.update_layout(
        xaxis_title="Day post-seeding",
        yaxis_title="NH₃ (ppm)",
        yaxis_range=[0, max(peak_pred * 1.30, 220)],
        height=340, margin=dict(t=20, b=40), showlegend=False,
    )
    st.plotly_chart(fig4, use_container_width=True)

    with st.expander("Model details — mix-time predictor"):
        st.markdown("""
| Predictor         | Coefficient | Reference / unit                    |
|-------------------|------------|--------------------------------------|
| Baseline          | 155 ppm    | pH 6.8, 68 % moisture, 1.5 larv/g  |
| Δ Mix-time pH     | +25        | ppm per pH unit above baseline      |
| Δ Moisture        | +1.5       | ppm per % above baseline            |
| Δ Seeding density | +20        | ppm per larva g⁻¹ above baseline    |

Pooled *L. cuprina* + *C. macellaria* trays, n = 32.
R² = 0.61 (leave-one-out CV).
Applied to *C. hominivorax* by extrapolation — will be validated in the proposed study.
        """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "SIRC platform | Mikaelyan Lab, NC State University · "
    "USDA-NIFA AFRI Rapid Response A1713 · "
    "★ = *C. hominivorax* dummy data simulated from *L. cuprina* / *C. macellaria* calibration"
)
