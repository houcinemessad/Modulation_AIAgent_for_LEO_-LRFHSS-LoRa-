"""Interactive dashboard (Streamlit).

Run (from the repo root):
    python -m streamlit run dashboard/dashboard.py

Requirements: pip install -r requirements.txt ; Ollama running only if an LLM stage is
selected. The default config rule/knn/rule is fully classical (no LLM, ~80 ms).

Two input modes:
  - Manual : enter the observed channel state.
  - TLE    : paste 2 TLE lines; skyfield (real SGP4 propagation) derives elevation /
             distance / radial speed for a ground station + time, and plots the pass.
"""
import os
import sys

import pandas as pd
import streamlit as st

# Put the repo root on the path (this file is in dashboard/, one level down).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Guarded import: clear message if a dependency is missing.
try:
    import numpy as np
    from skyfield.api import EarthSatellite, load, wgs84
    from src.agent import agent
    from src.agent.pipeline import best_under_tau, to_at_command, TAU
    _TS = load.timescale()
    _IMPORT_OK, _IMPORT_ERR = True, None
except Exception as exc:                       # noqa: BLE001
    _IMPORT_OK, _IMPORT_ERR = False, exc


# ── TLE -> geometry (skyfield, dashboard-only) ──────────────────────
_RE_KM = 6378.137
_MU = 398_600.4418


def make_satellite(line1, line2, name="SAT"):
    """Build the satellite (SGP4) + derived info for display."""
    sat = EarthSatellite(line1, line2, name, _TS)
    mean_motion_rad_s = sat.model.no_kozai / 60.0       # rad/min -> rad/s
    semi_major_axis_km = (_MU / mean_motion_rad_s ** 2) ** (1.0 / 3.0)
    return {
        "sat": sat,
        "altitude_km": semi_major_axis_km - _RE_KM,
        "orbital_velocity_kmps": float(np.sqrt(_MU / semi_major_axis_km)),
        "period_min": float(2.0 * np.pi / sat.model.no_kozai),
        "inclination_deg": float(np.degrees(sat.model.inclo)),
    }


def observe(sat, lat_deg, lon_deg, minutes_from_now=0.0):
    """Real station->satellite geometry -> (elevation_deg, distance_km, v_radial_kmps)."""
    station = wgs84.latlon(lat_deg, lon_deg)
    t = _TS.tt_jd(_TS.now().tt + minutes_from_now / 1440.0)
    topo = (sat - station).at(t)
    alt, az, dist = topo.altaz()
    pos = topo.position.km
    vel = topo.velocity.km_per_s
    range_km = float(dist.km)
    v_radial = abs(float(np.dot(pos, vel) / range_km))  # |range rate| km/s -> Doppler
    return float(alt.degrees), range_km, v_radial


def pass_profile(sat, lat_deg, lon_deg, minutes=90.0, step_s=30.0):
    """Elevation over time (for the pass plot). < 0 means out of pass."""
    station = wgs84.latlon(lat_deg, lon_deg)
    t0 = _TS.now()
    times, elevs = [], []
    n = int(minutes * 60 / step_s)
    for i in range(n + 1):
        t = _TS.tt_jd(t0.tt + (i * step_s) / 86400.0)
        alt, _, _ = (sat - station).at(t).altaz()
        times.append(round(i * step_s / 60.0, 1))
        elevs.append(float(alt.degrees))
    return times, elevs


# ── Presets (use cases) ─────────────────────────────────────────────
PRESETS = {
    "Smart city (very dense)":   dict(elevation_deg=45, v_rel_kmps=6.0, n_nodes=100_000),
    "Wildfire (low elevation)":  dict(elevation_deg=10, v_rel_kmps=7.5, n_nodes=10_000),
    "Remote area (sparse)":      dict(elevation_deg=60, v_rel_kmps=4.0, n_nodes=10_000),
}
DEFAULT = dict(elevation_deg=30, v_rel_kmps=5.0, n_nodes=50_000)
N_NODES_CHOICES = [10_000, 50_000, 100_000]   # densities present in the dataset
ISS_L1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9000"
ISS_L2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49114000563537"

# Per-stage technique options. Stage 2 uses "knn" (deterministic) instead of "rule".
STAGE1_OPTS = ["rule", "zero_shot", "few_shot", "cot"]
STAGE2_OPTS = ["knn", "zero_shot", "few_shot", "cot"]
STAGE3_OPTS = ["rule", "zero_shot", "few_shot", "cot"]
_LLM_TECHS = {"zero_shot", "few_shot", "cot"}

st.set_page_config(page_title="AI Agent - LEO Modulation", layout="wide")
st.title("AI Agent - LEO modulation selection (LoRa / LR-FHSS)")
st.caption("3-stage pipeline: geometry -> severity, PER prediction, decision. "
           "Each stage is rule/kNN (deterministic) or LLM (zero_shot/few_shot/cot).")

if not _IMPORT_OK:
    st.error("Missing dependencies:\n\n    pip install -r requirements.txt\n\n"
             f"Details: {_IMPORT_ERR}")
    st.stop()


@st.cache_resource
def _ensure_indexed():
    agent.startup()
    return True


def _mod_of(command: str) -> str:
    """'AT+MOD=LORA-SF12' -> 'sf12' ; 'AT+MOD=LR-FHSS-DR8' -> 'dr8'."""
    return command.split("=")[-1].split("-")[-1].lower()


for k in ("history", "last"):
    st.session_state.setdefault(k, [] if k == "history" else None)

# ── Inputs ──────────────────────────────────────────────────────────
trajectory = None
with st.sidebar:
    mode = st.radio("Input mode", ["Manual", "TLE (real-time)"])

    if mode == "Manual":
        preset = st.selectbox("Use case", ["(custom)"] + list(PRESETS))
        preset_values = PRESETS.get(preset, DEFAULT)
        elevation = st.slider("Elevation (deg)", 5, 90, int(preset_values["elevation_deg"]))
        v_rel     = st.slider("Relative speed (km/s)", 0.0, 7.7, float(preset_values["v_rel_kmps"]), 0.1)
        n_nodes   = st.select_slider("Density (nodes)", N_NODES_CHOICES, value=preset_values["n_nodes"])
        st.caption("distance / kappa / RSSI = metadata looked up from the dataset.")
    else:
        tle_line1 = st.text_area("TLE - line 1", ISS_L1, height=70)
        tle_line2 = st.text_area("TLE - line 2", ISS_L2, height=70)
        col_lat, col_lon = st.columns(2)
        lat = col_lat.number_input("Station latitude (deg)", -90.0, 90.0, 48.85, 0.1)
        lon = col_lon.number_input("Station longitude (deg)", -180.0, 180.0, 2.35, 0.1)
        try:
            sat_info = make_satellite(tle_line1, tle_line2)
        except Exception as exc:               # noqa: BLE001
            st.error(f"Invalid TLE: {exc}")
            st.stop()
        st.caption(f"alt {sat_info['altitude_km']:.0f} km - v_orb {sat_info['orbital_velocity_kmps']:.2f} km/s - "
                   f"incl {sat_info['inclination_deg']:.1f} deg - T {sat_info['period_min']:.0f} min")
        minutes = st.slider("Time (min from now)", 0, max(1, int(sat_info["period_min"])), 0)
        elevation, distance, v_rel = observe(sat_info["sat"], lat, lon, minutes)
        n_nodes = st.select_slider("Density (nodes)", N_NODES_CHOICES, value=50_000)
        trajectory = pass_profile(sat_info["sat"], lat, lon, minutes=sat_info["period_min"])
        if elevation <= 0:
            st.warning(f"Satellite below horizon at t+{minutes} min (elevation {elevation:.1f} deg) - no link.")
        else:
            st.caption(f"t+{minutes} min: elevation {elevation:.1f} deg - v_radial {v_rel:.2f} km/s "
                       f"(real distance {distance:.0f} km)")

    st.divider()
    st.markdown("**Pipeline configuration (per stage)**")
    technique1 = st.selectbox("Stage 1 - severity", STAGE1_OPTS, index=0)
    technique2 = st.selectbox("Stage 2 - PER prediction", STAGE2_OPTS, index=0)
    technique3 = st.selectbox("Stage 3 - decision", STAGE3_OPTS, index=0)
    st.caption("Default rule/knn/rule = classical agent (no LLM, ~80 ms). "
               "LLM stages need Ollama and add seconds of latency.")
    go = st.button("Decide modulation", type="primary")

# ── Pass plot (TLE mode) ────────────────────────────────────────────
if trajectory is not None:
    st.subheader("LEO pass - elevation angle over time")
    times, elevs = trajectory
    st.line_chart(pd.DataFrame({"elevation (deg)": elevs}, index=times))

# ── Decision ────────────────────────────────────────────────────────
if go and elevation <= 0:
    st.warning("Satellite below horizon: pick a time where elevation > 0.")
elif go:
    try:
        _ensure_indexed()
        decision = agent.run(elevation, v_rel, n_nodes, technique1, technique2, technique3)
    except Exception as exc:                   # noqa: BLE001
        st.error(f"Failed - is Ollama running (LLM stage selected)? models pulled?\n\n{exc}")
        st.stop()

    real_per    = decision.get("real_per") or {}
    max_packets = decision["state"].get("max_packets", {})
    oracle_mod  = best_under_tau(real_per, max_packets, TAU, decision["state"]["elevation_deg"])
    chosen_mod  = _mod_of(decision["command"])
    chosen_real = real_per.get("per_" + chosen_mod)
    st.session_state.last = {
        **decision, "elevation": elevation,
        "oracle_mod": oracle_mod, "oracle_command": to_at_command(oracle_mod),
        "chosen_mod": chosen_mod, "is_optimal": chosen_mod == oracle_mod,
        "regret_pkts": int((max_packets.get(oracle_mod, 0) or 0) - (max_packets.get(chosen_mod, 0) or 0)),
        "chosen_pkts": max_packets.get(chosen_mod), "chosen_real_per": chosen_real,
        "meets_tau": (chosen_real is not None and chosen_real <= TAU),
    }
    st.session_state.history.insert(0, {
        "elevation": round(float(elevation), 1),
        "v_rel (km/s)": round(float(v_rel), 2),
        "n_nodes": n_nodes,
        "config": decision.get("technique"),
        "command": decision["command"],
        "oracle": oracle_mod,
        "optimal": int(chosen_mod == oracle_mod),
        "severity": decision["severity"],
        "PER chosen (%)": chosen_real,
        "throughput (pkts)": max_packets.get(chosen_mod),
        "latency (ms)": decision.get("latency_ms"),
    })
    st.session_state.history = st.session_state.history[:100]

# ── Last decision ───────────────────────────────────────────────────
last = st.session_state.last
if last:
    n_llm = sum(t in _LLM_TECHS for t in last.get("technique", "").split("/"))
    mode_label = "Classical (no LLM)" if n_llm == 0 else ("Full LLM" if n_llm == 3 else "Hybrid")
    lat = last.get("latency_ms") or 0.0

    st.subheader("Decision")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recommended modulation", last["command"])
    c2.metric("vs oracle (real data)", "optimal" if last["is_optimal"] else "sub-optimal",
              delta=None if last["is_optimal"] else f"-{last['regret_pkts']} pkts/window",
              delta_color="off" if last["is_optimal"] else "inverse")
    c3.metric("Channel severity", f"{last['severity']} / 5" if last.get("severity") is not None else "-")
    c4.metric("Latency", f"{lat:.0f} ms", delta="< 500 ms" if lat < 500 else "> 500 ms",
              delta_color="normal" if lat < 500 else "inverse")

    per_chosen = last.get("chosen_real_per")
    tau_txt = (f"meets PER<={TAU:.0f}% (real {per_chosen:.1f}%)" if last.get("meets_tau")
               else f"violates PER<={TAU:.0f}% (real {per_chosen:.1f}%)" if per_chosen is not None
               else "no real PER")
    st.caption(f"Mode: **{mode_label}** ({last.get('technique')}) - {tau_txt} - "
               f"throughput {last.get('chosen_pkts')} pkts/window - "
               f"Doppler {last['state']['doppler_hz'] / 1000:.1f} kHz - "
               f"oracle (real-data optimum): {last.get('oracle_command')}")

    st.subheader("PER per modulation (%) - predicted vs real (nearest neighbour)")
    st.bar_chart(pd.DataFrame({
        "predicted": pd.Series(last["per"]),
        "real":      pd.Series(last.get("real_per") or {}),
    }).sort_index())

    if st.button("Apply command to modem"):
        st.success(f"AT command sent to modem: {last['command']}")
        st.code(last["command"], language="text")

# ── History ─────────────────────────────────────────────────────────
st.subheader(f"History ({len(st.session_state.history)} / 100 decisions)")
if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
else:
    st.info("No decision yet. Set the inputs and click 'Decide modulation'.")
