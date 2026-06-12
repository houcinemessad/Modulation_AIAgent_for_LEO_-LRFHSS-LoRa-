"""Dashboard interactif (Streamlit) — étape 6.

Lancement (depuis la racine du repo) :
    python -m streamlit run dashboard/dashboard.py

Prérequis : pip install -r requirements.txt ; Ollama lancé (llama3.2:3b,
nomic-embed-text). Sans Ollama, l'interface s'affiche mais « Décider » renvoie
une erreur claire.

Deux modes :
  - Manuel : on saisit l'état observé du canal.
  - TLE    : on colle 2 lignes TLE ; skyfield (propagation SGP4 réelle) en dérive
             élévation / distance / vitesse radiale pour une station sol + instant,
             et trace la trajectoire. Ce parsing/propagation est intégré ICI
             (make_satellite / observe / pass_profile) car il ne sert qu'au dashboard.
"""
import os
import sys

import pandas as pd
import streamlit as st

# Racine du repo sur le path (ce fichier est dans dashboard/, donc UN cran au-dessus)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import protégé : message clair si une dépendance manque.
try:
    import numpy as np
    from skyfield.api import EarthSatellite, load, wgs84
    from src.agent import agent
    _TS = load.timescale()
    _IMPORT_OK, _IMPORT_ERR = True, None
except Exception as exc:                       # noqa: BLE001
    _IMPORT_OK, _IMPORT_ERR = False, exc


# ── TLE -> géométrie (skyfield, intégré au dashboard) ───────────────
_RE_KM = 6378.137
_MU = 398_600.4418


def make_satellite(line1, line2, name="SAT"):
    """Construit le satellite (SGP4) + infos dérivées pour l'affichage."""
    sat = EarthSatellite(line1, line2, name, _TS)
    n_per_s = sat.model.no_kozai / 60.0                 # mean motion rad/min -> rad/s
    a = (_MU / n_per_s ** 2) ** (1.0 / 3.0)             # demi-grand axe (km)
    return {
        "sat": sat,
        "altitude_km": a - _RE_KM,
        "orbital_velocity_kmps": float(np.sqrt(_MU / a)),
        "period_min": float(2.0 * np.pi / sat.model.no_kozai),
        "inclination_deg": float(np.degrees(sat.model.inclo)),
        "epoch_age_days": float(_TS.now() - sat.epoch),  #corrected
    }


def observe(sat, lat_deg, lon_deg, minutes_from_now=0.0):
    """Géométrie RÉELLE station->satellite -> (elevation_deg, distance_km, v_radial_kmps)."""
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
    """Élévation au fil du temps (pour la trajectoire). < 0 hors passage."""
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


# ── Presets (cas d'usage) ───────────────────────────────────────────
PRESETS = {
    "Smart city (très dense)":    dict(elevation_deg=45, v_rel_kmps=6.0, n_nodes=100_000),
    "Feu de forêt (basse élév.)": dict(elevation_deg=10, v_rel_kmps=7.5, n_nodes=10_000),
    "Zone déserte (peu dense)":   dict(elevation_deg=60, v_rel_kmps=4.0, n_nodes=10_000),
}
DEFAULT = dict(elevation_deg=30, v_rel_kmps=5.0, n_nodes=50_000)
N_NODES_CHOICES = [10_000, 50_000, 100_000]   # densités présentes dans le dataset
ISS_L1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9000"
ISS_L2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49114000563537"

st.set_page_config(page_title="Agent IA — Modulation LEO", layout="wide")
st.title("🛰️ Agent IA — sélection de modulation LEO (LoRa / LR-FHSS)")

if not _IMPORT_OK:
    st.error("Dépendances manquantes :\n\n    pip install -r requirements.txt\n\n"
             f"Détail : {_IMPORT_ERR}")
    st.stop()


@st.cache_resource
def _ensure_indexed():
    agent.startup()
    return True


def _mod_key(command: str) -> str:
    """'AT+MOD=LORA-SF12' -> 'per_sf12' ; 'AT+MOD=LR-FHSS-DR8' -> 'per_dr8'."""
    return "per_" + command.split("=")[-1].split("-")[-1].lower()


for k in ("history", "last"):
    st.session_state.setdefault(k, [] if k == "history" else None)

# ── Entrées ─────────────────────────────────────────────────────────
trajectory = None
with st.sidebar:
    mode = st.radio("Mode d'entrée", ["Manuel", "TLE (temps réel)"])

    if mode == "Manuel":
        preset = st.selectbox("Cas d'usage", ["(personnalisé)"] + list(PRESETS))
        base = PRESETS.get(preset, DEFAULT)
        elevation = st.slider("Élévation (°)", 5, 90, int(base["elevation_deg"]))
        v_rel     = st.slider("Vitesse relative (km/s)", 0.0, 7.7, float(base["v_rel_kmps"]), 0.1)
        n_nodes   = st.select_slider("Densité (nœuds)", N_NODES_CHOICES, value=base["n_nodes"])
        st.caption("distance / kappa / RSSI = métadonnées (récupérées dans le dataset).")
    else:
        l1 = st.text_area("TLE — ligne 1", ISS_L1, height=70)
        l2 = st.text_area("TLE — ligne 2", ISS_L2, height=70)
        col_a, col_b = st.columns(2)
        lat = col_a.number_input("Station latitude (°)", -90.0, 90.0, 48.85, 0.1)
        lon = col_b.number_input("Station longitude (°)", -180.0, 180.0, 2.35, 0.1)
        try:
            info = make_satellite(l1, l2)
        except Exception as exc:               # noqa: BLE001
            st.error(f"TLE invalide : {exc}")
            st.stop()
        st.caption(f"alt {info['altitude_km']:.0f} km · v_orb {info['orbital_velocity_kmps']:.2f} km/s · "
                   f"incl {info['inclination_deg']:.1f}° · T {info['period_min']:.0f} min")
        if info["epoch_age_days"] > 30:  #corrected
            st.warning(f"TLE vieux de {info['epoch_age_days']:.0f} jours : la propagation SGP4 "
                       f"devient imprécise — récupère un TLE récent (celestrak.org).")  #corrected
        minutes = st.slider("Instant (min à partir de maintenant)", 0, max(1, int(info["period_min"])), 0)
        elevation, distance, v_rel = observe(info["sat"], lat, lon, minutes)
        n_nodes = st.select_slider("Densité (nœuds)", N_NODES_CHOICES, value=50_000)
        trajectory = pass_profile(info["sat"], lat, lon, minutes=info["period_min"])
        if elevation <= 0:
            st.warning(f"Satellite sous l'horizon à t+{minutes} min (élévation {elevation:.1f}°) — pas de lien.")
        else:
            st.caption(f"t+{minutes} min : élévation {elevation:.1f}° · v_radiale {v_rel:.2f} km/s "
                       f"(distance réelle {distance:.0f} km = info ; l'agent lit les métadonnées du dataset)")

    technique = st.selectbox("Technique de prompt", ["zero_shot", "few_shot", "cot"],
                             help="Stratégie de prompting (comparées à l'étape 5).")
    if st.button("🔄 Reconstruire l'index (après régénération du dataset)"):  #corrected
        agent.startup(force_rebuild=True)  #corrected
        st.success("Index ChromaDB reconstruit sur le dataset courant.")  #corrected
    go = st.button("Décider la modulation", type="primary")

# ── Trajectoire (mode TLE) ──────────────────────────────────────────
if trajectory is not None:
    st.subheader("Trajectoire LEO — évolution de l'angle d'élévation")
    times, elevs = trajectory
    st.line_chart(pd.DataFrame({"élévation (°)": elevs}, index=times))

# ── Décision ────────────────────────────────────────────────────────
if go and elevation <= 0:
    st.warning("Satellite sous l'horizon : choisis un instant où l'élévation est > 0.")
elif go:
    try:
        _ensure_indexed()
        out = agent.run(elevation, v_rel, n_nodes, technique)
    except Exception as exc:                   # noqa: BLE001
        st.error(f"Échec — Ollama lancé ? modèles pull ?\n\n{exc}")
        st.stop()
    st.session_state.last = {**out, "elevation": elevation}
    key = _mod_key(out["command"])
    st.session_state.history.insert(0, {
        "élévation": round(float(elevation), 1),
        "v_rel (km/s)": round(float(v_rel), 2),
        "n_nodes": n_nodes,
        "technique": out.get("technique"),
        "commande": out["command"],
        "sévérité": out["severity"],
        "PER prédit (%)": out["per"].get(key),
        "PER réel (%)": (out.get("real_per") or {}).get(key),
        "latence (ms)": out.get("latency_ms"),
    })
    st.session_state.history = st.session_state.history[:100]

# ── Dernière décision : modulation + PER prédit vs réel + Appliquer ─
last = st.session_state.last
if last:
    c1, c2, c3 = st.columns(3)
    c1.metric("Modulation recommandée", last["command"])
    c2.metric("Sévérité du canal",
              f"{last['severity']} / 5" if last.get("severity") is not None else "—")
    c3.metric("Doppler", f"{last['state']['doppler_hz'] / 1000:.1f} kHz")
    st.caption(f"technique : {last.get('technique')} · latence : "
               f"{last.get('latency_ms')} ms (cible < 500)")

    st.subheader("PER par modulation (%) — prédit vs réel (voisin le plus proche)")
    st.bar_chart(pd.DataFrame({
        "prédit": pd.Series(last["per"]),
        "réel":   pd.Series(last.get("real_per") or {}),
    }).sort_index())

    if st.button("✅ Appliquer la commande au modem"):
        st.success(f"Commande AT envoyée au modem : {last['command']}")
        st.code(last["command"], language="text")

# ── Historique des 100 dernières décisions (réel vs prédit) ─────────
st.subheader(f"Historique ({len(st.session_state.history)} / 100 décisions) — réel vs prédit")
if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
else:
    st.info("Aucune décision encore. Règle l'entrée et clique « Décider la modulation ».")