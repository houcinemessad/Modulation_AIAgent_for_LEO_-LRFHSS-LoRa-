"""L'agent : reçoit les 3 axes (élévation, v_rel, densité) puis DÉCIDE.

À altitude/canal fixes, les SEULES entrées libres sont (élévation, v_rel, n_nodes)
— exactement la clé du chunker. Le reste (distance, kappa, RSSI, SNR) sont des
MÉTADONNÉES du canal : build_state les RÉCUPÈRE dans le dataset (ligne
correspondante), il ne les simule pas et ne les demande pas à l'utilisateur.
Le Doppler se déduit de v_rel (définition Δf = v/c·fc).

  - build_state : (élévation, v_rel, n_nodes) -> état complet (métadonnées du dataset).
  - run         : idem -> {state, severity, per, command}.
"""
import os
import sys
import math

import pandas as pd

if __package__ is None or __package__ == "":
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from src.rag.chunker import chunk_dataset
from src.agent.pipeline import SatelliteAgentPipeline

_DATASET_PATH = "data/dataset_real.csv"
_PIPELINE = None


def _pipeline():
    """Instancie une seule fois la pipeline (classe SatelliteAgentPipeline)."""
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = SatelliteAgentPipeline()
    return _PIPELINE


def startup(dataset_path=_DATASET_PATH, force_rebuild=False):
    """Indexe les chunks dans ChromaDB (data/chroma) — à appeler UNE fois au lancement.
    Réutilise le store/embedder de la pipeline (le MÊME index que execute lira ensuite).
    index_chunks saute si déjà peuplé ; force_rebuild=True pour reconstruire (nouveau dataset)."""
    p = _pipeline()
    p.store.index_chunks(list(chunk_dataset(dataset_path)), p.embedder, force_rebuild=force_rebuild)

# Constantes radio STANDARD (pour le calcul des SNR ; PAS le simulateur)
_NF_DB   = 6.0          # facteur de bruit récepteur (dB)
_BW_LORA = 125_000.0    # bande LoRa (Hz)
_BW_FHSS = 488.0        # bande d'un fragment LR-FHSS (Hz)

_DATASET_CSV = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "dataset_real.csv"))
_DATASET = None


def _dataset():
    """Charge le dataset une fois — c'est la source des MÉTADONNÉES du canal."""
    global _DATASET
    if _DATASET is None:
        _DATASET = pd.read_csv(_DATASET_CSV)
    return _DATASET


def _noise_floor_dbm(bw_hz):
    """Plancher de bruit thermique : -174 dBm/Hz + 10·log10(BW) + NF."""
    return -174.0 + 10.0 * math.log10(bw_hz) + _NF_DB


def build_state(elevation_deg, v_rel_kmps, n_nodes):
    """3 axes -> état complet.

    distance / kappa / Doppler / RSSI = MÉTADONNÉES lues dans le dataset (ligne
    dont l'élévation et la densité sont les plus proches). Les SNR sont calculés
    depuis le RSSI. Aucune simulation, aucune saisie de ces champs.
    """
    df = _dataset()
    sub = df[df["n_nodes"] == n_nodes]
    if sub.empty:
        sub = df
    # élévation la plus proche, PUIS v_rel le plus proche à cette élévation.
    # v_rel est un vrai axe : il fixe le Doppler/RSSI lus en métadonnée.
    e_near = sub.loc[(sub["elevation_deg"] - elevation_deg).abs().idxmin(), "elevation_deg"]
    cand   = sub[sub["elevation_deg"] == e_near]
    row    = cand.loc[(cand["v_rel_kmps"] - v_rel_kmps).abs().idxmin()]
    grp    = cand[cand["v_rel_kmps"] == row["v_rel_kmps"]]   # les 8 modulations de cet état
    max_packets = {m.lower(): int(p)                          # débit déterministe (paquets/fenêtre)
                   for m, p in zip(grp["modulation"], grp["max_packets_in_window"])}
    rssi = float(row["RSSI_dBm"])
    return {
        "elevation_deg": int(elevation_deg),
        "v_rel_kmps":    float(v_rel_kmps),
        "n_nodes":       int(n_nodes),
        "distance_km":   float(row["distance_km"]),     # métadonnée (dataset)
        "kappa":         float(row["kappa"]),           # métadonnée (rice, dataset)
        "doppler_hz":    float(row["doppler_hz"]),      # métadonnée (dataset) — cohérent
        "rssi_mean_dbm": rssi,                          # métadonnée (dataset)
        "snr_lora_db":   rssi - _noise_floor_dbm(_BW_LORA),
        "snr_lrfhss_db": rssi - _noise_floor_dbm(_BW_FHSS),
        "max_packets":   max_packets,                   # {mod: paquets/fenêtre} pour la décision τ
    }


def _real_per(state):
    """PER 'réel' des 8 modulations = état (élévation, v_rel) le plus proche du dataset
    (vérité terrain pour le comparatif prédit-vs-réel du dashboard). Clés per_<mod> minuscules."""
    df = _dataset()
    sub = df[df["n_nodes"] == state["n_nodes"]]
    if sub.empty:
        sub = df
    e_near = sub.loc[(sub["elevation_deg"] - state["elevation_deg"]).abs().idxmin(), "elevation_deg"]
    cand   = sub[sub["elevation_deg"] == e_near]
    v_near = cand.loc[(cand["v_rel_kmps"] - state["v_rel_kmps"]).abs().idxmin(), "v_rel_kmps"]
    grp    = cand[cand["v_rel_kmps"] == v_near]
    return {f"per_{m.lower()}": round(float(p), 2) for m, p in zip(grp["modulation"], grp["PER_pct"])}


def run(elevation_deg, v_rel_kmps, n_nodes,
        technique1="zero_shot", technique2=None, technique3=None):
    """(élévation, v_rel, n_nodes) -> {state, command, severity, per, real_per, latency_ms, technique}.
    technique1/2/3 ∈ {"zero_shot", "few_shot", "cot"} = stratégie de chacun des 3 étages
    (géométrie->sévérité, prédiction PER, décision). Un seul argument -> appliqué aux 3 étages.
    Suppose startup() déjà appelé (chunks indexés) et Ollama lancé."""
    technique2 = technique2 or technique1
    technique3 = technique3 or technique1
    state = build_state(elevation_deg, v_rel_kmps, n_nodes)
    # Clés attendues par SatelliteAgentPipeline.execute (conventions de la classe).
    state["relative_velocity_ms"] = float(v_rel_kmps) * 1000.0
    state["rssi_dbm"]             = state["rssi_mean_dbm"]
    state["N"]                    = int(n_nodes)

    result = _pipeline().execute(state, technique1, technique2, technique3)
    return {
        "state":      state,
        "command":    result.get("selected_command"),
        "severity":   result.get("severity"),
        "per":        result.get("per", {}),
        "real_per":   _real_per(state),
        "latency_ms": result.get("latency_ms"),
        "technique":  result.get("technique_used", f"{technique1}/{technique2}/{technique3}"),
    }

import csv

MODS = ("sf7", "sf8", "sf9", "sf10", "sf11", "sf12", "dr8", "dr9")

def generate_dashboard(out_csv="data/dataset_agent_zs.csv"):
    startup()
    elevations = range(1, 91)                               # 1 à 90
    vrels      = [round(0.5 * k, 1) for k in range(2, 16)]  # 1.0, 1.5, ..., 7.5
    nnodes     = [1_000, 10_000, 100_000]
    fields = (["elevation_deg", "v_rel_kmps", "n_nodes", "severity", "command"]
              + [f"pred_{m}" for m in MODS] + [f"real_{m}" for m in MODS])
    L = []
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for elev in elevations:
            for v in vrels:
                for n in nnodes:
                    out = run(float(elev), float(v), int(n))
                    row = {"elevation_deg": elev, "v_rel_kmps": v, "n_nodes": n,
                           "severity": out["severity"], "command": out["command"]}
                    for m in MODS:
                        row[f"pred_{m}"] = out["per"].get(f"per_{m}")            # PER prédit
                        row[f"real_{m}"] = (out.get("real_per") or {}).get(f"per_{m}")  # PER réel
                    w.writerow(row); f.flush()              # écrit la ligne + sauve direct
                    L.append(row)
                    print(f"el={elev} v={v} n={n} -> {out['command']}")
    return L


# Test en ligne de commande (nécessite Ollama lancé + deps installées)
if __name__ == "__main__":
    startup()
    out = run(float(input("Élévation (°): ")), float(input("Vitesse relative (km/s): ")), int(input("Densité (nœuds): ")))
    print("État    :", out["state"])
    print("Sévérité:", out["severity"])
    print("PER     :", out["per"])
    print("Commande:", out["command"])
