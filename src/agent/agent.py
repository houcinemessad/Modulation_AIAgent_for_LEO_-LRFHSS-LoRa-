"""Implements the main agent. The agent receives the 3 axes (elevation, v_rel, density) and then DECIDES.

At fixed altitude/channel, the ONLY free inputs are (elevation, v_rel, n_nodes)
— this is exactly the chunker's key. The rest (distance, kappa, RSSI, SNR) are
channel METADATA: build_state retrieves them from the dataset (corresponding row).
"""
import csv
import math
import os
import sys

import pandas as pd

if __package__ is None or __package__ == "":
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from src.rag.chunker import chunk_dataset
from src.agent.pipeline import SatelliteAgentPipeline

# ── Constants ──────────────────────────────────────────────────────
_DATASET_CSV = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "dataset_real.csv"))

# STANDARD radio constants (for SNR calculation; NOT the simulator)
_NF_DB   = 6.0          # receiver noise figure (dB)
_BW_LORA = 125_000.0    # bande LoRa (Hz)
_BW_FHSS = 488.0        # bande d'un fragment LR-FHSS (Hz)

MODS = ("sf7", "sf8", "sf9", "sf10", "sf11", "sf12", "dr8", "dr9")

# ── Singletons (constructed ONCE, at import) ──────────────────────
# _DATASET  : source of channel METADATA (distance, kappa, doppler, RSSI, real PER).
# _PIPELINE : RAG + 3 prompts; reuse its store/embedder for indexing.
_DATASET  = pd.read_csv(_DATASET_CSV)
_PIPELINE = SatelliteAgentPipeline()


def startup(dataset_path=_DATASET_CSV, force_rebuild=False):
    """Index chunks into ChromaDB (data/chroma) — call ONCE at startup.
    Reuses the pipeline's store/embedder (the SAME index that execute will read later).
    index_chunks skips if already populated; set force_rebuild=True to rebuild (new dataset)."""
    _PIPELINE.store.index_chunks(list(chunk_dataset(dataset_path)), _PIPELINE.embedder,
                                 force_rebuild=force_rebuild)


def _noise_floor_dbm(bw_hz):
    """Thermal noise floor: -174 dBm/Hz + 10·log10(BW) + NF."""
    return -174.0 + 10.0 * math.log10(bw_hz) + _NF_DB


def build_state(elevation_deg, v_rel_kmps, n_nodes):
    """
    distance / kappa / Doppler / RSSI = METADATA read from the dataset (row whose
    elevation THEN v_rel are the closest, at fixed density).
    """
    density_subset = _DATASET[_DATASET["n_nodes"] == n_nodes]
    # nearest elevation first, THEN nearest v_rel at that elevation.
    # v_rel is a real axis: it fixes the Doppler/RSSI read from the metadata.
    nearest_elev = density_subset.loc[(density_subset["elevation_deg"] - elevation_deg).abs().idxmin(), "elevation_deg"]
    rows_at_elev = density_subset[density_subset["elevation_deg"] == nearest_elev]
    nearest_row  = rows_at_elev.loc[(rows_at_elev["v_rel_kmps"] - v_rel_kmps).abs().idxmin()]
    rssi = float(nearest_row["RSSI_dBm"])
    return {
        "elevation_deg": int(elevation_deg),
        "v_rel_kmps":    float(v_rel_kmps),
        "n_nodes":       int(n_nodes),
        "distance_km":   float(nearest_row["distance_km"]),   # metadata (dataset)
        "kappa":         float(nearest_row["kappa"]),         # metadata (rice, dataset)
        "doppler_hz":    float(nearest_row["doppler_hz"]),    # metadata (dataset) — consistent
        "rssi_mean_dbm": rssi,                                # metadata (dataset)
        "snr_lora_db":   rssi - _noise_floor_dbm(_BW_LORA),
        "snr_lrfhss_db": rssi - _noise_floor_dbm(_BW_FHSS),
    }


def _real_per(state):
    """Real PER for the 8 modulations = dataset row closest to the given state
    (ground truth for the predicted-vs-real comparison on the dashboard). Keys are per_<mod> lowercase."""
    density_subset = _DATASET[_DATASET["n_nodes"] == state["n_nodes"]]
    nearest_elev = density_subset.loc[(density_subset["elevation_deg"] - state["elevation_deg"]).abs().idxmin(), "elevation_deg"]
    rows_at_elev = density_subset[density_subset["elevation_deg"] == nearest_elev]
    nearest_vrel = rows_at_elev.loc[(rows_at_elev["v_rel_kmps"] - state["v_rel_kmps"]).abs().idxmin(), "v_rel_kmps"]
    rows_at_state = rows_at_elev[rows_at_elev["v_rel_kmps"] == nearest_vrel]
    return {f"per_{m.lower()}": round(float(p), 2) for m, p in zip(rows_at_state["modulation"], rows_at_state["PER_pct"])}


def run(elevation_deg, v_rel_kmps, n_nodes, technique1="zero_shot", technique2=None, technique3=None):
    """Build the state then run 3 LLM calls and return results."""

    technique2 = technique2 or technique1
    technique3 = technique3 or technique1
    state = build_state(elevation_deg, v_rel_kmps, n_nodes)
    # Keys expected by SatelliteAgentPipeline.execute.
    state["relative_velocity_ms"] = float(v_rel_kmps) * 1000.0
    state["rssi_dbm"]             = state["rssi_mean_dbm"]
    state["N"]                    = int(n_nodes)

    result = _PIPELINE.execute(state, technique1, technique2, technique3)
    return {
        "state":      state,
        "command":    result.get("selected_command"),
        "severity":   result.get("severity"),
        "per":        result.get("per", {}),
        "real_per":   _real_per(state),
        "latency_ms": result.get("latency_ms"),
        "technique":  result.get("technique_used", f"{technique1}/{technique2}/{technique3}"),
    }

# Command-line test (requires Ollama running + dependencies installed)
if __name__ == "__main__":
    startup()
    elev  = float(input("Élévation (°): "))
    vrel  = float(input("Vitesse relative (km/s): "))
    nodes = int(input("Densité (nœuds): "))
    t1 = input("Technique étage 1 [few_shot]: ") or "few_shot"
    t2 = input("Technique étage 2 [few_shot]: ") or "few_shot"
    t3 = input("Technique étage 3 [few_shot]: ") or "few_shot"
    out = run(elev, vrel, nodes, t1, t2, t3)
    print("État    :", out["state"])
    print("Technique:", out["technique"])
    print("Sévérité:", out["severity"])
    print("PER     :", out["per"])
    print("Commande:", out["command"])
    print("Latence :", out["latency_ms"], "ms")
