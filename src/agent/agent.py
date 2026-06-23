"""Agent: takes (elevation, v_rel, density) and returns a modulation decision.

The only free inputs are (elevation, v_rel, n_nodes) — the chunker key. Everything
else (distance, kappa, RSSI, SNR, throughput) is channel metadata looked up from the
dataset, not simulated or asked from the user.

  - build_state : (elevation, v_rel, n_nodes) -> full state (dataset metadata).
  - run         : same -> {state, command, severity, per, real_per, latency_ms, technique}.
"""
import os
import sys
import csv
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
    """Build the pipeline once (lazy singleton)."""
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = SatelliteAgentPipeline()
    return _PIPELINE


def startup(dataset_path=_DATASET_PATH, force_rebuild=False):
    """Index the chunks into ChromaDB (data/chroma). Call once at launch.
    Skips if the index is already populated; force_rebuild=True rebuilds it
    (use after regenerating the dataset)."""
    p = _pipeline()
    p.store.index_chunks(list(chunk_dataset(dataset_path)), p.embedder, force_rebuild=force_rebuild)


# Standard radio constants (only for the SNR estimate, not the simulator).
_NF_DB   = 6.0          # receiver noise figure (dB)
_BW_LORA = 125_000.0    # LoRa bandwidth (Hz)
_BW_FHSS = 488.0        # LR-FHSS fragment bandwidth (Hz)

_DATASET_CSV = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "dataset_real.csv"))
_DATASET = None


def _dataset():
    """Load the dataset once (the source of channel metadata)."""
    global _DATASET
    if _DATASET is None:
        _DATASET = pd.read_csv(_DATASET_CSV)
    return _DATASET


def _noise_floor_dbm(bw_hz):
    """Thermal noise floor: -174 dBm/Hz + 10*log10(BW) + NF."""
    return -174.0 + 10.0 * math.log10(bw_hz) + _NF_DB


def build_state(elevation_deg, v_rel_kmps, n_nodes):
    """Turn the 3 inputs into a full channel state.

    distance / kappa / doppler / RSSI / throughput are read from the dataset row
    whose (elevation, v_rel) is closest, at the given density. SNR is derived from RSSI.
    """
    df = _dataset()
    sub = df[df["n_nodes"] == n_nodes]
    if sub.empty:
        sub = df
    # Nearest elevation, then nearest v_rel at that elevation (v_rel is a real axis:
    # it sets the doppler/RSSI read from the dataset).
    e_near = sub.loc[(sub["elevation_deg"] - elevation_deg).abs().idxmin(), "elevation_deg"]
    cand   = sub[sub["elevation_deg"] == e_near]
    row    = cand.loc[(cand["v_rel_kmps"] - v_rel_kmps).abs().idxmin()]
    grp    = cand[cand["v_rel_kmps"] == row["v_rel_kmps"]]   # the 8 modulations of this state
    max_packets = {m.lower(): int(p)                          # throughput capacity (packets/window)
                   for m, p in zip(grp["modulation"], grp["max_packets_in_window"])}
    rssi = float(row["RSSI_dBm"])
    return {
        "elevation_deg": int(elevation_deg),
        "v_rel_kmps":    float(v_rel_kmps),
        "n_nodes":       int(n_nodes),
        "distance_km":   float(row["distance_km"]),
        "kappa":         float(row["kappa"]),
        "doppler_hz":    float(row["doppler_hz"]),
        "rssi_mean_dbm": rssi,
        "snr_lora_db":   rssi - _noise_floor_dbm(_BW_LORA),
        "snr_lrfhss_db": rssi - _noise_floor_dbm(_BW_FHSS),
        "max_packets":   max_packets,                   # {mod: packets/window} for the decision
    }


def _real_per(state):
    """Ground-truth PER of the 8 modulations = nearest (elevation, v_rel) dataset row.
    Used to compare predicted vs real PER. Keys are lowercase per_<mod>."""
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
        technique1="rule", technique2="cot", technique3="rule"):
    """(elevation, v_rel, n_nodes) -> {state, command, severity, per, real_per, latency_ms, technique}.
    Stages 1/3 accept "rule" (deterministic); stage 2 accepts "knn"; all accept "zero_shot"/
    "few_shot"/"cot" (LLM). No technique given -> classical agent rule/knn/rule (deterministic,
    ~80 ms). A single technique applies to all 3 stages. Assumes startup() ran and Ollama is up."""
    if technique1 is None and technique2 is None and technique3 is None:
        technique1, technique2, technique3 = "rule", "knn", "rule"     # classical default
    else:
        technique1 = technique1 or "cot"
        technique2 = technique2 or technique1
        technique3 = technique3 or technique1
    state = build_state(elevation_deg, v_rel_kmps, n_nodes)
    # Keys expected by SatelliteAgentPipeline.execute.
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


MODS = ("sf7", "sf8", "sf9", "sf10", "sf11", "sf12", "dr8", "dr9")


def generate_dashboard(out_csv="data/dataset_agent_zs.csv"):
    """Sweep the grid and dump each decision (predicted vs real PER) to a CSV."""
    startup()
    elevations = range(1, 91)
    vrels      = [round(0.5 * k, 1) for k in range(2, 16)]  # 1.0, 1.5, ..., 7.5
    nnodes     = [1_000, 10_000, 100_000]
    fields = (["elevation_deg", "v_rel_kmps", "n_nodes", "severity", "command"]
              + [f"pred_{m}" for m in MODS] + [f"real_{m}" for m in MODS])
    rows = []
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
                        row[f"pred_{m}"] = out["per"].get(f"per_{m}")                    # predicted PER
                        row[f"real_{m}"] = (out.get("real_per") or {}).get(f"per_{m}")   # real PER
                    w.writerow(row); f.flush()              # write + flush each row
                    rows.append(row)
                    print(f"el={elev} v={v} n={n} -> {out['command']}")
    return rows


# CLI test (needs Ollama running + deps installed)
if __name__ == "__main__":
    startup()
    out = run(float(input("Elevation (deg): ")),
              float(input("Relative speed (km/s): ")),
              int(input("Density (nodes): ")))
    print("State   :", out["state"])
    print("Severity:", out["severity"])
    print("PER     :", out["per"])
    print("Command :", out["command"])
