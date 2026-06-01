"""
Chunk the dataset into groups of rows that share the same 
(elevation_deg, v_rel_kmps, n_nodes) triple.

"""
from __future__ import annotations
import pandas as pd
from typing import Iterator

ELEVATION_BINS = [
    (0.0,  15.0, "extreme_low"),
    (15.0, 30.0, "low"),
    (30.0, 60.0, "mid"),
    (60.0, 91.0, "high"),
]

DOPPLER_BINS = [
    (0.0,  2.0,  "low"),
    (2.0,  5,  "mid"),
    (5,  7.6, "extreme"),
]

DENSITY_BINS = [
    (0,       50_000,    "not_busy"),
    (50_000,  100_000,   "busy"),
    (100_000, 1_000_000, "very_busy"),
]


def classify_into_bin(value, bins):
    """
    Take a numeric value and return the label of the bin it falls into.

    Bins are half-open intervals [low, high): the lower edge is included,
    the upper edge is not. That choice avoids ambiguity when a value
    lands exactly on a boundary — it always belongs to the bin on its
    right (the higher one).

    Edge case: a value sitting exactly on the very last upper edge (e.g.
    elevation = 90°) falls in no half-open interval, so we explicitly
    return the last bin's label rather than raising.
    """
    for low, high, label in bins:
        if low <= value < high:
            return label
    return bins[-1][2]


# Columns the chunker actually reads from the CSV. Anything beyond this is copied verbatim into `rows` but never inspected.
#If you dont want to change your columns you can just add the new ones to the list and it will work fine, but if you want to remove some of the columns you have to make sure that they are not in this list, otherwise the chunker will raise an error.

"real data set from bimodal simulator are : elevation_deg;mode;sf;N;tx_time_s;toa_s;channel;elevation_deg;distance_km;visibility_window_s;relative_velocity_ms;fspl_db;clutter_db;rician_k;fading_db;shadow_db;rssi_dbm;rx_power_dbm;snr_db;collision;n_colliders;doppler_lost;doppler_static_lost;doppler_dynamic_lost;doppler_joint_lost;doppler_shift_hz;doppler_rate_hz_s;carrier_change_hz;PS_success;headers_rx;fragments_rx;total_fragments;required_fragments;element_collision_rate;n_elements;n_headers;n_fragments;n_collided_headers;n_collided_fragments;success;doppler_snr_penalty_db"
"In the future an adapter who convert the column names to the ones expected by the chunker could be implemented, but for now the dataset has to have the exact columns expected by the chunker"

REQUIRED_COLUMNS = (
    "elevation_deg", "v_rel_kmps", "N",
    "mode", "PER_pct",
    "distance_km", "kappa", "doppler_hz", "doppler_rate_hz", "visibility_window_s",
    "RSSI_dBm", "SNR_dB",
    "toa_s", "max_packets_in_window"
)


def validate_required_columns(dataset: pd.DataFrame) -> None:
    """
    Fail fast (and loudly) if the CSV is missing fields the chunker needs.

    Each CSV row is one (channel state × modulation) measurement. If even
    one of the required columns is missing, every chunk would be
    malformed downstream — we prefer to crash here with a clear message
    than to let a broken vector store be built silently.
    """
    "real data set : elevation_deg;mode;sf;N;packet_id;node_id;tx_time_s;toa_s;channel;elevation_deg;pass_time_s;distance_km;visibility_window_s;orbital_velocity_ms;relative_velocity_ms;fspl_db;clutter_db;rician_k;fading_db;shadow_db;rssi_dbm;rx_power_dbm;noise_dbm;snr_db;collision;n_colliders;doppler_lost;doppler_static_lost;doppler_dynamic_lost;doppler_joint_lost;doppler_shift_hz;doppler_rate_hz_s;carrier_change_hz;PS_success;headers_rx;fragments_rx;total_fragments;required_fragments;element_collision_rate;n_elements;n_headers;n_fragments;n_collided_headers;n_collided_fragments;success;doppler_snr_penalty_db"
    
    missing = [column for column in REQUIRED_COLUMNS if column not in dataset.columns]
    if missing:
        raise KeyError(
            f"Dataset is missing required columns: {missing}. "
            f"The chunker needs all of: {list(REQUIRED_COLUMNS)}. "
            f"Got: {list(dataset.columns)}."
        )


# ─────────────────────────────────────────────────────────────────
# CHUNKER
# ─────────────────────────────────────────────────────────────────

def chunk_dataset(dataset_path: str) -> Iterator[dict]:
    """
    Yield one chunk per (elevation_deg, v_rel_kmps, n_nodes) grid point.

    The idea: for any given channel state, the agent should be able to
    pull back, in a single retrieval, the PER of every modulation it
    might pick. So each chunk bundles together the 8 modulations
    measured under those exact conditions.
    
    """
    dataset = pd.read_csv(dataset_path)
    validate_required_columns(dataset)

    # A "channel state" is fully determined by the satellite geometry
    # (elevation, relative speed) and the network load (n_nodes). Grouping
    # on these three columns gives us one group per state, containing
    # exactly the 8 modulation rows measured under that state.
    channel_state_columns = ["elevation_deg", "v_rel_kmps", "n_nodes"]

    for (elev, vrel, nodes), modulations_at_state in dataset.groupby(
        channel_state_columns, sort=True
    ):

        # Each row of the group is one of the 8 modulations at this state.
        per_by_modulation = dict(zip(
            modulations_at_state["mode"],
            modulations_at_state["PER_pct"].round(2),
        ))
        toa_by_modulation = dict(zip(
            modulations_at_state["mode"],
            modulations_at_state["toa_s"].round(4),
        ))
        max_packets_by_modulation = dict(zip(
            modulations_at_state["mode"],
            modulations_at_state["max_packets_in_window"].astype(int),
        ))

        # Channel descriptors below (distance, kappa, doppler, visibility)
        # depend ONLY on the channel state, not on the modulation, so they are
        # identical across the 8 rows of the group. We pick the first row as a
        # representative — any of the eight would give the same numbers.
        channel_state_row = modulations_at_state.iloc[0]
        lora_rows   = modulations_at_state[modulations_at_state["mode"].str.startswith("SF")]
        lrfhss_rows = modulations_at_state[modulations_at_state["mode"].str.startswith("DR")]
#############################################################mode###############################

        metadata = {
            # Exact grid coordinates
            "elevation_deg":     int(elev),
            "v_rel_kmps":        float(vrel),
            "n_nodes":           int(nodes),

            # Physical bin labels (classification, not decision)
            "elevation_bin":     classify_into_bin(elev,  ELEVATION_BINS),
            "doppler_bin":       classify_into_bin(vrel,  DOPPLER_BINS),
            "density_bin":       classify_into_bin(nodes, DENSITY_BINS),

            # Physical channel descriptors (single-valued at this channel state)
            "distance_km":       float(channel_state_row["distance_km"]),
            "kappa":             float(channel_state_row["kappa"]),
            "doppler_hz":        float(channel_state_row["doppler_hz"]),
            "doppler_rate_hz":   float(channel_state_row["doppler_rate_hz"]),
            "rssi_mean_dbm":     round(float(modulations_at_state["RSSI_dBm"].mean()), 2),
            "snr_lora_mean_db":  round(float(lora_rows["SNR_dB"].mean()),   2),
            "snr_lrfhss_mean_db":round(float(lrfhss_rows["SNR_dB"].mean()), 2),
            "visibility_s":      float(channel_state_row["visibility_window_s"]),

            # Per-modulation PER, raw measurement, not a decision.
            # Flat keys so ChromaDB can filter on them: per_DR8 < 20.
            **{f"per_{mod}":         float(per) for mod, per in per_by_modulation.items()},
            **{f"toa_{mod}":         float(toa) for mod, toa in toa_by_modulation.items()},
            **{f"max_packets_{mod}": int(n)     for mod, n   in max_packets_by_modulation.items()},
        }

        yield {
            "id":       f"chunk_el{int(elev):02d}_v{vrel:g}_n{int(nodes)}",
            "metadata": metadata,
            "rows":     modulations_at_state.to_dict(orient="records"),
        }


# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Main/test code: run this file to see how the dataset gets chunked, and to check the distribution of channel states across the chunks.
# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from collections import Counter
    dataset_path = "data/dataset.csv" 
    try:
        all_chunks = list(chunk_dataset(dataset_path))
    except FileNotFoundError:
        dataset_path = input("Enter the path to the dataset CSV file: ").strip()
        all_chunks = list(chunk_dataset(dataset_path))

    all_chunks = list(chunk_dataset(dataset_path))
    total_rows = sum(len(chunk["rows"]) for chunk in all_chunks)

    print(f"Built {len(all_chunks)} chunks covering {total_rows} rows "
          f"({total_rows / len(all_chunks):.1f} modulations/chunk).\n")

    bin_distribution = Counter(
        (chunk["metadata"]["elevation_bin"], chunk["metadata"]["doppler_bin"], chunk["metadata"]["density_bin"])
        for chunk in all_chunks
    )
    print("(elevation_bin, doppler_bin, density_bin) distribution:")
    for bin_pair, count in sorted(bin_distribution.items()):
        print(f"  {bin_pair}: {count}")
