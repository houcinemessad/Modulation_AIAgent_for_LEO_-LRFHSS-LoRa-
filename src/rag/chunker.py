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
    (60.0, 90.0, "high"),
]

DOPPLER_BINS = [
    (0.0,  2.0,  "low"),
    (2.0,  7.8,  "mid"),
    (7.8,  16.0, "extreme"),
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
REQUIRED_COLUMNS = (
    "elevation_deg", "v_rel_kmps", "n_nodes",
    "modulation", "PER_pct",
    "distance_km", "kappa", "doppler_hz", "visibility_window_s",
    "RSSI_dBm", "SNR_dB",
)


def validate_required_columns(dataset: pd.DataFrame) -> None:
    """
    Fail fast (and loudly) if the CSV is missing fields the chunker needs.

    Each CSV row is one (channel state × modulation) measurement. If even
    one of the required columns is missing, every chunk would be
    malformed downstream — we prefer to crash here with a clear message
    than to let a broken vector store be built silently.
    """
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