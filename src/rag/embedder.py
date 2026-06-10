from __future__ import annotations
from chunker import chunk_dataset, ELEVATION_BINS, DOPPLER_BINS, DENSITY_BINS, classify_into_bin
from langchain_ollama import OllamaEmbeddings
import math
from typing import Iterable
import pandas as pd

"""
Provides hybrid embedding methods for texts.
Architecture idea is based on :
- Muennighoff et al. (2022) « MTEB: Massive Text Embedding Benchmark
- Nomic AI (2024) « Nomic Embed: Training a Reproducible Long Context Text Embedder 
- Microsoft Azure AI Search — RAG best practices (2024)
- LangChain, ollama documentation and other » 

"""

embeddings = OllamaEmbeddings(model="nomic-embed-text")

"""

We want our chunks to be embedded in a way that captures both their content and their context. 
To do this, we can use a hybrid embedding approach that combines the scenario of the chunk with
metadata about its position in the original document.
The embedder algorithm should NOT mix the metadata of SF and DR when textually describing the chunk,
but should instead encode the metadata as separate dimensions in the embedding space.
We should do this by making two embeddings for each chunk: one for the Agent data and one for the LLM context.

"""
C_MPS = 3e8  # Speed of light in m/s, for Doppler calculations
def _doppler_ppm(v_rel_kmps: float) -> float:
    """Convert a relative speed (km/s) into Doppler shift in ppm at 868 MHz."""
    return (v_rel_kmps * 1000.0 / C_MPS) * 1e6


def agent_text_for_indexing(chunk: dict) -> str:
    """
    Build the string that will be FED TO THE EMBEDDER for one chunk.

    A chunk is a CONTAINER of 8 modulation rows measured at one channel
    state. Most physical fields (RSSI, distance, kappa, Doppler, geometry)
    are identical across the 8 rows averaging them is safe. The single
    exception is SNR, which depends on the demodulator bandwidth (125 kHz
    for LoRa, 488 Hz for LR-FHSS) and therefore differs between the two
    families. We split SNR explicitly to keep the embedding faithful to
    physics.

    The string is short (~50 tokens), hybrid natural-language + values.
    Its style MUST match ``agent_text_from_state`` exactly (symmetry rule).
    """
    m = chunk["metadata"]

    return (
        f"LEO uplink at elevation {m['elevation_deg']}° ({m['elevation_bin']}), "
        f"relative speed {m['v_rel_kmps']} km/s ({m['doppler_bin']} Doppler), "
        f"network of {m['n_nodes']} nodes ({m['density_bin']}). "
        f"Link: distance {m['distance_km']:.0f} km, kappa {m['kappa']:.1f}, "
        f"RSSI {m['rssi_mean_dbm']:.0f} dBm, "
        f"SNR LoRa {m['snr_lora_mean_db']:.0f} dB / LR-FHSS {m['snr_lrfhss_mean_db']:.0f} dB."
    )


def llm_text_for_context(chunk: dict) -> str:
    """
    Build the string that will be INJECTED INTO THE LLM PROMPT for one chunk
    once it has been retrieved.
    Longer (~100 tokens) and more verbose than ``agent_text_from_chunk``,
    because here we are talking to a generative model that will benefit
    from a natural-language description with physical interpretation.
    """
    m = chunk["metadata"]
    doppler_ppm = _doppler_ppm(m["v_rel_kmps"])
    kappa_quality = (
        "strong line-of-sight"  if m["kappa"] >= 10.0
        else "mixed multipath"  if m["kappa"] >= 3.0
        else "diffuse multipath"
    )

    return (
        f"LEO satellite uplink at elevation {m['elevation_deg']}° "
        f"({m['elevation_bin']} regime above horizon). "
        f"Satellite-terminal relative speed {m['v_rel_kmps']} km/s, "
        f"producing a Doppler shift of approximately {doppler_ppm:.1f} ppm "
        f"at 868 MHz ({m['doppler_bin']} regime). "
        f"The Doppler rate of change is {m['doppler_rate_hz']:.0f} Hz/s, which drifts "
        f"the carrier mid-packet and most degrades long high-spreading-factor "
        f"transmissions; this rate peaks near culmination (high elevation), exactly "
        f"where the shift itself is smallest. "
        f"Shared channel contended by {m['n_nodes']} nodes ({m['density_bin']}). "
        f"Slant range {m['distance_km']:.0f} km. "
        f"Rician K-factor {m['kappa']:.2f} indicates {kappa_quality}. "
        f"Mean RSSI {m['rssi_mean_dbm']:.1f} dBm at the gateway. "
        f"Demodulator-equivalent SNR is {m['snr_lora_mean_db']:.1f} dB for LoRa CSS "
        f"(125 kHz noise bandwidth) and {m['snr_lrfhss_mean_db']:.1f} dB for LR-FHSS "
        f"(488 Hz hop bandwidth) — the ~24 dB gap is the kTB difference "
        f"between the two receiver chains, not a real link advantage."
    )

def agent_text_for_query(state: dict) -> str:
    """
    Build the string that will be FED TO THE EMBEDDER for a single query.
    The agent calls this when it needs to find similar past scenarios.
    """
    el_bin  = state.get("elevation_bin") or classify_into_bin(state["elevation_deg"], ELEVATION_BINS)
    dop_bin = state.get("doppler_bin")   or classify_into_bin(state["v_rel_kmps"],    DOPPLER_BINS)
    den_bin = state.get("density_bin")   or classify_into_bin(state["n_nodes"],       DENSITY_BINS)

    return (
        f"LEO uplink at elevation {state['elevation_deg']}° ({el_bin}), "
        f"relative speed {state['v_rel_kmps']} km/s ({dop_bin} Doppler), "
        f"network of {state['n_nodes']} nodes ({den_bin}). "
        f"Link: distance {state['distance_km']:.0f} km, kappa {state['kappa']:.1f}, "
        f"RSSI {state['rssi_mean_dbm']:.0f} dBm, "
        f"SNR LoRa {state['snr_lora_db']:.0f} dB / LR-FHSS {state['snr_lrfhss_db']:.0f} dB."
    )

class Embedder:
    """
    A simple wrapper around the embedding model that provides a clean API for
    our specific use case. It also handles the hybrid embedding logic, ensuring
    that the metadata is encoded in a way that preserves its physical meaning.
    """

    def prepare_chunks_for_indexing(self, chunks):
        agent_texts = [agent_text_for_indexing(c) for c in chunks]
        llm_texts   = [llm_text_for_context(c)    for c in chunks]
        return agent_texts, llm_texts

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            vectors.extend(self._embeddings.embed_documents(texts[i:i + batch_size]))
            print(f"  embed {min(i + batch_size, len(texts))}/{len(texts)}")
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)

# Main/test code: run this file to see how the dataset gets chunked, and to check the distribution of channel states across the chunks.
if __name__ == "__main__":
    import sys
    try:
        from .chunker import chunk_dataset
    except ImportError:
        from chunker import chunk_dataset

    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "data/dataset.csv"
    first_chunk = next(chunk_dataset(dataset_path))

    agent_text = agent_text_for_indexing(first_chunk)
    llm_text   = llm_text_for_context(first_chunk)

    print("---- agent_text (what the embedder reads) ----")
    print(agent_text)
    print(f"  -> {len(agent_text.split())} words / {len(agent_text)} chars")
    print()
    print("---- llm_text (what the LLM reads after retrieval) ----")
    print(llm_text)
    print(f"  -> {len(llm_text.split())} words / {len(llm_text)} chars")
    print()

    # Build the runtime state from the chunk we just formatted, so we can
    # check that agent_text_for_query reproduces agent_text_for_indexing
    # exactly. The two SNR values are read straight from the chunker's
    # already-split metadata fields.
    m = first_chunk["metadata"]
    sample_state = {
        "elevation_deg":   m["elevation_deg"],
        "v_rel_kmps":      m["v_rel_kmps"],
        "n_nodes":         m["n_nodes"],
        "distance_km":     m["distance_km"],
        "kappa":           m["kappa"],
        "rssi_mean_dbm":   m["rssi_mean_dbm"],
        "snr_lora_db":     m["snr_lora_mean_db"],
        "snr_lrfhss_db":   m["snr_lrfhss_mean_db"],
    }
    query_text = agent_text_for_query(sample_state)
    print("---- agent_text_for_query on the same state ----")
    print(query_text)
    print()
    print("Symmetry check (state-derived text vs chunk-derived text):")
    print(f"  identical = {query_text == agent_text}")