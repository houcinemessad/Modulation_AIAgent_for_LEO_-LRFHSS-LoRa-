"""LangChain pipeline: geometry -> severity, RAG -> PER prediction, then decision."""
import json
import time
from pathlib import Path

from langchain_ollama import ChatOllama
from src.rag.embedder import Embedder, agent_text_for_query
from src.rag.vectorstore import VectorStore
from . import prompts

TAU = 10.0                                                                  # max PER (%) allowed
_MODS = ("sf7", "sf8", "sf9", "sf10", "sf11", "sf12", "dr8", "dr9")
_DENSITIES = (10000, 50000, 100000)                                         # for density-aware RAG context
_THROUGHPUT_ORDER = ["sf7", "sf8", "sf9", "sf10", "sf11", "dr9", "sf12", "dr8"]   # increasing airtime
_AT = {
    "sf7": "AT+MOD=LORA-SF7",   "sf8": "AT+MOD=LORA-SF8",   "sf9": "AT+MOD=LORA-SF9",
    "sf10": "AT+MOD=LORA-SF10", "sf11": "AT+MOD=LORA-SF11", "sf12": "AT+MOD=LORA-SF12",
    "dr8": "AT+MOD=LR-FHSS-DR8", "dr9": "AT+MOD=LR-FHSS-DR9",
}


def to_at_command(mod):
    """'dr8' -> 'AT+MOD=LR-FHSS-DR8'."""
    return _AT.get(mod, "AT+MOD=LR-FHSS-DR8")


def best_under_tau(per, packets=None, tau=TAU, elevation_deg=None):
    """Pick the highest-throughput modulation whose PER <= tau.

    Safety rule first: elevation < 15 deg -> DR8 (before the tau filter). If no
    modulation meets tau, fall back to the lowest-PER one.
    """
    if elevation_deg is not None and elevation_deg < 15:
        return "dr8"

    def per_of(m):
        v = per.get("per_" + m)
        return v if isinstance(v, (int, float)) else None

    cand = [m for m in _MODS if per_of(m) is not None and per_of(m) <= tau]
    if cand:
        if packets:
            return max(cand, key=lambda m: packets.get(m, 0))     # most packets
        return min(cand, key=_THROUGHPUT_ORDER.index)             # fallback: shortest airtime
    valid = {m: per_of(m) for m in _MODS if per_of(m) is not None}
    return min(valid, key=valid.get) if valid else "dr8"          # nothing under tau -> min PER


def _fill(template: str, **values) -> str:
    """Fill {PLACEHOLDERS} via str.replace, and strip Mistral [INST] tags / trailing '{'."""
    template = template.replace("[INST]", "").replace("[/INST]", "").rstrip()
    if template.endswith("{"):
        template = template[:-1].rstrip()
    for key, value in values.items():
        template = template.replace("{" + key + "}", str(value))
    return template


class SatelliteAgentPipeline:
    def __init__(self, model_name="llama3.2:3b", index_path="data/chroma"):
        self.llm = ChatOllama(model=model_name, temperature=0, format="json")
        print(f"[Pipeline] Initialized with model: {model_name}")
        self.embedder = Embedder()
        self.store = VectorStore(persist_dir=Path(index_path))

    def _parse_json_secure(self, raw_text: str) -> dict:
        """Extract the JSON object from the LLM response."""
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON block found in the response.")
        return json.loads(raw_text[start:end + 1])

    def execute(self, state: dict, technique1: str = "cot",
                technique2: str = "cot", technique3: str = "cot") -> dict:
        """Run the 3 stages and return {selected_command, severity, per, latency_ms, technique_used}.
        technique1/2/3 in {"rule", "zero_shot", "few_shot", "cot"} ('rule' = deterministic,
        available for stages 1 and 3)."""
        start_time = time.time()

        elevation = state.get("elevation_deg", 45)
        v_ms = state.get("relative_velocity_ms", 3000)
        # Real Doppler if available (dataset metadata), else physical fallback:
        # df ~ 2.893 kHz per km/s at 868 MHz (= fc/c).
        doppler_hz = state.get("doppler_hz")
        if doppler_hz is not None:
            v_khz = round(abs(doppler_hz) / 1000.0, 1)
        else:
            v_khz = round(v_ms / 1000.0 * 2.893, 1)
        rx_pwr = state.get("rssi_dbm", -100)
        nodes = state.get("N", 50000)

        try:
            # STAGE 1 — geometry -> severity (1-5).
            if technique1 == "zero_shot":
                p1 = _fill(prompts.geometry_analysis_prompt_zeroshot(), ELEVATION=elevation, DOPPLER=v_khz, RX_POWER=rx_pwr)
            elif technique1 == "cot":
                p1 = _fill(prompts.geometry_analysis_prompt_COT(), ELEVATION=elevation, DOPPLER=v_khz, RX_POWER=rx_pwr)
            else:
                p1 = _fill(prompts.geometry_analysis_prompt_fewshot(), ELEVATION=elevation, DOPPLER=v_khz, RX_POWER=rx_pwr)

            res1 = self._parse_json_secure(self.llm.invoke(p1).content)
            severity = res1.get("severity")
            if severity not in (1, 2, 3, 4, 5):
                severity = 5

            # STAGE 2 — RAG retrieval -> PER prediction.
            state_copy = state.copy()
            state_copy["n_nodes"] = nodes
            state_copy["v_rel_kmps"] = v_ms / 1000.0

            query_vector = self.embedder.embed_query(agent_text_for_query(state_copy))
            # Density-aware context: k nearest, then the nearest at EACH density (SF PER depends
            # heavily on contention) so the LLM sees the density->PER relation. Each SIM is
            # tagged (EL, v, N) and carries the neighbour's real PER for interpolation.
            hits = self.store.search(query_vector, k=3)
            seen = {h["id"] for h in hits}
            for d in _DENSITIES:
                for h in self.store.search(query_vector, k=1, where={"n_nodes": d}):
                    if h["id"] not in seen:
                        seen.add(h["id"]); hits.append(h)
            context_rag = "\n".join(
                f"SIM {i+1} (EL={h['metadata'].get('elevation_deg')}deg, "
                f"v={h['metadata'].get('v_rel_kmps')}km/s, N={h['metadata'].get('n_nodes')}): "
                f"{h['llm_text']} | " + ", ".join(
                    f"{m.upper()}_PER:{round(h['metadata']['per_' + m.upper()])}%"
                    for m in _MODS if h['metadata'].get('per_' + m.upper()) is not None
                )
                for i, h in enumerate(hits)
            )

            if technique2 == "zero_shot":
                p2 = _fill(prompts.per_prediction_via_embedding_prompt_zeroshot(), SEVERITY=severity, ELEVATION=elevation, DOPPLER=v_khz, RX_POWER=rx_pwr, RAG_CONTEXT=context_rag)
            elif technique2 == "cot":
                p2 = _fill(prompts.per_prediction_via_embedding_prompt_COT(), SEVERITY=severity, ELEVATION=elevation, DOPPLER=v_khz, RX_POWER=rx_pwr, RAG_CONTEXT=context_rag)
            else:
                p2 = _fill(prompts.per_prediction_via_embedding_prompt_fewshot(), SEVERITY=severity, ELEVATION=elevation, DOPPLER=v_khz, RX_POWER=rx_pwr, RAG_CONTEXT=context_rag)

            predicted_per = self._parse_json_secure(self.llm.invoke(p2).content)
            predicted_per_json = json.dumps(predicted_per)
            packets_json = json.dumps(state.get("max_packets", {}))

            # STAGE 3 — decision -> AT command.
            # 'rule' = deterministic: best_under_tau on the PREDICTED PER (+ elev<15 -> DR8).
            # zero_shot/few_shot/cot = LLM decision (compared in the ablation study).
            if technique3 == "rule":
                mod = best_under_tau(predicted_per, state.get("max_packets", {}), TAU, elevation)
                result = {"selected_command": to_at_command(mod)}
            else:
                if technique3 == "zero_shot":
                    p3 = _fill(prompts.final_decision_prompt_zeroshot(), ELEVATION=elevation, PREDICTED_PER_JSON=predicted_per_json, MAX_PACKETS_JSON=packets_json)
                elif technique3 == "cot":
                    p3 = _fill(prompts.final_decision_prompt_COT(), ELEVATION=elevation, PREDICTED_PER_JSON=predicted_per_json, MAX_PACKETS_JSON=packets_json)
                else:
                    p3 = _fill(prompts.final_decision_prompt_fewshot(), ELEVATION=elevation, PREDICTED_PER_JSON=predicted_per_json, MAX_PACKETS_JSON=packets_json)
                result = self._parse_json_secure(self.llm.invoke(p3).content)
            result["severity"] = severity
            result["per"] = {k: v for k, v in predicted_per.items() if str(k).startswith("per_")}

        except Exception as e:
            print(f"[Pipeline] Fatal error in mode {technique1}/{technique2}/{technique3}: {e}")
            result = {"selected_command": "AT+MOD=LR-FHSS-DR8", "severity": None, "per": {}}
            # On failure, default to the safest modulation.

        result["latency_ms"] = round((time.time() - start_time) * 1000, 2)
        result["technique_used"] = f"{technique1}/{technique2}/{technique3}"
        return result


if __name__ == "__main__":
    pipeline = SatelliteAgentPipeline()

    # Extreme values: low elevation, high speed, weak signal, many nodes.
    test_state = {
        "elevation_deg": 10.0,
        "relative_velocity_ms": 7500.0,
        "rssi_dbm": -125,
        "N": 50000,
    }

    for tech in ["zero_shot", "few_shot", "cot"]:
        print(f"\n==================== MODE {tech.upper()} ====================")
        output = pipeline.execute(test_state, technique1=tech, technique2=tech, technique3=tech)
        print(json.dumps(output, indent=4, ensure_ascii=False))
