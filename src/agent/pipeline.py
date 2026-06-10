"""Orchestrates the LangChain steps for request processing."""
import json
import time
from langchain_ollama import ChatOllama
from src.rag.embedder import Embedder, agent_text_for_query
from src.rag.vectorstore import VectorStore
from agent import prompts
from pathlib import Path

TAU = 10.0
_MODS = ("sf7", "sf8", "sf9", "sf10", "sf11", "sf12", "dr8", "dr9")
_THROUGHPUT_ORDER = ["sf7", "sf8", "sf9", "sf10", "sf11", "dr9", "sf12", "dr8"]   # increasing ToA
_AT = {
    "sf7": "AT+MOD=LORA-SF7",   "sf8": "AT+MOD=LORA-SF8",   "sf9": "AT+MOD=LORA-SF9",
    "sf10": "AT+MOD=LORA-SF10", "sf11": "AT+MOD=LORA-SF11", "sf12": "AT+MOD=LORA-SF12",
    "dr8": "AT+MOD=LR-FHSS-DR8", "dr9": "AT+MOD=LR-FHSS-DR9",
}

def to_at_command(mod):
    """'dr8' -> 'AT+MOD=LR-FHSS-DR8'."""
    return _AT.get(mod, "AT+MOD=LR-FHSS-DR8")


def best_under_tau(per, packets=None, tau=TAU):
    """ 
    """
    def per_of(m):
        v = per.get("per_" + m)
        return v if isinstance(v, (int, float)) else None

    cand = [m for m in _MODS if per_of(m) is not None and per_of(m) <= tau]   # PER <= tau
    if cand:
        if packets:
            return max(cand, key=lambda m: packets.get(m, 0))                # max packets
        return min(cand, key=_THROUGHPUT_ORDER.index)                        # backoff
    valid = {m: per_of(m) for m in _MODS if per_of(m) is not None}           # None -> PER min
    return min(valid, key=valid.get) if valid else "dr8"


def _fill(template: str, **values) -> str:
    """Replace the {PLACEHOLDERS} by .replace"""
    for key, value in values.items():
        template = template.replace("{" + key + "}", str(value))
    return template
    
class SatelliteAgentPipeline:
    def __init__(self, model_name="llama3.2:3b", index_path="data/chroma"):
        # Init the stuff
        self.llm = ChatOllama(model=model_name, temperature=0, format="json")
        print(f"[Pipeline] Initialisé avec le modèle : {model_name}")

        self.embedder = Embedder()
        self.store = VectorStore(persist_dir=Path(index_path))

    def _parse_json_secure(self, raw_text: str) -> dict:
        """Isolate and cleanly extract the JSON dict from LLAMA response"""
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("Pas de bloc JSON détecté dans la réponse.")
        return json.loads(raw_text[start:end + 1])

    def execute(self, state: dict, technique1: str = "few_shot",
                technique2: str = "few_shot", technique3: str = "few_shot") -> dict:
        """
        Runs the prompts
        """
        start_time = time.time()
        
        # To get the vars from the situation
        elevation = state.get("elevation_deg", 45)
        
        v_ms = state.get("relative_velocity_ms", 3000)
        # REAL Doppler if available (dataset metadata via build_state), else physical fallback:
        # Δf ≈ 2.893 kHz per km/s at 868 MHz (= fc/c).
        doppler_hz = state.get("doppler_hz")
        if doppler_hz is not None:
            v_khz = round(abs(doppler_hz) / 1000.0, 1)
        else:
            v_khz = round(v_ms / 1000.0 * 2.893, 1)
        rx_pwr = state.get("rssi_dbm", -100)
        nodes = state.get("N", 50000)

        try:
            # PROMPT 1
            if technique1 == "zero_shot":
                p1 = _fill(prompts.geometry_analysis_prompt_zeroshot(), ELEVATION=elevation, DOPPLER=v_khz, RX_POWER=rx_pwr)
            elif technique1 == "cot":
                p1 = _fill(prompts.geometry_analysis_prompt_COT(), ELEVATION=elevation, DOPPLER=v_khz, RX_POWER=rx_pwr)
            else:
                p1 = _fill(prompts.geometry_analysis_prompt_fewshot(), ELEVATION=elevation, DOPPLER=v_khz, RX_POWER=rx_pwr)

            res1_raw = self.llm.invoke(p1).content
            res1 = self._parse_json_secure(res1_raw)
            severity = res1.get("severity", 3)

            # PROMPT 2
          
            state_copie = state.copy()
            state_copie["n_nodes"] = nodes
            state_copie["v_rel_kmps"] = v_ms / 1000.0

            query_text = agent_text_for_query(state_copie)
            query_vector = self.embedder.embed_query(query_text)
            hits = self.store.search(query_vector, k=3)
            context_rag = "\n".join([f"SIM {i+1}: {hit['llm_text']}" for i, hit in enumerate(hits)])

            if technique2 == "zero_shot":
                p2 = _fill(prompts.per_prediction_via_embedding_prompt_zeroshot(), SEVERITY=severity, ELEVATION=elevation, DOPPLER=v_khz, RX_POWER=rx_pwr, RAG_CONTEXT=context_rag)
            elif technique2 == "cot":
                p2 = _fill(prompts.per_prediction_via_embedding_prompt_COT(), SEVERITY=severity, ELEVATION=elevation, DOPPLER=v_khz, RX_POWER=rx_pwr, RAG_CONTEXT=context_rag)
            else:
                p2 = _fill(prompts.per_prediction_via_embedding_prompt_fewshot(), SEVERITY=severity, ELEVATION=elevation, DOPPLER=v_khz, RX_POWER=rx_pwr, RAG_CONTEXT=context_rag)

            res2_raw = self.llm.invoke(p2).content
            predicted_per = self._parse_json_secure(res2_raw)
            predicted_per_json = json.dumps(predicted_per)
            packets_json = json.dumps(state.get("max_packets", {}))

            # STAGE 3 - LLM decision. Goal: max throughput (max_packets) under PER <= TAU.
            # We inject predicted PER AND throughput by modulation; the LLM applies the rule.
            if technique3 == "zero_shot":
                p3 = _fill(prompts.final_decision_prompt_zeroshot(), ELEVATION=elevation, PREDICTED_PER_JSON=predicted_per_json, MAX_PACKETS_JSON=packets_json)
            elif technique3 == "cot":
                p3 = _fill(prompts.final_decision_prompt_COT(), ELEVATION=elevation, PREDICTED_PER_JSON=predicted_per_json, MAX_PACKETS_JSON=packets_json)
            else:
                p3 = _fill(prompts.final_decision_prompt_fewshot(), ELEVATION=elevation, PREDICTED_PER_JSON=predicted_per_json, MAX_PACKETS_JSON=packets_json)

            res3_raw = self.llm.invoke(p3).content
            result = self._parse_json_secure(res3_raw)
            result["severity"] = severity
            result["per"] = {k: v for k, v in predicted_per.items() if str(k).startswith("per_")}

        except Exception as e:
            print(f"[Pipeline] Erreur critique en mode {technique1}/{technique2}/{technique3}: {e}")
            result = {"selected_command": "AT+MOD=LR-FHSS-DR8", "severity": None, "per": {}}
            # if we did not choose in time we take the safest modulation

        # to compute latency and what we use
        latency_ms = (time.time() - start_time) * 1000
        result["latency_ms"] = round(latency_ms, 2)
        result["technique_used"] = f"{technique1}/{technique2}/{technique3}"
        return result

if __name__ == "__main__":
    pipeline = SatelliteAgentPipeline()
    
    # test with extreme values to see if the pipeline holds up (low elevation, high speed, weak signal, many nodes)
    test_state = {
        "elevation_deg": 10.0,
        "relative_velocity_ms": 7500.0,
        "rssi_dbm": -125,
        "N": 50000
    }

    for tech in ["zero_shot", "few_shot", "cot"]:
        print(f"\n==================== MODE {tech.upper()} ====================")
        output = pipeline.execute(test_state, technique=tech)
        print(json.dumps(output, indent=4, ensure_ascii=False))