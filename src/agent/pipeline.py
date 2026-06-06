"""Orchestrates the LangChain steps for request processing."""
import json
import time
from langchain_ollama import ChatOllama
from src.rag.embedder import Embedder, agent_text_for_query
from src.rag.vectorstore import VectorStore
from agent import prompts
from pathlib import Path

class SatelliteAgentPipeline:
    def __init__(self, model_name="llama3.2:3b", index_path="data/chroma"):
        # Initialization
        self.llm = ChatOllama(model=model_name, temperature=0, format="json")
        print(f"[Pipeline] Initialisé avec le modèle : {model_name}")

        self.embedder = Embedder()
        self.store = VectorStore(persist_dir=Path(index_path))

    def _parse_json_secure(self, raw_text: str) -> dict:
        """Safely isolate and extract the JSON dict from LLAMA's response."""
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON block detected in the response.")
        return json.loads(raw_text[start:end + 1])

    def execute(self, state: dict, technique1: str = "few_shot",
                technique2: str = "few_shot", technique3: str = "few_shot") -> dict:
        """
        Execute the prompts
        """
        start_time = time.time()
        
        # To extract situation variables
        elevation = state.get("elevation_deg", 45)
        
        v_ms = state.get("relative_velocity_ms", 3000)
        # Real Doppler if available (from dataset metadata via build_state), otherwise fallback physics:
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
                p1 = prompts.geometry_analysis_prompt_zeroshot().format(ELEVATION=elevation, DOPPLER=v_khz, RX_POWER=rx_pwr)
            elif technique1 == "cot":
                p1 = prompts.geometry_analysis_prompt_COT().format(ELEVATION=elevation, DOPPLER=v_khz, RX_POWER=rx_pwr)
            else:
                p1 = prompts.geometry_analysis_prompt_fewshot().format(ELEVATION=elevation, DOPPLER=v_khz, RX_POWER=rx_pwr)

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
            # We REINJECT each neighbor's PER (present in metadata, excluded from llm_text
            # to avoid polluting the embedding): without this the LLM predicts PER BLINDLY.
            # Expected format for the few-shot prompt: "SF7_PER:78%, ..., DR8_PER:2%".
            _MODS = ("SF7", "SF8", "SF9", "SF10", "SF11", "SF12", "DR8", "DR9")
            context_rag = "\n".join(
                f"SIM {i+1}: {hit['llm_text']} | " + ", ".join(
                    f"{m}_PER:{round(hit['metadata']['per_' + m])}%"
                    for m in _MODS if hit['metadata'].get('per_' + m) is not None
                )
                for i, hit in enumerate(hits)
            )

            if technique2 == "zero_shot":
                p2 = prompts.per_prediction_via_embedding_prompt_zeroshot().format(SEVERITY=severity, ELEVATION=elevation, DOPPLER=v_khz, RX_POWER=rx_pwr, RAG_CONTEXT=context_rag)
            elif technique2 == "cot":
                p2 = prompts.per_prediction_via_embedding_prompt_COT().format(SEVERITY=severity, ELEVATION=elevation, DOPPLER=v_khz, RX_POWER=rx_pwr, RAG_CONTEXT=context_rag)
            else:
                p2 = prompts.per_prediction_via_embedding_prompt_fewshot().format(SEVERITY=severity, ELEVATION=elevation, DOPPLER=v_khz, RX_POWER=rx_pwr, RAG_CONTEXT=context_rag)

            res2_raw = self.llm.invoke(p2).content
            predicted_per = self._parse_json_secure(res2_raw)
            predicted_per_json = json.dumps(predicted_per)

            # PROMPT 3
            if technique3 == "zero_shot":
                p3 = prompts.final_decision_prompt_zeroshot().format(ELEVATION=elevation, PREDICTED_PER_JSON=predicted_per_json)
            elif technique3 == "cot":
                p3 = prompts.final_decision_prompt_COT().format(ELEVATION=elevation, PREDICTED_PER_JSON=predicted_per_json)
            else:
                p3 = prompts.final_decision_prompt_fewshot().format(ELEVATION=elevation, PREDICTED_PER_JSON=predicted_per_json)

            res3_raw = self.llm.invoke(p3).content
            result = self._parse_json_secure(res3_raw)
            result["severity"] = severity
            result["per"] = {k: v for k, v in predicted_per.items() if str(k).startswith("per_")}

        except Exception as e:
            print(f"[Pipeline] Erreur critique en mode {technique1}/{technique2}/{technique3}: {e}")
            result = {"selected_command": "AT+MOD=LR-FHSS-DR8", "severity": None, "per": {}}
            # if no choice was made in time, pick the safest modulation

        # to calculate latency and record which techniques were used
        latency_ms = (time.time() - start_time) * 1000
        result["latency_ms"] = round(latency_ms, 2)
        result["technique_used"] = f"{technique1}/{technique2}/{technique3}"
        return result

if __name__ == "__main__":
    pipeline = SatelliteAgentPipeline()
    
    # test with extreme values
    test_state = {
        "elevation_deg": 10.0,
        "relative_velocity_ms": 7500.0,
        "rssi_dbm": -125,
        "N": 50000
    }

    for tech in ["zero_shot", "few_shot", "cot"]:
        print(f"\n==================== MODE {tech.upper()} ====================")
        output = pipeline.execute(test_state, technique1=tech, technique2=tech, technique3=tech)
        print(json.dumps(output, indent=4, ensure_ascii=False))