"""Orchestre les étapes LangChain pour le traitement des requêtes."""
import json
import time
from langchain_ollama import ChatOllama
from src.rag.embedder import Embedder, agent_text_for_query
from src.rag.vectorstore import VectorStore
from agent import prompts
from pathlib import Path

class SatelliteAgentPipeline:
    def __init__(self, model_name="llama3.2:1b", index_path="data/chroma"):
        # Initialisation des trucs
        self.llm = ChatOllama(model=model_name, temperature=0)
        print(f"[Pipeline] Initialisé avec le modèle : {model_name}")

        self.embedder = Embedder()
        self.store = VectorStore(persist_dir=Path(index_path))

    def _parse_json_secure(self, raw_text: str) -> dict:
        """Isole et extrait proprement le dict JSON de la réponse de LLAMA"""
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("Pas de bloc JSON détecté dans la réponse.")
        return json.loads(raw_text[start:end + 1])

    def execute(self, state: dict, technique: str = "few_shot") -> dict:
        """
        Exécute les prompts
        """
        start_time = time.time()
        
        # Pour prendre les var de la situation 
        elevation = state.get("elevation_deg", 45)
        
        v_ms = state.get("relative_velocity_ms", 3000)
        v_khz = round(v_ms / 1000.0 * 2.5, 1) # calcul du dopp
        rx_pwr = state.get("rssi_dbm", -100)
        nodes = state.get("N", 50000)

        try:
            # PROMPT 1
            if technique == "zero_shot":
                p1 = prompts.geometry_analysis_prompt_zeroshot().format(ELEVATION=elevation, DOPPLER=v_khz, RX_POWER=rx_pwr)
            elif technique == "cot":
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
            hits = self.store.search_knn(query_vector, k=3) 
            context_rag = "\n".join([f"SIM {i+1}: {hit['llm_text']}" for i, hit in enumerate(hits)])

            if technique == "zero_shot":
                p2 = prompts.per_prediction_via_embedding_prompt_zeroshot().format(SEVERITY=severity, ELEVATION=elevation, DOPPLER=v_khz, RX_POWER=rx_pwr, RAG_CONTEXT=context_rag)
            elif technique == "cot":
                p2 = prompts.per_prediction_via_embedding_prompt_COT().format(SEVERITY=severity, ELEVATION=elevation, DOPPLER=v_khz, RX_POWER=rx_pwr, RAG_CONTEXT=context_rag)
            else:
                p2 = prompts.per_prediction_via_embedding_prompt_fewshot().format(SEVERITY=severity, ELEVATION=elevation, DOPPLER=v_khz, RX_POWER=rx_pwr, RAG_CONTEXT=context_rag)

            res2_raw = self.llm.invoke(p2).content
            
            predicted_per_json = res2_raw[res2_raw.find("{"):res2_raw.rfind("}")+1] 

            #PROMPT 3
            if technique == "zero_shot":
                p3 = prompts.final_decision_prompt_zeroshot().format(ELEVATION=elevation, PREDICTED_PER_JSON=predicted_per_json)
            elif technique == "cot":
                p3 = prompts.final_decision_prompt_COT().format(ELEVATION=elevation, PREDICTED_PER_JSON=predicted_per_json)
            else:
                p3 = prompts.final_decision_prompt_fewshot().format(ELEVATION=elevation, PREDICTED_PER_JSON=predicted_per_json)

            res3_raw = self.llm.invoke(p3).content
            result = self._parse_json_secure(res3_raw)

        except Exception as e:
            print(f"[Pipeline] Erreur critique en mode {technique}: {e}")
            result = {"selected_command": "AT+MOD=LR-FHSS-DR8"}
            # si on a pas choisi à temps on prend la modu la plus safe

        # pour calculer la latence et ce qu'on utilise
        latency_ms = (time.time() - start_time) * 1000
        result["latency_ms"] = round(latency_ms, 2)
        result["technique_used"] = technique
        return result

if __name__ == "__main__":
    pipeline = SatelliteAgentPipeline()
    
    # test avec des valeurs extrêmes
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