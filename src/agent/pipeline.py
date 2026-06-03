"""Orchestre les étapes LangChain pour le traitement des requêtes."""
import json
from langchain_ollama import ChatOllama
from src.rag.embedder import Embedder, agent_text_for_query
from src.rag.vectorstore import VectorStore
from src.rag.chunker import chunk_dataset
from src.agent import prompts

embedder = Embedder()
store    = VectorStore()
llm      = ChatOllama(model="llama3.2:3b", temperature=0)

#INIT
store.index_chunks(list(chunk_dataset("data/dataset_real.csv")), embedder)

def ask_llm(prompt_text):
    """Envoie le prompt à Ollama et renvoie le JSON parsé."""
    raw = llm.invoke(prompt_text).content
    return json.loads(raw[raw.find("{"): raw.rfind("}") + 1])

def decide(state): #state sont les métadonnés de la requête, a lier sur le dashboard.
    """Réponse du llm"""
    # 1. Geometry  analisis, severity estimation 
    sev = ask_llm(prompts.geometry_analysis_prompt_zeroshot())

    # 2. RAG for PER prediction
    hits = store.search(embedder.embed_query(agent_text_for_query(state)), k=3)
    per  = ask_llm(prompts.per_prediction_via_embedding_prompt_fewshot())

    # 3. final decision
    return ask_llm(prompts.final_decision_prompt_zeroshot())

if __name__ == "__main__":
    # Exemple de requête
    state = {
        chunk_dataset("data/dataset_real.csv").__next__()["metadata"]  # Prend les métadonnées du premier chunk pour tester
    }
    decision = decide(state)
    print("Final Decision:", decision)