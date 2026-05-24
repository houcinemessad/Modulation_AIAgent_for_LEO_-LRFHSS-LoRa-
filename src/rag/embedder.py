"""
Provides hybrid embedding methods for texts.
Architecture idea is based on :
- Muennighoff et al. (2022) « MTEB: Massive Text Embedding Benchmark
- Nomic AI (2024) « Nomic Embed: Training a Reproducible Long Context Text Embedder 
- Microsoft Azure AI Search — RAG best practices (2024)
- LangChain, ollama documentation and other » 

"""
from chunker import chunk_dataset, ELEVATION_BINS, DOPPLER_BINS, DENSITY_BINS
from langchain_ollama import OllamaEmbeddings
from __future__ import annotations
import math
from typing import Iterable

embeddings = OllamaEmbeddings(model="nomic-embed-text")

"""

We want our chunks to be embedded in a way that captures both their content and their context. 
To do this, we can use a hybrid embedding approach that combines the scenario of the chunk with
metadata about its position in the original document.
The embedder algorithm should NOT mix the metadata of SF and DR when textually describing the chunk,
but should instead encode the metadata as separate dimensions in the embedding space.
We should do this by making two embeddings for each chunk: one for the Agent data and one for the LLM context.

"""
def embed_chunk(chunk: dict) -> list[float]: