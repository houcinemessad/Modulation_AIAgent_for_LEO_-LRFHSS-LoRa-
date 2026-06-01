"""Interface avec ChromaDB pour la gestion des vecteurs."""

from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from .embedder import Embedder
from .chunker import chunk_dataset
import pandas as pd
from typing import Iterator
from pathlib import Path

##############################################################
#
#                    CONFIGURATION
#
##############################################################

DEFAULT_CHROMA_DIR = Path("data/chroma")
DEFAULT_COLLECTION_NAME = "leo_modulation_rag"

# Chroma's HNSW config, cosine is the standard for text embeddings
# (mxbai-embed-large, nomic-embed-text, etc. are all trained with it).
COLLECTION_METADATA = {"hnsw:space": "cosine"}

def _sanitize_metadata(meta: dict) -> dict:
    """
    Chroma metadata can only hold flat primitives (str / int / float / bool).
    If we accidentally put something else in there (ex a None column, or a nested dict), 
    Chroma will raise an error when we try to add the chunk.
    """
    return {
        k: v for k, v in meta.items()
        if v is not None and isinstance(v, (str, int, float, bool))
    }

# ─────────────────────────────────────────────────────────────────────
# VECTOR STORE
# ─────────────────────────────────────────────────────────────────────

class VectorStore:
    """
    Thin wrapper over a persistent ChromaDB collection.

    Responsibilities:
      - Open / create the on-disk collection (cosine space)
      - Store chunks (embed once, then collection.add)
      - Retrieve top-k chunks by cosine similarity + optional metadata filter
    """

    def __init__(self,persist_dir: Path = DEFAULT_CHROMA_DIR,collection_name: str = DEFAULT_COLLECTION_NAME):
        """
         Initialize the ChromaDB client and collection.
         Uses fuctions from chromadb, source : ChromaDB documentation : https://docs.trychroma.com/docs/overview/getting-started
         """     
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.persist_dir.mkdir(parents=True, exist_ok=True) # Ensure the directory exists

        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata=COLLECTION_METADATA,
        )

    def index_chunks(self, chunks: Iterable[dict], embedder: Embedder, force_rebuild: bool = False) -> None:
        """
        Embed the chunks and push them into Chroma.
        Skips the work if the collection is already populated (so re-running
        the agent doesn't re-embed for nothing, and ``add`` never trips over
        duplicate IDs). Pass ``force_rebuild=True`` to wipe and rebuild.
        """
        chunks = list(chunks)

        # ── 1. Skip / wipe decision ─────────────────────────────────────
        if force_rebuild:
            self._wipe_collection()
        elif self._collection.count() > 0:
            print(
                f"VectorStore: collection already has {self._collection.count()} "
                f"chunks, skipping re-index (pass force_rebuild=True to refresh)."
            )
            return

        # ── 2. Format + embed ───────────────────────────────────────────
        agent_texts, llm_texts = embedder.prepare_chunks_for_indexing(chunks)
        print(f"VectorStore: embedding {len(agent_texts)} chunks via Ollama ...")
        vectors = embedder.embed_documents(agent_texts)

        # ── 3. Push to Chroma ───────────────────────────────────────────
        self._collection.add(
            ids        = [c["id"] for c in chunks],
            embeddings = vectors,
            documents  = llm_texts,
            metadatas  = [_sanitize_metadata(c["metadata"]) for c in chunks],
        )

        print(f"VectorStore: indexed {self._collection.count()} chunks at {self.persist_dir}/")

    # ── Read (built for RAG queries) ──────────────────────────

    def search_knn(self, query_vector: list[float],  k: int = 5, where: Optional[dict] = None,) -> list[dict]:
        """
        Return the top-k chunks closest to ``query_vector`` in cosine space.
        """
        results = self._collection.query(
            query_embeddings = [query_vector],
            n_results = k,
            where = where,
        )

        # Chroma returns nested lists (one per query vector). We sent a single
        # vector, so we always read index [0].
        ids       = results["ids"][0]
        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0] if results.get("distances") else [None] * len(ids)

        return [
            {"id": ids[i], "llm_text": docs[i], "metadata": metas[i], "distance": distances[i]}
            for i in range(len(ids))
        ]

    def count(self) -> int:
        """Number of chunks currently stored. 0 means the index is empty."""
        return self._collection.count()

    # ── Internal plumbing ──────────────────────────────────────────────

    def _wipe_collection(self) -> None:
        """
        Drop and recreate the collection. ChromaDB has no cleanr way to delete all vectors than deleting the whole collection, so we do that. 
        This is only needed if you want to re-index from scratch, e.g. after changing the embedding logic
        for vectors, so we delete the named collection and recreate it.
        """
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass  # Collection may not exist yet on the very first run.
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata=COLLECTION_METADATA,
        )


if __name__ == "__main__":
    # Quick test: create the store, index the dataset chunks, and print the count.
    store = VectorStore()
    dataset_path = "data/dataset.csv" 
    dataset = pd.read_csv(dataset_path)
    chunks = list(chunk_dataset(dataset))
    embedder = Embedder()
    store.index_chunks(chunks, embedder, force_rebuild=True)
    print(f"Total chunks in store: {store.count()}")