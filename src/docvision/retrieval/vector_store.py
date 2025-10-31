"""Vector storage and search using FAISS."""

from typing import List, Dict
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os


class VectorStore:
    """Manage vector embeddings and similarity search."""

    def __init__(self, model_name: str):
        print(f"Loading embedding model: {model_name}...")
        # Set number of threads for FAISS to avoid segfault on macOS
        faiss.omp_set_num_threads(1)
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = []
        self.embeddings = None  # Store embeddings for serialization
        print(f"✓ Model loaded (dimension: {self.dimension})")

    def index_chunks(self, chunks: List[Dict]):
        """Create FAISS index from chunks."""
        print(f"Indexing {len(chunks)} chunks...")
        self.chunks = chunks

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        self.embeddings = np.array(embeddings).astype("float32")

        # Use a simpler index type that's more stable on macOS
        self.index = faiss.IndexFlatL2(self.dimension)
        faiss.omp_set_num_threads(1)  # Set again before adding
        self.index.add(self.embeddings)

        print(f"✓ Indexed {self.index.ntotal} vectors")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar chunks."""
        if self.index is None:
            raise ValueError("Index not built. Call index_chunks() first.")

        query_vector = self.model.encode([query])
        query_vector = np.array(query_vector).astype("float32")

        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(distance)
            results.append(chunk)

        return results
