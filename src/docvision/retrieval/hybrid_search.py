"""Hybrid search combining vector and keyword retrieval."""

from typing import List, Dict
from .vector_store import VectorStore
from .keyword_store import KeywordStore


class HybridSearch:
    """Combine vector and keyword search with RRF."""

    def __init__(
        self, vector_store: VectorStore, keyword_store: KeywordStore, alpha: float = 0.5
    ):
        self.vector_store = vector_store
        self.keyword_store = keyword_store
        self.alpha = alpha
        print(f"✓ Hybrid search initialized (α={alpha})")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Perform hybrid search."""
        # Get results from both stores
        vector_results = self.vector_store.search(query, top_k * 2)
        keyword_results = self.keyword_store.search(query, top_k * 2)

        # Apply RRF
        scores = self._reciprocal_rank_fusion(vector_results, keyword_results)

        # Get top-k
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_keys = [k for k, _ in sorted_items[:top_k]]

        # Return chunks
        results = []
        all_chunks = {self._chunk_key(c): c for c in vector_results + keyword_results}

        for key in top_keys:
            if key in all_chunks:
                chunk = all_chunks[key].copy()
                chunk["hybrid_score"] = scores[key]
                results.append(chunk)

        return results

    def _reciprocal_rank_fusion(
        self, vec_results: List[Dict], kw_results: List[Dict], k: int = 60
    ) -> Dict[str, float]:
        """RRF algorithm."""
        scores = {}

        for rank, chunk in enumerate(vec_results, 1):
            key = self._chunk_key(chunk)
            scores[key] = scores.get(key, 0) + self.alpha / (k + rank)

        for rank, chunk in enumerate(kw_results, 1):
            key = self._chunk_key(chunk)
            scores[key] = scores.get(key, 0) + (1 - self.alpha) / (k + rank)

        return scores

    def _chunk_key(self, chunk: Dict) -> str:
        """Generate unique key for chunk."""
        return f"{chunk['source']}_{chunk['page']}_{chunk['chunk_id']}"
