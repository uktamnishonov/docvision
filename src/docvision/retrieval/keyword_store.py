"""Keyword-based search using Elasticsearch."""

from typing import List, Dict
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


class KeywordStore:
    """Manage keyword search with Elasticsearch."""

    def __init__(self, host: str, index_name: str):
        self.client = Elasticsearch(hosts=[host])
        self.index_name = index_name
        print(f"✓ Connected to Elasticsearch: {host}")

    def create_index(self):
        """Create Elasticsearch index."""
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)

        mapping = {
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "source": {"type": "keyword"},
                    "page": {"type": "integer"},
                    "chunk_id": {"type": "integer"},
                }
            }
        }

        self.client.indices.create(index=self.index_name, mappings=mapping["mappings"])
        print(f"✓ Created index: {self.index_name}")

    def index_chunks(self, chunks: List[Dict]):
        """Index chunks in Elasticsearch."""
        actions = [
            {"_index": self.index_name, "_id": i, "_source": chunk}
            for i, chunk in enumerate(chunks)
        ]

        success, _ = bulk(self.client, actions)
        self.client.indices.refresh(index=self.index_name)
        print(f"✓ Indexed {success} documents in Elasticsearch")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Keyword search."""
        query_body = {"match": {"text": query}}

        response = self.client.search(
            index=self.index_name, query=query_body, size=top_k
        )

        results = []
        for hit in response["hits"]["hits"]:
            chunk = hit["_source"]
            chunk["score"] = hit["_score"]
            results.append(chunk)

        return results
