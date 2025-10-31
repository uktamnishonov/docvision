"""Text chunking utilities."""

from typing import List, Dict
import re


class TextChunker:
    """Split documents into chunks."""

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, source: str) -> List[Dict]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk_text.rfind(".")
                if last_period > self.chunk_size // 2:
                    end = start + last_period + 1
                    chunk_text = text[start:end]

            page_num = self._extract_page_number(chunk_text)

            chunks.append(
                {
                    "text": chunk_text.strip(),
                    "source": source,
                    "page": page_num,
                    "chunk_id": len(chunks),
                }
            )

            start = end - self.chunk_overlap

        return chunks

    def _extract_page_number(self, text: str) -> int:
        """Extract page number from text marker."""
        match = re.search(r"--- Page (\d+) ---", text)
        return int(match.group(1)) if match else 0

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """Chunk multiple documents."""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_text(doc["content"], doc["source"])
            all_chunks.extend(chunks)
            print(f"✓ Chunked {doc['source']}: {len(chunks)} chunks")
        return all_chunks
