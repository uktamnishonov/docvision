import pytest
from docvision.ingestion import TextChunker


def test_text_chunker():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    text = "Test sentence. " * 50
    chunks = chunker.chunk_text(text, "test.pdf")

    assert len(chunks) > 0
    assert all("text" in c for c in chunks)
    assert all("source" in c for c in chunks)
