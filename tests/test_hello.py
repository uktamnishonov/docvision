from docvision.ingestion import TextChunker
from tests.conftest import settings  # Import settings from conftest


def test_text_chunker():
    chunker = TextChunker(settings.chunk_size, settings.chunk_overlap)
    text = "This is a test text for chunking."
    source = "test_document.pdf"
    chunks = chunker.chunk_text(text, source)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(chunk, dict) for chunk in chunks)
