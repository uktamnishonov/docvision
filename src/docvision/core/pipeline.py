"""Main RAG pipeline orchestration."""

from typing import Dict

from docvision.config import settings
from docvision.ingestion import PDFLoader, TextChunker
from docvision.retrieval import VectorStore, KeywordStore, HybridSearch
from docvision.generation import LLMClient


class LegalGPT:
    """Main RAG pipeline."""

    def __init__(self, use_hybrid: bool = False):
        print("🚀 Initializing LegalGPT...")

        # Initialize components
        self.pdf_loader = PDFLoader()
        self.text_chunker = TextChunker(settings.chunk_size, settings.chunk_overlap)
        self.vector_store = VectorStore(settings.embedding_model)
        self.llm_client = LLMClient(settings.groq_api_key, settings.llm_model)

        self.use_hybrid = use_hybrid
        if use_hybrid:
            self.keyword_store = KeywordStore(
                settings.elasticsearch_host, settings.elasticsearch_index
            )
            self.hybrid_search = None
            print("✓ Hybrid mode enabled")

        self.is_ready = False
        print("✓ LegalGPT initialized")

    def ingest_documents(self, directory: str):
        """Ingest PDF documents."""
        print(f"\n📂 Ingesting from: {directory}")

        # Load PDFs
        documents = self.pdf_loader.load_directory(directory)
        if not documents:
            raise ValueError(f"No PDFs found in {directory}")

        # Chunk documents
        print("\n✂️  Chunking...")
        chunks = self.text_chunker.chunk_documents(documents)

        # Index in vector store
        print("\n🔍 Building vector index...")
        self.vector_store.index_chunks(chunks)

        # Index in keyword store if hybrid
        if self.use_hybrid:
            print("\n📝 Building keyword index...")
            self.keyword_store.create_index()
            self.keyword_store.index_chunks(chunks)
            self.hybrid_search = HybridSearch(self.vector_store, self.keyword_store)

        self.is_ready = True
        print("\n✅ Ingestion complete!")

    def query(self, question: str, top_k: int = 5) -> Dict:
        """Answer a question."""
        if not self.is_ready:
            raise ValueError("Pipeline not ready. Call ingest_documents() first.")

        print(f"\n❓ Question: {question}")

        # Retrieve chunks
        if self.use_hybrid and self.hybrid_search:
            print("🔍 Hybrid search...")
            chunks = self.hybrid_search.search(question, top_k)
            method = "hybrid"
        else:
            print("🔍 Vector search...")
            chunks = self.vector_store.search(question, top_k)
            method = "vector"

        # Generate answer
        print("💭 Generating answer...")
        result = self.llm_client.generate_answer(question, chunks)
        result["search_method"] = method
        result["retrieved_chunks"] = chunks

        return result
