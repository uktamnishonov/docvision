"""LLM client for answer generation."""

from typing import List, Dict
from groq import Groq


class LLMClient:
    """Handle LLM interactions."""

    def __init__(self, api_key: str, model: str):
        self.client = Groq(api_key=api_key)
        self.model = model
        print(f"✓ LLM client initialized: {model}")

    def generate_answer(self, query: str, context_chunks: List[Dict]) -> Dict[str, any]:
        """Generate answer from context."""
        context = self._build_context(context_chunks)
        prompt = self._create_prompt(query, context)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a legal assistant. Answer questions based on provided context. Always cite sources.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )

        answer = response.choices[0].message.content
        sources = self._extract_sources(context_chunks)

        return {"answer": answer, "sources": sources}

    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context from chunks."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[{i}] Source: {chunk['source']}, Page: {chunk['page']}\n"
                f"{chunk['text']}\n"
            )
        return "\n".join(parts)

    def _create_prompt(self, query: str, context: str) -> str:
        """Create prompt."""
        return f"""Context:
{context}

Question: {query}

Answer based on the context above. Cite sources with document name and page number."""

    def _extract_sources(self, chunks: List[Dict]) -> List[Dict]:
        """Extract unique sources."""
        sources = {}
        for chunk in chunks:
            key = f"{chunk['source']}_{chunk['page']}"
            if key not in sources:
                sources[key] = {"document": chunk["source"], "page": chunk["page"]}
        return list(sources.values())
