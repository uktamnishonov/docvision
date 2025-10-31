# DocVision

A **RAG (Retrieval-Augmented Generation)** system for intelligent document Q&A using semantic search and LLMs.

## Features

- **PDF Document Processing** with intelligent chunking
- **Vector Search** using sentence embeddings (FAISS)
- **Hybrid Search** with Elasticsearch (optional)
- **LLM Answer Generation** via Groq API with source citations
- **Streamlit Web UI** with chat interface

## Quick Start

### Prerequisites
- Python 3.10+
- [Groq API key](https://console.groq.com/)

### Installation

```bash
# Clone and setup
git clone https://github.com/yourusername/docvision.git
cd docvision

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Run the App

```bash
streamlit run app.py
```

Open `http://localhost:8501`, upload PDFs, and start asking questions!

## Hybrid Search (Optional)

For better accuracy, enable keyword search with Elasticsearch:

```bash
# Start Elasticsearch
docker-compose up -d elasticsearch

# Enable in UI sidebar: "Use Hybrid Search"
```

## Project Structure

```
docvision/
├── app.py                          # Streamlit UI
├── src/docvision/
│   ├── core/pipeline.py            # RAG pipeline
│   ├── ingestion/                  # PDF loading & chunking
│   ├── retrieval/                  # Vector & keyword search
│   └── generation/                 # LLM client
└── tests/                          # Unit tests
```

## How It Works

1. **Load** PDFs and extract text
2. **Chunk** documents with overlap
3. **Embed** chunks into vectors using sentence-transformers
4. **Index** in FAISS (+ Elasticsearch for hybrid mode)
5. **Retrieve** relevant chunks for user questions
6. **Generate** answers using Groq LLM with citations

## Technologies

- **[FAISS](https://github.com/facebookresearch/faiss)** - Vector search
- **[Sentence-Transformers](https://www.sbert.net/)** - Text embeddings
- **[Groq](https://groq.com/)** - Fast LLM inference
- **[Elasticsearch](https://www.elastic.co/)** - Keyword search (optional)
- **[Streamlit](https://streamlit.io/)** - Web UI

## Troubleshooting

**ModuleNotFoundError**: Install in dev mode: `pip install -e .`

**Groq API Error**: Check your API key in `.env`

**Elasticsearch Error**: Start it: `docker-compose up -d elasticsearch`

## Development

```bash
# Run tests
pytest tests/

# Format code
black src/ tests/
```

## License

MIT License

---

**Built for intelligent document search 🚀**