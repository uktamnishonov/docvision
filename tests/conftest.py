import os
import pytest
from pydantic_settings import BaseSettings

# Set dummy environment variables before defining Settings
os.environ["GROQ_API_KEY"] = "dummy_key"
os.environ["EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"
os.environ["LLM_MODEL"] = "llama-3.1-8b-instant"
os.environ["CHUNK_SIZE"] = "1000"
os.environ["CHUNK_OVERLAP"] = "200"
os.environ["ELASTICSEARCH_HOST"] = "localhost:9200"
os.environ["ELASTICSEARCH_INDEX"] = "documents"


class Settings(BaseSettings):
    groq_api_key: str
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "llama-3.1-8b-instant"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    elasticsearch_host: str
    elasticsearch_index: str = "documents"


settings = Settings()

# Remove the fixture since env vars are now set at import time
# @pytest.fixture(autouse=True)
# def set_env():
#     ...
