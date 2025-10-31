"""Configuration management using Pydantic."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""

    # API Keys
    groq_api_key: str = Field(..., env="GROQ_API_KEY")

    # Model Settings
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    llm_model: str = Field(default="llama-3.1-8b-instant", env="LLM_MODEL")

    # Chunking Settings
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")

    # Elasticsearch
    elasticsearch_host: str = Field(
        default="http://localhost:9200", env="ELASTICSEARCH_HOST"
    )
    elasticsearch_index: str = Field(default="documents", env="ELASTICSEARCH_INDEX")

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Allow extra env vars for libraries


# Singleton instance
settings = Settings()
