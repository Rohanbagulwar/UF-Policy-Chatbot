import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    model_name: str = "BAAI/bge-base-en-v1.5"
    dimension: Optional[int] = None


@dataclass
class ChromaDBConfig:
    """Configuration for ChromaDB."""
    persist_directory: str = "./chroma_db"
    collection_name: str = "policies"
    description: str = "UF Policy documents with embeddings"

# models=['llama-3.1-70b-instruct','gemma-3-27b-it']
@dataclass
class OpenAIConfig:
    """Configuration for OpenAI client."""
    # api_key: str = os.getenv("OPENAI_API_KEY", "sk-pgNP-BIHOtI8RPt-aC3Stg")
    api_key="sk-Jz7TVhKsaeJYJHRSaGl1ag"
    # api_key="sk-pgNP-BIHOtI8RPt-aC3Stg"
    base_url: str = "https://api.ai.it.ufl.edu"
    model: str = "gpt-oss-20b"
    temperature: float = 0.1
    max_tokens: int = 1000


@dataclass
class QueryConfig:
    """Configuration for query operations."""
    n_results: int = 5
    filter_by_type: Optional[str] = None