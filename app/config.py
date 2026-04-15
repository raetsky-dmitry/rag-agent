# app/config.py
from pydantic_settings import BaseSettings
from functools import lru_cache



class Settings(BaseSettings):
    # LLM
    openrouter_api_key: str
    openrouter_base_url: str 
    model_name_qwen: str 
    model_name_deepseek: str 
    model_name_free: str 

    # LangSmith (опционально)
    langsmith_tracing: bool 
    langsmith_api_key: str 
    langsmith_project: str 

    # RAG
    embedding_model_name: str = "qwen/qwen3-embedding-8b"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retriever_k: int = 4
    retriver_weights: list = [0.4, 0.6] # 40% BM25 + 60% векторный
    index_path: str = "data/_indexes"
    docs_path: str = "data/_docs"
    is_local_embeddings_model: bool = False

    # # Сессии
    # max_history_messages: int = 20
    # session_ttl_seconds: int = 3600  # 1 час

    # # API
    # max_tokens: int = 2000
    # request_timeout: int = 60

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
