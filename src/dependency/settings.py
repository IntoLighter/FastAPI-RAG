from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client.http.models import Distance


class RagSettings(BaseModel):
    chunk_size: int
    chunk_overlap: int
    separators: list[str]
    retrieve_top_k: int
    rerank_top_k: int


class AppSettings(BaseModel):
    port: int


class QdrantSettings(BaseModel):
    host: str
    port: int
    collection_name: str
    vector_size: int
    distance: Distance


class GenerativeSettings(BaseModel):
    name: str
    base_url: str


class EmbeddingSettings(BaseModel):
    name: str
    base_url: str


class RerankerSettings(BaseModel):
    name: str
    base_url: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        extra="ignore",
    )

    app: AppSettings
    generative: GenerativeSettings
    embedding: EmbeddingSettings
    reranker: RerankerSettings
    qdrant: QdrantSettings
    rag: RagSettings


settings = Settings()
