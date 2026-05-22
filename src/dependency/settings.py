from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client.http.models import Distance


class AppSettings(BaseModel):
    port: int


class QdrantSettings(BaseModel):
    host: str
    port: int
    collection_name: str
    vector_size: int
    distance: Distance
    k: int


class GenerativeSettings(BaseModel):
    name: str
    port: int
    gpu_memory_utilization: float
    max_model_len: int
    base_url: str


class EmbeddingSettings(BaseModel):
    name: str
    port: int
    gpu_memory_utilization: float
    max_model_len: int
    base_url: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app: AppSettings
    generative: GenerativeSettings
    embedding: EmbeddingSettings
    qdrant: QdrantSettings


settings = Settings()
