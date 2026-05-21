from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client.http.models import Distance


class QdrantSettings(BaseModel):
    host: str
    port: int
    collection_name: str
    vector_size: int
    distance: Distance
    k: int


class VllmSettings(BaseModel):
    llm_base_url: str
    embedding_base_url: str
    api_key: str = "EMPTY"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    host: str
    port: int

    embedding_model_name: str
    generative_model_name: str

    vllm: VllmSettings
    qdrant: QdrantSettings


settings = Settings()
