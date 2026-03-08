from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client.http.models import Distance


class QdrantVectorStoreSettings(BaseModel):
    host: str
    port: int
    collection_name: str
    vector_size: int
    distance: Distance
    k: int


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app_host: str
    app_port: int

    embedding_model_name: str
    generative_model_name: str

    qdrant_vector_store: QdrantVectorStoreSettings


settings = Settings()
