from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class RagSettings(BaseModel):
    chunk_size: int
    chunk_overlap: int
    separators: list[str]
    retrieve_top_k: int
    rerank_top_k: int


class AppSettings(BaseModel):
    port: int
    log_level: str
    log_json: bool


class PostgresSettings(BaseModel):
    host: str
    port: int
    user: str
    password: str
    database: str
    echo: bool = False
    vector_size: int
    chunk_table: str
    distance: str

    @property
    def url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


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
    postgres: PostgresSettings
    rag: RagSettings


settings = Settings()
