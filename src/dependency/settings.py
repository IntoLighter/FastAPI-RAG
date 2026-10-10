from functools import cached_property

from psycopg.conninfo import make_conninfo
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL


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
    database: str
    user: str
    password: str
    vector_size: int
    embedding_table_name: str
    chat_table_name: str

    @cached_property
    def psycopg_conninfo(self) -> str:
        """Connection string in libpq format for the psycopg async driver."""
        return make_conninfo(
            host=self.host,
            port=self.port,
            dbname=self.database,
            user=self.user,
            password=self.password,
        )

    @cached_property
    def sqlalchemy_conninfo(self) -> str:
        """SQLAlchemy URL using the psycopg async driver."""
        return URL.create(
            "postgresql+psycopg",
            username=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
        ).render_as_string(hide_password=False)


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
