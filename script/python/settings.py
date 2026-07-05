import os

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client.http.models import Distance

path1 = os.path.abspath(os.path.join(__file__, "..", "..", "..", ".env"))
path2 = os.path.abspath(os.path.join(__file__, "..", ".env"))

load_dotenv(path1)
load_dotenv(path2, override=True)


class QdrantSettings(BaseModel):
    host: str
    port: int
    collection_name: str
    vector_size: int
    distance: Distance


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # env_file=(
        #     os.path.abspath(os.path.join(__file__, "..", "..", "..", ".env")),
        #     os.path.abspath(os.path.join(__file__, "..", ".env")),
        # ),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    qdrant: QdrantSettings


settings = Settings()
