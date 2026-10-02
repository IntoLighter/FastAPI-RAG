import os

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

path1 = os.path.abspath(os.path.join(__file__, "..", "..", "..", ".env"))
path2 = os.path.abspath(os.path.join(__file__, "..", ".env"))

load_dotenv(path1)
load_dotenv(path2, override=True)


class PostgresSettings(BaseModel):
    host: str
    port: int
    user: str
    password: str
    database: str
    chunk_table: str

    @property
    def url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


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

    postgres: PostgresSettings


settings = Settings()
