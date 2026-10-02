from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from database import create_db_and_tables
from dependency.logging import configure_logging
from dependency.settings import settings
from middleware.logging import LoggingMiddleware
from routes import document, query

configure_logging(
    level=settings.app.log_level,
    json_logs=settings.app.log_json,
)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await create_db_and_tables()
    yield


app = FastAPI(title="FastAPI RAG", lifespan=lifespan)
app.add_middleware(LoggingMiddleware)
app.include_router(query.router)
app.include_router(document.router)
