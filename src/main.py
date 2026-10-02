from fastapi import FastAPI

from dependency.logging import configure_logging
from dependency.settings import settings
from middleware.logging import LoggingMiddleware
from routes import document, query

configure_logging(
    level=settings.app.log_level,
    json_logs=settings.app.log_json,
)

app = FastAPI(title="FastAPI RAG")
app.add_middleware(LoggingMiddleware)
app.include_router(query.router)
app.include_router(document.router)
