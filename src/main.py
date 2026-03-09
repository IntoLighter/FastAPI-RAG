from fastapi import FastAPI
import uvicorn

from dependency.settings import settings
from route import document, query

app = FastAPI(title="FastAPI RAG")
app.include_router(query.router)
app.include_router(document.router)

uvicorn.run(app, host=settings.host, port=settings.port)
