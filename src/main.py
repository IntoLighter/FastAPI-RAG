from fastapi import FastAPI

from route import document, query

app = FastAPI(title="FastAPI RAG")
app.include_router(query.router)
app.include_router(document.router)
