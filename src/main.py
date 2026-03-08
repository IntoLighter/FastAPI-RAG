from fastapi import FastAPI

from route import query, document

app = FastAPI(title="FastAPI RAG")
app.include_router(query.router)
app.include_router(document.router)
