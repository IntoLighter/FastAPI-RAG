from pydantic import BaseModel


class QueryRequest(BaseModel):
    message: str


class RetrievedContext(BaseModel):
    query: str
    chunks: list[str]


class QueryResponse(BaseModel):
    answer: str
    retrieved: list[RetrievedContext]
