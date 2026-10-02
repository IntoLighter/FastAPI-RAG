from pydantic import BaseModel


class QueryRequest(BaseModel):
    message: str
    conversation_id: str | None = None


class RetrievedContext(BaseModel):
    query: str
    chunks: list[str]


class QueryResponse(BaseModel):
    answer: str
    conversation_id: str
    retrieved: list[RetrievedContext]
