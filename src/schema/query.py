from pydantic import BaseModel


class QueryRequest(BaseModel):
    message: str


class RetrieveResult(BaseModel):
    query: str
    hyde: str | None = None
    chunks: list[str]


class QueryResponse(BaseModel):
    response: str
    knowledge: list[RetrieveResult]
