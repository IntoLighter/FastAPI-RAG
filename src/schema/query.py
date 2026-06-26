from pydantic import BaseModel


class RetrieveResult(BaseModel):
    query: str
    chunks: list[str]


class QueryResponse(BaseModel):
    response: str
    knowledge: list[RetrieveResult]
