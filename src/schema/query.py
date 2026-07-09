from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    message: str


class RetrieveResult(BaseModel):
    llm_query: str
    final_query: str | None = None
    chunks: list[str] = Field(default_factory=list)
    tool_response: str | None = None


class QueryResponse(BaseModel):
    response: str
    knowledge: list[RetrieveResult]
