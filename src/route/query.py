from typing import Annotated

from fastapi import APIRouter, Body, Depends
from langchain_core.vectorstores import VectorStore
from langgraph.graph.state import CompiledStateGraph

from dependency.agent import RetrieveKnowledgeContext, get_agent
from dependency.vector_store import get_vector_store
from schema.query import QueryResponse, RetrieveResult

router = APIRouter(tags=["query"])


@router.post(path="/query")
async def query(
    query: Annotated[str, Body],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    agent: Annotated[CompiledStateGraph, Depends(get_agent)],
) -> QueryResponse:
    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": query}]},
        context=RetrieveKnowledgeContext(vector_store),
    )

    results: dict[str, RetrieveResult] = {}

    for message in response["messages"]:
        if message.type == "ai" and message.tool_calls:
            tool_data = message.tool_calls[0]
            results[tool_data["id"]] = RetrieveResult(
                query=tool_data["args"]["query"], chunks=[],
            )
        elif message.type == "tool":
            results[message.tool_call_id].chunks = [
                chunk.page_content for chunk in message.artifact
            ]

    return QueryResponse(
        response=response["messages"][-1].text, knowledge=list(results.values()),
    )
