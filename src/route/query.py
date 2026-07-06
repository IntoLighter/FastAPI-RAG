from fastapi import APIRouter

from dependency.agent import agent
from schema.query import QueryRequest, QueryResponse, RetrieveResult

router = APIRouter(tags=["query"])


@router.post(path="/query")
async def query(
    query: QueryRequest,
) -> QueryResponse:
    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": query.message}]},
    )

    results: dict[str, RetrieveResult] = {}

    for message in response["messages"]:
        if message.type == "ai" and message.tool_calls:
            tool_data = message.tool_calls[0]
            results[tool_data["id"]] = RetrieveResult(
                llm_query=tool_data["args"]["query"],
            )
        elif message.type == "tool":
            results[message.tool_call_id].final_query = message.artifact.final_query
            results[message.tool_call_id].chunks = [
                chunk.page_content for chunk in message.artifact.docs
            ]

    return QueryResponse(
        response=response["messages"][-1].text,
        knowledge=list(results.values()),
    )
