from fastapi import APIRouter

from dependency.agent import agent
from schema.query import QueryRequest, QueryResponse, RetrievedContext

router = APIRouter(tags=["query"])


@router.post(path="/query")
async def query(
    query: QueryRequest,
) -> QueryResponse:
    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": query.message}]},
    )

    artifacts = [
        message.artifact for message in response["messages"] if message.type == "tool"
    ]

    return QueryResponse(
        answer=response["messages"][-1].text,
        retrieved=[
            RetrievedContext(query=artifact.query, chunks=artifact.chunks)
            for artifact in artifacts
        ],
    )
