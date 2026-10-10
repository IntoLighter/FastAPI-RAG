import uuid

import psycopg
from fastapi import APIRouter
from langchain_postgres import PostgresChatMessageHistory

from dependency.agent import agent
from dependency.settings import settings
from schema.query import QueryRequest, QueryResponse, RetrievedContext

router = APIRouter(tags=["query"])


@router.post(path="/query")
async def query(
    query: QueryRequest,
) -> QueryResponse:
    session_id = query.session_id or uuid.uuid4()

    async_connection = psycopg.AsyncConnection.connect(settings.postgres.psycopg_conninfo)
    chat_message_history_client = PostgresChatMessageHistory(
        table_name=settings.postgres.chat_table_name,
        session_id=str(session_id),
        async_connection=async_connection
    )

    messages = await chat_message_history_client.aget_messages()
    messages.append({"role": "user", "content": query.message})

    response = await agent.ainvoke(
        messages,
        config={"configurable": {"thread_id": str(session_id)}},
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
