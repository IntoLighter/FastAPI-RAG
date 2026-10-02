from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_session
from dependency.agent import agent
from dependency.history import ensure_conversation, load_history, save_messages
from schema.query import QueryRequest, QueryResponse, RetrievedContext

router = APIRouter(tags=["query"])


@router.post(path="/query")
async def query(
    query: QueryRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> QueryResponse:
    conversation_id = await ensure_conversation(session, query.conversation_id)
    history = await load_history(session, conversation_id)

    response = await agent.ainvoke(
        {"messages": [*history, {"role": "user", "content": query.message}]},
    )

    artifacts = [
        message.artifact for message in response["messages"] if message.type == "tool"
    ]

    answer = response["messages"][-1].text
    await save_messages(
        session,
        conversation_id,
        [("user", query.message), ("assistant", answer)],
    )

    return QueryResponse(
        answer=answer,
        conversation_id=conversation_id,
        retrieved=[
            RetrievedContext(query=artifact.query, chunks=artifact.chunks)
            for artifact in artifacts
        ],
    )
