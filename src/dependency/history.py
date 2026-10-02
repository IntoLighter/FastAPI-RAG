from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import Conversation, Message


async def ensure_conversation(
    session: AsyncSession,
    conversation_id: str | None,
) -> str:
    """Return a usable conversation id, creating the conversation if needed."""
    if conversation_id:
        conversation = await session.get(Conversation, conversation_id)
        if conversation is None:
            conversation = Conversation(id=conversation_id)
            session.add(conversation)
            await session.commit()
        return conversation.id

    conversation = Conversation()
    session.add(conversation)
    await session.commit()
    return conversation.id


async def load_history(
    session: AsyncSession,
    conversation_id: str,
    limit: int | None = None,
) -> list[dict[str, str]]:
    """Load the last ``limit`` messages of a conversation as agent input."""
    statement = (
        select(Message.role, Message.content)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc(), Message.id.desc())
    )
    if limit is not None:
        statement = statement.limit(limit)

    result = await session.execute(statement)
    history = list(reversed(result.all()))
    return [{"role": role, "content": content} for role, content in history]


async def save_messages(
    session: AsyncSession,
    conversation_id: str,
    messages: Sequence[tuple[str, str]],
) -> None:
    """Append ``(role, content)`` pairs to a conversation."""
    if not messages:
        return

    session.add_all(
        Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
        )
        for role, content in messages
    )
    await session.commit()
