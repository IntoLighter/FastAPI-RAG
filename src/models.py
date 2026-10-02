from datetime import UTC, datetime
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Index, text
from sqlmodel import Field, SQLModel

from dependency.settings import settings


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Chunk(SQLModel, table=True):
    """A single embedded text chunk of an ingested document."""

    __tablename__ = settings.postgres.chunk_table

    id: str = Field(
        primary_key=True,
        default_factory=lambda: str(uuid4()),
        sa_column_kwargs={"server_default": text("gen_random_uuid()::text")},
    )
    content: str
    source: str | None = None
    embedding: list[float] = Field(
        sa_column=Column(Vector(settings.postgres.vector_size)),
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column_kwargs={"server_default": text("now()")},
    )

    __table_args__ = (
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class Conversation(SQLModel, table=True):
    """Dialog session grouping user/assistant exchanges."""

    __tablename__ = "conversations"

    id: str = Field(
        primary_key=True,
        default_factory=lambda: str(uuid4()),
        sa_column_kwargs={"server_default": text("gen_random_uuid()::text")},
    )
    title: str | None = None
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column_kwargs={"server_default": text("now()")},
    )


class Message(SQLModel, table=True):
    """A single message inside a conversation."""

    __tablename__ = "messages"

    id: str = Field(
        primary_key=True,
        default_factory=lambda: str(uuid4()),
        sa_column_kwargs={"server_default": text("gen_random_uuid()::text")},
    )
    conversation_id: str = Field(foreign_key="conversations.id", index=True)
    role: str
    content: str
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column_kwargs={"server_default": text("now()")},
    )
