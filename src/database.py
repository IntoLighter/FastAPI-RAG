from collections.abc import AsyncIterator
from contextvars import ContextVar

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from dependency.settings import settings
from models import Conversation, Message  # noqa: F401  # register models

_session_var: ContextVar[AsyncSession | None] = ContextVar("db_session", default=None)


def build_engine() -> create_async_engine:
    """Create the async engine used across the application."""
    return create_async_engine(
        settings.postgres.url,
        echo=settings.postgres.echo,
        pool_pre_ping=True,
    )


engine = build_engine()
session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def create_db_and_tables() -> None:
    """Create the pgvector extension and every table declared on ``SQLModel``."""
    async with engine.begin() as connection:
        await connection.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
        await connection.run_sync(SQLModel.metadata.create_all)


async def get_session() -> AsyncIterator[AsyncSession]:
    """Yield an :class:`AsyncSession` bound to the current request."""
    async with session_factory() as session:
        token = _session_var.set(session)
        try:
            yield session
        finally:
            _session_var.reset(token)


def current_session() -> AsyncSession:
    """
    Return the session bound to the running request.

    Allows tools and services to reuse the request session without
    threading it through every call.
    """
    session = _session_var.get()
    if session is None:
        msg = "No database session is bound to the current context"
        raise RuntimeError(msg)
    return session
