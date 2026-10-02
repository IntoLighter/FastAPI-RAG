from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dependency.settings import settings
from models import Chunk

_DISTANCE_OPERATOR = {
    "cosine": "<=>",
    "l2": "<->",
    "inner_product": "<#>",
}

_embeddings = OpenAIEmbeddings(
    model=settings.embedding.name,
    base_url=settings.embedding.base_url,
    api_key="EMPTY",
    check_embedding_ctx_length=False,
)


def _embed(texts: list[str]) -> list[list[float]]:
    return _embeddings.embed_documents(texts)


async def add_texts(
    session: AsyncSession,
    texts: list[str],
    source: str | None = None,
) -> int:
    """Embed ``texts`` and store them as chunks."""
    if not texts:
        return 0

    embeddings = _embed(texts)
    session.add_all(
        Chunk(content=text, source=source, embedding=vector)
        for text, vector in zip(texts, embeddings, strict=True)
    )
    await session.commit()
    return len(texts)


async def similarity_search(
    session: AsyncSession,
    query: str,
    k: int | None = None,
) -> list[Document]:
    """Return the ``k`` nearest chunks ordered by vector distance."""
    k = k or settings.rag.retrieve_top_k
    operator = _DISTANCE_OPERATOR[settings.postgres.distance]

    query_embedding = _embed([query])[0]
    statement = (
        select(Chunk.content, Chunk.source)
        .order_by(getattr(Chunk.embedding, operator)(query_embedding))
        .limit(k)
    )

    result = await session.execute(statement)
    return [
        Document(page_content=content, metadata={"source": source})
        for content, source in result.all()
    ]
