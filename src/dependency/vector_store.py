from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from dependency.settings import settings

client = QdrantClient(
    host=settings.qdrant.host, port=settings.qdrant.port,
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name=settings.qdrant.collection_name,
    embedding=OpenAIEmbeddings(
        model=settings.embedding.name,
        base_url=settings.embedding.base_url,
        api_key="EMPTY",
    ),
)
