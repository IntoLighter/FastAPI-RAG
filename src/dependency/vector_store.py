from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams

from dependency.settings import settings

client = QdrantClient(
    host=settings.qdrant.host, port=settings.qdrant.port,
)

if not client.collection_exists(
    collection_name=settings.qdrant.collection_name,
):
    client.create_collection(
        collection_name=settings.qdrant.collection_name,
        vectors_config=VectorParams(
            size=settings.qdrant.vector_size,
            distance=settings.qdrant.distance,
        ),
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
