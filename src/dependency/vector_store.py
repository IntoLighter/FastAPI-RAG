from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams

from dependency.settings import settings


client = QdrantClient(
    host=settings.qdrant.host, port=settings.qdrant.port
)

if not client.collection_exists(
    collection_name=settings.qdrant.collection_name
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
        model=settings.embedding_model_name,
        base_url=settings.vllm.embedding_base_url,
        api_key=settings.vllm.api_key,
    ),
)


def get_vector_store() -> VectorStore:
    return vector_store
