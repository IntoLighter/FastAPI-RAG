from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams

from dependency.settings import settings


client = QdrantClient(
    host=settings.qdrant_vector_store.host, port=settings.qdrant_vector_store.port
)

if not client.collection_exists(
    collection_name=settings.qdrant_vector_store.collection_name
):
    client.create_collection(
        collection_name=settings.qdrant_vector_store.collection_name,
        vectors_config=VectorParams(
            size=settings.qdrant_vector_store.vector_size,
            distance=settings.qdrant_vector_store.distance,
        ),
    )

vector_store = QdrantVectorStore(
    client=client,
    collection_name=settings.qdrant_vector_store.collection_name,
    embedding=HuggingFaceEmbeddings(model_name=settings.embedding_model_name),
)


def get_vector_store() -> VectorStore:
    return vector_store
