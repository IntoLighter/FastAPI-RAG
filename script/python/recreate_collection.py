from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
from settings import settings

client = QdrantClient(
    host=settings.qdrant.host,
    port=settings.qdrant.port,
)

if client.collection_exists(
    collection_name=settings.qdrant.collection_name,
):
    client.delete_collection(settings.qdrant.collection_name)

client.create_collection(
    collection_name=settings.qdrant.collection_name,
    vectors_config=VectorParams(
        size=settings.qdrant.vector_size,
        distance=settings.qdrant.distance,
    ),
)