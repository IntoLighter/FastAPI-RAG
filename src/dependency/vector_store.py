from langchain_openai import OpenAIEmbeddings
from langchain_postgres.v2.engine import PGEngine
from langchain_postgres.v2.vectorstores import PGVectorStore

from dependency.settings import settings

pg_engine = PGEngine.from_engine(settings.postgres.psycopg_conninfo)

pg_engine.init_vectorstore_table(
    table_name=settings.postgres.embedding_table_name,
    vector_size=settings.postgres.vector_size,
)

embedding = OpenAIEmbeddings(
    model=settings.embedding.name,
    base_url=settings.embedding.base_url,
    api_key="EMPTY",
    check_embedding_ctx_length=False,
)

vector_store = PGVectorStore.create(
    pg_engine,
    embedding_service=embedding,
    table_name=settings.postgres.embedding_table_name,
)
