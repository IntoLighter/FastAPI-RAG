# FastAPI RAG

A RAG service built with FastAPI, LangChain, PostgreSQL with pgvector, and Qwen models served with vLLM.

## Architecture

```text
                     ┌─────────────┐
                     │   FastAPI   │
                     └──────┬──────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌───────────┐ ┌───────────┐ ┌──────────┐
        │  Postgres │ │ Embedding │ │ Reranker │
        │  pgvector │ └───────────┘ └──────────┘
        │checkpoints│
        └───────────┘       ┌─────────────┐
                            │ Qwen3-4B-AWQ│
                            └─────────────┘
```

Postgres stores two things: the document chunks with their embeddings in the
`fastapi_rag` pgvector table (`PGVectorStore`), and the conversation state of
every chat session in the LangGraph checkpoint tables, so requests sharing a
`session_id` keep their history.

## Stack

* Python 3.13+
* FastAPI
* LangChain
* langchain-postgres (`PGVectorStore`, v2 engine)
* PostgreSQL with pgvector
* psycopg (async driver)
* vLLM
* Docker Compose
* Qwen3-4B-AWQ
* Qwen3-Embedding-0.6B
* Qwen3-Reranker-0.6B

## RAG Pipeline

1. Documents are split into chunks.
2. Embeddings are generated and stored in PostgreSQL with pgvector.
3. The query retrieves relevant chunks.
4. Retrieved chunks are reranked.
5. The most relevant chunks are passed to the generative model.
6. The model generates the final answer.

## Run

Create a `.env` file and start the services:

```bash
docker compose up -d
```

Check the service status:

```bash
docker compose ps
```

Stop the services:

```bash
docker compose down
```

## Conversations

Each query accepts an optional `session_id`. Omit it to start a new
conversation; reuse the returned one to continue it:

```bash
curl -X POST http://localhost/query \
  -H 'Content-Type: application/json' \
  -d '{"message": "How do I deploy the service?"}'
```

```bash
curl -X POST http://localhost/query \
  -H 'Content-Type: application/json' \
  -d '{"message": "And in debug mode?", "session_id": "<session_id>"}'
```

The second call sees the previous messages: the agent is created with an
`AsyncPostgresSaver` checkpointer backed by an async psycopg connection pool.

## Development

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Run linting and formatting:

```bash
uv run ruff check .
uv run ruff format .
```

To drop the stored vectors and conversations (locally, override the host from
`.env`, which points at the compose service):

```bash
POSTGRES__HOST=localhost uv run python scripts/python/recreate_collection.py
```

Note: the psycopg async driver (used by both the vector store and the
checkpointer) cannot run on the Windows proactor event loop that the plain
`fastapi run` entry point forces, so start the API locally with `fastapi dev`
(or `uvicorn --reload`).
