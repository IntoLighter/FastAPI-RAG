# FastAPI RAG

A RAG service built with FastAPI, LangChain, Qdrant, and Qwen models served with vLLM.

## Architecture

```text
                    ┌─────────────┐
                    │   FastAPI   │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         ┌─────────┐ ┌───────────┐ ┌──────────┐
         │ Qdrant  │ │ Embedding │ │ Reranker │
         └─────────┘ └───────────┘ └──────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ Qwen3-4B-AWQ│
                    └─────────────┘
```

## Stack

* Python 3.13+
* FastAPI
* LangChain
* Qdrant
* vLLM
* Docker Compose
* Qwen3-4B-AWQ
* Qwen3-Embedding-0.6B
* Qwen3-Reranker-0.6B

## RAG Pipeline

1. Documents are split into chunks.
2. Embeddings are generated and stored in Qdrant.
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
