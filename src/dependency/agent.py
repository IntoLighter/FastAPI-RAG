from dataclasses import dataclass
from typing import Any

import httpx
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI, BaseModel
from pydantic import field_validator

from dependency.settings import settings
from dependency.vector_store import vector_store

client = AsyncOpenAI(
    base_url=settings.reranker.base_url,
    api_key="EMPTY",
)


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: str

    @field_validator("document", mode="before")
    @classmethod
    def extract_document(cls, v: Any) -> str:
        return v.get("text")


class RerankResponse(BaseModel):
    results: list[RerankResult]


async def rerank(query: str, docs: list[str]) -> list[RerankResult]:
    raw = await client.post(
        "/rerank",
        cast_to=httpx.Response,
        body={
            "model": settings.reranker.name,
            "query": query,
            "documents": docs,
        },
    )

    response = RerankResponse.model_validate(raw.json())
    return response.results


@dataclass
class RetrieveArtifact:
    docs: list[Document]
    final_query: str


async def get_hyde_query(query: str) -> str:
    hyde_prompt = f"""
    Write a detailed textbook-style passage that answers the question.

    Question: {query}

    Answer:
    """
    return await generative.ainvoke(hyde_prompt).content


@tool(response_format="content_and_artifact")
async def retrieve_knowledge(
    query: str,
) -> tuple[str, RetrieveArtifact]:
    """
    Mandatory tool for retrieving factual context from internal knowledge base.

    ALWAYS use this tool before answering any question.
    Input `query` should be a rewritten version of the user question optimized for search.
    """

    # final_query = query
    # final_query = await get_hyde_query(query)
    task_description = "Given a user question, retrieve relevant passages from a knowledge base that answer the question."
    final_query = f"Instruct: {task_description}\nQuery: {query}"

    docs = vector_store.similarity_search(
        final_query,
        k=settings.rag.retrieve_top_k,
    )

    doc_texts = [d.page_content for d in docs]
    reranked_documents = await rerank(query, doc_texts)

    rerank_top_k_documents = reranked_documents[:settings.rag.rerank_top_k]
    rerank_top_k_texts = [d.document for d in rerank_top_k_documents]

    response = "\n\n".join(
        f"[Document {i}]\n{doc}"
        for i, doc in enumerate(rerank_top_k_texts, start=1)
    )

    return response, RetrieveArtifact(
        docs=rerank_top_k_texts,
        final_query=final_query
    )


prompt = """
You are a retrieval-augmented assistant.

You MUST follow these rules:

1. For EVERY user question, you MUST first call the tool `retrieve_knowledge`.
2. You are NOT allowed to answer from your internal knowledge without using the tool first.
3. Even if you think you know the answer, you still MUST use the tool.
4. Only after receiving tool output, you may generate the final answer.
5. If the tool returns no relevant information, say that the information was not found in the knowledge base.

Violation of these rules is not allowed.

Tool usage format:
- Always call `retrieve_knowledge` with a relevant search query derived from the user question.
"""

generative = ChatOpenAI(
    model=settings.generative.name,
    base_url=settings.generative.base_url,
    api_key="EMPTY",
)

agent = create_agent(generative, [retrieve_knowledge], system_prompt=prompt)
