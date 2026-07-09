from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from dependency.settings import settings
from dependency.vector_store import vector_store


@dataclass
class RetrieveArtifcat:
    docs: list[Document]
    final_query: str


def get_hyde_query(query: str) -> str:
    hyde_prompt = f"""
    Write a detailed textbook-style passage that answers the question.

    Question: {query}

    Answer:
    """
    return generative.invoke(hyde_prompt).content


@tool(response_format="content_and_artifact")
def retrieve_knowledge(
    query: str,
) -> tuple[str, list[Document]]:
    """
    Mandatory tool for retrieving factual context from internal knowledge base.

    ALWAYS use this tool before answering any question.
    Input `query` should be a rewritten version of the user question optimized for search.
    """

    # final_query = query
    # final_query = get_hyde_query(query)
    task_description = "Given a user question, retrieve relevant passages from a knowledge base that answer the question."
    final_query = f"Instruct: {task_description}\nQuery: {query}"

    retrieved_docs = vector_store.similarity_search(
        final_query,
        k=settings.rag.top_k,
    )
    response = "\n\n".join(
        f"[Document {i}]\n{doc.page_content}"
        for i, doc in enumerate(retrieved_docs, start=1)
    )
    return response, RetrieveArtifcat(docs=retrieved_docs, final_query=final_query)


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
