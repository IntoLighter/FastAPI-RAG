from langchain.agents import create_agent
from langchain.retrievers import ContextualCompressionRetriever
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from dependency import vector_store
from dependency.settings import settings

base_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": settings.rag.top_k},
)

reranker_runnable = RunnableLambda(
    lambda inputs: rerank(inputs["query"], inputs["docs"])
)

compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=reranker_runnable,
)


def rerank(query: str, docs: list[Document]) -> list[Document]:
    pairs = [(query, d.page_content) for d in docs]

    scores = reranker_model.predict(pairs)

    sorted_docs = [
        doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    ]

    return sorted_docs[:5]


@tool(response_format="content_and_artifact")
def retrieve_knowledge(
    query: str,
) -> tuple[str, list[Document]]:
    """
    Mandatory tool for retrieving factual context from internal knowledge base.

    ALWAYS use this tool before answering any question.
    Input `query` should be a rewritten version of the user question optimized for search.
    """
    retrieved_docs = vector_store.similarity_search(
        query,
        k=settings.rag.top_k,
    )
    response = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return response, retrieved_docs


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

model = ChatOpenAI(
    model=settings.generative.name,
    base_url=settings.generative.base_url,
    api_key="EMPTY",
)

agent = create_agent(model, [retrieve_knowledge], system_prompt=prompt)
