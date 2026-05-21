from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph

from dependency.settings import settings


@dataclass
class RetrieveKnowledgeContext:
    vector_store: VectorStore


@tool(response_format="content_and_artifact")
def retrieve_knowledge(runtime: ToolRuntime[RetrieveKnowledgeContext], query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = runtime.context.vector_store.similarity_search(
        query, k=settings.qdrant.k
    )
    response = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return response, retrieved_docs


prompt = (
    "You have access to a tool that retrieves context from a database. "
    "Use the tool to help answer user queries."
)

model = ChatOpenAI(
    model=settings.generative_model_name,
    base_url=settings.vllm.llm_base_url,
    api_key=settings.vllm.api_key,
    temperature=0,
)

agent = create_agent(model, [retrieve_knowledge], system_prompt=prompt)


def get_agent() -> CompiledStateGraph:
    return agent
