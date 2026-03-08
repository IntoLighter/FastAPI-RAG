from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.vectorstores import VectorStore
from langchain_ollama import ChatOllama
from langgraph.graph.state import CompiledStateGraph

from dependency.settings import settings


@dataclass
class RetrieveKnowledgeContext:
    vector_store: VectorStore


@tool(response_format="content_and_artifact")
def retrieve_knowledge(runtime: ToolRuntime[RetrieveKnowledgeContext], query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = runtime.context.vector_store.similarity_search(
        query, k=settings.qdrant_vector_store.k
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

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype="float16",
#     bnb_4bit_use_double_quant=True,
# )

# pipeline = HuggingFacePipeline.from_model_id(
#     model_id="Qwen/Qwen3.5-4B",
#     task="text-generation",
#     pipeline_kwargs=dict(
#         max_new_tokens=1024,
#         do_sample=False,
#         return_full_text=False,
#     ),
#     model_kwargs={"quantization_config": quantization_config},
# )

# model = ChatHuggingFace(llm=pipeline)

model = ChatOllama(
    model=settings.generative_model_name,
    validate_model_on_init=True,
    temperature=0,
)

agent = create_agent(model, [retrieve_knowledge], system_prompt=prompt)


def get_agent() -> CompiledStateGraph:
    return agent
