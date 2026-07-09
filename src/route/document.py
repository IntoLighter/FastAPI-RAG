import re
import tempfile

from fastapi import APIRouter, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from dependency.settings import settings
from dependency.vector_store import vector_store

router = APIRouter(prefix="/documents", tags=["documents"])


def pre_clean(text: str) -> str:
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def post_clean(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^[\.\s]+", "", text)
    text = re.sub(r"\s+", " ", text)
    text += "."
    return text


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload(
    file: UploadFile,
) -> JSONResponse:
    """To upload russian or english books."""

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Unsupported file type")

    content = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = await loader.aload()

    full_text = "\n".join([doc.page_content for doc in docs])
    full_text = pre_clean(full_text)

    tokenizer = AutoTokenizer.from_pretrained(
        settings.embedding.name,
    )

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=settings.rag.chunk_size,
        chunk_overlap=settings.rag.chunk_overlap,
        separators=settings.rag.separators,
    )
    splits = text_splitter.split_text(full_text)
    splits = [post_clean(split) for split in splits]

    await vector_store.aadd_texts(splits)

    return JSONResponse(
        content={"status": "success", "message": "Document successfully uploaded"},
    )
