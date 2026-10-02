import asyncio
import re

import pymupdf
from fastapi import APIRouter, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from dependency.settings import settings
from dependency.vector_store import vector_store

router = APIRouter(prefix="/documents", tags=["documents"])

tokenizer = AutoTokenizer.from_pretrained(settings.embedding.name)

text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=settings.rag.chunk_size,
    chunk_overlap=settings.rag.chunk_overlap,
    separators=settings.rag.separators,
)


def _extract_text(content: bytes) -> str:
    with pymupdf.open(stream=content, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


def pre_clean(text: str) -> str:
    text = re.sub(r"[ \t\xa0\f\v]+", " ", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(?<![.!?…:;\n])\n(?![\n#*\-»«\"'(\[{])", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def post_clean(text: str) -> str:
    text = text.strip()
    return re.sub(r"\s+", " ", text)


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload(
    file: UploadFile,
) -> JSONResponse:
    """To upload russian or english books."""

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Unsupported file type")

    content = await file.read()

    full_text = await asyncio.to_thread(_extract_text, content)
    full_text = pre_clean(full_text)

    splits = text_splitter.split_text(full_text)
    splits = [post_clean(split) for split in splits]

    await vector_store.aadd_texts(splits)

    return JSONResponse(
        content={"status": "success", "message": "Document successfully uploaded"},
    )
