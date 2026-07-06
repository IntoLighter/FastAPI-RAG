import re
import tempfile

from fastapi import APIRouter, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dependency import vector_store
from dependency.settings import settings

router = APIRouter(prefix="/documents", tags=["documents"])


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

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=settings.rag.chunk_size,
        chunk_overlap=settings.rag.chunk_overlap,
        separators=settings.rag.separators,
    )
    all_splits = text_splitter.split_text(full_text)

    await vector_store.aadd_texts(all_splits)

    return JSONResponse(
        content={"status": "success", "message": "Document successfully uploaded"},
    )
