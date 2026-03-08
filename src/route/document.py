import tempfile

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dependency.vector_store import get_vector_store
from langchain_community.document_loaders import PyPDFLoader

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload(
    file: UploadFile,
    vector_store: QdrantVectorStore = Depends(get_vector_store),
) -> JSONResponse:
    """
    To upload russian or english books.
    """
    if file.content_type not in ["application/pdf"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    content = await file.read()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(content)
    tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    vector_store.add_documents(all_splits)

    return JSONResponse(
        content={"status": "success", "message": "Document successfully uploaded"}
    )
