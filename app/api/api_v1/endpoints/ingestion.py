import os
import traceback
from typing import Any, List


from fastapi import APIRouter

from app.vectorstore.chromadb_db import Chroma as RAGChroma

from langchain.text_splitter import RecursiveCharacterTextSplitter
from starlette.responses import Response


from app.core.config import settings, logger

from app.schemas.ingestion import DocumentChunker, StoreInVectoDB

router = APIRouter()
DEFAULT_K = 4
# default_collection_name: str = "default_collection"


@router.post("/chunking", status_code=200)
def split_documents(body: DocumentChunker) -> Any:
    """
       This endpoint is used to split a document into multiple documents based on the chunk size
    """
    try:
        chunk_size = body.chunk_size
        chunk_overlap = body.chunk_overlap
        if body.chunk_size == 0:
            chunk_size = 1000
        if body.chunk_overlap == 0:
            chunk_overlap = 100
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""]
        )
        documents = text_splitter.split_documents(documents=body.documents)
    except Exception as e:
        logger.error(e)
        return Response(status_code=400, content=str(e))
    return documents


@router.post("/store", status_code=200)
def create_embeddings(properties: StoreInVectoDB) -> Any:
    """
       This endpoint is used to create embeddings for a list of documents, and store them in a vectorstore, via connectors
    """
    try:
        vectorstore = properties.vectorstore
        if properties.documents is None:
            return Response(status_code=400, content="Documents are empty")
        if properties.embedding_model is None:
            return Response(status_code=400, content="Embedding model is empty")

        texts = [doc.page_content for doc in properties.documents]
        metadatas = [doc.metadata for doc in properties.documents]

        # supports only ChromaDB, for now
        if vectorstore == "chromadb":
            chromadb_obj = RAGChroma(embedding_function="openai")
            chromadb_obj.add_texts(texts=texts, metadatas=metadatas, ids=None)

        logger.info("****** Added to ChromaDB vectorstore vectors")
    except Exception as e:
        logger.error(e)
        traceback.print_exc()
        return Response(status_code=400, content=str(e))
    return "Success"
