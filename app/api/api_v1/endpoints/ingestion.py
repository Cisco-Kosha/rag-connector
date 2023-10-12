import os
import traceback
from typing import Any, List

import openai
from chromadb.utils import embedding_functions
from fastapi import APIRouter
import chromadb
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

from app.helper.utility import text_embedding
from app.vectorstore.chromadb_db import Chroma as RAGChroma

from langchain.text_splitter import RecursiveCharacterTextSplitter
from starlette.responses import Response

from app.connectors.chroma import ChromaDBConnector

from app.core.config import settings, logger

from app.schemas.ingestion import DocumentChunker, StoreInVectoDB, ChatBotParameters

router = APIRouter()
DEFAULT_K = 4
default_collection_name: str = "default_collection"


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
       This endpoint is used to create embeddings for a list of documents, and store them in a vectorstore
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


@router.post("/chatbot", status_code=200)
def chatbot(properties: ChatBotParameters) -> Any:
    """
       This endpoint is used to fetch the top K documents from a vectorstore, based on a query and then send it as context to the LLM model
    """
    try:
        os.environ['OPENAI_API_KEY'] = ""
        if properties.embedding_model is None:
            return Response(status_code=400, content="Embedding model is empty")

        if properties.embedding_model == "openai":
            chroma_connector = ChromaDBConnector()
            collection = chroma_connector.get_or_create_collection("default_collection")

            print(collection)
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ["OPENAI_API_KEY"],
                model_name="text-embedding-ada-002"
            )
            # vector = text_embedding(properties.prompt)
            #
            # print(vector)
            # results = chroma_connector.query(name=str(collection['id']), vector=[vector], include=["documents"],
            #                                  n_results=10)
            # res = "\n".join(str(item) for item in results['documents'][0])

            documents = chroma_connector.get_collection(collection['id'])
            print(documents['documents'])
            vector = text_embedding(documents['documents'])

            results = chroma_connector.query(name=str(collection['id']), vector=[vector], include=["documents"],
                                             n_results=15)

            embeddings = OpenAIEmbeddings()

            chromadb_client = chromadb.HttpClient(
                host="localhost", port="8211", headers={"Authorization": "Bearer test-token"})

            chat = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", verbose=True)
            db = Chroma(embedding_function=embeddings,
                        collection_name='default_collection', client=chromadb_client)
            qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=db.as_retriever())
            # res = "\n".join(str(item) for item in results['documents'][0])
            # prompt = f'```{res}```'
            #
            # messages = [
            #     {"role": "system", "content": "You are an API Expert. You are helping a customer with an API issue. Do not worry about missing parts and formatting issues. Do your best to help the customer."},
            #     {"role": "user", "content": prompt}
            # ]
            # response = openai.ChatCompletion.create(
            #     model="gpt-3.5-turbo",
            #     messages=messages,
            #     temperature=0
            # )
            # response_message = response["choices"][0]["message"]["content"]
            #
            # print(response_message)
            # return response_message

            return qa.run(properties.prompt)
    except Exception as e:
        logger.error(e)
        traceback.print_exc()
        return Response(status_code=400, content=str(e))
    return "Success"
