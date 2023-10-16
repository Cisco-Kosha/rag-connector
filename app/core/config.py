import logging
import os
from logging.config import dictConfig

from pydantic import BaseSettings

from app.utils.logging import LogConfig


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "rag-connector"

    # env variables
    DEFAULT_COLLECTION_NAME: str = os.getenv('DEFAULT_COLLECTION_NAME', 'default_collection')
    VECTORSTORE: str = os.getenv('VECTORSTORE', 'chromadb')
    CHROMA_PERSIST_DIRECTORY: str = os.getenv('CHROMA_PERSIST_DIRECTORY', '')
    CHROMADB_CONNECTOR_SERVER_URL: str = os.getenv('CHROMADB_SERVER_URL', '')
    OPENAI_CONNECTOR_SERVER_URL: str = os.getenv('OPENAI_CONNECTOR_SERVER_URL', '')

    # needed only for the retrieval part, as we still use langchain
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    JWT_TOKEN: str = os.getenv('JWT_TOKEN', '')
    CHROMADB_SERVER_URL: str = os.getenv('CHROMADB_SERVER_URL', '')
    CHROMADB_SERVER_API_KEY: str = os.getenv('CHROMADB_SERVER_API_KEY', '')


settings = Settings()

dictConfig(LogConfig().dict())
logger = logging.getLogger(settings.PROJECT_NAME)
