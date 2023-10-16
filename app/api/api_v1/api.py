from fastapi import APIRouter
from app.api.api_v1.endpoints import ingestion, retrieval

api_v1_router = APIRouter()

api_v1_router.include_router(ingestion.router, prefix="/ingest", tags=["Data Ingestion"])
api_v1_router.include_router(retrieval.router, prefix="/retrieve", tags=["Data Retrieval"])
