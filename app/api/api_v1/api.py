from fastapi import APIRouter
from app.api.api_v1.endpoints import ingestion

api_v1_router = APIRouter()

api_v1_router.include_router(ingestion.router, prefix="/ingest", tags=["Data Ingestion"])
