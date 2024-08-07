from fastapi import APIRouter, Depends, HTTPException, status
from app.core.dependencies import get_db
from sqlalchemy.orm import Session
import logging


health_module = APIRouter()


@health_module.get("/healthz")
async def health():
    logging.debug("Health check")
    return {"status": "ok"}


@health_module.get("/healthdb")
async def healthdb(db: Session = Depends(get_db)):
    try:
        logging.debug("database health check init")
        db.execute("SELECT 1")
        logging.debug("database health check done")
        return {"status": "ok"}
    except Exception as e:
        logging.error(f"Database connection error: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Database connection error: {e}")
