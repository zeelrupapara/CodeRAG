from fastapi import APIRouter
from app.api.endpoints.health.health import health_module


health_router = APIRouter()

health_router.include_router(
    health_module,
    prefix="",
    tags=["health"],
    responses={404: {"description": "Not found"}},
)
