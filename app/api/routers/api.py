from fastapi import APIRouter
from app.api.routers.user import user_router
from app.api.routers.health import health_router

router = APIRouter(prefix="/api/v1")

router.include_router(user_router)
router.include_router(health_router)
