from fastapi import APIRouter
from coderag.v1.routers.user import user_router
from coderag.v1.routers.health import health_router

router = APIRouter(prefix="/api/v1")

router.include_router(user_router)
router.include_router(health_router)
