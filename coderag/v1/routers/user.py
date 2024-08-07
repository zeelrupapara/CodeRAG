from fastapi import APIRouter
from coderag.v1.endpoints.user.user import user_module
from coderag.v1.endpoints.user.auth import auth_module

user_router = APIRouter()

user_router.include_router(
    user_module,
    prefix="/users",
    tags=["users"],
    responses={404: {"description": "Not found"}},
)

user_router.include_router(
    auth_module,
    prefix="",
    tags=["auth"],
    responses={404: {"description": "Not found"}},
)
