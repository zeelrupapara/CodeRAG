# fastapi
from fastapi import FastAPI
from coderag.core.modules import init_routers, make_middleware
from coderag.core.logger import load_logger
from dotenv import load_dotenv
import logging
import os


def create_app() -> FastAPI:

    # load environment variables
    load_dotenv()

    # initialize logger
    load_logger()
    logging.info("loaded environment variables")

    # initialize app
    app_ = FastAPI(
        title="CodeRAG",
        description="Talk with your code using LLM",
        version="1.0.0",
        docs_url="/api/v1/docs",
        redoc_url="/api/v1/redoc",
        openapi_url="/api/v1/openapi.json",
        middleware=make_middleware(),
        debug=os.getenv("DEBUG"),
    )

    # initialize routers
    init_routers(app_=app_)
    return app_


app = create_app()
