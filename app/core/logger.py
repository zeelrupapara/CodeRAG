import os
from logging.config import dictConfig


def load_logger():
    # Get debug flag from environment variable or default to False
    debug = os.getenv("DEBUG")

    # Define logging configuration
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            },
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s',
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'default',
            },
            # 'file': {
            #     'class': 'logging.FileHandler',
            #     'filename': 'app.log',
            #     'formatter': 'detailed',
            # },
        },
        'loggers': {
            '': {
                'handlers': ['console'],
                'level': 'DEBUG' if debug else 'INFO',  # Set to DEBUG if debug is True, else INFO
                'propagate': True,
            },
            'uvicorn.error': {
                'level': 'DEBUG' if debug else 'INFO',
                'handlers': ['console'],
                'propagate': False,
            },
            'uvicorn.access': {
                'level': 'DEBUG' if debug else 'INFO',
                'handlers': ['console'],
                'propagate': False,
            },
        },
    }

    # Apply the logging configuration
    dictConfig(LOGGING_CONFIG)
