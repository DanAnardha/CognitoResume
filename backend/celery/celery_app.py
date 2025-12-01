# backend/celery/celery_app.py

import os
import logging
from celery import Celery

logging.basicConfig(level=logging.WARNING)

logging.info("[CELERY] Initializing Celery...")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery = Celery(
    "worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["backend.celery.tasks"]
)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    worker_hijack_root_logger=False,
    worker_redirect_stdouts_level="WARNING"
)

logging.info("[CELERY] Celery is ready.")