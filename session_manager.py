"""
Session Manager & Logger Utilities
"""
import uuid
import logging
import sys
from datetime import datetime


class SessionManager:
    def generate_id(self) -> str:
        return f"session_{datetime.utcnow().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
