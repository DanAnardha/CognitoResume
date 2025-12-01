# backend/app/services/extractor.py

import os
import logging
from typing import Optional
import spacy
from spacy_layout import spaCyLayout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("✅ Model spaCy-layout 'en_core_web_sm' loaded successfully.")
    layout = spaCyLayout(nlp)
except OSError:
    logger.error(
        "❌ spaCy model 'en_core_web_sm' not found. "
        "Run command: python -m spacy download en_core_web_sm"
    )
    nlp = None

def extract_text(file_path: str) -> Optional[str]:
    if not os.path.exists(file_path):
        logger.error(f"CV file not found: {file_path}")
        return None

    if nlp is None:
        logger.error("spaCy-layout model not found. Can't proceed extraction.")
        return None

    ext = os.path.splitext(file_path)[1].lower()
    text = None

    try:
        doc = layout(file_path)
        logger.info(f"Successfully load document {file_path}")
        text = doc._.markdown
        return text
    except Exception as e:
        logger.warning(f"No text extracted from {file_path}")
        return None