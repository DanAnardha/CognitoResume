# backend/celery/tasks.py

import logging
from backend.app.services.extractor import extract_text
from backend.app.database import SessionLocal
from backend.app.crud import get_candidate, update_extraction, set_status
from .celery_app import celery

logger = logging.getLogger(__name__)

@celery.task(name="process_cv")
def process_cv(candidate_id: int):
    logger.info(f"[TASK] Starting CV extraction for candidate {candidate_id}")
    db = SessionLocal()
    try:
        candidate = get_candidate(db, candidate_id)
        if not candidate:
            logger.error(f"[TASK] Candidate with id {candidate_id} not found")
            return {"status": "error", "message": "Candidate not found"}

        filepath = candidate.cv_filepath

        if not filepath:
            logger.error(f"[TASK] No CV filepath for candidate {candidate_id}")
            set_status(db, candidate_id, "extraction_failed")
            return {"status": "error", "message": "No CV filepath"}

        logger.info(f"[TASK] Extracting text from {filepath}...")
        text = extract_text(filepath)

        logger.info(f"[TASK] Updating DB for candidate {candidate_id}...")
        update_extraction(db, candidate_id, text)
        set_status(db, candidate_id, "extraction_success")

        return {"status": "success", "candidate_id": candidate_id}

    except Exception as e:
        logger.error(f"[TASK] Extraction error for candidate {candidate_id}: {e}", exc_info=True)
        set_status(db, candidate_id, "extraction_failed")
        return {"status": "error", "message": str(e)}
    finally:
        db.close()