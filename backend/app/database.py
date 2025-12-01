# backend/app/database.py

import logging
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.WARNING)

# Read DB URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/cognitoresume")

logging.info(f"[DB] Using database URL: {DATABASE_URL}")

try:
    engine = create_engine(DATABASE_URL, echo=False)
    logging.info("[DB] PostgreSQL engine created successfully")
except Exception as e:
    logging.error(f"[DB] Failed to create engine: {e}")
    raise

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def get_db():
    db = None
    try:
        logging.info("[DB] Opening new DB session")
        db = SessionLocal()
        yield db
    except SQLAlchemyError as e:
        logging.error(f"[DB] Session failed: {e}")
        raise
    finally:
        if db:
            logging.info("[DB] Closing DB session")
            db.close()
