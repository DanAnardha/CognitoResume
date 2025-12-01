# backend/app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import logging
from backend.app.database import Base, engine
from . import models
from backend.app.api.vacancies import router as vacancies_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Cognitive Resume Screening API")
templates = Jinja2Templates(directory="backend/app/templates")
app.mount("/static", StaticFiles(directory="backend/app/static"), name="static")

# Unneeded since runing the server with jinja2
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

@app.on_event("startup")
def on_startup():
    logging.info("[APP] Starting FastAPI application...")
    logging.info("[DB] Creating all database tables if not present...")
    Base.metadata.create_all(bind=engine)
    logging.info("[DB] Database ready.")

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Job Portal API is running"}

# load routers
app.include_router(vacancies_router, prefix="/vacancies", tags=["Vacancies"])

logging.info("[APP] Routers loaded.")