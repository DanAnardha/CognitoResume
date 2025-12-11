# app/api/vacancies.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from fastapi import Request
from typing import Dict, Any, List
from fastapi.templating import Jinja2Templates
from app.database import get_db
from app.crud import create_vacancy, get_vacancy
from app.schemas import VacancyCreate, WeightCreate, VacancyResponse
from app.models import Vacancy, Weight

router = APIRouter(tags=["Vacancies"])
templates = Jinja2Templates(directory="backend/app/templates")

@router.get("/")
def list_vacancies(request: Request, db: Session = Depends(get_db)):
    vacancies = db.query(Vacancy).all()
    return templates.TemplateResponse(
        "vacancies.html",
        {"request": request, "vacancies": vacancies}
    )
    
@router.post("/", response_model=VacancyResponse)
def create_vacancy_api(vacancy_data: VacancyCreate, db: Session = Depends(get_db)):
    try:
        new_vacancy = create_vacancy(db=db, vacancy_data=vacancy_data)
        return new_vacancy.to_dict()
    except Exception as e:
        print(f"Error creating vacancy: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating vacancy: {e}")

@router.get("/add")
def add_vacancy_form(request: Request):
    return templates.TemplateResponse("add_vacancy.html", {
        "request": request
})
    
@router.post("/{vacancy_id}/toggle_status")
def toggle_vacancy_status(vacancy_id: str, db: Session = Depends(get_db)):
    vacancy = db.query(Vacancy).filter(Vacancy.id == vacancy_id).first()
    if not vacancy:
        raise HTTPException(status_code=404, detail="Vacancy not found")
    if vacancy.status.lower() == "active":
        vacancy.status = "inactive"
    else:
        vacancy.status = "active"
    db.commit()
    db.refresh(vacancy)
    return {"id": vacancy.id, "new_status": vacancy.status}

@router.get("/vacancies")
def list_vacancies(request: Request, db=Depends(get_db)):
    vacancies = db.query(Vacancy).all()
    return templates.TemplateResponse(
        "vacancies.html",
        {"request": request, "vacancies": vacancies}
    )

@router.post("/vacancy-weights/", response_model=Dict[str, Any])
def create_vacancy_weights(
    weights: WeightCreate, 
    db: Session = Depends(get_db)
):
    vacancy = db.query(Vacancy).filter(Vacancy.id == weights.vacancy_id).first()
    if not vacancy:
        raise HTTPException(status_code=404, detail="Vacancy not found")
    
    # Create new weights record
    db_weights = Weight(
        vacancy_id=weights.vacancy_id,
        education_weight=weights.education_weight,
        experience_weight=weights.experience_weight,
        projects_weight=weights.projects_weight,
        certifications_weight=weights.certifications_weight,
        required_skills_weight=weights.required_skills_weight,
        optional_skills_weight=weights.optional_skills_weight
    )
    
    db.add(db_weights)
    db.commit()
    db.refresh(db_weights)
    
    return {"id": db_weights.id, "message": "Weights created successfully"}

@router.get("/{public_id}")
def recruiter_get(request: Request, public_id: str, db: Session = Depends(get_db)):
    vacancy = db.query(Vacancy).filter_by(public_id=public_id).first()
    if not vacancy:
        raise HTTPException(404, "Vacancy not found")
    return templates.TemplateResponse(
        "details.html",
        {"request": request, "vacancy": vacancy, "weights": vacancy.weight}
    )