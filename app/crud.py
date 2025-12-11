# app/crud.py

from sqlalchemy.orm import Session
import logging
from sqlalchemy.exc import SQLAlchemyError
from app.models import Vacancy, VacancyCertifications, VacancyOptionalSkills, VacancyRequiredSkills, VacancyResponsibilities
from app.schemas import VacancyCreate

logger = logging.getLogger(__name__)

# Vacancy CRUD
def create_vacancy(db: Session, vacancy_data: VacancyCreate):
    vacancy = Vacancy(
        job_position = vacancy_data.job_position,
        job_description = vacancy_data.job_description,
        department = vacancy_data.department,
        education_requirements = vacancy_data.education_requirements,
        experience_level = vacancy_data.experience_level,
        min_years_experience = vacancy_data.min_years_experience,
        job_location = vacancy_data.job_location,
        work_type = vacancy_data.work_type,
        salary_min = vacancy_data.salary_min,
        salary_max = vacancy_data.salary_max,
        application_deadline = vacancy_data.application_deadline,
        max_applicants = vacancy_data.max_applicants,
        status = vacancy_data.status
    )
    db.add(vacancy)
    db.flush()

    for item in vacancy_data.responsibilities:
        db.add(VacancyResponsibilities(vacancy_id=vacancy.id, responsibility=item))

    if vacancy_data.certification_requirements:
        for cert in vacancy_data.certification_requirements:
            db.add(VacancyCertifications(vacancy_id=vacancy.id, certification=cert))

    for skill in vacancy_data.required_skills:
        db.add(VacancyRequiredSkills(vacancy_id=vacancy.id, skill=skill))

    if vacancy_data.optional_skills:
        for skill in vacancy_data.optional_skills:
            db.add(VacancyOptionalSkills(vacancy_id=vacancy.id, skill=skill))

    db.commit()
    db.refresh(vacancy)
    return vacancy


def get_vacancy(db: Session, vacancy_id: str):
    return db.query(Vacancy).filter(Vacancy.id == vacancy_id).first()