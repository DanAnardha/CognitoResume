# backend/app/schemas.py

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import date, datetime

class VacancyCreate(BaseModel):
    job_position: str = Field(..., min_length=1, max_length=255)
    job_description: str = Field(..., min_length=1)
    department: str = Field(..., min_length=1, max_length=100)
    responsibilities: List[str] = Field(...)
    education_requirements: str = Field(...)
    certification_requirements: Optional[List[str]] = None
    required_skills: List[str] = Field(...)
    optional_skills: Optional[List[str]] = None
    experience_level: str = Field(...)
    min_years_experience: Optional[int] = Field(None, ge=0)
    job_location: str = Field(...)
    work_type: str = Field(...)
    salary_min: int = Field(..., gt=0)
    salary_max: int = Field(..., gt=0)
    application_deadline: date
    max_applicants: Optional[int] = Field(None, ge=0)
    status: str = "active"
    @validator('salary_max')
    def salary_must_be_greater_than_min(cls, v, values):
        if 'salary_min' in values and v <= values['salary_min']:
            raise ValueError('salary_max must be greater than salary_min')
        return v
    class Config:
        orm_mode = True


class WeightCreate(BaseModel):
    vacancy_id: str
    education_weight: float
    experience_weight: float
    projects_weight: float
    certifications_weight: float
    required_skills_weight: float
    optional_skills_weight: float
    class Config:
        orm_mode = True


class VacancyResponse(BaseModel):
    id: str
    job_position: str
    job_description: str
    department: str
    responsibilities: List[str]
    certifications: Optional[List[str]]
    required_skills: List[str]
    optional_skills: Optional[List[str]]
    education_requirements: Optional[str]
    experience_level: str
    min_years_experience: Optional[int]
    job_location: str
    work_type: str
    salary_min: int
    salary_max: int
    status: str
    application_deadline: date
    posting_date: Optional[datetime]
    class Config:
        orm_mode = True