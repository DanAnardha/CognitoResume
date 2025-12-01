# backend/app/models.py

from sqlalchemy import Column, Integer, String, Text, ForeignKey, TIMESTAMP, Sequence
from sqlalchemy.orm import relationship
from backend.app.database import Base
from sqlalchemy.sql import func
from sqlalchemy import event, text

class Vacancy(Base):
    __tablename__ = "vacancy"
    seq_id = Column(Integer, Sequence('job_vacancy_seq', start=1, increment=1), primary_key=True)
    id = Column(String, unique=True, index=True, nullable=False)
    job_position = Column(Text, nullable=False)
    job_description = Column(Text, nullable=False)
    department = Column(Text, nullable=False)
    education_requirements = Column(Text)
    experience_level = Column(Text, nullable=False)
    min_years_experience = Column(Integer)
    job_location = Column(Text, nullable=False)
    work_type = Column(Text, nullable=False)
    salary_min = Column(Integer, nullable=False)
    salary_max = Column(Integer, nullable=False)
    status = Column(Text, default="active")
    application_deadline = Column(TIMESTAMP)
    max_applicants = Column(Integer)
    posting_date = Column(TIMESTAMP, server_default=func.now())

    # Relationships
    candidates = relationship("Candidate", back_populates="vacancy")
    weight = relationship("Weight", back_populates="vacancy", uselist=False)
    responsibilities = relationship("VacancyResponsibilities", cascade="all, delete-orphan", back_populates="vacancy")
    certifications = relationship("VacancyCertifications", cascade="all, delete-orphan", back_populates="vacancy")
    required_skills = relationship("VacancyRequiredSkills", cascade="all, delete-orphan", back_populates="vacancy")
    optional_skills = relationship("VacancyOptionalSkills", cascade="all, delete-orphan", back_populates="vacancy")

    @staticmethod
    def generate_id(mapper, connection, target):
        if not target.id:
            result = connection.execute(text("SELECT nextval('job_vacancy_seq')"))
            seq_number = result.scalar()
            target.id = f"VAC-{seq_number:05d}"
            
    def to_dict(self):
        return {
            "id": self.id,
            "job_position": self.job_position,
            "job_description": self.job_description,
            "department": self.department,

            "responsibilities": [r.responsibility for r in self.responsibilities],
            "certifications": [c.certification for c in self.certifications],
            "required_skills": [s.skill for s in self.required_skills],
            "optional_skills": [s.skill for s in self.optional_skills],

            "education_requirements": self.education_requirements,
            "experience_level": self.experience_level,
            "min_years_experience": self.min_years_experience,
            "job_location": self.job_location,
            "work_type": self.work_type,
            "salary_min": self.salary_min,
            "salary_max": self.salary_max,
            "status": self.status,
            "application_deadline": self.application_deadline,
            "posting_date": self.posting_date,
        }
event.listen(Vacancy, 'before_insert', Vacancy.generate_id)

class VacancyResponsibilities(Base):
    __tablename__ = "vacancy_responsibilities"
    id = Column(Integer, primary_key=True, autoincrement=True)
    vacancy_id = Column(String, ForeignKey("vacancy.id"), nullable=False)
    responsibility = Column(Text, nullable=False)
    vacancy = relationship("Vacancy", back_populates="responsibilities")

class VacancyCertifications(Base):
    __tablename__ = "vacancy_certifications"
    id = Column(Integer, primary_key=True, autoincrement=True)
    vacancy_id = Column(String, ForeignKey("vacancy.id"), nullable=False)
    certification = Column(Text, nullable=False)
    vacancy = relationship("Vacancy", back_populates="certifications")

class VacancyRequiredSkills(Base):
    __tablename__ = "vacancy_required_skills"
    id = Column(Integer, primary_key=True, autoincrement=True)
    vacancy_id = Column(String, ForeignKey("vacancy.id"), nullable=False)
    skill = Column(Text, nullable=False)
    vacancy = relationship("Vacancy", back_populates="required_skills")

class VacancyOptionalSkills(Base):
    __tablename__ = "vacancy_optional_skills"
    id = Column(Integer, primary_key=True, autoincrement=True)
    vacancy_id = Column(String, ForeignKey("vacancy.id"), nullable=False)
    skill = Column(Text, nullable=False)
    vacancy = relationship("Vacancy", back_populates="optional_skills")


class Weight(Base):
    __tablename__ = "weight"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    vacancy_id = Column(String, ForeignKey("vacancy.id"), unique=True, nullable=False)
    education_weight = Column(Integer, nullable=False, default=15)
    experience_weight = Column(Integer, nullable=False, default=25)
    projects_weight = Column(Integer, nullable=False, default=20)
    certifications_weight = Column(Integer, nullable=False, default=10)
    required_skills_weight = Column(Integer, nullable=False, default=30)
    optional_skills_weight = Column(Integer, nullable=False, default=0)
    vacancy = relationship("Vacancy", back_populates="weight")


class Candidate(Base):
    __tablename__ = "candidate"
    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(Text, nullable=False)
    email = Column(Text, nullable=False)
    phone = Column(Text)
    portfolio_link = Column(Text)
    vacancy_id = Column(String, ForeignKey("vacancy.id"))
    experience_years = Column(Text)
    cv_filepath = Column(Text)
    expected_salary = Column(Integer)
    additional_info = Column(Text)
    status = Column(Text, default="pending")
    extracted_text = Column(Text)
    submission_date = Column(TIMESTAMP, server_default=func.now())
    vacancy = relationship("Vacancy", back_populates="candidates")