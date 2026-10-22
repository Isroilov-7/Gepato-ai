"""Patients route"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from db.database import get_db
from db import models

router = APIRouter()


class PatientCreate(BaseModel):
    full_name: str
    age: int
    gender: str = "male"
    diagnosis: str = "Liver cirrhosis"
    child_pugh_class: str
    bilirubin: Optional[float] = None
    albumin: Optional[float] = None
    inr: Optional[float] = None
    lactulose_dose: Optional[str] = None
    ammonia_level: Optional[float] = None
    doctor_id: Optional[int] = None
    notes: Optional[str] = None


@router.post("/", status_code=201)
async def create_patient(body: PatientCreate, db: AsyncSession = Depends(get_db)):
    patient = models.Patient(**body.model_dump())
    db.add(patient)
    await db.commit()
    await db.refresh(patient)
    return {"id": patient.id, "name": patient.full_name}


@router.get("/")
async def list_patients(
    risk_level: Optional[str] = None,
    doctor_id: Optional[int] = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    q = select(models.Patient).where(models.Patient.is_active == True)
    if risk_level:
        q = q.where(models.Patient.current_risk_level == risk_level)
    if doctor_id:
        q = q.where(models.Patient.doctor_id == doctor_id)
    q = q.order_by(models.Patient.last_risk_score.desc().nullslast()).limit(limit)
    result = await db.execute(q)
    patients = result.scalars().all()
    return [_patient_summary(p) for p in patients]


@router.get("/{patient_id}")
async def get_patient(patient_id: int, db: AsyncSession = Depends(get_db)):
    p = await db.get(models.Patient, patient_id)
    if not p:
        raise HTTPException(404)
    return _patient_detail(p)


@router.patch("/{patient_id}")
async def update_patient(patient_id: int, body: dict, db: AsyncSession = Depends(get_db)):
    p = await db.get(models.Patient, patient_id)
    if not p:
        raise HTTPException(404)
    for k, v in body.items():
        if hasattr(p, k):
            setattr(p, k, v)
    await db.commit()
    return {"success": True}


def _patient_summary(p: models.Patient) -> dict:
    return {
        "id": p.id,
        "name": p.full_name,
        "age": p.age,
        "child_pugh_class": p.child_pugh_class.value if p.child_pugh_class else None,
        "risk_level": p.current_risk_level.value if p.current_risk_level else "low",
        "risk_score": p.last_risk_score,
        "last_assessment": p.last_assessment.isoformat() if p.last_assessment else None,
        "baseline_computed": p.baseline_computed,
    }


def _patient_detail(p: models.Patient) -> dict:
    d = _patient_summary(p)
    d.update({
        "diagnosis": p.diagnosis,
        "bilirubin": p.bilirubin,
        "albumin": p.albumin,
        "inr": p.inr,
        "lactulose_dose": p.lactulose_dose,
        "ammonia_level": p.ammonia_level,
        "rifaximin": p.rifaximin,
        "encephalopathy_grade": p.encephalopathy_grade,
        "notes": p.notes,
        "enrolled_at": p.enrolled_at.isoformat() if p.enrolled_at else None,
    })
    return d
