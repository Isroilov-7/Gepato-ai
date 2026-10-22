"""Admin route"""
from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from db.database import get_db
from db import models

router = APIRouter()


@router.get("/stats")
async def system_stats(db: AsyncSession = Depends(get_db)):
    patients_total = (await db.execute(select(func.count()).select_from(models.Patient))).scalar()
    recordings_total = (await db.execute(select(func.count()).select_from(models.VoiceRecording))).scalar()
    labeled_total = (await db.execute(
        select(func.count()).select_from(models.AcousticFeatures)
        .where(models.AcousticFeatures.is_labeled == True)
    )).scalar()
    alerts_pending = (await db.execute(
        select(func.count()).select_from(models.Alert)
        .where(models.Alert.status == models.AlertStatus.PENDING)
    )).scalar()
    high_risk = (await db.execute(
        select(func.count()).select_from(models.Patient)
        .where(models.Patient.current_risk_level == models.RiskLevel.HIGH)
    )).scalar()

    # Active model
    model_result = await db.execute(
        select(models.ModelVersion).where(models.ModelVersion.is_active == True)
    )
    active_model = model_result.scalar_one_or_none()

    return {
        "patients_total": patients_total,
        "recordings_total": recordings_total,
        "labeled_total": labeled_total,
        "alerts_pending": alerts_pending,
        "high_risk_patients": high_risk,
        "active_model": {
            "version": active_model.version if active_model else "v1.0-pretrain",
            "auc_roc": active_model.auc_roc if active_model else None,
            "sensitivity": active_model.sensitivity if active_model else None,
            "train_samples": active_model.train_samples if active_model else None,
        }
    }


@router.get("/alerts")
async def get_alerts(status: str = None, limit: int = 50, db: AsyncSession = Depends(get_db)):
    q = select(models.Alert).order_by(models.Alert.created_at.desc()).limit(limit)
    if status:
        q = q.where(models.Alert.status == status)
    result = await db.execute(q)
    alerts = result.scalars().all()
    return [
        {
            "id": a.id,
            "patient_id": a.patient_id,
            "title": a.title,
            "message": a.message,
            "severity": a.severity,
            "status": a.status.value,
            "created_at": a.created_at.isoformat(),
        }
        for a in alerts
    ]


@router.post("/model/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    """Manual retrain triggeri."""
    from core.tasks import nightly_retrain
    background_tasks.add_task(nightly_retrain)
    return {"message": "Retrain navbatga qo'shildi"}


@router.get("/model/versions")
async def model_versions(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(models.ModelVersion).order_by(models.ModelVersion.created_at.desc())
    )
    versions = result.scalars().all()
    return [
        {
            "version": v.version,
            "is_active": v.is_active,
            "auc_roc": v.auc_roc,
            "sensitivity": v.sensitivity,
            "train_samples": v.train_samples,
            "created_at": v.created_at.isoformat() if v.created_at else None,
        }
        for v in versions
    ]
