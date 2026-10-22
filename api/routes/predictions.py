"""Predictions route"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from db.database import get_db
from db import models

router = APIRouter()


@router.get("/{patient_id}")
async def get_predictions(patient_id: int, limit: int = 30, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(models.Prediction)
        .where(models.Prediction.patient_id == patient_id)
        .order_by(models.Prediction.predicted_at.desc())
        .limit(limit)
    )
    preds = result.scalars().all()
    return [
        {
            "id": p.id,
            "risk_score": p.risk_score,
            "risk_level": p.risk_level.value if p.risk_level else None,
            "confidence": p.confidence,
            "trend_direction": p.trend_direction,
            "top_factors": p.top_factors,
            "explanation": p.explanation,
            "predicted_at": p.predicted_at.isoformat(),
            "doctor_confirmed": p.doctor_confirmed,
            "model_version": p.model_version,
        }
        for p in preds
    ]
