from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from db.database import get_db
from db import models
from agents.agents import atlas_query, oracle_query

router = APIRouter()


class ChatRequest(BaseModel):
    question: str
    patient_id: Optional[int] = None
    history: Optional[list] = None


@router.post("/atlas")
async def atlas(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    patient_data = None
    if req.patient_id:
        patient_data = await _build_patient_context(req.patient_id, db)
    result = await atlas_query(req.question, patient_data, req.history)
    await _log_agent(db, "atlas", req.question, result, req.patient_id)
    return result


@router.post("/oracle")
async def oracle(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    trend_data = None
    if req.patient_id:
        patient_data = await _build_patient_context(req.patient_id, db)
        scores = await _get_7day_scores(req.patient_id, db)
        trend_data = {"patients": [{"name": patient_data.get("name",""), "scores_7d": scores,
                                     "child_pugh": patient_data.get("child_pugh_class",""),
                                     "trend_direction": patient_data.get("trend_direction","stable"),
                                     "trend_magnitude": 0}]}
    result = await oracle_query(req.question, trend_data, req.history)
    await _log_agent(db, "oracle", req.question, result, req.patient_id)
    return result


async def _build_patient_context(patient_id: int, db: AsyncSession) -> dict:
    patient = await db.get(models.Patient, patient_id)
    if not patient:
        return {}

    # Oxirgi features
    result = await db.execute(
        select(models.AcousticFeatures)
        .join(models.VoiceRecording,
              models.AcousticFeatures.recording_id == models.VoiceRecording.id)
        .where(models.AcousticFeatures.patient_id == patient_id)
        .order_by(models.VoiceRecording.recorded_at.desc())
        .limit(1)
    )
    feat = result.scalar_one_or_none()

    # Oxirgi prediction
    pred_result = await db.execute(
        select(models.Prediction)
        .where(models.Prediction.patient_id == patient_id)
        .order_by(models.Prediction.predicted_at.desc())
        .limit(1)
    )
    pred = pred_result.scalar_one_or_none()

    return {
        "name": patient.full_name,
        "age": patient.age,
        "child_pugh_class": patient.child_pugh_class.value if patient.child_pugh_class else None,
        "risk_score": f"{patient.last_risk_score:.1%}" if patient.last_risk_score else None,
        "jitter_local": feat.jitter_local if feat else None,
        "shimmer_local": feat.shimmer_local if feat else None,
        "hnr": feat.hnr if feat else None,
        "speech_rate": feat.speech_rate if feat else None,
        "tremor_index": feat.tremor_index if feat else None,
        "ammonia": patient.bilirubin,
        "bilirubin": patient.bilirubin,
        "inr": patient.inr,
        "lactulose_dose": patient.lactulose_dose,
        "top_factors": pred.top_factors if pred else [],
        "trend_direction": pred.trend_direction if pred else "stable",
    }


async def _get_7day_scores(patient_id: int, db: AsyncSession) -> list:
    from datetime import datetime, timedelta
    result = await db.execute(
        select(models.Prediction.risk_score)
        .where(
            models.Prediction.patient_id == patient_id,
            models.Prediction.predicted_at >= datetime.utcnow() - timedelta(days=7)
        )
        .order_by(models.Prediction.predicted_at)
    )
    return [row[0] for row in result.fetchall()]


async def _log_agent(db, agent, query, result, patient_id=None):
    log = models.AgentLog(
        agent_name=agent,
        patient_id=patient_id,
        query=query[:1000],
        response=result.get("response","")[:2000],
        tokens_used=result.get("tokens_used", 0),
        latency_ms=result.get("latency_ms", 0),
    )
    db.add(log)
    await db.commit()
