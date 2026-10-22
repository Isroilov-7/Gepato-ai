"""Predictions route"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from db.database import get_db
from db import models

router = APIRouter()


@router.get("/labeling-queue")
async def labeling_queue(limit: int = 50, db: AsyncSession = Depends(get_db)):
    """Shifokor belgilash navbati (frontend/doctor_labeling_ui.html).
    Belgilanmagan yozuvlar, eng noaniq bashorat (0.5 ga yaqin) birinchi — Active Learning.
    `id` = recording_id: belgi POST /api/voice/{id}/label ga yuboriladi.
    (Ilgari bu endpoint yo'q edi — sahifa doim namunaviy ma'lumot ko'rsatardi.)"""
    from sqlalchemy import func as F
    from ml.feature_extraction import STRATA_NORMS
    q = (select(models.Prediction, models.AcousticFeatures, models.Patient)
         .join(models.AcousticFeatures, models.AcousticFeatures.recording_id == models.Prediction.recording_id)
         .join(models.Patient, models.Patient.id == models.Prediction.patient_id)
         .where(models.AcousticFeatures.is_labeled == False)  # noqa: E712
         .order_by(F.abs(models.Prediction.risk_score - 0.5))
         .limit(limit))
    rows = (await db.execute(q)).all()
    items = []
    for pred, feat, pt in rows:
        extra = feat.extra_features or {}
        raw = {k: (getattr(feat, k, None) if getattr(feat, k, None) is not None else extra.get(k))
               for k in ("jitter_local", "shimmer_local", "hnr", "speech_rate", "nhr", "dvb",
                         "f0_mean", "tremor_index", "pause_rate")}
        norms = STRATA_NORMS["unknown"]
        dev = {k: round((float(raw[k]) - m) / (sd or 1), 2)
               for k, (m, sd) in norms.items() if raw.get(k) is not None}
        hist = (await db.execute(
            select(models.Prediction.risk_score).where(models.Prediction.patient_id == pt.id)
            .order_by(models.Prediction.predicted_at.desc()).limit(7))).scalars().all()[::-1]
        slope = round((hist[-1] - hist[0]) / max(len(hist) - 1, 1), 3) if len(hist) > 1 else 0.0
        shap_top = dict(sorted((pred.shap_values or {}).items(), key=lambda x: abs(x[1]), reverse=True)[:5])
        items.append({
            "id": feat.recording_id, "prediction_id": pred.id, "patient_id": pt.id,
            "name": pt.full_name, "age": pt.age, "stage": "unknown",
            "child_pugh": pt.child_pugh_class.value if pt.child_pugh_class else "B",
            "date": pred.predicted_at.strftime("%Y-%m-%d") if pred.predicted_at else "",
            "he_index": pred.risk_score, "model_score": pred.risk_score,
            "confidence": pred.confidence or 0.0,
            "risk_level": pred.risk_level.value if pred.risk_level else "low", "label": None,
            "features": {k: (round(float(v), 3) if v is not None else 0) for k, v in raw.items()},
            "deviations": dev, "shap": shap_top, "trend_7d": hist, "trend_slope": slope,
        })
    return {"items": items, "count": len(items)}


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
