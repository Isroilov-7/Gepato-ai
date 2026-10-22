"""
Gepato-AI — Voice API
POST /api/voice/upload   → fayl yuklash + feature extraction
POST /api/voice/analyze  → bashorat + SHAP + alert yaratish
GET  /api/voice/{patient_id}/history → 7 kunlik trend
"""
import os
import uuid
import asyncio
import aiofiles
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from db.database import get_db
from db import models
from ml.feature_extraction import get_extractor, QualityError
from ml.model import get_model

router = APIRouter()

ALLOWED_TYPES = {
    "audio/wav", "audio/wave", "audio/x-wav",
    "audio/mpeg", "audio/mp3",
    "audio/ogg", "audio/webm", "audio/m4a",
    "application/octet-stream",  # ba'zi brauzerlar shu yuboradi
}
UPLOAD_DIR = Path(settings.UPLOAD_DIR)
UPLOAD_DIR.mkdir(exist_ok=True)


# ── Schemas ─────────────────────────────────────────────

class UploadResponse(BaseModel):
    recording_id: int
    quality: str
    quality_notes: str
    duration: float
    snr_db: float
    features_ready: bool


class AnalyzeResponse(BaseModel):
    recording_id: int
    patient_id: int
    risk_score: float
    risk_level: str
    confidence: float
    top_factors: list
    explanation: str
    trend_direction: str
    trend_magnitude: float
    needs_labeling: bool
    alert_created: bool
    model_version: str


class LabelRequest(BaseModel):
    label: int          # 0=normal, 1=mHE, 2=overt HE
    child_pugh_class: str
    notes: str = ""


# ── Endpoints ───────────────────────────────────────────

@router.post("/upload/{patient_id}", response_model=UploadResponse)
async def upload_voice(
    patient_id: int,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    1. Faylni saqlash
    2. Sifat tekshiruvi (SNR, davomiylik)
    3. Feature extraction (fon taskda)
    4. VoiceRecording yozuvi yaratish
    """
    # Hajm tekshiruvi
    content = await file.read()
    if len(content) > settings.MAX_AUDIO_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, f"Fayl hajmi {settings.MAX_AUDIO_SIZE_MB}MB dan oshmasin")

    # Saqlash
    ext = Path(file.filename or "audio.wav").suffix or ".wav"
    filename = f"p{patient_id}_{uuid.uuid4().hex[:8]}{ext}"
    file_path = UPLOAD_DIR / filename

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    # VoiceRecording yozuvi
    recording = models.VoiceRecording(
        patient_id=patient_id,
        file_path=str(file_path),
        file_size=len(content),
        recorded_at=datetime.utcnow(),
    )
    db.add(recording)
    await db.commit()
    await db.refresh(recording)

    # Feature extraction (background)
    background_tasks.add_task(
        _extract_features_bg, recording.id, str(file_path), patient_id
    )

    # Darhol sifat tekshiruvi qaytarish
    try:
        extractor = get_extractor()
        # Tezkor sifat baholash — to'liq extraction emas
        import librosa
        y, sr = librosa.load(str(file_path), sr=16000, mono=True)
        duration = len(y) / sr
        snr = extractor._compute_snr(y, sr)
        quality, notes = extractor._assess_quality(snr, duration)

        recording.duration = duration
        recording.sample_rate = sr
        recording.snr_db = snr
        recording.quality = quality
        recording.quality_notes = notes
        await db.commit()

        return UploadResponse(
            recording_id=recording.id,
            quality=quality,
            quality_notes=notes,
            duration=round(duration, 2),
            snr_db=round(snr, 2),
            features_ready=False,
        )
    except Exception as e:
        return UploadResponse(
            recording_id=recording.id,
            quality="unknown",
            quality_notes=str(e),
            duration=0,
            snr_db=0,
            features_ready=False,
        )


@router.post("/analyze/{recording_id}", response_model=AnalyzeResponse)
async def analyze_recording(
    recording_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    1. AcousticFeatures ni olish (extraction tugagan bo'lishi kerak)
    2. Baseline deviation hisoblash
    3. Model bashorat
    4. Alert yaratish (kerak bo'lsa)
    5. Baseline yangilash (agar yangi bemor)
    """
    from sqlalchemy import select

    # Recording
    rec = await db.get(models.VoiceRecording, recording_id)
    if not rec:
        raise HTTPException(404, "Yozuv topilmadi")

    # Features tayyor?
    if not rec.features_extracted:
        # Kutish (max 30 soniya)
        for _ in range(30):
            await asyncio.sleep(1)
            await db.refresh(rec)
            if rec.features_extracted:
                break
        else:
            raise HTTPException(202, "Feature extraction hali tugamagan. Qayta urinib ko'ring.")

    # Features
    result = await db.execute(
        select(models.AcousticFeatures).where(
            models.AcousticFeatures.recording_id == recording_id
        )
    )
    feat = result.scalar_one_or_none()
    if not feat:
        raise HTTPException(500, "Feature extraction xatosi")

    # Patient va baseline
    patient = await db.get(models.Patient, rec.patient_id)

    baseline = {}
    if patient.baseline_computed:
        baseline = {
            "jitter_local":  patient.baseline_jitter,
            "shimmer_local": patient.baseline_shimmer,
            "hnr":           patient.baseline_hnr,
        }

    # Deviation
    extractor = get_extractor()
    feat_dict = _acoustic_to_dict(feat)
    deviation = extractor.compute_deviation(
        type("F", (), feat_dict)(), baseline
    ) if baseline else {}
    feat_dict.update(deviation)

    # Oxirgi 7 kun skori (trend uchun)
    history_result = await db.execute(
        select(models.Prediction.risk_score)
        .where(models.Prediction.patient_id == rec.patient_id)
        .order_by(models.Prediction.predicted_at.desc())
        .limit(7)
    )
    history = [row[0] for row in history_result.fetchall()]

    # Bashorat
    model = get_model()
    pred_result = model.predict(feat_dict, history)

    # Prediction saqlash
    prediction = models.Prediction(
        patient_id=rec.patient_id,
        recording_id=recording_id,
        risk_score=pred_result.risk_score,
        risk_level=pred_result.risk_level,
        confidence=pred_result.confidence,
        shap_values=pred_result.shap_values,
        top_factors=pred_result.top_factors,
        explanation=pred_result.explanation,
        trend_direction=pred_result.trend_direction,
        trend_magnitude=pred_result.trend_magnitude,
        model_version=pred_result.model_version,
    )
    db.add(prediction)

    # Patient risk yangilash
    from db.models import RiskLevel
    patient.current_risk_level = RiskLevel(pred_result.risk_level)
    patient.last_risk_score = pred_result.risk_score
    patient.last_assessment = datetime.utcnow()

    await db.commit()
    await db.refresh(prediction)

    # Alert (fon taskda)
    alert_created = False
    if pred_result.risk_level in ("high", "medium"):
        background_tasks.add_task(
            _create_alert_bg, prediction.id, rec.patient_id, patient.doctor_id,
            pred_result
        )
        alert_created = True

    # Baseline yangilash (birinchi 7 yozuv bo'lsa)
    if not patient.baseline_computed:
        background_tasks.add_task(
            _update_baseline_bg, rec.patient_id
        )

    return AnalyzeResponse(
        recording_id=recording_id,
        patient_id=rec.patient_id,
        risk_score=pred_result.risk_score,
        risk_level=pred_result.risk_level,
        confidence=pred_result.confidence,
        top_factors=pred_result.top_factors,
        explanation=pred_result.explanation,
        trend_direction=pred_result.trend_direction,
        trend_magnitude=pred_result.trend_magnitude,
        needs_labeling=pred_result.needs_labeling,
        alert_created=alert_created,
        model_version=pred_result.model_version,
    )


@router.post("/{recording_id}/label")
async def label_recording(
    recording_id: int,
    body: LabelRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Shifokor label beradi — Active Learning uchun.
    Bu endpoint modelni qayta o'qitish uchun ma'lumot yig'adi.
    """
    from sqlalchemy import select, func

    feat_result = await db.execute(
        select(models.AcousticFeatures).where(
            models.AcousticFeatures.recording_id == recording_id
        )
    )
    feat = feat_result.scalar_one_or_none()
    if not feat:
        raise HTTPException(404, "Topilmadi")

    feat.is_labeled = True
    feat.label = body.label
    feat.child_pugh_at_time = body.child_pugh_class
    feat.labeled_at = datetime.utcnow()
    await db.commit()

    # Yangi labeled sample soni
    count_result = await db.execute(
        select(func.count()).select_from(models.AcousticFeatures)
        .where(
            models.AcousticFeatures.is_labeled == True,
            models.AcousticFeatures.used_in_training == False,
        )
    )
    new_labeled = count_result.scalar()

    # Retrain threshold yetdimi?
    should_retrain = new_labeled >= settings.MODEL_RETRAIN_THRESHOLD

    return {
        "success": True,
        "new_labeled_count": new_labeled,
        "retrain_triggered": should_retrain,
        "message": (
            "Retrain navbatga qo'shildi!" if should_retrain
            else f"Label saqlandi. Retrain uchun {settings.MODEL_RETRAIN_THRESHOLD - new_labeled} ta ko'proq kerak."
        )
    }


@router.get("/{patient_id}/history")
async def get_voice_history(
    patient_id: int,
    days: int = 7,
    db: AsyncSession = Depends(get_db),
):
    """Bemor ovoz tarixini qaytaradi (grafik uchun)."""
    from sqlalchemy import select
    from datetime import timedelta

    since = datetime.utcnow() - timedelta(days=days)

    result = await db.execute(
        select(models.AcousticFeatures, models.VoiceRecording.recorded_at)
        .join(models.VoiceRecording,
              models.AcousticFeatures.recording_id == models.VoiceRecording.id)
        .where(
            models.AcousticFeatures.patient_id == patient_id,
            models.VoiceRecording.recorded_at >= since,
        )
        .order_by(models.VoiceRecording.recorded_at)
    )

    rows = result.fetchall()
    return {
        "patient_id": patient_id,
        "days": days,
        "data": [
            {
                "date": row[1].strftime("%Y-%m-%d"),
                "jitter": row[0].jitter_local,
                "shimmer": row[0].shimmer_local,
                "hnr": row[0].hnr,
                "speech_rate": row[0].speech_rate,
                "composite_deviation": row[0].composite_deviation,
            }
            for row in rows
        ]
    }


# ── Background Tasks ────────────────────────────────────

async def _extract_features_bg(recording_id: int, file_path: str, patient_id: int):
    """Fon rejimida feature extraction."""
    from db.database import async_session
    from sqlalchemy import select

    async with async_session() as db:
        try:
            extractor = get_extractor()
            result = extractor.extract(file_path)

            # Patient baseline
            patient = await db.get(models.Patient, patient_id)
            baseline = {}
            if patient and patient.baseline_computed:
                baseline = {
                    "jitter_local":  patient.baseline_jitter,
                    "shimmer_local": patient.baseline_shimmer,
                    "hnr":           patient.baseline_hnr,
                }

            # Deviation hisoblash
            result_dict = {
                "jitter_local": result.jitter_local,
                "jitter_ppq5": result.jitter_ppq5,
                "shimmer_local": result.shimmer_local,
                "shimmer_apq11": result.shimmer_apq11,
                "hnr": result.hnr,
                "speech_rate": result.speech_rate,
                "tremor_index": result.tremor_index,
                "f0_mean": result.f0_mean,
                "f0_std": result.f0_std,
                "f0_range": result.f0_range,
                "pause_rate": result.pause_rate,
                "voiced_fraction": result.voiced_fraction,
            }
            deviation = extractor.compute_deviation(result, baseline)

            # AcousticFeatures saqlash
            feat = models.AcousticFeatures(
                recording_id=recording_id,
                patient_id=patient_id,
                extracted_at=datetime.utcnow(),
                **result_dict,
                **deviation,
                mfcc=result.mfcc,
                mfcc_delta=result.mfcc_delta,
            )
            db.add(feat)

            # Recording yangilash
            rec = await db.get(models.VoiceRecording, recording_id)
            rec.features_extracted = True
            rec.snr_db = result.snr_db
            rec.duration = result.duration
            rec.quality = result.quality
            rec.quality_notes = result.quality_notes

            await db.commit()

        except QualityError as e:
            async with async_session() as db2:
                rec = await db2.get(models.VoiceRecording, recording_id)
                if rec:
                    rec.features_extracted = True
                    rec.quality = "failed"
                    rec.quality_notes = str(e)
                    rec.extraction_error = str(e)
                    await db2.commit()
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"BG extraction xatosi: {e}")


async def _create_alert_bg(prediction_id, patient_id, doctor_id, pred):
    """Alert yaratish."""
    from db.database import async_session

    if not doctor_id:
        return

    async with async_session() as db:
        patient = await db.get(models.Patient, patient_id)
        name = patient.full_name if patient else f"Bemor #{patient_id}"

        severity = "critical" if pred.risk_level == "high" else "warning"
        top = pred.top_factors[0] if pred.top_factors else {}

        alert = models.Alert(
            patient_id=patient_id,
            prediction_id=prediction_id,
            doctor_id=doctor_id,
            title=f"{'🚨 Kritik' if severity == 'critical' else '⚠️ Ogohlantirish'}: {name}",
            message=(
                f"AI xavf skori: {pred.risk_score:.1%}. "
                f"{pred.explanation} "
                f"Trend: {pred.trend_direction}."
            ),
            severity=severity,
            trigger_type="ml_prediction",
        )
        db.add(alert)
        await db.commit()


async def _update_baseline_bg(patient_id: int):
    """Birinchi 7 yozuvdan baseline hisoblash."""
    from db.database import async_session
    from sqlalchemy import select

    async with async_session() as db:
        result = await db.execute(
            select(models.AcousticFeatures)
            .where(models.AcousticFeatures.patient_id == patient_id)
            .order_by(models.AcousticFeatures.extracted_at)
            .limit(7)
        )
        feats = result.scalars().all()

        if len(feats) < 7:
            return

        import numpy as np
        patient = await db.get(models.Patient, patient_id)
        patient.baseline_jitter   = float(np.median([f.jitter_local  for f in feats if f.jitter_local]))
        patient.baseline_shimmer  = float(np.median([f.shimmer_local for f in feats if f.shimmer_local]))
        patient.baseline_hnr      = float(np.median([f.hnr           for f in feats if f.hnr]))
        patient.baseline_computed = True
        patient.baseline_date     = datetime.utcnow()
        await db.commit()


def _acoustic_to_dict(feat: models.AcousticFeatures) -> dict:
    return {
        "jitter_local": feat.jitter_local or 0,
        "jitter_ppq5": feat.jitter_ppq5 or 0,
        "shimmer_local": feat.shimmer_local or 0,
        "shimmer_apq11": feat.shimmer_apq11 or 0,
        "hnr": feat.hnr or 0,
        "speech_rate": feat.speech_rate or 0,
        "tremor_index": feat.tremor_index or 0,
        "f0_mean": feat.f0_mean or 0,
        "f0_std": feat.f0_std or 0,
        "f0_range": feat.f0_range or 0,
        "pause_rate": feat.pause_rate or 0,
        "voiced_fraction": feat.voiced_fraction or 0,
        "mfcc": feat.mfcc or [0] * 13,
        "jitter_deviation": feat.jitter_deviation or 0,
        "shimmer_deviation": feat.shimmer_deviation or 0,
        "hnr_deviation": feat.hnr_deviation or 0,
        "composite_deviation": feat.composite_deviation or 0,
    }
