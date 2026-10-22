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
    # Multimodal HE indeks (ovoz + oxirgi 7 kundagi tremor va kognitiv test)
    he_index: float | None = None
    he_risk_level: str | None = None
    he_components: dict | None = None


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

    # Darhol sifat tekshiruvi (P1 AudioQualityPipeline). Ilgari bu yerda mavjud bo'lmagan
    # extractor._compute_snr chaqirilardi -> har yuklash "unknown" va tahlil hech qachon tugamasdi.
    from ml.audio_quality import get_audio_quality_pipeline
    try:
        rep = await asyncio.to_thread(
            get_audio_quality_pipeline().process, str(file_path), "sustained_vowel",
            settings.MIN_AUDIO_DURATION_SEC, settings.MAX_AUDIO_DURATION_SEC,
        )
        passed, snr, duration = rep.passed, rep.ambient_snr_db, rep.final_duration_sec
        notes = rep.block_reason or ("; ".join(rep.warnings) if rep.warnings else "OK")
    except Exception as e:  # sifat moduli xatosi yozuvni to'xtatmasin
        passed, snr, duration, notes = True, 0.0, 0.0, f"Sifat tekshiruvi o'tkazilmadi: {e}"

    quality = models.AudioQuality.GOOD if passed and notes == "OK" else (
        models.AudioQuality.POOR if passed else models.AudioQuality.FAILED)
    recording.duration = duration
    recording.sample_rate = 16000
    recording.snr_db = snr
    recording.quality = quality
    recording.quality_notes = notes[:500]
    if not passed:
        recording.features_extracted = True          # tahlil kutib qolmasin — darhol sababini aytadi
        recording.extraction_error = notes[:500]
    await db.commit()

    if passed:
        background_tasks.add_task(_extract_features_bg, recording.id, str(file_path), patient_id)

    return UploadResponse(
        recording_id=recording.id,
        quality=quality.value,
        quality_notes=notes,
        duration=round(duration, 2),
        snr_db=round(snr, 2),
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

    if rec.extraction_error:
        raise HTTPException(422, f"Yozuv tahlil qilinmadi: {rec.extraction_error}")

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
    feat_dict = {**(feat.extra_features or {}), **_acoustic_to_dict(feat)}
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
    history = [row[0] for row in history_result.fetchall()][::-1]   # eskidan yangiga (_trend shuni kutadi)

    # Bashorat. Ilgari predict(feat_dict, history) — history disease_stage o'rniga tushib,
    # trend hech qachon hisoblanmasdi.
    model = get_model()
    stage = getattr(patient, "disease_stage", None) or "unknown"
    pred_result = model.predict(feat_dict, disease_stage=stage, history=history)

    he = await _he_index(db, rec.patient_id, pred_result.risk_score, stage)

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

    if he:  # multimodal indeksni ham saqlaymiz (006-migratsiya ustunlari)
        from sqlalchemy import text as _t
        try:
            await db.execute(_t(
                "UPDATE predictions SET composite_he_index=:h, voice_component=:v, "
                "tremor_component=:t, cognitive_component=:c WHERE id=:i"),
                {"h": he["he_index"], "v": he["components"]["voice"], "t": he["components"]["tremor"],
                 "c": he["components"]["cognitive"], "i": prediction.id})
            await db.commit()
        except Exception:
            await db.rollback()

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
        he_index=he.get("he_index") if he else None,
        he_risk_level=he.get("risk_level") if he else None,
        he_components=he.get("components") if he else None,
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

            # Patient baseline
            patient = await db.get(models.Patient, patient_id)
            baseline = {}
            if patient and patient.baseline_computed:
                baseline = {
                    "jitter_local":  patient.baseline_jitter,
                    "shimmer_local": patient.baseline_shimmer,
                    "hnr":           patient.baseline_hnr,
                }
            stage = (getattr(patient, "disease_stage", None) if patient else None) or "unknown"
            # Ilgari extractor.extract(file_path) — extract() audio massiv kutadi -> har doim xato.
            # Praat/librosa CPU-og'ir: event loop'ni bloklamasin.
            result = await asyncio.to_thread(
                extractor.extract_from_file, file_path, "sustained_vowel", stage, baseline or None)

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

            feat.extra_features = {k: v for k, v in result.to_dict().items()
                                   if isinstance(v, (int, float)) and not isinstance(v, bool)}

            # Recording yangilash (snr/duration/quality upload paytida yozilgan)
            rec = await db.get(models.VoiceRecording, recording_id)
            rec.features_extracted = True

            await db.commit()

        except QualityError as e:
            async with async_session() as db2:
                rec = await db2.get(models.VoiceRecording, recording_id)
                if rec:
                    rec.features_extracted = True
                    rec.quality = models.AudioQuality.FAILED
                    rec.quality_notes = str(e)
                    rec.extraction_error = str(e)
                    await db2.commit()
        except Exception as e:
            import logging
            logging.getLogger(__name__).exception("BG extraction xatosi: %s", e)
            # Tahlil 30 soniya behuda kutmasin — xatoni yozuvga belgilaymiz
            async with async_session() as db2:
                rec = await db2.get(models.VoiceRecording, recording_id)
                if rec:
                    rec.features_extracted = True
                    rec.quality = models.AudioQuality.FAILED
                    rec.extraction_error = f"Feature extraction xatosi: {e}"[:500]
                    await db2.commit()


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


async def _he_index(db: AsyncSession, patient_id: int, voice_score: float, stage: str) -> dict:
    """Ovoz + oxirgi 7 kundagi tremor va kognitiv natijadan HEScoreEngine indeksi.
    Kognitiv ball 'yuqori = yaxshi' (1 = normal), engine esa xavf kutadi -> 1 - ball."""
    from sqlalchemy import text
    from ml.scoring import get_scoring_engine
    try:
        t = (await db.execute(text(
            "SELECT tremor_score FROM tremor_readings WHERE patient_id=:p "
            "AND recorded_at >= now() - interval '7 days' ORDER BY recorded_at DESC LIMIT 1"),
            {"p": patient_id})).scalar()
        c = (await db.execute(text(
            "SELECT cognitive_score FROM cognitive_tests WHERE patient_id=:p "
            "AND test_date >= now() - interval '7 days' ORDER BY test_date DESC LIMIT 1"),
            {"p": patient_id})).scalar()
        r = get_scoring_engine().compute(
            voice_score=voice_score,
            tremor_score=float(t) if t is not None else None,
            cognitive_score=(1.0 - float(c)) if c is not None else None,
            disease_stage=stage,
        )
        return {"he_index": r.total_he_index, "risk_level": r.risk_level,
                "components": {"voice": r.voice_component, "tremor": r.tremor_component,
                               "cognitive": r.cognitive_component}}
    except Exception as e:  # indeks qo'shimcha — ovoz bashoratini to'xtatmasin
        import logging
        logging.getLogger(__name__).warning("HE indeks hisoblanmadi: %s", e)
        return {}


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
