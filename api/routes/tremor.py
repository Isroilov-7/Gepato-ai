"""
api/routes/tremor.py
Gepato-AI v2.0 — P4: Tremor / Akselerometr Endpoint

POST /api/tremor/{patient_id}
  Body: {samples: [{x,y,z,t}], duration_sec: float, disease_stage?: str}

Pipeline:
  1. Sample validation
  2. Feature extraction (RMS, dominant_freq, tremor_power_8_12hz, stability_score)
  3. Etiology-adjusted normalization → tremor_score 0–1
  4. DB save (tremor_readings table)
  5. HEScoreEngine ga uzatish uchun tremor_score qaytarish

DB: migration 004 → tremor_readings table (master prompt Q7)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_db
from ml.scoring import get_scoring_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tremor", tags=["tremor"])

# ── Etiology-adjusted normalizatsiya parametrlari ─────────────────────────
# post_transplant: immunosuppressant tremor → past threshold
# cirrhosis_decompensated: yomon holat → past threshold
# fibrosis_early: sog'lom yaqin → yuqori threshold
ETIOLOGY_SENSITIVITY: dict[str, float] = {
    "post_transplant":         1.35,   # yuqori sezgirlik
    "cirrhosis_decompensated": 1.20,
    "cirrhosis_compensated":   1.00,
    "fibrosis_advanced":       0.90,
    "chronic_hepatitis":       0.85,
    "fibrosis_early":          0.75,   # past sezgirlik
    "unknown":                 1.00,
}

# Tremor score normalizatsiya chegaralari
TREMOR_NORMAL_RMS       = 0.08   # g — sog'lom qo'l titrashi
TREMOR_PATHOLOGIC_RMS   = 0.40   # g — patologik daraja
TREMOR_FREQ_LOW         = 8.0    # Hz — patologik tremor diapazoni boshi
TREMOR_FREQ_HIGH        = 12.0   # Hz — patologik tremor diapazoni oxiri


# ── Pydantic schemas ───────────────────────────────────────────────────────

class AccelSample(BaseModel):
    x: float = Field(default=0.0, description="X-o'q akseleratsi (g)")
    y: float = Field(default=0.0, description="Y-o'q akseleratsi (g)")
    z: float = Field(default=0.0, description="Z-o'q akseleratsi (g)")
    t: float = Field(default=0.0, description="Vaqt (ms, yozuv boshlanishidan)")


class TremorRequest(BaseModel):
    samples:      list[AccelSample] = Field(..., min_length=10)
    duration_sec: float             = Field(..., gt=0.0, le=60.0)
    disease_stage: str              = Field(default="unknown")
    etiology:     str               = Field(default="unknown")

    @field_validator("samples")
    @classmethod
    def check_samples(cls, v):
        if len(v) < 10:
            raise ValueError("Kamida 10 ta sample kerak")
        return v


class TremorFeatures(BaseModel):
    rms_amplitude:      float
    dominant_freq_hz:   float
    tremor_power_8_12hz: float
    stability_score:    float
    tremor_score:       float       # 0.0–1.0, HEScoreEngine uchun


class TremorResponse(BaseModel):
    patient_id:     int
    tremor_score:   float           # HEScoreEngine ga uzatiladigan asosiy qiymat
    features:       TremorFeatures
    risk_level:     str             # "low"|"medium"|"high"|"critical"
    interpretation: str
    sample_count:   int
    sample_rate_hz: float
    saved_id:       Optional[int] = None


# ── Feature extraction ─────────────────────────────────────────────────────

def extract_tremor_features(
    samples:       list[AccelSample],
    duration_sec:  float,
    disease_stage: str = "unknown",
    etiology:      str = "unknown",
) -> TremorFeatures:
    """
    Akselerometr ma'lumotlaridan tremor featurelari.

    Formulalar (master prompt P4):
        magnitude          = sqrt(x² + y² + z²)
        rms_amplitude      = sqrt(mean(magnitude²))
        dominant_freq_hz   = argmax(FFT of z-axis, resample 50Hz)
        tremor_power_8_12  = power[8–12Hz] / total_power
        stability_score    = 1 – std(magnitude) / mean(magnitude)
        tremor_score       = normalize(rms, freq_power) × etiology_sensitivity
    """
    if len(samples) < 2:
        return TremorFeatures(
            rms_amplitude=0.0, dominant_freq_hz=0.0,
            tremor_power_8_12hz=0.0, stability_score=1.0, tremor_score=0.0,
        )

    # Numpy array ga o'tkazish
    arr = np.array([[s.x, s.y, s.z] for s in samples], dtype=np.float32)
    t_ms = np.array([s.t for s in samples], dtype=np.float32)

    # ── Magnitude ─────────────────────────────────────────────────────────
    magnitude = np.sqrt(np.sum(arr ** 2, axis=1))

    # ── RMS amplitude ─────────────────────────────────────────────────────
    rms_amplitude = float(np.sqrt(np.mean(magnitude ** 2)))

    # ── Stability score ───────────────────────────────────────────────────
    mag_mean = float(np.mean(magnitude)) + 1e-10
    mag_std  = float(np.std(magnitude))
    stability_score = float(np.clip(1.0 - mag_std / mag_mean, 0.0, 1.0))

    # ── FFT — z-axis, 50Hz resample ───────────────────────────────────────
    dominant_freq_hz   = 0.0
    tremor_power_8_12  = 0.0

    try:
        z_axis = arr[:, 2]
        sample_rate_hz = len(samples) / max(duration_sec, 0.1)

        # 50 Hz ga resample
        target_sr = 50.0
        n_target  = int(duration_sec * target_sr)
        if n_target >= 4 and len(z_axis) >= 4:
            t_sec = t_ms / 1000.0
            t_uniform = np.linspace(0, duration_sec, n_target)
            z_resampled = np.interp(t_uniform, t_sec, z_axis)

            # FFT
            fft_vals  = np.abs(np.fft.rfft(z_resampled)) ** 2
            fft_freqs = np.fft.rfftfreq(n_target, d=1.0 / target_sr)

            # Dominant frequency (DC component ni o'tkazib yuborish)
            fft_no_dc = fft_vals.copy()
            fft_no_dc[fft_freqs < 0.5] = 0.0
            dominant_freq_hz = float(fft_freqs[np.argmax(fft_no_dc)])

            # Tremor power 8–12 Hz
            mask_tremor = (fft_freqs >= TREMOR_FREQ_LOW) & (fft_freqs <= TREMOR_FREQ_HIGH)
            total_power = float(fft_vals.sum()) + 1e-10
            tremor_power_8_12 = float(fft_vals[mask_tremor].sum() / total_power)

    except Exception as exc:
        logger.warning("Tremor FFT hisoblash xatosi: %s", exc)

    # ── Tremor score (0–1) ────────────────────────────────────────────────
    # RMS component: 0 → normal, 1 → patologik
    rms_component = float(np.clip(
        (rms_amplitude - TREMOR_NORMAL_RMS) / (TREMOR_PATHOLOGIC_RMS - TREMOR_NORMAL_RMS),
        0.0, 1.0,
    ))

    # Frequency component: 8–12 Hz da kuchli signal → patologik
    freq_component = float(np.clip(tremor_power_8_12 * 3.0, 0.0, 1.0))

    # Vazn: 60% RMS, 40% frequency power
    raw_score = 0.60 * rms_component + 0.40 * freq_component

    # Etiology adjustment
    sensitivity = ETIOLOGY_SENSITIVITY.get(
        disease_stage, ETIOLOGY_SENSITIVITY.get(etiology, 1.0)
    )
    tremor_score = float(np.clip(raw_score * sensitivity, 0.0, 1.0))

    logger.info(
        "Tremor features | rms=%.4f freq=%.2fHz power_8_12=%.4f "
        "stability=%.3f score=%.4f stage=%s",
        rms_amplitude, dominant_freq_hz, tremor_power_8_12,
        stability_score, tremor_score, disease_stage,
    )

    return TremorFeatures(
        rms_amplitude=round(rms_amplitude, 5),
        dominant_freq_hz=round(dominant_freq_hz, 2),
        tremor_power_8_12hz=round(tremor_power_8_12, 5),
        stability_score=round(stability_score, 4),
        tremor_score=round(tremor_score, 4),
    )


def _interpret_tremor(score: float, stage: str) -> tuple[str, str]:
    """tremor_score → (risk_level, interpretation)"""
    if score < 0.25:
        return "low",      "Tremor darajasi normal. Qo'l barqaror."
    if score < 0.50:
        return "medium",   "Yengil tremor belgilari. Kuzatuvni davom ettiring."
    if score < 0.75:
        return "high",     "O'rtacha tremor. Shifokor bilan maslahatlashing."
    return "critical",     "Kuchli tremor. Darhol shifokorga murojaat qiling."


# ── DB save helper ─────────────────────────────────────────────────────────

async def _save_tremor_to_db(
    db:            AsyncSession,
    patient_id:    int,
    features:      TremorFeatures,
    disease_stage: str,
    etiology:      str,
    raw_samples:   list[dict],
) -> Optional[int]:
    """
    tremor_readings jadvaliga yozadi.
    Migration 004 (master prompt Q7) talab qiladi.

    Agar jadval mavjud bo'lmasa — xatolik log qilinadi, crash bo'lmaydi.
    """
    try:
        from sqlalchemy import text
        # raw_samples hajmini 50KB ga cheklaymiz
        import json
        raw_json = json.dumps(raw_samples[:500])  # max 500 sample

        result = await db.execute(
            text("""
                INSERT INTO tremor_readings (
                    patient_id, rms_amplitude, dominant_freq_hz,
                    tremor_power_8_12hz, stability_score, tremor_score,
                    disease_stage, etiology, raw_samples
                ) VALUES (
                    :pid, :rms, :freq, :power, :stab, :score,
                    :stage, :etio, :raw
                )
                RETURNING id
            """),
            {
                "pid":   patient_id,
                "rms":   features.rms_amplitude,
                "freq":  features.dominant_freq_hz,
                "power": features.tremor_power_8_12hz,
                "stab":  features.stability_score,
                "score": features.tremor_score,
                "stage": disease_stage,
                "etio":  etiology,
                "raw":   raw_json,
            },
        )
        await db.commit()
        row = result.fetchone()
        return int(row[0]) if row else None

    except Exception as exc:
        logger.error("Tremor DB saqlash xatosi: %s", exc)
        await db.rollback()
        return None


# ── Router ─────────────────────────────────────────────────────────────────

@router.post("/{patient_id}", response_model=TremorResponse)
async def submit_tremor(
    patient_id: int,
    body:       TremorRequest,
    db:         AsyncSession = Depends(get_db),
):
    """
    Bemorning akselerometr ma'lumotlarini qabul qiladi.

    Frontend (patient_daily_flow.html) P4 bosqichida chaqiradi:
        POST /api/tremor/{patient_id}
        {samples: [{x,y,z,t}, ...], duration_sec: 10}

    Javob:
        {tremor_score: 0.0–1.0, features: {...}, risk_level: str}

    tremor_score → HEScoreEngine.compute(tremor_score=...) ga uzatiladi.
    """
    if patient_id <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Noto'g'ri patient_id",
        )

    if body.duration_sec < 1.0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="duration_sec kamida 1 soniya bo'lishi kerak",
        )

    # ── Feature extraction ────────────────────────────────────────────
    features = extract_tremor_features(
        samples=body.samples,
        duration_sec=body.duration_sec,
        disease_stage=body.disease_stage,
        etiology=body.etiology,
    )

    risk_level, interpretation = _interpret_tremor(
        features.tremor_score, body.disease_stage
    )

    # ── DB save ───────────────────────────────────────────────────────
    raw_samples = [
        {"x": s.x, "y": s.y, "z": s.z, "t": s.t}
        for s in body.samples
    ]
    saved_id = await _save_tremor_to_db(
        db=db,
        patient_id=patient_id,
        features=features,
        disease_stage=body.disease_stage,
        etiology=body.etiology,
        raw_samples=raw_samples,
    )

    sample_rate = round(len(body.samples) / max(body.duration_sec, 0.01), 1)

    return TremorResponse(
        patient_id=patient_id,
        tremor_score=features.tremor_score,
        features=features,
        risk_level=risk_level,
        interpretation=interpretation,
        sample_count=len(body.samples),
        sample_rate_hz=sample_rate,
        saved_id=saved_id,
    )


@router.get("/{patient_id}/latest")
async def get_latest_tremor(
    patient_id: int,
    db:         AsyncSession = Depends(get_db),
):
    """
    Bemorning so'nggi tremor o'lchovini qaytaradi.
    Doctor dashboard uchun.
    """
    try:
        from sqlalchemy import text
        result = await db.execute(
            text("""
                SELECT id, recorded_at, rms_amplitude, dominant_freq_hz,
                       tremor_power_8_12hz, stability_score, tremor_score,
                       disease_stage, etiology
                FROM tremor_readings
                WHERE patient_id = :pid
                ORDER BY recorded_at DESC
                LIMIT 1
            """),
            {"pid": patient_id},
        )
        row = result.fetchone()
        if not row:
            return {"patient_id": patient_id, "data": None}
        return {
            "patient_id": patient_id,
            "data": {
                "id":                  row[0],
                "recorded_at":         str(row[1]),
                "rms_amplitude":       row[2],
                "dominant_freq_hz":    row[3],
                "tremor_power_8_12hz": row[4],
                "stability_score":     row[5],
                "tremor_score":        row[6],
                "disease_stage":       row[7],
                "etiology":            row[8],
            },
        }
    except Exception as exc:
        logger.error("Tremor latest fetch xatosi: %s", exc)
        raise HTTPException(status_code=500, detail="DB xatosi")


@router.get("/{patient_id}/history")
async def get_tremor_history(
    patient_id: int,
    days:       int = 30,
    db:         AsyncSession = Depends(get_db),
):
    """
    Trend engine uchun: so'nggi N kunlik tremor tarixi.
    ml/trend_engine.py → TrendEngine.update() ga uzatiladi.
    """
    try:
        from sqlalchemy import text
        result = await db.execute(
            text("""
                SELECT recorded_at, tremor_score, disease_stage
                FROM tremor_readings
                WHERE patient_id = :pid
                  AND recorded_at >= NOW() - MAKE_INTERVAL(days => :days)
                ORDER BY recorded_at ASC
            """),
            {"pid": patient_id, "days": days},
        )
        rows = result.fetchall()
        return {
            "patient_id": patient_id,
            "days":       days,
            "history":    [
                {"date": str(r[0]), "tremor_score": r[1], "disease_stage": r[2]}
                for r in rows
            ],
        }
    except Exception as exc:
        logger.error("Tremor history xatosi: %s", exc)
        raise HTTPException(status_code=500, detail="DB xatosi")
