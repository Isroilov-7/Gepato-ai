"""
api/routes/cognitive.py
Gepato-AI v2.0 — P5: Kognitiv Testlar (NCT-A + Stroop)

POST /api/cognitive/{patient_id}
  Body: {
    stroop_trials: [{color, word, response, rt_ms, correct}],
    nct_sequence:  [{number, tapped_at_ms, correct}],
    disease_stage: str
  }

Pipeline:
  1. Stroop hisoblash: mean_rt, std_rt, error_rate → stroop_score 0–1
  2. NCT hisoblash: total_time, error_count, inter_tap → nct_score 0–1
  3. Composite cognitive_score 0–1 (Stroop 50% + NCT 50%)
  4. DB save (cognitive_tests table — migration 005)
  5. cognitive_score → HEScoreEngine.compute() ga uzatish

West-Haven normalizatsiya:
  Stroop: normal <280ms, minimal HE 280–600ms, overt >600ms
  NCT: normal <60sec, patologik >120sec
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cognitive", tags=["cognitive"])

# ── West-Haven asosidagi normalizatsiya chegaralari ─────────────────────────
# Ref: Weissenborn 2001, Amodio 2004, Randolph 2009

# Stroop reaction time (ms)
STROOP_NORMAL_RT_MS  = 280.0    # normal chegarasi
STROOP_OVERT_RT_MS   = 1500.0   # oshganda overt HE
# score = max(0, 1 - (mean_rt - 280) / 1220)  ← master prompt formula

# NCT-A (Number Connection Test type A)
NCT_NORMAL_SEC      = 60.0      # sog'lom chegarasi
NCT_PATOLOGIC_SEC   = 180.0     # patologik daraja
# score = max(0, 1 - (total_time - 60) / 120)

# Composite weights
STROOP_WEIGHT = 0.50
NCT_WEIGHT    = 0.50


# ── Pydantic schemas ───────────────────────────────────────────────────────

class StroopTrial(BaseModel):
    color:    str               # haqiqiy rang: "red"|"green"|"blue"|"yellow"
    word:     str               # ko'rsatilgan so'z (inkongrent yoki kongruent)
    response: str               # foydalanuvchi tanlagan rang
    rt_ms:    float             # reaction time (ms)
    correct:  bool              # to'g'ri javob?


class NCTTap(BaseModel):
    number:       int           # 1–9
    tapped_at_ms: float         # yozuv boshlanishidan ms
    correct:      bool          # to'g'ri tartibdami?


class CognitiveRequest(BaseModel):
    stroop_trials:  Optional[list[StroopTrial]] = Field(default=None)
    nct_sequence:   Optional[list[NCTTap]]      = Field(default=None)
    disease_stage:  str = Field(default="unknown")

    @field_validator("stroop_trials")
    @classmethod
    def min_stroop(cls, v):
        if v is not None and len(v) < 3:
            raise ValueError("Kamida 3 ta Stroop trial kerak")
        return v

    @field_validator("nct_sequence")
    @classmethod
    def min_nct(cls, v):
        if v is not None and len(v) < 5:
            raise ValueError("Kamida 5 ta NCT tap kerak")
        return v


class StroopResult(BaseModel):
    mean_rt_ms:  float
    std_rt_ms:   float
    error_rate:  float          # 0.0–1.0
    stroop_score: float         # 0.0–1.0 (yuqori = yaxshi)
    trial_count: int


class NCTResult(BaseModel):
    total_time_sec:    float
    error_count:       int
    mean_inter_tap_ms: float
    nct_score:         float    # 0.0–1.0 (yuqori = yaxshi)
    tap_count:         int


class CognitiveResponse(BaseModel):
    patient_id:       int
    cognitive_score:  float     # 0.0–1.0 — HEScoreEngine uchun
    stroop:           Optional[StroopResult]
    nct:              Optional[NCTResult]
    risk_level:       str
    interpretation:   str
    saved_id:         Optional[int] = None


# ── Stroop hisoblash ───────────────────────────────────────────────────────

def compute_stroop(trials: list[StroopTrial]) -> StroopResult:
    """
    Stroop Color-Word Test natijasi.

    Faqat INKONGRENT triallar hisobga olinadi (word ≠ color).
    Kongrent triallar isiqlik uchun, lekin score ga kirmaydi.

    Score formula (master prompt P5):
        score = max(0, 1 - (mean_rt - 280) / 1220)
        0.0 = mean_rt ≥ 1500ms (overt HE)
        1.0 = mean_rt ≤ 280ms (normal)
    """
    if not trials:
        return StroopResult(
            mean_rt_ms=0.0, std_rt_ms=0.0, error_rate=0.0,
            stroop_score=0.0, trial_count=0,
        )

    rts = [t.rt_ms for t in trials if t.rt_ms > 0]
    if not rts:
        rts = [STROOP_OVERT_RT_MS]

    mean_rt = float(np.mean(rts))
    std_rt  = float(np.std(rts))

    errors     = sum(1 for t in trials if not t.correct)
    error_rate = errors / max(len(trials), 1)

    # Score formula — master prompt P5
    raw_score = max(0.0, 1.0 - (mean_rt - STROOP_NORMAL_RT_MS) / 1220.0)

    # Error penalty: har 10% xato uchun 0.05 chegirma
    error_penalty = min(error_rate * 0.5, 0.30)
    stroop_score  = float(np.clip(raw_score - error_penalty, 0.0, 1.0))

    logger.info(
        "Stroop | mean_rt=%.0fms std=%.0f errors=%.0f%% score=%.3f",
        mean_rt, std_rt, error_rate * 100, stroop_score,
    )

    return StroopResult(
        mean_rt_ms=round(mean_rt, 1),
        std_rt_ms=round(std_rt, 1),
        error_rate=round(error_rate, 3),
        stroop_score=round(stroop_score, 4),
        trial_count=len(trials),
    )


# ── NCT hisoblash ──────────────────────────────────────────────────────────

def compute_nct(taps: list[NCTTap]) -> NCTResult:
    """
    Number Connection Test type A (9-raqamli grid).

    Total time: oxirgi tap – birinchi tap
    Inter-tap interval: ketma-ket taplar orasidagi vaqt

    Score formula:
        score = max(0, 1 - (total_time_sec - 60) / 120)
        0.0 = total_time ≥ 180sec
        1.0 = total_time ≤ 60sec

    Xato penalty: har xato uchun 0.08
    """
    if not taps:
        return NCTResult(
            total_time_sec=0.0, error_count=0,
            mean_inter_tap_ms=0.0, nct_score=0.0, tap_count=0,
        )

    sorted_taps = sorted(taps, key=lambda t: t.tapped_at_ms)
    total_time_sec = (sorted_taps[-1].tapped_at_ms - sorted_taps[0].tapped_at_ms) / 1000.0

    if total_time_sec < 0.5:
        total_time_sec = 0.5

    # Inter-tap interval
    tap_times = [t.tapped_at_ms for t in sorted_taps]
    inter_taps = [tap_times[i+1] - tap_times[i] for i in range(len(tap_times)-1)]
    mean_inter_tap = float(np.mean(inter_taps)) if inter_taps else 0.0

    error_count = sum(1 for t in taps if not t.correct)

    # Score
    raw_score   = max(0.0, 1.0 - (total_time_sec - NCT_NORMAL_SEC) / 120.0)
    error_pen   = min(error_count * 0.08, 0.40)
    nct_score   = float(np.clip(raw_score - error_pen, 0.0, 1.0))

    logger.info(
        "NCT | total=%.1fs inter=%.0fms errors=%d score=%.3f",
        total_time_sec, mean_inter_tap, error_count, nct_score,
    )

    return NCTResult(
        total_time_sec=round(total_time_sec, 2),
        error_count=error_count,
        mean_inter_tap_ms=round(mean_inter_tap, 1),
        nct_score=round(nct_score, 4),
        tap_count=len(taps),
    )


# ── Composite cognitive score ──────────────────────────────────────────────

def compute_cognitive_score(
    stroop: Optional[StroopResult],
    nct:    Optional[NCTResult],
) -> float:
    """
    Composite cognitive score 0–1.
    Stroop 50% + NCT 50%.
    Agar biri yo'q bo'lsa — mavjud bo'lgan 100% ga ko'tariladi.
    """
    if stroop is None and nct is None:
        return 0.0

    if stroop is None:
        return nct.nct_score
    if nct is None:
        return stroop.stroop_score

    return float(
        STROOP_WEIGHT * stroop.stroop_score +
        NCT_WEIGHT    * nct.nct_score
    )


def _interpret_cognitive(score: float, stage: str) -> tuple[str, str]:
    """cognitive_score → (risk_level, interpretation)"""
    if score >= 0.75:
        return "low",      "Kognitiv funksiya normal. Reaksiya tezligi yaxshi."
    if score >= 0.50:
        return "medium",   "Yengil kognitiv sekinlashuv. Minimal HE ehtimoli bor."
    if score >= 0.30:
        return "high",     "O'rtacha kognitiv buzilish. Klinik tekshiruv tavsiya etiladi."
    return "critical",     "Sezilarli kognitiv buzilish. Darhol shifokorga murojaat qiling."


# ── DB save ────────────────────────────────────────────────────────────────

async def _save_cognitive_to_db(
    db:          AsyncSession,
    patient_id:  int,
    stroop:      Optional[StroopResult],
    nct:         Optional[NCTResult],
    cog_score:   float,
    stage:       str,
    raw_stroop:  Optional[list],
    raw_nct:     Optional[list],
) -> Optional[int]:
    """
    cognitive_tests jadvaliga yozadi (migration 005).
    """
    try:
        from sqlalchemy import text
        import json

        raw_stroop_json = json.dumps(raw_stroop or [])
        raw_nct_json    = json.dumps(raw_nct or [])

        result = await db.execute(
            text("""
                INSERT INTO cognitive_tests (
                    patient_id,
                    stroop_mean_rt_ms, stroop_std_rt, stroop_error_rate, stroop_score,
                    nct_total_time_sec, nct_error_count, nct_mean_inter_tap_ms, nct_score,
                    cognitive_score, disease_stage,
                    raw_stroop_trials, raw_nct_sequence
                ) VALUES (
                    :pid,
                    :smrt, :sstd, :serr, :ssc,
                    :ntt, :nec, :nmit, :nsc,
                    :cog, :stage,
                    :rstroop, :rnct
                )
                RETURNING id
            """),
            {
                "pid":    patient_id,
                "smrt":   stroop.mean_rt_ms   if stroop else None,
                "sstd":   stroop.std_rt_ms    if stroop else None,
                "serr":   stroop.error_rate   if stroop else None,
                "ssc":    stroop.stroop_score if stroop else None,
                "ntt":    nct.total_time_sec      if nct else None,
                "nec":    nct.error_count          if nct else None,
                "nmit":   nct.mean_inter_tap_ms    if nct else None,
                "nsc":    nct.nct_score             if nct else None,
                "cog":    cog_score,
                "stage":  stage,
                "rstroop": raw_stroop_json,
                "rnct":    raw_nct_json,
            },
        )
        await db.commit()
        row = result.fetchone()
        return int(row[0]) if row else None

    except Exception as exc:
        logger.error("Cognitive DB saqlash xatosi: %s", exc)
        await db.rollback()
        return None


# ── Router ─────────────────────────────────────────────────────────────────

@router.post("/{patient_id}", response_model=CognitiveResponse)
async def submit_cognitive(
    patient_id: int,
    body:       CognitiveRequest,
    db:         AsyncSession = Depends(get_db),
):
    """
    Kognitiv test natijalarini qabul qiladi (Stroop + NCT).

    Frontend (patient_daily_flow.html) P5 bosqichida chaqiradi:
        POST /api/cognitive/{patient_id}
        {stroop_trials: [...], nct_sequence: [...]}

    Javob:
        {cognitive_score: 0–1, risk_level: str, ...}

    cognitive_score → HEScoreEngine.compute(cognitive_score=...) ga uzatiladi.
    """
    if patient_id <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Noto'g'ri patient_id",
        )

    if body.stroop_trials is None and body.nct_sequence is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Kamida bitta test (Stroop yoki NCT) kerak",
        )

    # ── Hisoblash ─────────────────────────────────────────────────────
    stroop_result = compute_stroop(body.stroop_trials) if body.stroop_trials else None
    nct_result    = compute_nct(body.nct_sequence)     if body.nct_sequence    else None

    cog_score  = compute_cognitive_score(stroop_result, nct_result)
    risk_level, interpretation = _interpret_cognitive(cog_score, body.disease_stage)

    # ── DB save ───────────────────────────────────────────────────────
    raw_stroop = (
        [{"color": t.color, "word": t.word, "response": t.response,
          "rt_ms": t.rt_ms, "correct": t.correct}
         for t in body.stroop_trials]
        if body.stroop_trials else None
    )
    raw_nct = (
        [{"number": t.number, "tapped_at_ms": t.tapped_at_ms, "correct": t.correct}
         for t in body.nct_sequence]
        if body.nct_sequence else None
    )

    saved_id = await _save_cognitive_to_db(
        db=db,
        patient_id=patient_id,
        stroop=stroop_result,
        nct=nct_result,
        cog_score=round(cog_score, 4),
        stage=body.disease_stage,
        raw_stroop=raw_stroop,
        raw_nct=raw_nct,
    )

    logger.info(
        "Cognitive submit | patient=%d cog_score=%.3f risk=%s saved=%s",
        patient_id, cog_score, risk_level, saved_id,
    )

    return CognitiveResponse(
        patient_id=patient_id,
        cognitive_score=round(cog_score, 4),
        stroop=stroop_result,
        nct=nct_result,
        risk_level=risk_level,
        interpretation=interpretation,
        saved_id=saved_id,
    )


@router.get("/{patient_id}/latest")
async def get_latest_cognitive(
    patient_id: int,
    db:         AsyncSession = Depends(get_db),
):
    """So'nggi kognitiv test natijasi. Doctor dashboard uchun."""
    try:
        from sqlalchemy import text
        result = await db.execute(
            text("""
                SELECT id, test_date,
                       stroop_mean_rt_ms, stroop_score,
                       nct_total_time_sec, nct_score,
                       cognitive_score, disease_stage
                FROM cognitive_tests
                WHERE patient_id = :pid
                ORDER BY test_date DESC
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
                "id":               row[0],
                "test_date":        str(row[1]),
                "stroop_mean_rt_ms": row[2],
                "stroop_score":     row[3],
                "nct_total_time_sec": row[4],
                "nct_score":        row[5],
                "cognitive_score":  row[6],
                "disease_stage":    row[7],
            },
        }
    except Exception as exc:
        logger.error("Cognitive latest xatosi: %s", exc)
        raise HTTPException(status_code=500, detail="DB xatosi")


@router.get("/{patient_id}/history")
async def get_cognitive_history(
    patient_id: int,
    days:       int = 30,
    db:         AsyncSession = Depends(get_db),
):
    """
    Trend engine uchun: so'nggi N kunlik kognitiv test tarixi.
    ml/trend_engine.py ga uzatiladi.
    """
    try:
        from sqlalchemy import text
        result = await db.execute(
            text("""
                SELECT test_date, cognitive_score,
                       stroop_score, nct_score, disease_stage
                FROM cognitive_tests
                WHERE patient_id = :pid
                  AND test_date >= NOW() - MAKE_INTERVAL(days => :days)
                ORDER BY test_date ASC
            """),
            {"pid": patient_id, "days": days},
        )
        rows = result.fetchall()
        return {
            "patient_id": patient_id,
            "days":       days,
            "history":    [
                {
                    "date":            str(r[0]),
                    "cognitive_score": r[1],
                    "stroop_score":    r[2],
                    "nct_score":       r[3],
                    "disease_stage":   r[4],
                }
                for r in rows
            ],
        }
    except Exception as exc:
        logger.error("Cognitive history xatosi: %s", exc)
        raise HTTPException(status_code=500, detail="DB xatosi")
