"""
ml/trend_engine.py
Gepato-AI v2.0 — P6: Multi-Timescale Trend Engine

Uch darajali tahlil:
  A) Baseline establishment  — dastlabki 5 o'lchov → personal reference
  B) Individual delta        — (score - baseline) / baseline_std → z-score
  C) Trend velocity (7 kun)  — linear regression slope → HE early warning
  D) Progression (30 kun)    — uzoq muddatli kasallik progressiyasi

Foydalanish (api/routes/voice.py ichida):
    engine = get_trend_engine()
    result = await engine.update(db, patient_id, he_score, voice_score,
                                  tremor_score, cognitive_score, disease_stage)
    if result.alert_level == "critical":
        send_alert(patient_id, result.alert_message)

DB: patients jadvaliga migration 006 (master prompt Q7) qo'shilishi kerak.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

logger = logging.getLogger(__name__)

# ── Alert chegaralari (z-score asosida) ──────────────────────────────────
DELTA_WARN     = 1.5    # z > 1.5σ → ogohlantirish
DELTA_HIGH     = 2.0    # z > 2.0σ → yuqori xavf
DELTA_CRITICAL = 2.5    # z > 2.5σ → kritik

# Trend velocity (7 kun linear slope)
TREND_SLOPE_ALERT  = 0.03    # score units/kun
TREND_R2_MIN       = 0.60    # regression aniqligi (ishonchlilik)
TREND_DAYS_MIN     = 3       # kamida N kun davom etganda alert

# Progression (30 kun)
PROGRESSION_SLOPE = 0.008   # score units/kun → kasallik progressiyasi

# Outlier detection
OUTLIER_Z = 3.0

# Baseline window
BASELINE_WINDOW = 5          # dastlabki N o'lchov baseline uchun


# ── Data classlar ─────────────────────────────────────────────────────────

@dataclass
class DeltaResult:
    """Bir komponent uchun individual delta tahlili natijasi."""
    z_score:        float = 0.0
    delta_pct:      float = 0.0     # % o'zgarish baselinedan
    is_outlier:     bool  = False
    alert_level:    str   = "ok"    # "ok"|"warn"|"high"|"critical"
    message:        str   = ""


@dataclass
class TrendResult:
    """7-kunlik trend natijasi."""
    slope_per_day:  float = 0.0
    r_squared:      float = 0.0
    direction:      str   = "stable"    # "stable"|"worsening"|"improving"
    alert:          bool  = False
    message:        str   = ""


@dataclass
class ProgressionResult:
    """30-kunlik progressiya natijasi."""
    slope_per_day:  float = 0.0
    progressing:    bool  = False
    message:        str   = ""


@dataclass
class TrendEngineResult:
    """engine.update() qaytaradigan to'liq natija."""
    patient_id:         int
    he_delta:           DeltaResult
    voice_delta:        DeltaResult
    tremor_delta:       DeltaResult
    cognitive_delta:    DeltaResult
    trend_7d:           TrendResult
    progression_30d:    ProgressionResult
    alert_level:        str     = "ok"      # umumiy eng yuqori daraja
    alert_message:      str     = ""
    baseline_ready:     bool    = False
    outlier_detected:   bool    = False
    outlier_message:    str     = ""
    component_deltas:   dict    = field(default_factory=dict)


# ── Asosiy sinf ────────────────────────────────────────────────────────────

class TrendEngine:
    """
    Bemorning shaxsiy baseline asosida HE indeksini kuzatadi.

    Har yangi o'lchov kelganda:
      1. Outlier tekshiruvi
      2. Baseline yangilash (agar kerak)
      3. Delta z-score hisoblash
      4. 7-kunlik trend
      5. 30-kunlik progressiya
      6. Alert qaror qilish
    """

    # ── PUBLIC ────────────────────────────────────────────────────────────

    async def update(
        self,
        db:              AsyncSession,
        patient_id:      int,
        he_score:        float,
        voice_score:     float,
        tremor_score:    Optional[float] = None,
        cognitive_score: Optional[float] = None,
        disease_stage:   str = "unknown",
    ) -> TrendEngineResult:
        """
        Yangi o'lchov kiritilganda trend tahlilini yangilaydi.

        voice.py pipeline'da chaqiriladi:
            result = await trend_engine.update(
                db, patient_id,
                he_score=pred.risk_score,
                voice_score=voice_score,
                tremor_score=tremor_score,
                cognitive_score=cognitive_score,
                disease_stage=patient.disease_stage,
            )
        """
        # ── Bemorning mavjud baseline va tarixini yuklash ────────────
        baseline = await self._load_baseline(db, patient_id)
        history  = await self._load_history(db, patient_id, days=35)

        # ── Outlier detection ─────────────────────────────────────────
        is_outlier = self._detect_outlier(he_score, baseline)
        outlier_msg = ""
        if is_outlier:
            outlier_msg = (
                "Bu natija g'ayrioddiy. "
                "Yaxshi sezasizmi? Qayta test o'tkazasizmi?"
            )
            logger.warning(
                "Outlier detected | patient=%d he_score=%.3f", patient_id, he_score
            )

        # ── Baseline yangilash (outlier bo'lmasa) ─────────────────────
        if not is_outlier:
            baseline = await self._update_baseline(
                db, patient_id, he_score, voice_score,
                tremor_score, cognitive_score, baseline,
            )

        baseline_ready = baseline.get("n", 0) >= BASELINE_WINDOW

        # ── Delta z-score ─────────────────────────────────────────────
        he_delta   = self._compute_delta(he_score,        baseline, "he_score")
        voice_delta = self._compute_delta(voice_score,    baseline, "voice_score")
        trem_delta  = self._compute_delta(
            tremor_score or 0.0, baseline, "tremor_score"
        ) if tremor_score is not None else DeltaResult()
        cog_delta   = self._compute_delta(
            cognitive_score or 0.0, baseline, "cognitive_score"
        ) if cognitive_score is not None else DeltaResult()

        # ── 7-kunlik trend ────────────────────────────────────────────
        trend_7d = self._trend_velocity(history, days=7)

        # ── 30-kunlik progressiya ─────────────────────────────────────
        prog_30d = self._progression(history, days=30)

        # ── Umumiy alert daraja ───────────────────────────────────────
        alert_level, alert_msg = self._decide_alert(
            he_delta, trend_7d, prog_30d, baseline_ready
        )

        # ── DB yangilash ───────────────────────────────────────────────
        if not is_outlier:
            await self._save_trend_to_patients(
                db, patient_id,
                trend_slope_7d=trend_7d.slope_per_day,
                trend_r2_7d=trend_7d.r_squared,
                progression_slope_30d=prog_30d.slope_per_day,
                progression_flag=prog_30d.progressing,
            )

        component_deltas = {
            "jitter_deviation":   0.0,
            "shimmer_deviation":  0.0,
            "hnr_deviation":      0.0,
        }

        return TrendEngineResult(
            patient_id=patient_id,
            he_delta=he_delta,
            voice_delta=voice_delta,
            tremor_delta=trem_delta,
            cognitive_delta=cog_delta,
            trend_7d=trend_7d,
            progression_30d=prog_30d,
            alert_level=alert_level,
            alert_message=alert_msg,
            baseline_ready=baseline_ready,
            outlier_detected=is_outlier,
            outlier_message=outlier_msg,
            component_deltas=component_deltas,
        )

    # ── PRIVATE: Baseline ─────────────────────────────────────────────────

    async def _load_baseline(
        self, db: AsyncSession, patient_id: int
    ) -> dict:
        """patients jadvalidan baseline qiymatlarini yuklaydi."""
        try:
            result = await db.execute(
                text("""
                    SELECT
                        baseline_he_score, baseline_voice_score,
                        baseline_tremor_score, baseline_cognitive_score,
                        baseline_std, baseline_window_n
                    FROM patients
                    WHERE id = :pid
                """),
                {"pid": patient_id},
            )
            row = result.fetchone()
            if not row:
                return {}
            return {
                "he_score":        row[0] or 0.0,
                "voice_score":     row[1] or 0.0,
                "tremor_score":    row[2] or 0.0,
                "cognitive_score": row[3] or 0.0,
                "std":             row[4] or 0.10,
                "n":               row[5] or 0,
            }
        except Exception as exc:
            logger.warning("Baseline yuklash xatosi: %s", exc)
            return {}

    async def _update_baseline(
        self,
        db:              AsyncSession,
        patient_id:      int,
        he_score:        float,
        voice_score:     float,
        tremor_score:    Optional[float],
        cognitive_score: Optional[float],
        current:         dict,
    ) -> dict:
        """
        Exponential moving average bilan baseline yangilash.

        Dastlabki BASELINE_WINDOW o'lchov: oddiy o'rtacha (pretrain)
        Undan keyin: EMA α=0.15 (yangi o'lchov oz ta'sir qiladi)
        """
        n    = current.get("n", 0)
        alpha = 1.0 / (n + 1) if n < BASELINE_WINDOW else 0.15

        def ema(old, new):
            return old * (1 - alpha) + new * alpha if old else new

        new_he    = ema(current.get("he_score", he_score), he_score)
        new_voice = ema(current.get("voice_score", voice_score), voice_score)
        new_trem  = ema(current.get("tremor_score", 0.0), tremor_score or 0.0)
        new_cog   = ema(current.get("cognitive_score", 0.0), cognitive_score or 0.0)

        # Std: running std taxminiy hisoblash
        old_std  = current.get("std", 0.10)
        deviation = abs(he_score - current.get("he_score", he_score))
        new_std  = max(old_std * 0.90 + deviation * 0.10, 0.01)

        new_n = n + 1

        try:
            await db.execute(
                text("""
                    UPDATE patients SET
                        baseline_he_score        = :he,
                        baseline_voice_score     = :voice,
                        baseline_tremor_score    = :trem,
                        baseline_cognitive_score = :cog,
                        baseline_std             = :std,
                        baseline_window_n        = :n,
                        baseline_last_updated    = NOW()
                    WHERE id = :pid
                """),
                {
                    "he":    round(new_he, 4),
                    "voice": round(new_voice, 4),
                    "trem":  round(new_trem, 4),
                    "cog":   round(new_cog, 4),
                    "std":   round(new_std, 4),
                    "n":     new_n,
                    "pid":   patient_id,
                },
            )
            await db.commit()
        except Exception as exc:
            logger.warning("Baseline yangilash xatosi: %s", exc)
            await db.rollback()

        return {
            "he_score":        new_he,
            "voice_score":     new_voice,
            "tremor_score":    new_trem,
            "cognitive_score": new_cog,
            "std":             new_std,
            "n":               new_n,
        }

    # ── PRIVATE: Delta ────────────────────────────────────────────────────

    def _detect_outlier(self, score: float, baseline: dict) -> bool:
        """z > 3.0σ → outlier, baseline YANGILANMAYDI."""
        if not baseline or baseline.get("n", 0) < BASELINE_WINDOW:
            return False
        b_mean = baseline.get("he_score", score)
        b_std  = max(baseline.get("std", 0.10), 0.01)
        z      = abs(score - b_mean) / b_std
        return z > OUTLIER_Z

    def _compute_delta(
        self, current: float, baseline: dict, key: str
    ) -> DeltaResult:
        """
        Individual delta z-score.

        Formula (master prompt P6):
            delta_z = (current - baseline_mean) / baseline_std

        Alert levels:
            delta_z > 1.5σ → "Kuzatuvni kuchaytiring"
            delta_z > 2.0σ → "Yuqori xavf"
            delta_z > 2.5σ → "Kritik — darhol shifokorga"
        """
        if not baseline or baseline.get("n", 0) < 2:
            return DeltaResult()

        b_mean = baseline.get(key, current)
        b_std  = max(baseline.get("std", 0.10), 0.01)
        z      = (current - b_mean) / b_std

        # % o'zgarish
        delta_pct = ((current - b_mean) / max(abs(b_mean), 1e-6)) * 100

        # Alert level
        if abs(z) >= DELTA_CRITICAL:
            level = "critical"
            msg   = "Kritik o'zgarish — darhol shifokorga murojaat qiling."
        elif abs(z) >= DELTA_HIGH:
            level = "high"
            msg   = f"Yuqori xavf — kuzatuvni kuchaytiring ({z:+.1f}σ)."
        elif abs(z) >= DELTA_WARN:
            level = "warn"
            msg   = f"Baselinedan og'ish ({z:+.1f}σ). Kuzatuv tavsiya."
        else:
            level = "ok"
            msg   = ""

        return DeltaResult(
            z_score=round(z, 3),
            delta_pct=round(delta_pct, 1),
            is_outlier=False,
            alert_level=level,
            message=msg,
        )

    # ── PRIVATE: Trend & Progression ─────────────────────────────────────

    async def _load_history(
        self, db: AsyncSession, patient_id: int, days: int
    ) -> list[dict]:
        """predictions jadvalidan so'nggi N kunlik HE score tarixi."""
        try:
            result = await db.execute(
                text("""
                    SELECT
                        created_at,
                        risk_score,
                        EXTRACT(EPOCH FROM (created_at - NOW())) / 86400.0 AS days_ago
                    FROM predictions
                    WHERE patient_id = :pid
                      AND created_at >= NOW() - MAKE_INTERVAL(days => :days)
                    ORDER BY created_at ASC
                """),
                {"pid": patient_id, "days": days},
            )
            rows = result.fetchall()
            return [
                {"date": str(r[0]), "score": float(r[1]), "days_ago": float(r[2])}
                for r in rows
            ]
        except Exception as exc:
            logger.warning("History yuklash xatosi: %s", exc)
            return []

    def _trend_velocity(self, history: list[dict], days: int = 7) -> TrendResult:
        """
        7-kunlik linear regression slope.

        Formula (master prompt P6):
            slope    = polyfit(days, scores, 1)[0]
            r2       = R² coefficient
            alert    = slope > 0.03/kun AND r2 > 0.6 AND 3+ kun

        slope > 0 → yomonlashuv
        slope < 0 → yaxshilanish
        """
        recent = [h for h in history if abs(h.get("days_ago", 999)) <= days]
        if len(recent) < TREND_DAYS_MIN:
            return TrendResult(message="Yetarli ma'lumot yo'q.")

        scores = np.array([h["score"] for h in recent], dtype=float)
        times  = np.array([abs(h["days_ago"]) for h in recent], dtype=float)
        times  = max(times) - times          # 0 = eng eski, N = bugun

        if len(np.unique(times)) < 2:
            return TrendResult()

        coeffs    = np.polyfit(times, scores, 1)
        slope     = float(coeffs[0])
        predicted = np.polyval(coeffs, times)
        ss_res    = float(np.sum((scores - predicted) ** 2))
        ss_tot    = float(np.sum((scores - scores.mean()) ** 2))
        r2        = 1.0 - ss_res / (ss_tot + 1e-10)
        r2        = float(np.clip(r2, 0.0, 1.0))

        # Direction
        if slope > 0.01:
            direction = "worsening"
        elif slope < -0.01:
            direction = "improving"
        else:
            direction = "stable"

        # Alert
        alert = (
            slope > TREND_SLOPE_ALERT and
            r2    > TREND_R2_MIN and
            len(recent) >= TREND_DAYS_MIN
        )
        message = ""
        if alert:
            message = (
                f"7 kunlik trend: +{slope*100:.1f}% har kuni (R²={r2:.2f}). "
                "Shifokor bilan maslahatlashing."
            )
        elif direction == "improving":
            message = f"Holat yaxshilanmoqda (slope={slope*100:+.1f}%/kun)."

        return TrendResult(
            slope_per_day=round(slope, 5),
            r_squared=round(r2, 4),
            direction=direction,
            alert=alert,
            message=message,
        )

    def _progression(self, history: list[dict], days: int = 30) -> ProgressionResult:
        """
        30-kunlik uzoq muddatli progressiya.

        Formula (master prompt P6):
            progression_flag = True agar 30 kunlik slope > 0.008/kun

        Bu kasallikning sekin, doimiy yomonlashuvini ko'rsatadi.
        """
        recent = [h for h in history if abs(h.get("days_ago", 999)) <= days]
        if len(recent) < 5:
            return ProgressionResult(message="Yetarli ma'lumot yo'q.")

        scores = np.array([h["score"] for h in recent], dtype=float)
        times  = np.array([abs(h["days_ago"]) for h in recent], dtype=float)
        times  = max(times) - times

        if len(np.unique(times)) < 2:
            return ProgressionResult()

        coeffs = np.polyfit(times, scores, 1)
        slope  = float(coeffs[0])

        progressing = slope > PROGRESSION_SLOPE
        message = ""
        if progressing:
            message = (
                f"30 kunlik progressiya: +{slope*100:.2f}% har kuni. "
                "Kasallik yomonlashishi belgilari."
            )

        return ProgressionResult(
            slope_per_day=round(slope, 5),
            progressing=progressing,
            message=message,
        )

    # ── PRIVATE: Alert qaror ─────────────────────────────────────────────

    @staticmethod
    def _decide_alert(
        he_delta:      DeltaResult,
        trend:         TrendResult,
        progression:   ProgressionResult,
        baseline_ready: bool,
    ) -> tuple[str, str]:
        """
        Barcha signallarni birlashtiradi → bitta alert level.
        Tartibi: critical > high > warn > ok
        """
        if not baseline_ready:
            return "ok", ""

        # Critical: delta > 2.5σ YOKI trend alert
        if he_delta.alert_level == "critical":
            return "critical", he_delta.message
        if trend.alert and he_delta.alert_level in ("high", "critical"):
            return "critical", (
                f"Trend + delta birgalikda kritik signal. {trend.message}"
            )

        # High: delta > 2.0σ
        if he_delta.alert_level == "high":
            return "high", he_delta.message

        # High: faqat trend (delta mo'tadil bo'lsa ham)
        if trend.alert:
            return "high", trend.message

        # Warn: delta > 1.5σ
        if he_delta.alert_level == "warn":
            return "warn", he_delta.message

        # Warn: progressiya
        if progression.progressing:
            return "warn", progression.message

        return "ok", ""

    # ── PRIVATE: DB yozish ────────────────────────────────────────────────

    @staticmethod
    async def _save_trend_to_patients(
        db:                    AsyncSession,
        patient_id:            int,
        trend_slope_7d:        float,
        trend_r2_7d:           float,
        progression_slope_30d: float,
        progression_flag:      bool,
    ) -> None:
        """patients jadvalidagi trend ustunlarini yangilaydi (migration 006)."""
        try:
            await db.execute(
                text("""
                    UPDATE patients SET
                        trend_slope_7d        = :s7,
                        trend_r2_7d           = :r7,
                        progression_slope_30d = :s30,
                        progression_flag      = :pf
                    WHERE id = :pid
                """),
                {
                    "s7":  trend_slope_7d,
                    "r7":  trend_r2_7d,
                    "s30": progression_slope_30d,
                    "pf":  progression_flag,
                    "pid": patient_id,
                },
            )
            await db.commit()
        except Exception as exc:
            logger.warning("Trend DB yangilash xatosi: %s", exc)
            await db.rollback()


# ── Singleton ──────────────────────────────────────────────────────────────
_engine: Optional[TrendEngine] = None


def get_trend_engine() -> TrendEngine:
    global _engine
    if _engine is None:
        _engine = TrendEngine()
    return _engine
