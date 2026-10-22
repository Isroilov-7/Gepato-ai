"""
ml/scoring.py
Gepato-AI v2.0 — Composite HE Score Engine

Formula:
  Total_HE_Index = Voice × 0.75 + Tremor × 0.15 + Cognitive × 0.10

Har kasallik bosqichi uchun alohida threshold (adabiyot asosida).
Confidence interval: bootstrap emas, Wilson score interval.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Vazn koeffitsientlari ──────────────────────────────────────────────────
WEIGHTS = {"voice": 0.75, "tremor": 0.15, "cognitive": 0.10}

# ── Strata-specific risk thresholds ───────────────────────────────────────
# (low_max, medium_max, high_max) → critical = > high_max
# Manbalar: West-Haven + Gazda 2025 + klinik konsensus
STAGE_THRESHOLDS: dict[str, tuple[float, float, float]] = {
    "fibrosis_early":          (0.30, 0.55, 0.75),
    "fibrosis_advanced":       (0.35, 0.58, 0.78),
    "cirrhosis_compensated":   (0.40, 0.65, 0.80),
    "cirrhosis_decompensated": (0.45, 0.68, 0.82),
    "post_transplant":         (0.35, 0.60, 0.78),
    "chronic_hepatitis":       (0.32, 0.57, 0.76),
    "unknown":                 (0.40, 0.65, 0.80),
}

# Risk level har bir West-Haven grade bilan klinnik mos kelishi
RISK_LEVEL_MAP = {
    "low":      "West-Haven Grade 0 (minimal risk)",
    "medium":   "West-Haven Grade 1 (minimal HE ehtimoli)",
    "high":     "West-Haven Grade 2 (klinik tekshiruv tavsiya)",
    "critical": "West-Haven Grade 3–4 (darhol kuzatuv zarur)",
}


@dataclass
class HEIndexResult:
    """HEScoreEngine.compute() qaytaradigan natija."""

    # ── Asosiy natija ──
    total_he_index: float       # 0.0–1.0  (display: × 100)
    risk_level: str             # "low" | "medium" | "high" | "critical"
    risk_label: str             # West-Haven mos klinik tavsif

    # ── Komponentlar ──
    voice_component:    float   # voice_score × 0.75
    tremor_component:   float   # tremor_score × 0.15
    cognitive_component: float  # cognitive_score × 0.10

    # ── % ulush (vizualizatsiya uchun) ──
    voice_pct:    float         # 0–100 %
    tremor_pct:   float
    cognitive_pct: float

    # ── Threshold (strata ga qarab) ──
    stage_threshold_low:    float
    stage_threshold_medium: float
    stage_threshold_high:   float
    disease_stage: str

    # ── Wilson 95% CI ──
    ci_lower: float
    ci_upper: float

    # ── Klinik izoh ──
    clinical_interpretation: str
    action_required: str        # shifokorga ko'rsatma

    # ── Komponentlar mavjudmi? ──
    has_voice:    bool = True
    has_tremor:   bool = True
    has_cognitive: bool = True


class HEScoreEngine:
    """
    Composite HE Index hisoblash.

    Foydalanish:
        engine = HEScoreEngine()
        result = engine.compute(
            voice_score=0.62,
            tremor_score=0.18,
            cognitive_score=0.45,
            disease_stage="cirrhosis_compensated",
        )
        print(result.total_he_index, result.risk_level)
    """

    def compute(
        self,
        voice_score: Optional[float],
        tremor_score: Optional[float],
        cognitive_score: Optional[float],
        disease_stage: str = "unknown",
    ) -> HEIndexResult:
        """
        Composite HE Index hisoblash.

        Agar biror komponent mavjud bo'lmasa (None) → weights qayta taqsimlanadi.
        Hech bo'lmasa voice_score majburiy.
        """
        stage = disease_stage if disease_stage in STAGE_THRESHOLDS else "unknown"
        thresholds = STAGE_THRESHOLDS[stage]

        # ── Mavjud komponentlarni aniqlash ────────────────────────────
        has_v = voice_score    is not None
        has_t = tremor_score   is not None
        has_c = cognitive_score is not None

        if not has_v:
            raise ValueError("voice_score majburiy — boshqa komponentlar ixtiyoriy.")

        voice_score    = float(np.clip(voice_score,    0.0, 1.0))
        tremor_score   = float(np.clip(tremor_score,   0.0, 1.0)) if has_t else None
        cognitive_score = float(np.clip(cognitive_score, 0.0, 1.0)) if has_c else None

        # ── Weights qayta taqsimlash (yo'q komponentlar uchun) ────────
        w = self._adjusted_weights(has_v, has_t, has_c)

        # ── HE Index hisoblash ────────────────────────────────────────
        vc = voice_score    * w["voice"]
        tc = (tremor_score  * w["tremor"])   if has_t else 0.0
        cc = (cognitive_score * w["cognitive"]) if has_c else 0.0

        total = float(np.clip(vc + tc + cc, 0.0, 1.0))

        # ── Risk level ────────────────────────────────────────────────
        risk_level = self._score_to_risk(total, thresholds)
        risk_label = RISK_LEVEL_MAP[risk_level]

        # ── % ulush ───────────────────────────────────────────────────
        total_nonzero = total + 1e-10
        v_pct = round(vc / total_nonzero * 100, 1)
        t_pct = round(tc / total_nonzero * 100, 1)
        c_pct = round(cc / total_nonzero * 100, 1)

        # ── Wilson 95% CI ─────────────────────────────────────────────
        ci_l, ci_u = self._wilson_ci(total, n_samples=50)

        # ── Klinik tushuntirish ───────────────────────────────────────
        interpretation = self._interpret(
            total, risk_level, stage,
            voice_score, tremor_score, cognitive_score,
            has_v, has_t, has_c,
        )
        action = self._action(risk_level)

        logger.info(
            "HE Index: %.3f (%s) | stage=%s | voice=%.3f tremor=%s cog=%s",
            total, risk_level, stage,
            voice_score,
            f"{tremor_score:.3f}" if has_t else "N/A",
            f"{cognitive_score:.3f}" if has_c else "N/A",
        )

        return HEIndexResult(
            total_he_index=round(total, 4),
            risk_level=risk_level,
            risk_label=risk_label,
            voice_component=round(vc, 4),
            tremor_component=round(tc, 4),
            cognitive_component=round(cc, 4),
            voice_pct=v_pct,
            tremor_pct=t_pct,
            cognitive_pct=c_pct,
            stage_threshold_low=thresholds[0],
            stage_threshold_medium=thresholds[1],
            stage_threshold_high=thresholds[2],
            disease_stage=stage,
            ci_lower=round(ci_l, 4),
            ci_upper=round(ci_u, 4),
            clinical_interpretation=interpretation,
            action_required=action,
            has_voice=has_v,
            has_tremor=has_t,
            has_cognitive=has_c,
        )

    # ── PRIVATE ───────────────────────────────────────────────────────────

    @staticmethod
    def _adjusted_weights(has_v: bool, has_t: bool, has_c: bool) -> dict[str, float]:
        """Mavjud bo'lmagan komponentlar weightlarini boshqalarga taqsimlash."""
        base = {"voice": 0.75, "tremor": 0.15, "cognitive": 0.10}
        active = {k for k, flag in [("voice",has_v),("tremor",has_t),("cognitive",has_c)] if flag}
        missing_weight = sum(base[k] for k in base if k not in active)
        if missing_weight == 0:
            return base
        total_active = sum(base[k] for k in active)
        return {
            k: base[k] + base[k] / total_active * missing_weight if k in active else 0.0
            for k in base
        }

    @staticmethod
    def _score_to_risk(score: float, thresholds: tuple) -> str:
        lo, med, hi = thresholds
        if score < lo:    return "low"
        if score < med:   return "medium"
        if score < hi:    return "high"
        return "critical"

    @staticmethod
    def _wilson_ci(p: float, n_samples: int = 50, z: float = 1.96) -> tuple[float, float]:
        """
        Wilson score 95% confidence interval.
        n_samples: taxminiy yig'ilgan test soni (dastlabki pilot uchun 50).
        """
        n = max(n_samples, 1)
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denominator
        margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denominator
        return float(np.clip(center - margin, 0, 1)), float(np.clip(center + margin, 0, 1))

    @staticmethod
    def _interpret(
        total: float,
        risk: str,
        stage: str,
        vs: float,
        ts: Optional[float],
        cs: Optional[float],
        has_v: bool,
        has_t: bool,
        has_c: bool,
    ) -> str:
        stage_names = {
            "fibrosis_early": "erta fibroz",
            "fibrosis_advanced": "ilg'or fibroz/steatoz",
            "cirrhosis_compensated": "kompensirlangan sirroz",
            "cirrhosis_decompensated": "dekompensirlangan sirroz",
            "post_transplant": "transplantatsiyadan keyingi holat",
            "chronic_hepatitis": "surunkali gepatit",
            "unknown": "jigar kasalligi",
        }
        sname = stage_names.get(stage, "jigar kasalligi")
        parts = [f"Umumiy HE indeksi: {total*100:.0f}/100. {sname} uchun."]

        if vs > 0.65:
            parts.append(f"Ovoz tahlili yuqori xavf ko'rsatmoqda ({vs*100:.0f}%).")
        if has_t and ts is not None and ts > 0.5:
            parts.append(f"Tremor indeksi oshgan ({ts*100:.0f}%).")
        if has_c and cs is not None and cs < 0.4:
            parts.append(f"Kognitiv tezlik pasaygan ({(1-cs)*100:.0f}% sekinlashish).")
        if not has_t or not has_c:
            missing = []
            if not has_t: missing.append("tremor")
            if not has_c: missing.append("kognitiv")
            parts.append(f"Eslatma: {', '.join(missing)} ma'lumoti yo'q — aniqlik cheklangan.")
        return " ".join(parts)

    @staticmethod
    def _action(risk: str) -> str:
        return {
            "low":      "Kundalik monitoring davom ettirilsin. Keyingi test ertaga.",
            "medium":   "Dori dozasini tekshiring. 3 kun kuzatuv tavsiya etiladi.",
            "high":     "Shifokor bilan kechiktirilmay maslahatlashing.",
            "critical": "DARHOL shifokorga murojaat qiling yoki kasalxonaga qo'ng'iroq qiling.",
        }[risk]


# ── Singleton ──────────────────────────────────────────────────────────────
_engine: Optional[HEScoreEngine] = None

def get_scoring_engine() -> HEScoreEngine:
    global _engine
    if _engine is None:
        _engine = HEScoreEngine()
    return _engine
