"""
ml/audio_quality.py
Gepato-AI v2.0 — Audio Quality Pipeline

Bosqichlar:
  1. Environment noise detection  (yozishdan OLDIN)
  2. Audio normalization          (-3 dBFS peak)
  3. Silence stripping            (bosh/oxir + ichki)
  4. Clipping detection           (mikrofon juda yaqin?)
  5. Final quality report

Foydalanish:
    pipeline = AudioQualityPipeline()

    # Yozishdan oldin (ambient check):
    ambient = pipeline.check_ambient(ambient_wav_path)
    if not ambient.ok_to_record:
        show_warning(ambient.message)

    # Yozuvdan keyin (full check + process):
    report = pipeline.process(audio_path)
    if not report.passed:
        ask_patient_to_retry(report.block_reason)
    else:
        processed_audio = report.processed_audio  # normalize qilingan
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# ── ITU-T P.800 asosida chegara qiymatlari ─────────────────────────────────
AMBIENT_WARN_DBFS  = -45.0   # shu darajadan yuqori → ogohlantirish
AMBIENT_BLOCK_DBFS = -35.0   # shu darajadan yuqori → yozishni bloklash
CLIP_FRACTION      = 0.001   # > 0.1 % samples near ±1 → clipped
CLIP_THRESHOLD     = 0.98    # ±1.0 ga yaqin hisoblash chegarasi
NORM_TARGET        = 0.708   # -3 dBFS ≈ 0.708 linear peak
TARGET_SR          = 16_000  # barcha audio shu sample rate ga resample
INTERNAL_SILENCE_S = 2.0     # text reading uchun: ichki jimlik chegarasi
TRIM_TOP_DB        = 25      # librosa.effects.trim uchun
SPLIT_TOP_DB       = 30      # ichki jimlik aniqlash uchun


# ── Data classlar ──────────────────────────────────────────────────────────

@dataclass
class AmbientReport:
    """check_ambient() qaytaradigan natija."""
    ambient_db: float
    ok_to_record: bool
    warning: bool          # ogohlantirish (lekin yozish mumkin)
    blocked: bool          # yozish MUMKIN EMAS
    message: str           # foydalanuvchiga ko'rsatiladigan xabar (O'zbekcha)
    message_uz: str = ""   # qo'shimcha O'zbekcha varianti


@dataclass
class AudioQualityReport:
    """process() qaytaradigan to'liq natija."""

    # ── O'lchov natijalari ──
    ambient_snr_db: float = 0.0
    peak_db: float = 0.0
    is_clipped: bool = False
    silence_trimmed_sec: float = 0.0
    final_duration_sec: float = 0.0
    sample_rate: int = TARGET_SR

    # ── Qaror ──
    passed: bool = True
    warnings: list[str] = field(default_factory=list)
    block_reason: Optional[str] = None   # None = passed

    # ── Qayta ishlangan audio (normalize + trim qilingan) ──
    processed_audio: Optional[np.ndarray] = None   # float32, TARGET_SR

    # ── Audio turiga xos natijalar ──
    voiced_fraction: float = 0.0        # sustained vowel uchun


# ── Asosiy sinf ────────────────────────────────────────────────────────────

class AudioQualityPipeline:
    """
    Gepato-AI ovoz sifatini tekshirish pipeline'i.

    Barcha metodlar xatolarni "yutib" yuboradi va qayta urinishga
    imkon beradi — tizim HECH QACHON audio xatosi sababli crash bo'lmaydi.
    """

    def __init__(
        self,
        target_sr: int = TARGET_SR,
        ambient_warn_db: float = AMBIENT_WARN_DBFS,
        ambient_block_db: float = AMBIENT_BLOCK_DBFS,
        norm_target: float = NORM_TARGET,
        trim_top_db: float = TRIM_TOP_DB,
        clip_threshold: float = CLIP_THRESHOLD,
        clip_fraction: float = CLIP_FRACTION,
        internal_silence_s: float = INTERNAL_SILENCE_S,
    ):
        self.target_sr         = target_sr
        self.ambient_warn_db   = ambient_warn_db
        self.ambient_block_db  = ambient_block_db
        self.norm_target       = norm_target
        self.trim_top_db       = trim_top_db
        self.clip_threshold    = clip_threshold
        self.clip_fraction     = clip_fraction
        self.internal_silence_s = internal_silence_s

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────

    def check_ambient(self, audio_path: str) -> AmbientReport:
        """
        Yozishdan OLDIN chaqiriladi (2 soniyali ambient yozuv).
        Foydalanuvchi bu yozuvdan xabarsiz — avtomatik bo'ladi.

        Returns: AmbientReport (ok_to_record + xabar)
        """
        try:
            y, _ = self._load(audio_path)
            db = self._peak_db(y)

            if db >= self.ambient_block_db:
                return AmbientReport(
                    ambient_db=db,
                    ok_to_record=False,
                    warning=False,
                    blocked=True,
                    message=(
                        "Atrofdagi shovqin juda baland. "
                        "Iltimos, tinchroq xonaga o'ting va qayta urinib ko'ring."
                    ),
                    message_uz=(
                        "Shovqin darajasi: {:.0f} dB — ruxsat etilgan chegara: {} dB. "
                        "Ovoz tahlili sifati past bo'ladi.".format(abs(db), abs(self.ambient_block_db))
                    ),
                )

            if db >= self.ambient_warn_db:
                return AmbientReport(
                    ambient_db=db,
                    ok_to_record=True,
                    warning=True,
                    blocked=False,
                    message=(
                        "Atrofda biroz shovqin bor. "
                        "Iloji bo'lsa, tinchroq joyga o'ting."
                    ),
                    message_uz=(
                        "Shovqin darajasi: {:.0f} dB — tavsiya etilgan chegara: {} dB.".format(
                            abs(db), abs(self.ambient_warn_db)
                        )
                    ),
                )

            return AmbientReport(
                ambient_db=db,
                ok_to_record=True,
                warning=False,
                blocked=False,
                message="Muhit yaxshi. Yozishni boshlashingiz mumkin.",
                message_uz="Shovqin darajasi: {:.0f} dB — optimal.".format(abs(db)),
            )

        except Exception as exc:
            logger.warning("Ambient tekshiruv xatosi: %s", exc)
            # Xatolik bo'lsa — foydalanuvchini to'sib qo'ymaymiz
            return AmbientReport(
                ambient_db=-99.0,
                ok_to_record=True,
                warning=False,
                blocked=False,
                message="",
                message_uz="Muhit tekshiruvi o'tkazilmadi.",
            )

    def process(
        self,
        audio_path: str,
        audio_type: str = "sustained_vowel",  # "sustained_vowel" | "text_reading"
        min_duration_sec: float = 5.0,
        max_duration_sec: float = 60.0,
    ) -> AudioQualityReport:
        """
        Asosiy metod. Faylni yuklaydi, tekshiradi, qayta ishlaydi.

        audio_type:
          "sustained_vowel" → voiced_fraction tekshiruvi (> 0.85 kerak)
          "text_reading"    → ichki jimliklarni ham kesadi

        Returns: AudioQualityReport
          .passed          = True/False
          .processed_audio = normalize + trim qilingan numpy array
          .block_reason    = None (passed) yoki xato sababi
        """
        report = AudioQualityReport()

        try:
            y, sr = self._load(audio_path)
        except Exception as exc:
            logger.error("Audio yuklash xatosi (%s): %s", audio_path, exc)
            report.passed = False
            report.block_reason = "Audio faylni o'qib bo'lmadi. Qayta yozing."
            return report

        original_duration = len(y) / sr
        report.sample_rate = sr

        # ── 1. Davomiylik tekshiruvi ────────────────────────────────────
        if original_duration < min_duration_sec:
            report.passed = False
            report.block_reason = (
                f"Yozuv juda qisqa: {original_duration:.1f} soniya. "
                f"Kamida {min_duration_sec:.0f} soniya kerak."
            )
            return report

        if original_duration > max_duration_sec:
            # Kesib olish — to'xtatmaymiz
            y = y[: int(max_duration_sec * sr)]
            report.warnings.append(
                f"Yozuv uzun ({original_duration:.0f}s), "
                f"birinchi {max_duration_sec:.0f} soniya ishlatiladi."
            )

        # ── 2. Peak dB va clipping ──────────────────────────────────────
        report.peak_db  = self._peak_db(y)
        report.is_clipped = self._detect_clipping(y)

        if report.is_clipped:
            report.warnings.append(
                "Ovoz qirqilgan (clipping). Mikrofonni biroz uzoqroq ushlab, "
                "qayta yozing."
            )

        # ── 3. SNR hisoblash ────────────────────────────────────────────
        report.ambient_snr_db = self._compute_snr(y, sr)

        # ── 4. Normalizatsiya (-3 dBFS) ─────────────────────────────────
        y_norm = self._normalize(y)

        # ── 5. Jimlik kesish ────────────────────────────────────────────
        y_trimmed, trimmed_sec = self._trim_silence(
            y_norm, sr, audio_type=audio_type
        )
        report.silence_trimmed_sec = trimmed_sec
        report.final_duration_sec  = len(y_trimmed) / sr

        # Trim dan keyin yana davomiylik tekshiruvi
        if report.final_duration_sec < min_duration_sec:
            report.passed = False
            report.block_reason = (
                f"Jimlik kesib bo'lgach yozuv juda qisqa: "
                f"{report.final_duration_sec:.1f}s. Gapirish qismini uzaytiring."
            )
            return report

        # ── 6. Voiced fraction (faqat sustained vowel uchun) ────────────
        if audio_type == "sustained_vowel":
            report.voiced_fraction = self._voiced_fraction(y_trimmed, sr)
            if report.voiced_fraction < 0.85:
                report.warnings.append(
                    f"Ovozli qism kam ({report.voiced_fraction:.0%}). "
                    f"'Aaaa' tovushini uzluksiz ayting."
                )

        # ── 7. Yakuniy qaror ────────────────────────────────────────────
        # Clipping yoki past SNR → ogohlantirish, lekin bloklamaymiz
        # (Ma'lumot yo'qligidan yomon sifat yaxshiroq)
        report.processed_audio = y_trimmed
        report.passed = True

        logger.info(
            "Audio tekshiruv OK | type=%s dur=%.1fs snr=%.1fdB "
            "peak=%.1fdB clipped=%s voiced=%.2f",
            audio_type,
            report.final_duration_sec,
            report.ambient_snr_db,
            report.peak_db,
            report.is_clipped,
            report.voiced_fraction,
        )
        return report

    def process_from_bytes(
        self,
        audio_bytes: bytes,
        audio_type: str = "sustained_vowel",
        min_duration_sec: float = 5.0,
    ) -> AudioQualityReport:
        """
        Bytes dan to'g'ridan-to'g'ri qayta ishlash.
        Faylga saqlash zarur bo'lmagan holatlarda.
        """
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            return self.process(tmp_path, audio_type, min_duration_sec)
        finally:
            os.unlink(tmp_path)

    # ──────────────────────────────────────────────────────────────────────
    # PRIVATE METODLAR
    # ──────────────────────────────────────────────────────────────────────

    def _load(self, path: str) -> tuple[np.ndarray, int]:
        """
        Audiofaylni yuklaydi.
        - Mono ga aylantiradi
        - TARGET_SR ga resample qiladi
        - float32 formatga keltiradi
        """
        y, sr = librosa.load(str(path), sr=self.target_sr, mono=True, dtype=np.float32)
        return y, sr

    def _peak_db(self, y: np.ndarray) -> float:
        """Peak amplitudani dBFS ga aylantiradi."""
        peak = np.max(np.abs(y))
        if peak < 1e-10:
            return -99.0
        return float(20 * np.log10(peak + 1e-10))

    def _detect_clipping(self, y: np.ndarray) -> bool:
        """
        Clipping aniqlash.
        Agar > CLIP_FRACTION ta sample CLIP_THRESHOLD dan yuqori bo'lsa — clipped.
        """
        clipped = np.sum(np.abs(y) >= self.clip_threshold)
        return bool(clipped / (len(y) + 1) > self.clip_fraction)

    def _compute_snr(self, y: np.ndarray, sr: int) -> float:
        """
        Signal-to-Noise Ratio (dB).
        Ovozli qismlar = signal, jimlik qismlari = noise.
        """
        try:
            intervals = librosa.effects.split(y, top_db=20)
            if len(intervals) == 0:
                return 0.0

            signal_mask = np.zeros(len(y), dtype=bool)
            for s, e in intervals:
                signal_mask[s:e] = True

            signal_rms = np.sqrt(np.mean(y[signal_mask] ** 2)) if signal_mask.any() else 1e-10
            noise_rms  = np.sqrt(np.mean(y[~signal_mask] ** 2)) if (~signal_mask).any() else 1e-10

            if noise_rms < 1e-10:
                return 40.0

            snr = 20 * np.log10(signal_rms / (noise_rms + 1e-10))
            return float(np.clip(snr, -20.0, 60.0))

        except Exception as exc:
            logger.warning("SNR hisoblash xatosi: %s", exc)
            return 0.0

    def _normalize(self, y: np.ndarray) -> np.ndarray:
        """
        Peak normalizatsiya → -3 dBFS (0.708 linear).
        Original faylni o'zgartirmaydi — yangi array qaytaradi.
        """
        peak = np.max(np.abs(y))
        if peak < 1e-10:
            return y.copy()
        return (y / peak * self.norm_target).astype(np.float32)

    def _trim_silence(
        self,
        y: np.ndarray,
        sr: int,
        audio_type: str,
    ) -> tuple[np.ndarray, float]:
        """
        Jimlik kesish.
        - Bosh/oxir: har ikkala audio tur uchun
        - Ichki uzun jimlik (>2s): faqat text_reading uchun

        Returns: (trimmed_audio, trimmed_seconds)
        """
        original_len = len(y)

        # Bosh va oxir jimlik
        try:
            y_trimmed, _ = librosa.effects.trim(y, top_db=self.trim_top_db)
        except Exception:
            y_trimmed = y

        # Text reading: ichki uzun jimliklarni ham kesish
        if audio_type == "text_reading":
            y_trimmed = self._remove_long_internal_silences(y_trimmed, sr)

        trimmed_sec = (original_len - len(y_trimmed)) / sr
        return y_trimmed, max(0.0, trimmed_sec)

    def _remove_long_internal_silences(
        self, y: np.ndarray, sr: int
    ) -> np.ndarray:
        """
        Text reading da 2 soniyadan uzun ichki jimliklarni olib tashlaydi.
        HE bemorlarida pauzalar uzayadi — lekin juda uzun jimlik
        (bemor to'xtab qolgan) tahlilni buzadi.
        """
        try:
            intervals = librosa.effects.split(y, top_db=SPLIT_TOP_DB)
            if len(intervals) == 0:
                return y

            max_silence_samples = int(self.internal_silence_s * sr)
            segments = []

            for i, (start, end) in enumerate(intervals):
                segments.append(y[start:end])

                # Keyingi segment boshlanadigan joyga qadar jimlik
                if i < len(intervals) - 1:
                    gap_len = intervals[i + 1][0] - end
                    if gap_len <= max_silence_samples:
                        # Qisqa jimlik — saqlab qolamiz
                        segments.append(y[end: intervals[i + 1][0]])
                    else:
                        # Uzun jimlik — maksimal miqdorga qisqartiramiz
                        segments.append(np.zeros(max_silence_samples, dtype=y.dtype))

            return np.concatenate(segments) if segments else y

        except Exception as exc:
            logger.warning("Ichki jimlik kesish xatosi: %s", exc)
            return y

    def _voiced_fraction(self, y: np.ndarray, sr: int) -> float:
        """
        Ovozli qismlar ulushi (sustained vowel sifati uchun).
        1.0 = to'liq ovozli | 0.0 = to'liq jim
        """
        try:
            intervals = librosa.effects.split(y, top_db=SPLIT_TOP_DB)
            if len(intervals) == 0:
                return 0.0
            voiced_samples = sum(e - s for s, e in intervals)
            return float(voiced_samples / (len(y) + 1))
        except Exception:
            return 0.0


# ── Singleton ──────────────────────────────────────────────────────────────
_pipeline: Optional[AudioQualityPipeline] = None


def get_audio_quality_pipeline() -> AudioQualityPipeline:
    """Modul darajasidagi singleton — har safar yangi ob'ekt yaratmaydi."""
    global _pipeline
    if _pipeline is None:
        _pipeline = AudioQualityPipeline()
    return _pipeline
