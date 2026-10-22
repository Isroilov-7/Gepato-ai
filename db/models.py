"""
Gepato-AI — PostgreSQL Schema
Barcha jadvallar shu yerda.
"""
from datetime import datetime
from enum import Enum as PyEnum
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime,
    ForeignKey, Text, JSON, Enum, Index
)
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


# ── Enums ──────────────────────────────────────────────

class UserRole(str, PyEnum):
    PATIENT = "patient"
    DOCTOR  = "doctor"
    ADMIN   = "admin"

class ChildPughClass(str, PyEnum):
    A = "A"
    B = "B"
    C = "C"

class RiskLevel(str, PyEnum):
    LOW     = "low"      # <40%
    MEDIUM  = "medium"   # 40-70%
    HIGH    = "high"     # >70%

class AlertStatus(str, PyEnum):
    PENDING  = "pending"
    REVIEWED = "reviewed"
    RESOLVED = "resolved"

class AudioQuality(str, PyEnum):
    GOOD = "good"
    POOR = "poor"      # SNR past
    FAILED = "failed"  # qayta yozish kerak


# ── 1. Users ──────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    email         = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name     = Column(String(255), nullable=False)
    role          = Column(Enum(UserRole), default=UserRole.PATIENT, nullable=False)
    phone         = Column(String(20))
    is_active     = Column(Boolean, default=True)
    created_at    = Column(DateTime, server_default=func.now())
    last_login    = Column(DateTime)

    # Relations
    patients_as_doctor = relationship("Patient", back_populates="doctor",
                                       foreign_keys="Patient.doctor_id")
    patient_profile    = relationship("Patient", back_populates="user",
                                       foreign_keys="Patient.user_id", uselist=False)


# ── 2. Patients ────────────────────────────────────────

class Patient(Base):
    __tablename__ = "patients"

    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, ForeignKey("users.id"), unique=True)
    doctor_id       = Column(Integer, ForeignKey("users.id"))
    full_name       = Column(String(255), nullable=False)
    age             = Column(Integer, nullable=False)
    gender          = Column(String(10))
    diagnosis       = Column(String(500))

    # Child-Pugh
    child_pugh_class = Column(Enum(ChildPughClass), nullable=False)
    # Lab values (Child-Pugh komponenti)
    bilirubin        = Column(Float)   # mg/dL
    albumin          = Column(Float)   # g/dL
    inr              = Column(Float)
    ascites_grade    = Column(Integer, default=0)  # 0,1,2
    encephalopathy_grade = Column(Integer, default=0)  # West-Haven 0-4

    # Current medication
    lactulose_dose   = Column(String(100))
    rifaximin        = Column(Boolean, default=False)
    other_meds       = Column(Text)

    # Baseline voice (birinchi 7 yozuvdan hisoblangan)
    baseline_jitter   = Column(Float)
    baseline_shimmer  = Column(Float)
    baseline_hnr      = Column(Float)
    baseline_computed = Column(Boolean, default=False)
    baseline_date     = Column(DateTime)

    # Risk
    current_risk_level = Column(Enum(RiskLevel), default=RiskLevel.LOW)
    last_risk_score    = Column(Float, default=0.0)
    last_assessment    = Column(DateTime)

    is_active   = Column(Boolean, default=True)
    enrolled_at = Column(DateTime, server_default=func.now())
    notes       = Column(Text)

    # Relations
    user        = relationship("User", back_populates="patient_profile",
                               foreign_keys=[user_id])
    doctor      = relationship("User", back_populates="patients_as_doctor",
                               foreign_keys=[doctor_id])
    recordings  = relationship("VoiceRecording", back_populates="patient",
                               order_by="VoiceRecording.recorded_at.desc()")
    predictions = relationship("Prediction", back_populates="patient")
    alerts      = relationship("Alert", back_populates="patient")

    __table_args__ = (
        Index("ix_patients_doctor_id", "doctor_id"),
        Index("ix_patients_risk", "current_risk_level"),
    )


# ── 3. Voice Recordings ────────────────────────────────

class VoiceRecording(Base):
    __tablename__ = "voice_recordings"

    id          = Column(Integer, primary_key=True, index=True)
    patient_id  = Column(Integer, ForeignKey("patients.id"), nullable=False)
    file_path   = Column(String(500), nullable=False)
    file_size   = Column(Integer)          # bytes
    duration    = Column(Float)            # seconds
    sample_rate = Column(Integer)          # Hz
    recorded_at = Column(DateTime, server_default=func.now())

    # Sifat
    quality     = Column(Enum(AudioQuality), default=AudioQuality.GOOD)
    snr_db      = Column(Float)            # Signal-to-noise ratio
    quality_notes = Column(String(500))    # Nima muammo bo'lsa

    # Feature extraction holati
    features_extracted = Column(Boolean, default=False)
    extraction_error   = Column(String(500))

    # Relations
    patient  = relationship("Patient", back_populates="recordings")
    features = relationship("AcousticFeatures", back_populates="recording",
                            uselist=False)
    prediction = relationship("Prediction", back_populates="recording",
                              uselist=False)

    __table_args__ = (
        Index("ix_recordings_patient_date", "patient_id", "recorded_at"),
    )


# ── 4. Acoustic Features ──────────────────────────────

class AcousticFeatures(Base):
    """
    Har bir yozuvdan chiqarilgan biomarkerlar.
    Bu jadval modelni o'qitish uchun asosiy ma'lumot.
    """
    __tablename__ = "acoustic_features"

    id           = Column(Integer, primary_key=True, index=True)
    recording_id = Column(Integer, ForeignKey("voice_recordings.id"),
                          unique=True, nullable=False)
    patient_id   = Column(Integer, ForeignKey("patients.id"), nullable=False)
    extracted_at = Column(DateTime, server_default=func.now())

    # ── Asosiy 5 biomarker ──
    jitter_local    = Column(Float)   # % — chastota beqarorligi
    jitter_ppq5     = Column(Float)   # smoothed jitter
    shimmer_local   = Column(Float)   # % — amplituda beqarorligi
    shimmer_apq11   = Column(Float)
    hnr             = Column(Float)   # dB — harmonik-shovqin nisbati
    speech_rate     = Column(Float)   # syl/sec
    tremor_index    = Column(Float)   # Praat tremor

    # ── Kengaytirilgan ──
    f0_mean         = Column(Float)   # Hz — asosiy chastota
    f0_std          = Column(Float)
    f0_range        = Column(Float)
    pause_rate      = Column(Float)   # pauzalar soni/soniya
    voiced_fraction = Column(Float)   # ovozli qismlar ulushi

    # MFCC (13 koeffitsient) — JSON array sifatida
    mfcc            = Column(JSON)    # [c1, c2, ..., c13]
    mfcc_delta      = Column(JSON)    # MFCC birinchi hosilasi

    # Bemorning o'z baseline'idan og'ish (% da)
    jitter_deviation   = Column(Float)
    shimmer_deviation  = Column(Float)
    hnr_deviation      = Column(Float)
    composite_deviation = Column(Float)  # umumiy og'ish skori

    # Active Learning uchun
    # True = shifokor label bergan, False = avtomatik
    is_labeled    = Column(Boolean, default=False)
    label         = Column(Integer)   # 0=normal, 1=mHE, 2=overt HE
    child_pugh_at_time = Column(Enum(ChildPughClass))
    labeled_by    = Column(Integer, ForeignKey("users.id"))
    labeled_at    = Column(DateTime)

    # Model training uchun ishlatildimi?
    used_in_training = Column(Boolean, default=False)
    model_version    = Column(String(50))

    # Relations
    recording = relationship("VoiceRecording", back_populates="features")

    __table_args__ = (
        Index("ix_features_patient_labeled", "patient_id", "is_labeled"),
        Index("ix_features_training", "used_in_training", "is_labeled"),
    )


# ── 5. Predictions ────────────────────────────────────

class Prediction(Base):
    """Model chiqargan har bir bashorat."""
    __tablename__ = "predictions"

    id           = Column(Integer, primary_key=True, index=True)
    patient_id   = Column(Integer, ForeignKey("patients.id"), nullable=False)
    recording_id = Column(Integer, ForeignKey("voice_recordings.id"))
    predicted_at = Column(DateTime, server_default=func.now())

    # Natija
    risk_score   = Column(Float, nullable=False)   # 0.0 – 1.0
    risk_level   = Column(Enum(RiskLevel), nullable=False)
    confidence   = Column(Float)                   # model confidence

    # SHAP tushuntirish
    shap_values  = Column(JSON)    # {feature: shap_value}
    top_factors  = Column(JSON)    # [{"feature": "shimmer", "contribution": 0.31}]
    explanation  = Column(Text)    # Inson o'qiydigan tushuntirish

    # Trend (oldingi 7 kun bilan taqqoslash)
    trend_direction = Column(String(20))  # "improving", "stable", "deteriorating"
    trend_magnitude = Column(Float)

    # Qaysi model versiyasi
    model_version = Column(String(50))

    # Shifokor tasdig'i (active learning)
    doctor_confirmed  = Column(Boolean)
    doctor_feedback   = Column(String(500))
    confirmed_by      = Column(Integer, ForeignKey("users.id"))
    confirmed_at      = Column(DateTime)

    # Relations
    patient   = relationship("Patient", back_populates="predictions")
    recording = relationship("VoiceRecording", back_populates="prediction")
    alert     = relationship("Alert", back_populates="prediction", uselist=False)

    __table_args__ = (
        Index("ix_predictions_patient_date", "patient_id", "predicted_at"),
    )


# ── 6. Alerts ─────────────────────────────────────────

class Alert(Base):
    __tablename__ = "alerts"

    id           = Column(Integer, primary_key=True, index=True)
    patient_id   = Column(Integer, ForeignKey("patients.id"), nullable=False)
    prediction_id = Column(Integer, ForeignKey("predictions.id"))
    doctor_id    = Column(Integer, ForeignKey("users.id"), nullable=False)

    created_at   = Column(DateTime, server_default=func.now())
    status       = Column(Enum(AlertStatus), default=AlertStatus.PENDING)

    # Alert sababi
    title        = Column(String(255), nullable=False)
    message      = Column(Text, nullable=False)
    severity     = Column(String(20))   # "critical", "warning", "info"
    trigger_type = Column(String(50))   # "threshold", "trend", "composite"

    # Shifokor javobi
    reviewed_at  = Column(DateTime)
    reviewed_by  = Column(Integer, ForeignKey("users.id"))
    action_taken = Column(Text)

    # Relations
    patient    = relationship("Patient", back_populates="alerts")
    prediction = relationship("Prediction", back_populates="alert")

    __table_args__ = (
        Index("ix_alerts_doctor_status", "doctor_id", "status"),
        Index("ix_alerts_patient_date", "patient_id", "created_at"),
    )


# ── 7. Model Versions ─────────────────────────────────

class ModelVersion(Base):
    """Har bir train qilingan model versiyasi."""
    __tablename__ = "model_versions"

    id            = Column(Integer, primary_key=True, index=True)
    version       = Column(String(50), unique=True, nullable=False)  # "v1.0", "v1.1"
    created_at    = Column(DateTime, server_default=func.now())
    is_active     = Column(Boolean, default=False)

    # Training stats
    train_samples = Column(Integer)
    test_samples  = Column(Integer)
    sensitivity   = Column(Float)
    specificity   = Column(Float)
    auc_roc       = Column(Float)
    f1_score      = Column(Float)

    # Feature importances
    feature_importances = Column(JSON)

    # File
    model_path    = Column(String(500))
    notes         = Column(Text)

    __table_args__ = (
        Index("ix_model_active", "is_active"),
    )


# ── 8. Agent Logs ─────────────────────────────────────

class AgentLog(Base):
    """AI agent so'rovlari tarixi."""
    __tablename__ = "agent_logs"

    id         = Column(Integer, primary_key=True, index=True)
    agent_name = Column(String(50), nullable=False)   # "atlas", "oracle"
    user_id    = Column(Integer, ForeignKey("users.id"))
    patient_id = Column(Integer, ForeignKey("patients.id"))

    query      = Column(Text, nullable=False)
    response   = Column(Text, nullable=False)
    tokens_used = Column(Integer)
    latency_ms  = Column(Integer)
    created_at  = Column(DateTime, server_default=func.now())


# ── Migration 004: Tremor Readings ────────────────────────────────────────

class TremorReading(Base):
    """Akselerometr tremor o'lchovlari. api/routes/tremor.py dan to'ldiriladi."""
    __tablename__ = "tremor_readings"

    id                  = Column(Integer, primary_key=True, index=True)
    patient_id          = Column(Integer, ForeignKey("patients.id"), nullable=False)
    recorded_at         = Column(DateTime, server_default=func.now())
    rms_amplitude       = Column(Float)
    dominant_freq_hz    = Column(Float)
    tremor_power_8_12hz = Column(Float)
    stability_score     = Column(Float)
    tremor_score        = Column(Float)         # 0–1, HEScoreEngine uchun
    disease_stage       = Column(String(50))
    etiology            = Column(String(50))
    raw_samples         = Column(JSON)

    __table_args__ = (
        Index("ix_tremor_patient_date", "patient_id", "recorded_at"),
    )


# ── Migration 005: Cognitive Tests ───────────────────────────────────────

class CognitiveTest(Base):
    """Stroop + NCT-A natijalar. api/routes/cognitive.py dan to'ldiriladi."""
    __tablename__ = "cognitive_tests"

    id                    = Column(Integer, primary_key=True, index=True)
    patient_id            = Column(Integer, ForeignKey("patients.id"), nullable=False)
    test_date             = Column(DateTime, server_default=func.now())
    stroop_mean_rt_ms     = Column(Float)
    stroop_std_rt         = Column(Float)
    stroop_error_rate     = Column(Float)
    stroop_score          = Column(Float)
    nct_total_time_sec    = Column(Float)
    nct_error_count       = Column(Integer)
    nct_mean_inter_tap_ms = Column(Float)
    nct_score             = Column(Float)
    cognitive_score       = Column(Float)       # 0–1 composite
    disease_stage         = Column(String(50))
    raw_stroop_trials     = Column(JSON)
    raw_nct_sequence      = Column(JSON)

    __table_args__ = (
        Index("ix_cognitive_patient_date", "patient_id", "test_date"),
    )


# ── Migration 006: Patients trend fields ─────────────────────────────────
# Quyidagi ustunlar patients jadvaliga Alembic migration orqali qo'shiladi.
# models.py da Patient klasiga ham qo'shish:
# (SQLAlchemy init_db() bilan avtomatik yaratiladi — faqat yangi jadvallar uchun)
