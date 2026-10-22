-- Migration 006: patients jadvaliga trend + stratification ustunlar
-- Gepato-AI v2.0

ALTER TABLE patients
    ADD COLUMN IF NOT EXISTS disease_stage       VARCHAR(50),
    ADD COLUMN IF NOT EXISTS etiology            VARCHAR(50),
    ADD COLUMN IF NOT EXISTS meld_na             FLOAT,
    ADD COLUMN IF NOT EXISTS alt_level           FLOAT,
    ADD COLUMN IF NOT EXISTS ast_level           FLOAT,
    ADD COLUMN IF NOT EXISTS fibrosis_stage      VARCHAR(10),
    -- Trend engine fields (P6)
    ADD COLUMN IF NOT EXISTS baseline_he_score        FLOAT,
    ADD COLUMN IF NOT EXISTS baseline_voice_score     FLOAT,
    ADD COLUMN IF NOT EXISTS baseline_tremor_score    FLOAT,
    ADD COLUMN IF NOT EXISTS baseline_cognitive_score FLOAT,
    ADD COLUMN IF NOT EXISTS baseline_std             FLOAT,
    ADD COLUMN IF NOT EXISTS baseline_window_n        INTEGER DEFAULT 5,
    ADD COLUMN IF NOT EXISTS baseline_last_updated    TIMESTAMP,
    ADD COLUMN IF NOT EXISTS trend_slope_7d           FLOAT,
    ADD COLUMN IF NOT EXISTS trend_r2_7d              FLOAT,
    ADD COLUMN IF NOT EXISTS progression_slope_30d    FLOAT,
    ADD COLUMN IF NOT EXISTS progression_flag         BOOLEAN DEFAULT FALSE;

ALTER TABLE voice_recordings
    ADD COLUMN IF NOT EXISTS audio_type VARCHAR(20) DEFAULT 'sustained_vowel';

ALTER TABLE predictions
    ADD COLUMN IF NOT EXISTS composite_he_index          FLOAT,
    ADD COLUMN IF NOT EXISTS voice_component             FLOAT,
    ADD COLUMN IF NOT EXISTS tremor_component            FLOAT,
    ADD COLUMN IF NOT EXISTS cognitive_component         FLOAT,
    ADD COLUMN IF NOT EXISTS stage_adjusted_threshold    FLOAT,
    ADD COLUMN IF NOT EXISTS confidence_interval_lower   FLOAT,
    ADD COLUMN IF NOT EXISTS confidence_interval_upper   FLOAT;
