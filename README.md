# Gepato-AI v2.0 — Ishga Tushirish

## Tezkor boshlash
```bash
cp .env.example .env   # ANTHROPIC_API_KEY kiriting
docker-compose up --build
# API: http://localhost:8000/docs
# Bemor: http://localhost:8000/patient
# Labeling: http://localhost:8000/doctor/labeling
```

## Modullar holati
| Modul | Holat |
|-------|-------|
| P1 AudioQualityPipeline (ml/audio_quality.py) | TAYYOR |
| P2 FeatureExtractor - 23 feature, 6 strata (ml/feature_extraction.py) | TAYYOR |
| P3 HEScoreEngine + StratifiedHEModel (ml/scoring.py, ml/model.py) | TAYYOR |
| P4 Tremor API (api/routes/tremor.py) | TAYYOR |
| P5 Cognitive API - Stroop+NCT (api/routes/cognitive.py) | TAYYOR |
| P6 TrendEngine - z-score, 7d/30d (ml/trend_engine.py) | TAYYOR |
| P7 Patient Daily Flow PWA (frontend/patient_daily_flow.html) | TAYYOR |
| P8 Doctor Labeling UI (frontend/doctor_labeling_ui.html) | TAYYOR |
| P9 Service Worker + manifest (frontend/sw.js, manifest.json) | TAYYOR |

## Migration
```bash
docker-compose exec api python -c "from db.database import init_db; import asyncio; asyncio.run(init_db())"
docker-compose exec db psql -U gepato -d gepato < migrations/006_patients_trend_fields.sql
```
