# Gepato-AI v2.0 — Ishga Tushirish

## Serverga joylash
Bitta buyruq (NafasNazorat bilan bir xil tartib): **[deploy/README.md](deploy/README.md)** —
`sudo bash deploy/install.sh gepato.plagiatx.uz <email>`.

## Lokal ishga tushirish
```bash
cp .env.example .env
docker compose up --build
# Sayt: http://127.0.0.1:3200   API sxema: http://127.0.0.1:3200/docs (faqat development)
# Bemor: /patient   Labeling: /doctor/labeling
# Kirish: .env dagi BOOTSTRAP_ADMIN_EMAIL / BOOTSTRAP_ADMIN_PASSWORD
docker compose exec -T api python scripts/selfcheck.py
```
API faqat kirgan shifokor/admin uchun; ro'yxatdan o'tkazish — faqat admin (`scripts/create_user.py`).

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
