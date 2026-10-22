"""
Gepato-AI — Celery Tasks
- Har kecha 02:00: model retrain
- Har soat: trend tahlili
- Yangi label kelganda: retrain tekshiruvi
"""
from celery import Celery
from celery.schedules import crontab
from core.config import settings

celery_app = Celery(
    "gepato",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_expires=3600,
    timezone="Asia/Tashkent",
    beat_schedule={
        # Har kecha 02:00 — retrain
        "nightly-retrain": {
            "task": "core.tasks.nightly_retrain",
            "schedule": crontab(hour=2, minute=0),
        },
        # Har 4 soatda — trend alert tekshiruvi
        "trend-check": {
            "task": "core.tasks.check_trends",
            "schedule": crontab(minute=0, hour="*/4"),
        },
    },
)


@celery_app.task(name="core.tasks.nightly_retrain")
def nightly_retrain():
    """
    Har kecha yangi labeled datalar bilan qayta trening.
    """
    import asyncio
    from db.database import async_session
    from db.models import AcousticFeatures
    from sqlalchemy import select
    import pandas as pd
    from ml.model import get_model, FEATURE_NAMES

    async def _run():
        async with async_session() as db:
            # Ishlatilmagan labeled featurelar
            result = await db.execute(
                select(AcousticFeatures)
                .where(
                    AcousticFeatures.is_labeled == True,
                    AcousticFeatures.used_in_training == False,
                )
            )
            new_feats = result.scalars().all()

            if len(new_feats) < settings.MIN_SAMPLES_FOR_TRAINING:
                return {"skipped": True, "reason": f"Yetarli data yo'q: {len(new_feats)}"}

            # DataFrame
            rows = []
            for f in new_feats:
                row = {name: getattr(f, name, 0) or 0 for name in FEATURE_NAMES if not name.startswith("mfcc")}
                mfcc = f.mfcc or [0] * 13
                for i in range(6):
                    row[f"mfcc_{i}"] = mfcc[i] if i < len(mfcc) else 0
                row["label"] = f.label
                rows.append(row)

            df = pd.DataFrame(rows)
            model = get_model()
            train_result = model.retrain(df)

            if train_result["success"]:
                # used_in_training = True qilib belgilash
                for f in new_feats:
                    f.used_in_training = True
                    f.model_version = train_result["metrics"]["version"]
                await db.commit()

            return train_result

    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(_run())
    loop.close()
    return result


@celery_app.task(name="core.tasks.check_trends")
def check_trends():
    """
    Barcha faol bemorlarning trendini tekshirib,
    zarur bo'lsa alert yaratadi.
    """
    import asyncio

    async def _run():
        from db.database import async_session
        from db.models import Patient, Prediction, Alert, RiskLevel
        from sqlalchemy import select
        from datetime import datetime, timedelta

        async with async_session() as db:
            # Faol bemorlar
            result = await db.execute(
                select(Patient).where(Patient.is_active == True)
            )
            patients = result.scalars().all()
            alerts_created = 0

            for patient in patients:
                # Oxirgi 3 kunlik skor
                hist = await db.execute(
                    select(Prediction.risk_score, Prediction.predicted_at)
                    .where(
                        Prediction.patient_id == patient.id,
                        Prediction.predicted_at >= datetime.utcnow() - timedelta(days=3)
                    )
                    .order_by(Prediction.predicted_at.desc())
                    .limit(3)
                )
                scores = [row[0] for row in hist.fetchall()]

                if len(scores) < 2:
                    continue

                # Deterioration aniqlash: 3 kun ketma-ket oshish
                if all(scores[i] < scores[i+1] for i in range(len(scores)-1)):
                    slope = scores[0] - scores[-1]
                    if slope > 0.1 and patient.doctor_id:
                        alert = Alert(
                            patient_id=patient.id,
                            doctor_id=patient.doctor_id,
                            title=f"📈 Trend ogohlantiruv: {patient.full_name}",
                            message=(
                                f"3 kun ketma-ket yomonlashish aniqlandi. "
                                f"Skor o'zgarishi: {scores[-1]:.2f} → {scores[0]:.2f}. "
                                f"Klinik tekshiruv tavsiya etiladi."
                            ),
                            severity="warning",
                            trigger_type="trend",
                        )
                        db.add(alert)
                        alerts_created += 1

            await db.commit()
            return {"alerts_created": alerts_created, "patients_checked": len(patients)}

    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(_run())
    loop.close()
    return result
