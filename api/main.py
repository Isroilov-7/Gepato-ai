"""
Gepato-AI v2.0 — FastAPI Application
"""
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
from pathlib import Path

from core.config import settings
from db.database import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


problems = settings.production_problems()
if problems:  # xato sozlangan server umuman ochilmasin (ochiq qolgandan ko'ra ishlamagani yaxshi)
    raise RuntimeError("Production sozlamalari xato: " + "; ".join(problems))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Gepato-AI v2.0 ishga tushmoqda...")
    await init_db()
    from db.database import async_session
    from api.routes.auth import ensure_bootstrap_admin
    async with async_session() as s:
        await ensure_bootstrap_admin(s)
    from ml.model import get_model
    model = get_model()
    logger.info("✅ Model yuklandi: %s", model.version)
    yield
    logger.info("👋 Gepato-AI to'xtatilmoqda")


app = FastAPI(
    title="Gepato-AI API v2.0",
    version=settings.VERSION,
    description="Multimodal Hepatic Encephalopathy Monitoring Platform",
    lifespan=lifespan,
    # Productionda API sxemasi ochiq e'lon qilinmaydi (ENABLE_DOCS=true bilan yoqiladi)
    docs_url="/docs" if (settings.ENABLE_DOCS or not settings.is_production) else None,
    redoc_url=None,
    openapi_url="/openapi.json" if (settings.ENABLE_DOCS or not settings.is_production) else None,
)

# Frontend shu domendan — CORS kerak emas. Ilgari allow_origins=["*"] + credentials edi.
if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,  # token Authorization sarlavhasida, cookie yo'q
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

# ── Static files (frontend) ──────────────────────────────────────────────
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
    if (frontend_dir / "vendor").exists():  # index.html: vendor/chart.umd.js (CDN'siz)
        app.mount("/vendor", StaticFiles(directory=str(frontend_dir / "vendor")), name="vendor")

# ── Routes ────────────────────────────────────────────────────────────────
# Bemor ma'lumoti va AI — faqat kirgan shifokor/admin; admin bo'limi — faqat admin.
# Himoya router darajasida: shu routerlarga qo'shilgan yangi endpoint ham tokensiz ochilmaydi.
from api.routes import auth, patients, voice, predictions, admin, agents
from api.routes import tremor, cognitive   # P4, P5 — yangi
from api.security import require_admin, require_clinician

CLIN = [Depends(require_clinician)]
app.include_router(auth.router,        prefix="/api/auth",        tags=["Auth"])
app.include_router(patients.router,    prefix="/api/patients",    tags=["Patients"],    dependencies=CLIN)
app.include_router(voice.router,       prefix="/api/voice",       tags=["Voice"],       dependencies=CLIN)
app.include_router(predictions.router, prefix="/api/predictions", tags=["Predictions"], dependencies=CLIN)
app.include_router(agents.router,      prefix="/api/agents",      tags=["AI Agents"],   dependencies=CLIN)
app.include_router(admin.router,       prefix="/api/admin",       tags=["Admin"],
                   dependencies=[Depends(require_admin)])
app.include_router(tremor.router,      tags=["Tremor"],    dependencies=CLIN)  # P4 (/api/tremor)
app.include_router(cognitive.router,   tags=["Cognitive"], dependencies=CLIN)  # P5 (/api/cognitive)


@app.get("/")
async def root():
    """Bosh sahifa: bemor / shifokor / admin paneli (frontend/index.html)."""
    from fastapi.responses import FileResponse
    p = frontend_dir / "index.html"
    if p.exists():
        return FileResponse(str(p))
    return {"service": "Gepato-AI", "version": "2.0", "status": "running"}


@app.get("/api/status")
async def status():
    return {"service": "Gepato-AI", "version": "2.0", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/patient")
async def patient_app():
    """PWA patient portal."""
    from fastapi.responses import FileResponse
    p = frontend_dir / "patient_daily_flow.html"
    if p.exists():
        return FileResponse(str(p))
    return {"error": "Frontend not found"}

@app.get("/doctor/labeling")
async def doctor_labeling():
    """Doctor labeling UI."""
    from fastapi.responses import FileResponse
    p = frontend_dir / "doctor_labeling_ui.html"
    if p.exists():
        return FileResponse(str(p))
    return {"error": "Frontend not found"}
