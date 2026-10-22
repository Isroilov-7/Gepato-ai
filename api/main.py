"""
Gepato-AI v2.0 — FastAPI Application
"""
from fastapi import FastAPI
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Gepato-AI v2.0 ishga tushmoqda...")
    await init_db()
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
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Production: domenni cheklash
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files (frontend) ──────────────────────────────────────────────
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

# ── Routes ────────────────────────────────────────────────────────────────
from api.routes import auth, patients, voice, predictions, admin, agents
from api.routes import tremor, cognitive   # P4, P5 — yangi

app.include_router(auth.router,        prefix="/api/auth",        tags=["Auth"])
app.include_router(patients.router,    prefix="/api/patients",    tags=["Patients"])
app.include_router(voice.router,       prefix="/api/voice",       tags=["Voice"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["Predictions"])
app.include_router(agents.router,      prefix="/api/agents",      tags=["AI Agents"])
app.include_router(admin.router,       prefix="/api/admin",       tags=["Admin"])
app.include_router(tremor.router,                                  tags=["Tremor"])    # P4
app.include_router(cognitive.router,                               tags=["Cognitive"]) # P5


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
