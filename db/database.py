from pathlib import Path
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from core.config import settings
from db.models import Base

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_size=10,
    max_overflow=20,
)

async_session = async_sessionmaker(engine, expire_on_commit=False)


MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"

# create_all mavjud jadvalga yangi ustun QO'SHMAYDI — eski bazada ham ishlashi uchun
_EXTRA_COLUMNS = (
    "ALTER TABLE patients ADD COLUMN IF NOT EXISTS ammonia_level FLOAT",
    "ALTER TABLE acoustic_features ADD COLUMN IF NOT EXISTS extra_features JSON",
)


async def init_db():
    """Jadvallar + migrations/*.sql (idempotent: ADD COLUMN IF NOT EXISTS).
    Ilgari 006-migratsiya qo'lda bajarilishi kerak edi; bajarilmasa tremor/kognitiv/trend xato berardi."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        for sql in _EXTRA_COLUMNS:
            await conn.execute(text(sql))
        for f in sorted(MIGRATIONS_DIR.glob("*.sql")):
            body = "\n".join(l for l in f.read_text(encoding="utf-8").splitlines()
                             if not l.strip().startswith("--"))
            for stmt in body.split(";"):
                if stmt.strip():
                    await conn.execute(text(stmt))


async def get_db() -> AsyncSession:
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
