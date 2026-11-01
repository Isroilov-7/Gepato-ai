"""Auth route.

Ro'yxatdan o'tkazish FAQAT admin tokeni bilan (ilgari ochiq edi va `role` ni so'rovchi o'zi tanlardi —
har kim o'zini admin qilib ro'yxatdan o'tkaza olardi). Birinchi admin .env dagi BOOTSTRAP_ADMIN_* dan
ishga tushishda yaratiladi (`ensure_bootstrap_admin`); keyingi shifokorlarni admin qo'shadi
(`POST /api/auth/register` yoki `python scripts/create_user.py`).
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, select
from passlib.context import CryptContext
from jose import JWTError, jwt

from core.config import settings
from db.database import get_db
from db import models

logger = logging.getLogger(__name__)
router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")
# Topilmagan email uchun ham bcrypt tekshiruvi — javob vaqtidan hisob borligi bilinmasin
_DUMMY_HASH = pwd_context.hash("gepato-timing-guard")


class Token(BaseModel):
    access_token: str
    token_type: str
    role: str
    full_name: str


class UserCreate(BaseModel):
    email: str = Field(..., min_length=5, max_length=255)
    password: str = Field(..., min_length=10, max_length=72)  # bcrypt 72 baytdan keyinini kesadi
    full_name: str = Field(..., min_length=2, max_length=255)
    role: models.UserRole = models.UserRole.DOCTOR
    phone: str = Field("", max_length=20)

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        v = v.strip().lower()
        if "@" not in v or v.startswith("@") or v.endswith("@") or " " in v:
            raise ValueError("Email noto'g'ri")
        return v


def create_token(data: dict) -> str:
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode({**data, "exp": expire}, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


_UNAUTH = HTTPException(status_code=401, detail="Kirish talab qilinadi",
                        headers={"WWW-Authenticate": "Bearer"})


async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id = int(payload.get("sub") or 0)
    except (JWTError, ValueError, TypeError):
        raise _UNAUTH
    user = await db.get(models.User, user_id) if user_id else None
    if not user or user.is_active is False:  # o'chirilgan xodimning eski tokeni ham ishlamaydi
        raise _UNAUTH
    return user


async def _require_admin(current_user: models.User = Depends(get_current_user)) -> models.User:
    if current_user.role != models.UserRole.ADMIN:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Faqat admin")
    return current_user


async def create_user(db: AsyncSession, body: UserCreate) -> models.User:
    exists = await db.execute(select(models.User).where(func.lower(models.User.email) == body.email))
    if exists.scalar_one_or_none():
        raise HTTPException(400, "Email allaqachon ro'yxatdan o'tgan")
    user = models.User(email=body.email, hashed_password=pwd_context.hash(body.password),
                       full_name=body.full_name.strip(), role=body.role, phone=body.phone)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@router.post("/register", status_code=201)
async def register(body: UserCreate, db: AsyncSession = Depends(get_db),
                   admin: models.User = Depends(_require_admin)):
    user = await create_user(db, body)
    logger.info("Foydalanuvchi qo'shildi: id=%s role=%s (admin id=%s)", user.id, user.role.value, admin.id)
    return {"id": user.id, "email": user.email, "role": user.role}


class UserPatch(BaseModel):
    is_active: Optional[bool] = None
    password: Optional[str] = Field(None, min_length=10, max_length=72)


@router.get("/users")
async def list_users(db: AsyncSession = Depends(get_db), admin: models.User = Depends(_require_admin)):
    rows = (await db.execute(select(models.User).order_by(models.User.id))).scalars().all()
    return [{"id": u.id, "email": u.email, "full_name": u.full_name, "role": u.role,
             "is_active": u.is_active, "last_login": u.last_login} for u in rows]


@router.patch("/users/{user_id}")
async def update_user(user_id: int, body: UserPatch, db: AsyncSession = Depends(get_db),
                      admin: models.User = Depends(_require_admin)):
    user = await db.get(models.User, user_id)
    if not user:
        raise HTTPException(404)
    if body.is_active is False and user.id == admin.id:
        raise HTTPException(400, "O'zingizni o'chira olmaysiz")
    if body.is_active is not None:
        user.is_active = body.is_active
    if body.password:
        user.hashed_password = pwd_context.hash(body.password)
    await db.commit()
    return {"id": user.id, "is_active": user.is_active}


async def ensure_bootstrap_admin(db: AsyncSession) -> Optional[int]:
    """BOOTSTRAP_ADMIN_EMAIL/PASSWORD berilgan va bunday hisob yo'q bo'lsa — admin yaratadi.
    Mavjud hisobga tegmaydi (parol .env dan qayta yozilmaydi)."""
    email = settings.BOOTSTRAP_ADMIN_EMAIL.strip().lower()
    password = settings.BOOTSTRAP_ADMIN_PASSWORD
    if not email or not password:
        return None
    exists = await db.execute(select(models.User.id).where(func.lower(models.User.email) == email))
    if exists.scalar_one_or_none():
        return None
    if len(password) < 10:
        logger.error("BOOTSTRAP_ADMIN_PASSWORD juda qisqa (kamida 10 belgi) — admin yaratilmadi")
        return None
    user = models.User(email=email, hashed_password=pwd_context.hash(password),
                       full_name="Administrator", role=models.UserRole.ADMIN)
    db.add(user)
    await db.commit()
    logger.info("Birinchi admin yaratildi: id=%s", user.id)
    return user.id


@router.post("/login", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    email = (form.username or "").strip().lower()
    result = await db.execute(select(models.User).where(func.lower(models.User.email) == email))
    user = result.scalar_one_or_none()
    ok = pwd_context.verify(form.password or "", user.hashed_password if user else _DUMMY_HASH)
    if not user or not ok or user.is_active is False:
        raise HTTPException(status_code=401, detail="Email yoki parol noto'g'ri")
    user.last_login = datetime.utcnow()
    await db.commit()
    token = create_token({"sub": str(user.id), "role": user.role.value})
    return Token(access_token=token, token_type="bearer", role=user.role.value, full_name=user.full_name)


@router.get("/me")
async def me(current_user: models.User = Depends(get_current_user)):
    return {"id": current_user.id, "email": current_user.email,
            "full_name": current_user.full_name, "role": current_user.role}
