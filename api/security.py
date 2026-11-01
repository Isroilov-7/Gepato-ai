"""Rol tekshiruvi — barcha bemor ma'lumoti va AI chaqiruvlari faqat kirgan xodim uchun.

Ilgari /api/patients, /api/voice, /api/agents, /api/admin ... tokensiz ochiq edi (internetdagi har kim
bemorlar ro'yxatini o'qiy, yozuv qo'sha, pullik AI'ni chaqira va modelni qayta o'qitishi mumkin edi).
Endi routerlar main.py da shu dependency'lar bilan ulanadi — yangi endpoint ham avtomatik himoyalanadi.
"""
from fastapi import Depends, HTTPException, status

from api.routes.auth import get_current_user
from db import models


def require_roles(*roles: models.UserRole):
    allowed = set(roles)

    async def _dep(user: models.User = Depends(get_current_user)) -> models.User:
        if user.role not in allowed:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Bu amal uchun ruxsat yo'q")
        return user

    return _dep


# Shifokor yoki admin: bemorlar, ovoz, tremor, kognitiv, bashorat, AI agentlar
require_clinician = require_roles(models.UserRole.DOCTOR, models.UserRole.ADMIN)
# Faqat admin: statistika, modelni qayta o'qitish, foydalanuvchi qo'shish
require_admin = require_roles(models.UserRole.ADMIN)
