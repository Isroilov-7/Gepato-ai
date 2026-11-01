"""Shifokor yoki admin qo'shish (parol ekranda ko'rinmaydi, logga yozilmaydi).

    docker compose exec api python scripts/create_user.py doctor ism@klinika.uz "Ism Familiya"
    docker compose exec api python scripts/create_user.py admin  boshqa@admin.uz "Ism Familiya"

Parolni tiklash (hisob bor bo'lsa): --reset-password qo'shing.
"""
import argparse
import asyncio
import getpass
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sqlalchemy import func, select  # noqa: E402

from api.routes.auth import UserCreate, create_user, pwd_context  # noqa: E402
from db import models  # noqa: E402
from db.database import async_session, init_db  # noqa: E402


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=["doctor", "admin"])
    ap.add_argument("email")
    ap.add_argument("full_name")
    ap.add_argument("--reset-password", action="store_true")
    a = ap.parse_args()
    pw = getpass.getpass("Parol (kamida 10 belgi): ")
    if pw != getpass.getpass("Parolni takrorlang: "):
        print("Parollar mos emas"); return 1
    body = UserCreate(email=a.email, password=pw, full_name=a.full_name, role=a.role)
    await init_db()
    async with async_session() as db:
        u = (await db.execute(select(models.User).where(func.lower(models.User.email) == body.email))).scalar_one_or_none()
        if u and not a.reset_password:
            print("Bu email bilan hisob bor (parolni almashtirish: --reset-password)"); return 1
        if u:
            u.hashed_password = pwd_context.hash(pw)
            u.is_active = True
            await db.commit()
            print(f"Parol yangilandi: id={u.id}")
            return 0
        u = await create_user(db, body)
        print(f"Yaratildi: id={u.id} role={u.role.value} email={u.email}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
