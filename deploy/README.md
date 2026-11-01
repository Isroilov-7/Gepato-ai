# Gepato-AI — serverga joylash (plagiatx.uz bilan bir serverda, alohida sayt)

NafasNazorat bilan bir xil tartib: alohida papka, alohida compose loyiha, faqat `127.0.0.1` port,
xotira chegarasi, o'z nginx sayti, HTTPS, kunlik backup.

| | PlagiatX | NafasNazorat | Gepato-AI |
|---|---|---|---|
| Papka | `/opt/plagiatx` | `/opt/nafasnazorat` | `/opt/gepato` |
| Compose loyiha | `plagiatx` | `nafasnazorat` | `gepato` (alohida) |
| Port (faqat 127.0.0.1) | 8000 | 3100 | **3200** |
| Baza | Postgres 16 | SQLite | o'z Postgres 15 (`gepato_gepato_pg` volume) |
| nginx | mavjud blok | o'z bloki | o'z bloki (`sites-available/gepato`) |
| RAM | ~3 GB | ~40–200 MB | **~650 MB** odatda; chegara: api 900m, worker 512m, beat 160m, db 256m, redis 80m |

`oom_score_adj: 800` — server xotirasi tugasa yadro avval Gepato konteynerlarini to'xtatadi, PlagiatX'ni emas.
`memswap_limit` = `mem_limit` — swap'ni ham yemaydi. PlagiatX servislariga hech narsa tegmaydi — downtime yo'q.

## Xavfsizlik (2026-11-01 da qo'shildi)

- Bemor ma'lumoti, ovoz, tremor, kognitiv, bashorat va AI agentlar — **faqat kirgan shifokor/admin**
  (tokensiz 401). Admin bo'limi (statistika, modelni qayta o'qitish) — faqat admin.
- Ro'yxatdan o'tkazish ochiq EMAS: shifokorni admin qo'shadi. Birinchi admin `.env` dagi
  `BOOTSTRAP_ADMIN_*` dan avtomatik yaratiladi.
- `ENVIRONMENT=production` da zaif `SECRET_KEY`, standart baza paroli yoki `ALLOWED_ORIGINS=*` bo'lsa
  API umuman ishga tushmaydi. `/docs` productionda yopiq.
- Bosh sahifadagi shifokor/admin paneli — **namunaviy (demo) ma'lumot**, parolsiz ko'rinadi (hay'atga
  ko'rsatish uchun). Haqiqiy ma'lumot API va "Belgilash navbati" (`/doctor/labeling`) orqali, kirgandan keyin.

## O'rnatish

**1. Cloudflare DNS** (plagiatx.uz zonasi): `A` yozuv — Name: `gepato`, IPv4: serverning IP manzili
(plagiatx.uz bilan bir xil).

**2. Kodni serverga olish** (bir marta):
```bash
git clone -b claude/gracious-pascal-7b310w https://github.com/Isroilov-7/Gepato-ai.git /opt/gepato
```
Repo Private bo'lsa git login so'raydi: username — `Isroilov-7`, parol o'rniga GitHub token
(github.com → Settings → Developer settings → Fine-grained tokens, faqat shu repo, "Contents: Read").

**3. O'rnatish — bitta buyruq:**
```bash
cd /opt/gepato && sudo bash deploy/install.sh gepato.plagiatx.uz <sizning-email>
```
Skript: joy/xotira/portni tekshiradi → `.env` ni tasodifiy sirlar bilan yaratadi (chmod 600; Gemini kalitini
so'raydi — ekranda ko'rinmaydi, Enter bilan o'tkazsa bo'ladi) → image build (birinchi marta 5–10 daqiqa) →
nginx sayt (`nginx -t` xato bersa orqaga qaytaradi) → HTTPS → kunlik backup (02:40).
Kutilgan oxirgi qator: `✔ Tayyor: https://gepato.plagiatx.uz`

**4. Tekshirish:**
```bash
cd /opt/gepato && docker compose exec -T api python scripts/selfcheck.py
```
Kutilgan: `JAMI: N/N o'tdi` (AI kalit bo'lmasa "AI agentlar" haqida DIQQAT — bu xato emas).

## Shifokor qo'shish

Admin login: `grep ADMIN /opt/gepato/.env` → saytda "Shifokor Paneli" → **Kirish**.
Shifokor hisobi (parol ekranda ko'rinmaydi):
```bash
cd /opt/gepato && docker compose exec api python scripts/create_user.py doctor ism@klinika.uz "Ism Familiya"
```
Parolni tiklash: oxiriga `--reset-password`. Hisobni o'chirish: `PATCH /api/auth/users/<id>` `{"is_active": false}`
(admin tokeni bilan) — eski tokeni ham darhol ishlamay qoladi.

## AI kalitini keyin qo'shish

```bash
cd /opt/gepato && read -r -s -p "Gemini kaliti: " K && sed -i "s|^GEMINI_API_KEY=.*|GEMINI_API_KEY=$K|" .env && unset K && echo
docker compose up -d --force-recreate api worker
```

## Yangilash (yangi versiya chiqqanda)

```bash
cd /opt/gepato && git pull origin claude/gracious-pascal-7b310w && docker compose up -d --build
```

## Domen olingandan keyin

DNS'da yangi domenning `A` yozuvini server IP'siga yo'naltiring, keyin:
```bash
cd /opt/gepato && sudo bash deploy/install.sh yangi-domen.uz <sizning-email>
```
Ma'lumotlar va `.env` saqlanadi, nginx va sertifikat yangi domenga o'tadi.

## Backup va tiklash

Har kuni 02:40 — `/var/backups/gepato/db_*.dump` (baza) va `files_*.tar.gz` (ovoz yozuvlari + modellar),
14 kun. Qo'lda: `sudo bash /opt/gepato/deploy/backup.sh`. Log: `/var/log/gepato-backup.log`.

Bazani tiklash (avval yangi backup oling!):
```bash
cd /opt/gepato && docker compose exec -T db pg_restore -U gepato -d gepato --clean --if-exists < /var/backups/gepato/db_<SANA>.dump
```

## Foydali buyruqlar

| Nima | Buyruq |
|---|---|
| Holat | `cd /opt/gepato && docker compose ps` |
| Xotira | `docker stats --no-stream $(docker compose ps -q)` |
| Loglar | `docker compose logs --tail=100 api` (worker, beat, db) |
| Admin login | `grep ADMIN /opt/gepato/.env` |
| Qayta ishga tushirish | `docker compose restart api worker beat` |

## Xavfli joylar

- `docker compose down -v` va `docker system prune --volumes` — bazani o'chiradi. Ishlatmang
  (tozalash kerak bo'lsa faqat `docker builder prune -f`, `docker image prune -f`).
- `.env` o'zgarsa: `docker compose up -d --force-recreate api worker beat` (oddiy restart yangi env'ni o'qimaydi).
- `POSTGRES_PASSWORD` ni baza yaratilgandan keyin `.env` da o'zgartirmang — baza eski parolni kutadi.
- Sirlarni (kalit, parol) chatga yoki git'ga yozmang; faqat `read -s` bilan kiriting.
