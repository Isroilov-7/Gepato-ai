#!/usr/bin/env bash
# Gepato-AI — bitta buyruq bilan o'rnatish / yangilash / domen almashtirish (umumiy serverda).
#
#   sudo bash deploy/install.sh <domen> <https-uchun-email>
#   masalan: sudo bash deploy/install.sh gepato.plagiatx.uz admin@plagiatx.uz
#
# Qayta ishga tushirish xavfsiz: mavjud .env va baza volume'i saqlanadi; yangi domen bilan ishga
# tushirilsa nginx sayti va HTTPS sertifikati o'sha domenga o'tadi.
# PlagiatX (/opt/plagiatx), NafasNazorat va ularning nginx konfiguratsiyasiga TEGMAYDI.
set -euo pipefail

DOMAIN="${1:?Foydalanish: sudo bash deploy/install.sh <domen> <email>}"
EMAIL="${2:?Foydalanish: sudo bash deploy/install.sh <domen> <email>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SITE=/etc/nginx/sites-available/gepato
PORT=3200
say() { printf '\n\033[1;36m▶ %s\033[0m\n' "$*"; }

[[ $EUID -eq 0 ]] || { echo "root kerak: sudo bash deploy/install.sh ..."; exit 1; }
[[ "$DOMAIN" =~ ^[A-Za-z0-9.-]+\.[A-Za-z]{2,}$ ]] || { echo "Domen noto'g'ri: $DOMAIN"; exit 1; }
[[ "$EMAIL" =~ ^[^@[:space:]]+@[^@[:space:]]+\.[A-Za-z]{2,}$ ]] || { echo "Email noto'g'ri: $EMAIL"; exit 1; }
cd "$ROOT"

say "0/6 Joy, xotira va port tekshiruvi"
AVAIL_KB=$(df -Pk /var/lib/docker 2>/dev/null | awk 'NR==2{print $4}')
AVAIL_KB=${AVAIL_KB:-$(df -Pk / | awk 'NR==2{print $4}')}
if [[ "$AVAIL_KB" -lt 4194304 ]]; then
  echo "  Diskda 4 GB dan kam joy ($((AVAIL_KB/1024)) MB). Avval tozalang: docker builder prune -f && docker image prune -f"
  echo "  (docker system prune --volumes ISHLATMANG — bazalar o'chadi)"
  exit 1
fi
echo "  disk: $((AVAIL_KB/1024/1024)) GB bo'sh"
MEM_AVAIL=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
echo "  xotira: ${MEM_AVAIL} MB bo'sh (Gepato odatda ~650 MB, chegara jami ~1.9 GB)"
[[ "$MEM_AVAIL" -lt 900 ]] && echo "  ⚠ Bo'sh xotira kam — Gepato ishlaydi, lekin og'ir tahlilda o'zi qayta ishga tushishi mumkin (PlagiatX'ga tegmaydi)"
port_busy() { if command -v ss >/dev/null; then ss -ltn "sport = :$PORT" | grep -q LISTEN; else curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/"; fi; }
if port_busy && [[ -z "$(docker compose ps -q api 2>/dev/null)" ]]; then
  echo "  127.0.0.1:$PORT band (boshqa dastur). Kim ishlatayotganini ko'ring: ss -ltnp 'sport = :$PORT'"
  exit 1
fi

say "1/6 .env (sirlar) tekshirilmoqda"
if [[ ! -f .env ]]; then
  umask 077
  rnd() { openssl rand -base64 48 | tr -d '/+=' | cut -c1-"$1"; }
  GEMINI_KEY=""
  if [[ -t 0 ]]; then
    # Kalit ekranda ko'rinmaydi va logga tushmaydi. Bo'sh qoldirsangiz AI agent namunaviy javob beradi.
    read -r -s -p "  Gemini API kaliti (ixtiyoriy, Enter — keyinroq): " GEMINI_KEY || true
    echo
  fi
  cat > .env <<ENV
ENVIRONMENT=production
DEBUG=false
SECRET_KEY=$(openssl rand -hex 32)
POSTGRES_PASSWORD=$(rnd 32)
BOOTSTRAP_ADMIN_EMAIL=admin@${DOMAIN}
BOOTSTRAP_ADMIN_PASSWORD=$(rnd 20)
AI_PROVIDER=auto
GEMINI_MODEL=gemini-2.5-flash
GEMINI_API_KEY=${GEMINI_KEY}
ANTHROPIC_API_KEY=
ENV
  unset GEMINI_KEY
  chmod 600 .env
  echo "  .env yaratildi (chmod 600). Admin login: grep ADMIN $ROOT/.env"
else
  chmod 600 .env
  echo "  mavjud .env saqlandi"
fi

say "2/6 Docker image build va ishga tushirish (birinchi marta 5–10 daqiqa)"
docker compose up -d --build
for i in $(seq 1 90); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then echo "  health: ok"; break; fi
  if [[ $i -eq 90 ]]; then
    echo "  Ilova javob bermadi. Sababi: docker compose logs --tail=60 api"
    exit 1
  fi
  sleep 2
done

say "3/6 nginx sayti: $DOMAIN"
install -m 644 deploy/gepato-proxy.conf /etc/nginx/snippets/gepato-proxy.conf
sed "s/__DOMAIN__/${DOMAIN}/g" deploy/nginx-gepato.conf > "$SITE.new"
[[ -f "$SITE" ]] && cp "$SITE" "$SITE.bak"
mv "$SITE.new" "$SITE"
ln -sfn "$SITE" /etc/nginx/sites-enabled/gepato
if ! nginx -t; then
  echo "  nginx -t xato — eski konfiguratsiya tiklanmoqda"
  if [[ -f "$SITE.bak" ]]; then mv "$SITE.bak" "$SITE"; else rm -f /etc/nginx/sites-enabled/gepato "$SITE"; fi
  nginx -t && systemctl reload nginx
  exit 1
fi
systemctl reload nginx

say "4/6 HTTPS sertifikati (certbot)"
CERT_OK=0
# DNS yo'q bo'lsa certbot chaqirilmaydi: har muvaffaqiyatsiz urinish Let's Encrypt limitini yeydi (soatiga 5 ta)
if [[ -z "$(getent ahostsv4 "$DOMAIN" | awk 'NR==1{print $1}')" ]]; then
  echo "  DNS'da $DOMAIN yo'q — certbot o'tkazib yuborildi."
  echo "  Cloudflare -> DNS -> Add record: A, Name: ${DOMAIN%%.*}, IPv4: $(curl -4 -fsS --max-time 5 https://api.ipify.org 2>/dev/null || echo '<server IP>'), Proxied (to'q sariq bulut)"
  echo "  (DNS only ISHLAMAYDI: server 80/443 ni faqat Cloudflare'ga ochgan)"
elif command -v certbot >/dev/null; then
  if certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "$EMAIL" --redirect; then
    CERT_OK=1
  else
    echo "  certbot muvaffaqiyatsiz. 'Timeout during connect' bo'lsa — Cloudflare yozuvi Proxied (to'q sariq) bo'lsin:"
    echo "  server 80/443 ni faqat Cloudflare IP'lariga ochgan (to'g'ridan IP / sslip.io / DNS only ishlamaydi)."
  fi
else
  echo "  certbot topilmadi: apt install -y certbot python3-certbot-nginx"
fi

say "5/6 Kunlik backup (02:40, 14 kun) -> /var/backups/gepato"
CRON="40 2 * * * bash $ROOT/deploy/backup.sh >> /var/log/gepato-backup.log 2>&1"
# crontab bo'sh bo'lsa `crontab -l`/`grep -v` xato qaytaradi — set -e skriptni uzmasin
{ { crontab -l 2>/dev/null || true; } | { grep -v 'gepato/deploy/backup.sh' || true; }; echo "$CRON"; } | crontab -
echo "  cron: $CRON"

say "6/6 Holat"
docker compose ps --format 'table {{.Service}}\t{{.Status}}'
docker stats --no-stream --format 'table {{.Name}}\t{{.MemUsage}}' $(docker compose ps -q) 2>/dev/null || true

if [[ $CERT_OK -eq 1 ]]; then
  printf '\n\033[1;32m✔ Tayyor: https://%s\033[0m\n' "$DOMAIN"
else
  printf '\n\033[1;33m⚠ Ilova ishlayapti, lekin HTTPS sertifikati YO‘Q (parol ochiq tarmoqdan o‘tadi — kirmang).\033[0m\n'
  echo "  DNS: getent hosts $DOMAIN  -> server IP chiqishi kerak; keyin shu skriptni qayta ishga tushiring."
fi
echo "  Demo: bosh sahifa — parolsiz, faqat namunaviy bemorlar (tepada \"Demo rejim\")."
echo "  Admin login: grep ADMIN $ROOT/.env   (shifokor qo'shish: deploy/README.md)"
echo "  To'liq tekshiruv: docker compose exec -T api python scripts/selfcheck.py"
