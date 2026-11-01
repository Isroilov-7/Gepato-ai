#!/usr/bin/env bash
# Gepato-AI — kunlik backup (install.sh cron'ga qo'yadi: 02:40).
#   sudo bash /opt/gepato/deploy/backup.sh
# Nima: Postgres (pg_dump, custom format) + ovoz yozuvlari (uploads) + modellar. 14 kun saqlanadi.
# Tiklash: deploy/README.md "Backup'dan tiklash".
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="${GEPATO_BACKUP_DIR:-/var/backups/gepato}"
KEEP_DAYS="${GEPATO_BACKUP_KEEP_DAYS:-14}"
TS="$(date +%Y%m%d_%H%M%S)"
cd "$ROOT"
umask 077
mkdir -p "$DIR"

# Disk to'lib qolsa PlagiatX ham to'xtaydi — 2 GB dan kam joy bo'lsa backup qilinmaydi
FREE_KB=$(df -Pk "$DIR" | awk 'NR==2{print $4}')
if [[ "${FREE_KB:-0}" -lt 2097152 ]]; then
  echo "$(date -Is) XATO: diskda 2 GB dan kam joy — backup o'tkazib yuborildi" >&2
  exit 1
fi

docker compose exec -T db pg_dump -U gepato -d gepato --format=custom --compress=6 > "$DIR/db_$TS.dump.tmp"
[[ -s "$DIR/db_$TS.dump.tmp" ]] || { echo "$(date -Is) XATO: pg_dump bo'sh" >&2; rm -f "$DIR/db_$TS.dump.tmp"; exit 1; }
mv "$DIR/db_$TS.dump.tmp" "$DIR/db_$TS.dump"

docker compose exec -T api tar -C /app -czf - uploads models > "$DIR/files_$TS.tar.gz.tmp"
mv "$DIR/files_$TS.tar.gz.tmp" "$DIR/files_$TS.tar.gz"

find "$DIR" -maxdepth 1 -type f \( -name 'db_*.dump' -o -name 'files_*.tar.gz' \) -mtime +"$KEEP_DAYS" -delete
find "$DIR" -maxdepth 1 -type f -name '*.tmp' -mmin +120 -delete

echo "$(date -Is) OK: $(du -h "$DIR/db_$TS.dump" | cut -f1) baza, $(du -h "$DIR/files_$TS.tar.gz" | cut -f1) fayllar -> $DIR"
