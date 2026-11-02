"""Gepato-AI — to'liq ichki tekshiruv (bitta buyruq).

    docker-compose exec api python scripts/selfcheck.py

API konteyner ichida ishlaydi (http://127.0.0.1:8000). Admin (.env BOOTSTRAP_ADMIN_*) bilan kiradi,
har safar yangi sinov shifokorini qo'shadi va oxirida o'chiradi (is_active=false),
sinov bemori ("TEST tekshiruv") bitta — qayta ishlatiladi. Belgilash (label) va qayta o'qitish
chaqirilmaydi: soxta yozuvlar modelni o'qitish ma'lumotiga tushmasin.
"""
import io
import math
import os
import random
import sys
import time

import httpx
import numpy as np
import soundfile as sf

BASE = os.environ.get("SELFCHECK_URL", "http://127.0.0.1:8000")
c = httpx.Client(base_url=BASE, timeout=180, trust_env=False)
results = []


def check(name, cond, info=""):
    results.append(bool(cond))
    print(("  OK  " if cond else "  XATO") + f"  {name}" + (f"  — {info}" if info else ""))
    return cond


def req(method, url, **kw):
    try:
        return c.request(method, url, **kw)
    except Exception as e:  # noqa: BLE001
        class R:  # server javob bermadi
            status_code, text = 0, str(e)
            def json(self): return {}
        return R()


def wav(seconds=10, silent=False):
    sr = 16000
    t = np.arange(int(sr * seconds)) / sr
    if silent:
        x = np.zeros_like(t)
    else:  # "aaa" ga o'xshash unli tovush: 120 Hz asosiy ton + garmonikalar
        ph = 2 * np.pi * np.cumsum(120 + 3 * np.sin(2 * np.pi * 5 * t)) / sr
        x = sum((0.6 / k) * np.sin(k * ph) for k in range(1, 12)) * (0.8 + 0.2 * np.sin(2 * np.pi * 1.5 * t))
        x = x + 0.002 * np.random.randn(len(t))
    b = io.BytesIO()
    sf.write(b, x.astype(np.float32), sr, format="WAV")
    b.seek(0)
    return b


# Build/restartdan keyin API modellarni yuklaguncha (~10 s) javob bermaydi — "0" xatolar chalg'itmasin
for _ in range(45):
    if req("GET", "/health").status_code == 200:
        break
    time.sleep(2)

print("\n1) Server va sahifalar")
for u in ["/health", "/", "/patient", "/doctor/labeling", "/vendor/chart.umd.js"]:
    r = req("GET", u)
    check(f"GET {u}", r.status_code == 200, str(r.status_code))

print("\n2) Himoya: tokensiz kirish yopiq")
for m, u in [("GET", "/api/patients/"), ("GET", "/api/admin/stats"), ("POST", "/api/agents/atlas"),
             ("GET", "/api/predictions/labeling-queue"), ("GET", "/api/tremor/1/latest")]:
    r = req(m, u, json={} if m == "POST" else None)
    check(f"{m} {u} tokensiz -> 401", r.status_code == 401, str(r.status_code))
r = req("POST", "/api/auth/register", json={"email": "x@x.uz", "password": "Selfcheck2025!",
                                            "full_name": "X", "role": "admin"})
check("tokensiz ro'yxat (admin bo'lib) -> 401", r.status_code == 401, str(r.status_code))

print("\n3) Kirish: admin (.env BOOTSTRAP_ADMIN_*) -> shifokor qo'shadi")
a_email, a_pw = os.environ.get("BOOTSTRAP_ADMIN_EMAIL", ""), os.environ.get("BOOTSTRAP_ADMIN_PASSWORD", "")
r = req("POST", "/api/auth/login", data={"username": a_email, "password": a_pw})
atoken = r.json().get("access_token", "") if r.status_code == 200 else ""
if not check("admin kirishi", bool(atoken), f"{r.status_code} (BOOTSTRAP_ADMIN_EMAIL/PASSWORD env'da bormi?)"):
    sys.exit(1)
HA = {"Authorization": f"Bearer {atoken}"}
email = f"selfcheck{int(time.time())}@gepato.local"
dpw = "Selfcheck-" + os.urandom(6).hex()
r = req("POST", "/api/auth/register", headers=HA, json={"email": email, "password": dpw,
                                                         "full_name": "Selfcheck", "role": "doctor"})
check("admin shifokor qo'shdi", r.status_code in (200, 201), str(r.status_code))
doc_id = r.json().get("id") if r.status_code in (200, 201) else None
r = req("POST", "/api/auth/login", data={"username": email, "password": dpw})
token = r.json().get("access_token", "") if r.status_code == 200 else ""
check("shifokor kirishi (token)", bool(token), str(r.status_code))
H = {"Authorization": f"Bearer {token}"}
check("/api/auth/me", req("GET", "/api/auth/me", headers=H).status_code == 200)
check("shifokor admin bo'limiga -> 403", req("GET", "/api/admin/stats", headers=H).status_code == 403)
r = req("POST", "/api/auth/register", headers=H, json={"email": "y@y.uz", "password": "Selfcheck2025!",
                                                        "full_name": "Y", "role": "admin"})
check("shifokor admin qo'sha olmaydi -> 403", r.status_code == 403, str(r.status_code))
check("noto'g'ri parol -> 401", req("POST", "/api/auth/login",
                                    data={"username": email, "password": "notogri-parol"}).status_code == 401)

print("\n4) Bemor")
r = req("GET", "/api/patients/?limit=500", headers=H)
check("bemorlar ro'yxati", r.status_code == 200, str(r.status_code))
items = r.json() if r.status_code == 200 else []
items = items.get("patients", items.get("items", [])) if isinstance(items, dict) else items
pid = next((p["id"] for p in items if p.get("name") == "TEST tekshiruv"), None)
if pid is None:
    r = req("POST", "/api/patients/", headers=H, json={
        "full_name": "TEST tekshiruv", "age": 58, "gender": "male", "diagnosis": "cirrhosis",
        "child_pugh_class": "B", "bilirubin": 2.1, "albumin": 3.1, "inr": 1.5, "ammonia_level": 95})
    check("bemor yaratish (ammiak bilan)", r.status_code in (200, 201), f"{r.status_code} {r.text[:120]}")
    pid = r.json().get("id") if r.status_code in (200, 201) else None
if not pid:
    print("\nBemor yo'q — davom etib bo'lmaydi.")
    sys.exit(1)
check("bemor ma'lumoti", req("GET", f"/api/patients/{pid}", headers=H).status_code == 200)
check("bemorni tahrirlash", req("PATCH", f"/api/patients/{pid}", headers=H,
                                json={"ammonia_level": 96}).status_code == 200)
check("ichki maydonni tahrirlash (id) -> 422", req("PATCH", f"/api/patients/{pid}", headers=H,
                                                   json={"id": 999999}).status_code == 422)

print("\n5) Tremor (telefon akselerometri)")
still = [{"x": random.gauss(0, .003), "y": random.gauss(0, .003), "z": 1 + random.gauss(0, .003), "t": i * 20}
         for i in range(500)]
r = req("POST", f"/api/tremor/{pid}", headers=H, json={"samples": still, "duration_sec": 10})
check("qimirlamagan telefon -> past", r.status_code == 200 and r.json().get("risk_level") == "low",
      f"{r.json().get('risk_level')} {r.json().get('tremor_score')}")
shake = [{"x": .25 * math.sin(2 * math.pi * 9 * i / 50), "y": random.gauss(0, .005),
          "z": 1 + .25 * math.sin(2 * math.pi * 9 * i / 50), "t": i * 20} for i in range(500)]
r = req("POST", f"/api/tremor/{pid}", headers=H, json={"samples": shake, "duration_sec": 10})
check("9 Hz kuchli titroq -> yuqori", r.status_code == 200 and r.json().get("risk_level") in ("high", "moderate"),
      f"{r.json().get('risk_level')} {r.json().get('tremor_score')}")
check("tremor tarixi", req("GET", f"/api/tremor/{pid}/history", headers=H).status_code == 200)

print("\n6) Kognitiv test (Stroop + raqamlar)")
cols = ["red", "blue", "green", "yellow"]
st = [{"color": random.choice(cols), "word": random.choice(cols), "response": "red",
       "rt_ms": 700 + random.random() * 300, "correct": random.random() > .2} for _ in range(20)]
nct = [{"number": i + 1, "tapped_at_ms": (i + 1) * 1500, "correct": True} for i in range(25)]
r = req("POST", f"/api/cognitive/{pid}", headers=H, json={"stroop_trials": st, "nct_sequence": nct})
check("kognitiv natija saqlandi", r.status_code in (200, 201), str(r.status_code))

print("\n7) Ovoz tahlili (yuklash -> 23 belgi -> model -> HE indeksi)")
r = req("POST", f"/api/voice/upload/{pid}", headers=H, files={"file": ("v.wav", wav(), "audio/wav")})
check("ovoz yuklash", r.status_code in (200, 201), f"{r.status_code} {r.text[:120]}")
rid = r.json().get("recording_id") if r.status_code in (200, 201) else None
if rid:
    r = req("POST", f"/api/voice/analyze/{rid}", headers=H)
    d = r.json() if r.status_code == 200 else {}
    check("tahlil natijasi", r.status_code == 200, f"{r.status_code} xavf={d.get('risk_level')} {d.get('risk_score')}")
    check("HE indeksi (ovoz+tremor+kognitiv)", d.get("he_index") is not None,
          f"{d.get('he_index')} {d.get('he_risk_level')}")
    check("SHAP sabablari", bool(d.get("top_factors") or d.get("shap_explanation")))
r = req("POST", f"/api/voice/upload/{pid}", headers=H, files={"file": ("s.wav", wav(2, silent=True), "audio/wav")})
sid = r.json().get("recording_id") if r.status_code in (200, 201) else None
r2 = req("POST", f"/api/voice/analyze/{sid}", headers=H) if sid else None
check("jim yozuv rad etiladi", r2 is not None and r2.status_code == 422,
      (r2.text[:100] if r2 is not None else "yuklanmadi"))
r = req("GET", f"/api/voice/{pid}/history", headers=H)
check("ovoz tarixi", r.status_code == 200)
hist = (r.json() or {}).get("data", []) if r.status_code == 200 else []
# Regressiya: jitter Praat chaqiruvi xato bo'lib doim 0 edi (2026-11-02)
check("jitter hisoblanadi (> 0)", bool(hist) and (hist[-1].get("jitter") or 0) > 0,
      f"jitter={hist[-1].get('jitter') if hist else None}")
check("bashoratlar", req("GET", f"/api/predictions/{pid}", headers=H).status_code == 200)
check("belgilash navbati", req("GET", "/api/predictions/labeling-queue", headers=H).status_code == 200)

print("\n8) Admin")
for u in ["/api/admin/stats", "/api/admin/alerts", "/api/admin/model/versions"]:
    check(u, req("GET", u, headers=HA).status_code == 200)

print("\n9) AI agentlar (Atlas, Oracle)")
ai_ok = True
for agent in ("atlas", "oracle"):
    r = req("POST", f"/api/agents/{agent}", headers=H, json={"question": "Qisqa xulosa bering", "patient_id": pid})
    d = r.json() if r.status_code == 200 else {}
    if r.status_code == 200 and not d.get("error"):
        print(f"  OK    {agent} ({d.get('provider')}): {d.get('response', '')[:80]!r}")
    else:
        ai_ok = False
        print(f"  DIQQAT {agent}: AI javob bermadi — {str(d.get('error') or r.text)[:160]}")

if doc_id:  # sinov shifokori faol qolmasin (bazada izi qoladi, kira olmaydi)
    req("PATCH", f"/api/auth/users/{doc_id}", headers=HA, json={"is_active": False})
    check("sinov shifokori o'chirildi -> eski token 401",
          req("GET", "/api/auth/me", headers=H).status_code == 401)

ok, n = sum(results), len(results)
print(f"\nJAMI: {ok}/{n} o'tdi" + ("" if ai_ok else "  (AI agentlar: kalit/balansni tekshiring — sayt demo javob beradi)"))
sys.exit(0 if ok == n else 1)
