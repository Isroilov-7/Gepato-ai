"""Gepato-AI — to'liq ichki tekshiruv (bitta buyruq).

    docker-compose exec api python scripts/selfcheck.py

API konteyner ichida ishlaydi (http://127.0.0.1:8000). Har safar yangi sinov shifokori ochiladi,
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


print("\n1) Server va sahifalar")
for u in ["/health", "/", "/patient", "/doctor/labeling", "/vendor/chart.umd.js"]:
    r = req("GET", u)
    check(f"GET {u}", r.status_code == 200, str(r.status_code))

print("\n2) Ro'yxatdan o'tish va kirish")
email = f"selfcheck{int(time.time())}@gepato.uz"
r = req("POST", "/api/auth/register", json={"email": email, "password": "Selfcheck2025!",
                                            "full_name": "Selfcheck", "role": "doctor"})
check("ro'yxatdan o'tish", r.status_code in (200, 201), str(r.status_code))
r = req("POST", "/api/auth/login", data={"username": email, "password": "Selfcheck2025!"})
token = r.json().get("access_token", "") if r.status_code == 200 else ""
check("kirish (token)", bool(token), str(r.status_code))
H = {"Authorization": f"Bearer {token}"}
check("/api/auth/me", req("GET", "/api/auth/me", headers=H).status_code == 200)

print("\n3) Bemor")
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

print("\n4) Tremor (telefon akselerometri)")
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

print("\n5) Kognitiv test (Stroop + raqamlar)")
cols = ["red", "blue", "green", "yellow"]
st = [{"color": random.choice(cols), "word": random.choice(cols), "response": "red",
       "rt_ms": 700 + random.random() * 300, "correct": random.random() > .2} for _ in range(20)]
nct = [{"number": i + 1, "tapped_at_ms": (i + 1) * 1500, "correct": True} for i in range(25)]
r = req("POST", f"/api/cognitive/{pid}", headers=H, json={"stroop_trials": st, "nct_sequence": nct})
check("kognitiv natija saqlandi", r.status_code in (200, 201), str(r.status_code))

print("\n6) Ovoz tahlili (yuklash -> 23 belgi -> model -> HE indeksi)")
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
check("ovoz tarixi", req("GET", f"/api/voice/{pid}/history", headers=H).status_code == 200)
check("bashoratlar", req("GET", f"/api/predictions/{pid}", headers=H).status_code == 200)
check("belgilash navbati", req("GET", "/api/predictions/labeling-queue", headers=H).status_code == 200)

print("\n7) Admin")
for u in ["/api/admin/stats", "/api/admin/alerts", "/api/admin/model/versions"]:
    check(u, req("GET", u, headers=H).status_code == 200)

print("\n8) AI agentlar (Atlas, Oracle)")
ai_ok = True
for agent in ("atlas", "oracle"):
    r = req("POST", f"/api/agents/{agent}", headers=H, json={"question": "Qisqa xulosa bering", "patient_id": pid})
    d = r.json() if r.status_code == 200 else {}
    if r.status_code == 200 and not d.get("error"):
        print(f"  OK    {agent} ({d.get('provider')}): {d.get('response', '')[:80]!r}")
    else:
        ai_ok = False
        print(f"  DIQQAT {agent}: AI javob bermadi — {str(d.get('error') or r.text)[:160]}")

ok, n = sum(results), len(results)
print(f"\nJAMI: {ok}/{n} o'tdi" + ("" if ai_ok else "  (AI agentlar: kalit/balansni tekshiring — sayt demo javob beradi)"))
sys.exit(0 if ok == n else 1)
