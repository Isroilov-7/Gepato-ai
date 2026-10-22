"""
Gepato-AI — AI Agentlar
Atlas: Klinik tahlil
Oracle: Trend bashorat
"""
import time
import logging
from typing import Optional
import anthropic
from core.config import settings

logger = logging.getLogger(__name__)
client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)


# ── ATLAS — Klinik Tahlil ─────────────────────────────

ATLAS_SYSTEM = """Siz Gepato-AI platformasining klinik tahlil agenti "Atlas"siz.
Jigar ensefalopiyasi (HE) bo'yicha mutaxassis klinik AI bo'lib, 
gastroenterologlarga va gepatologlar-ga yordam berasiz.

Vazifangiz:
- Bemorning ovoz biomarkerlari va kognitiv test natijalarini tahlil qilish
- Child-Pugh sinfi, ammiak darajasi va boshqa klinik ma'lumotlar bilan integratsiya
- SHAP tushuntirish asosida klinik qarorlar uchun dalillar keltirish
- Dori dozasi tavsiyalari (rifaximin, laktuloza)
- Kasalxonaga yotqizish zarurligini baholash

Qoidalar:
- Har doim klinik dalillarga asoslaning
- Noaniq holatlarda shifokorga murojaat qilishni tavsiya eting
- Uzbekcha yoki ruscha javob bering (so'rovga qarab)
- Qisqa, aniq, amaliy javoblar bering
- Raqamlarni ko'rsating: % o'zgarish, mg/dL, ms
- Oxirida: "⚕️ Bu ma'lumot mutaxassis shifokor qarorini almashtirmaydi" deb yozing
"""

ORACLE_SYSTEM = """Siz Gepato-AI platformasining trend va bashorat agenti "Oracle"siz.
7 kunlik ovoz biomarkerlari trendini tahlil qilib ertangi prognozlar berasiz.

Vazifangiz:
- 7 kunlik jitter, shimmer, HNR trendlarini tahlil qilish
- Deterioratsiya tezligini hisoblash
- 24-48 soat oldidan ogohlantirish
- Kasalxonaga yotqizilish ehtimolini baholash (%)
- Guruhlash: qaysi bemorlar prioritet kuzatuvga muhtoj

Qoidalar:
- Trend yo'nalishi + tezligini ko'rsating
- Raqamli prognoz bering: "48 soatda 78% ehtimol"
- Ertangi kun uchun aniq tavsiyalar bering
- Uzbekcha yoki ruscha javob bering
"""


async def atlas_query(
    question: str,
    patient_data: Optional[dict] = None,
    conversation_history: Optional[list] = None,
) -> dict:
    """
    Atlas agentiga so'rov.
    
    patient_data: {name, age, child_pugh, risk_score, features, lab_values, ...}
    conversation_history: [{role, content}, ...]
    """
    start = time.time()

    # Kontekst tayyorlash
    context = ""
    if patient_data:
        context = _format_patient_context(patient_data)

    messages = []
    if conversation_history:
        messages.extend(conversation_history[-6:])  # Oxirgi 6 xabar

    user_content = f"{context}\n\nSavol: {question}" if context else question
    messages.append({"role": "user", "content": user_content})

    try:
        response = await client.messages.create(
            model=settings.CLAUDE_MODEL,
            max_tokens=1000,
            system=ATLAS_SYSTEM,
            messages=messages,
        )

        text = response.content[0].text
        latency = int((time.time() - start) * 1000)

        return {
            "response": text,
            "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
            "latency_ms": latency,
            "agent": "atlas",
        }

    except Exception as e:
        logger.error(f"Atlas xatosi: {e}")
        return {
            "response": "Kechirasiz, hozir javob berib bo'lmaydi. Iltimos qayta urinib ko'ring.",
            "tokens_used": 0,
            "latency_ms": 0,
            "agent": "atlas",
            "error": str(e),
        }


async def oracle_query(
    question: str,
    trend_data: Optional[dict] = None,
    conversation_history: Optional[list] = None,
) -> dict:
    """
    Oracle agentiga so'rov.
    
    trend_data: {
        patients: [{name, 7_day_scores, trend_direction, trend_magnitude}, ...]
    }
    """
    start = time.time()

    context = ""
    if trend_data:
        context = _format_trend_context(trend_data)

    messages = []
    if conversation_history:
        messages.extend(conversation_history[-6:])

    user_content = f"{context}\n\nSavol: {question}" if context else question
    messages.append({"role": "user", "content": user_content})

    try:
        response = await client.messages.create(
            model=settings.CLAUDE_MODEL,
            max_tokens=1000,
            system=ORACLE_SYSTEM,
            messages=messages,
        )

        text = response.content[0].text
        latency = int((time.time() - start) * 1000)

        return {
            "response": text,
            "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
            "latency_ms": latency,
            "agent": "oracle",
        }

    except Exception as e:
        logger.error(f"Oracle xatosi: {e}")
        return {
            "response": "Kechirasiz, hozir javob berib bo'lmaydi.",
            "tokens_used": 0,
            "latency_ms": 0,
            "agent": "oracle",
            "error": str(e),
        }


# ── Kontekst formatlash ────────────────────────────────

def _format_patient_context(p: dict) -> str:
    lines = [
        f"=== BEMOR MA'LUMOTLARI ===",
        f"Ism: {p.get('name', 'N/A')}",
        f"Yosh: {p.get('age', 'N/A')}",
        f"Child-Pugh: {p.get('child_pugh_class', 'N/A')}",
        f"Joriy xavf skori: {p.get('risk_score', 'N/A')}",
        "",
        "Ovoz biomarkerlari (bugun):",
        f"  Jitter: {p.get('jitter_local', 'N/A')}% (normal: <1.04%)",
        f"  Shimmer: {p.get('shimmer_local', 'N/A')}% (normal: <3.07%)",
        f"  HNR: {p.get('hnr', 'N/A')} dB (normal: >20dB)",
        f"  Nutq tezligi: {p.get('speech_rate', 'N/A')} syl/sec",
        f"  Tremor: {p.get('tremor_index', 'N/A')}",
        "",
        "Laboratoriya:",
        f"  Ammiak: {p.get('ammonia', 'N/A')} μg/dL",
        f"  Bilirubin: {p.get('bilirubin', 'N/A')} mg/dL",
        f"  INR: {p.get('inr', 'N/A')}",
        "",
        f"Dori: {p.get('lactulose_dose', 'N/A')}",
        f"Trend: {p.get('trend_direction', 'N/A')} (7 kun)",
    ]

    if p.get("top_factors"):
        lines.append("\nSHAP — asosiy omillar:")
        for f in p["top_factors"][:3]:
            lines.append(f"  {f['feature']}: {f['contribution']:.0f}% hissa")

    return "\n".join(lines)


def _format_trend_context(data: dict) -> str:
    patients = data.get("patients", [])
    if not patients:
        return ""

    lines = ["=== 7 KUNLIK TREND MA'LUMOTLARI ==="]
    for p in patients[:10]:  # Max 10 bemor
        scores = p.get("scores_7d", [])
        lines.append(
            f"{p['name']} (CP-{p.get('child_pugh','?')}): "
            f"skorlar={[round(s,2) for s in scores]}, "
            f"trend={p.get('trend_direction','?')}, "
            f"tezlik={p.get('trend_magnitude',0):.3f}"
        )
    return "\n".join(lines)
