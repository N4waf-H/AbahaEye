# openai_severity.py  — Arabic output version (backward-compatible)
import base64, json, os, re
from typing import Dict, Any, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# --- Arabic enums -------------------------------------------------------------
AR_SEVERITY = ["لا يوجد", "بسيط", "متوسط", "شديد", "كارثي"]
AR_HAZARDS = [
    "حريق أو دخان", "تسرب وقود", "سقوط عمود أو أسلاك كهرباء", "مسار مغلق",
    "مركبة مشتعلة", "انقلاب", "مخلفات على الطريق", "رؤية ضعيفة", "أخرى"
]
AR_DISPATCH = ["مقيّم الحوادث", "الشرطة", "الإسعاف", "الدفاع المدني", "إدارة المرور", "فريق المرافق"]

# Optional: fallbacks if the model returned English words
EN2AR_SEVERITY = {
    "none": "لا يوجد", "minor": "بسيط", "moderate": "متوسط", "severe": "شديد", "catastrophic": "كارثي",
}
EN2AR_HAZARDS = {
    "fire_or_smoke": "حريق أو دخان", "fuel_spill": "تسرب وقود",
    "fallen_pole_or_power_lines": "سقوط عمود أو أسلاك كهرباء",
    "blocked_lanes": "مسار مغلق", "vehicle_on_fire": "مركبة مشتعلة",
    "rollover": "انقلاب", "debris_on_road": "مخلفات على الطريق",
    "poor_visibility": "رؤية ضعيفة", "other": "أخرى",
}
EN2AR_DISPATCH = {
    "accident_rating_agent": "مقيّم الحوادث", "police": "الشرطة", "ambulance": "الإسعاف",
    "fire_department": "الدفاع المدني", "traffic_management": "إدارة المرور", "utility_crew": "فريق المرافق",
}

# --- JSON schema in Arabic ----------------------------------------------------
SEVERITY_SCHEMA: Dict[str, Any] = {
    "name": "abhaeye_incident_assessment_ar",
    "schema": {
        "type": "object",
        "properties": {
            "severity_level": {"type": "string", "enum": AR_SEVERITY},
            "injuries_likely": {"type": "boolean"},
            "hazards": {"type": "array", "items": {"type": "string", "enum": AR_HAZARDS}},
            "recommended_dispatch": {
                "type": "array",
                "items": {"type": "string", "enum": AR_DISPATCH},
                "minItems": 0, "uniqueItems": True
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reasoning_short": {"type": "string", "maxLength": 280}
        },
        "required": [
            "severity_level", "injuries_likely", "hazards",
            "recommended_dispatch", "confidence", "reasoning_short"
        ],
        "additionalProperties": False
    },
    "strict": True,
}

DEFAULT_INSTRUCTIONS = (
    "أنت خبير في تحليل الحوادث المرورية ضمن مشروع عين أبها (AbhaEye). "
    "حلّل صورة كاميرا المراقبة وأعد STRICT JSON يطابق المخطط المحدد أدناه. "
    "يجب أن تكون جميع القيم والمفاتيح بالعربية فقط باستخدام القوائم المعطاة في المخطط. "
    "اعتمد فقط على ما هو ظاهر في الصورة. عند الشك اختر الخيار الأكثر أمانًا مع تقليل قيمة الثقة."
)

# --- helpers ------------------------------------------------------------------
def _encode_image_to_data_url(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def _normalize_to_arabic(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure Arabic labels even if the model returned English by mistake."""
    out = dict(obj)

    sev = str(out.get("severity_level", "")).strip()
    if sev in EN2AR_SEVERITY:
        out["severity_level"] = EN2AR_SEVERITY[sev]

    haz: List[str] = out.get("hazards", []) or []
    out["hazards"] = [
        EN2AR_HAZARDS.get(h, h) for h in haz
    ]

    disp: List[str] = out.get("recommended_dispatch", []) or []
    out["recommended_dispatch"] = [
        EN2AR_DISPATCH.get(d, d) for d in disp
    ]

    # Clamp confidence to [0,1]
    try:
        conf = float(out.get("confidence", 0))
        out["confidence"] = max(0.0, min(1.0, conf))
    except Exception:
        out["confidence"] = 0.0

    return out

def _fallback_chat_completion(data_url: str, instructions: str) -> Dict[str, Any]:
    """Older Chat Completions vision path: ask for Arabic JSON only and parse."""
    sys = instructions + (
        "\nأعد JSON مُصغّرًا بالعربية فقط بالمفاتيح التالية: "
        "severity_level, injuries_likely, hazards, recommended_dispatch, confidence, reasoning_short. "
        "لا تُدخل أي نصوص إضافية أو Markdown."
    )

    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": [
                {"type": "text", "text": "حلّل هذه الصورة وأعد JSON بالعربية فقط."},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    obj = json.loads(text)
    return _normalize_to_arabic(obj)

def analyze_image(image_path: str, system_prompt: str | None = None) -> Dict[str, Any]:
    """
    Returns Arabic dict:
        severity_level, injuries_likely, hazards, recommended_dispatch, confidence, reasoning_short
    Tries Responses API (structured) -> falls back to Chat Completions (vision).
    """
    instructions = system_prompt or DEFAULT_INSTRUCTIONS
    data_url = _encode_image_to_data_url(image_path)

    # Try new Responses API with Arabic structured output
    try:
        rsp = client.responses.create(
            model=MODEL,
            instructions=instructions,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "حلّل هذه الصورة وأعد JSON بالعربية فقط."},
                    {"type": "input_image", "image_url": data_url},
                ],
            }],
            response_format={"type": "json_schema", "json_schema": SEVERITY_SCHEMA},
        )
        try:
            parsed = rsp.output_parsed  # new SDKs
            if isinstance(parsed, dict):
                return _normalize_to_arabic(parsed)
        except Exception:
            txt = getattr(rsp, "output_text", None)
            if txt:
                return _normalize_to_arabic(json.loads(txt))
    except TypeError:
        # SDK too old for response_format -> fallback
        pass
    except Exception:
        # Any other error -> fallback
        pass

    return _fallback_chat_completion(data_url, instructions)

# --- convenience for your GUI -------------------------------------------------
def format_for_gui(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a display-friendly dict with Arabic-joined dispatch string to avoid letter-by-letter splitting.
    """
    r = _normalize_to_arabic(result)
    dispatch_list = r.get("recommended_dispatch", []) or []
    dispatch_str = "".join(dispatch_list) if isinstance(dispatch_list, list) else str(dispatch_list)

    return {
        "severity_level": r.get("severity_level", ""),
        "injuries_likely": bool(r.get("injuries_likely", False)),
        "hazards": r.get("hazards", []),
        "recommended_dispatch_list": dispatch_list,
        "recommended_dispatch_str": dispatch_str,  # <- use this in your UI
        "confidence": r.get("confidence", 0.0),
        "reasoning_short": r.get("reasoning_short", ""),
    }
