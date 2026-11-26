import os
import time
import requests
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
JARVIS_SECRET = os.getenv("JARVIS_SECRET", "zizo2026").strip()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()

device_telemetry = {
    "Redmi Note 15 Pro+": {"battery": "--%", "network": "Waiting...", "last_seen": "Never", "status": "standby"},
    "Redmi Note 13": {"battery": "--%", "network": "Waiting...", "last_seen": "Never", "status": "standby"},
    "HP EliteBook x360": {"battery": "AC/Cloud", "network": "Online", "last_seen": "Active", "status": "online"},
}

MODELS = {
    "auto": {
        "name": "🧠 Auto-Router (Smart)",
        "provider": "auto",
        "model_id": "auto",
        "desc": "يختار موديل مجاني نشط تلقائياً حسب نوع سؤالك",
    },
    "coder_qwen": {
        "name": "💻 Coding / Qwen & Free Code",
        "provider": "openrouter",
        "model_id": "auto_free_coder",
        "desc": "يبحث تلقائياً عن أفضل موديل مجاني نشط للبرمجة على OpenRouter",
    },
    "deepseek_v3": {
        "name": "🔬 DeepSeek / Reasoning Free",
        "provider": "openrouter",
        "model_id": "auto_free_chat",
        "desc": "يبحث تلقائياً عن موديل مجاني نشط للتحليل والبرمجة على OpenRouter",
    },
    "fast_llama70b": {
        "name": "⚡ Groq Active Model",
        "provider": "groq",
        "model_id": "auto_groq",
        "desc": "يجيب أول موديل شغال فعلياً على حسابك في Groq تلقائياً",
    },
    "speed_llama8b": {
        "name": "🚀 Groq Instant Free",
        "provider": "groq",
        "model_id": "auto_groq_fast",
        "desc": "موديل سريع من Groq",
    },
}

SYSTEM_PROMPT = """أنت J.A.R.V.I.S.، المساعد الشخصي والهندسي الخاص بـ "زيزو" (محمود).
- تتحدث باللهجة المصرية الذكية والواضحة عند الشرح والدردشة، وتكتب المصطلحات التقنية بالإنجليزية.
- عندما يطلب زيزو كوداً برمجياً: اكتب الكود كاملاً متكاملاً جاهزاً للنسخ والتشغيل فوراً بدون أي اختصارات مثل (// rest of code) أو (...).
- التزم بالدقة الهندسية العالية واقترح دائماً أفضل الممارسات (Best Practices)."""

# كاش بسيط للموديلات النشطة عشان ميطلبش القايمة في كل رسالة
_cached_openrouter_free = {"models": [], "ts": 0}
_cached_groq_models = {"models": [], "ts": 0}


def get_free_openrouter_models(prefer_coder: bool = False) -> list:
    """يجلب قائمة الموديلات المجانية النشطة حالياً من OpenRouter API"""
    now = time.time()
    if _cached_openrouter_free["models"] and (now - _cached_openrouter_free["ts"] < 600):
        pool = list(_cached_openrouter_free["models"])
    else:
        try:
            res = requests.get("https://openrouter.ai/api/v1/models", timeout=20)
            if res.status_code == 200:
                data = res.json().get("data", [])
                free_ids = []
                for m in data:
                    mid = m.get("id", "")
                    pricing = m.get("pricing") or {}
                    prompt_price = str(pricing.get("prompt", "1"))
                    completion_price = str(pricing.get("completion", "1"))
                    if mid.endswith(":free") or (prompt_price == "0" and completion_price == "0"):
                        free_ids.append(mid)
                _cached_openrouter_free["models"] = free_ids
                _cached_openrouter_free["ts"] = now
                pool = list(free_ids)
            else:
                pool = []
        except Exception:
            pool = []

    if not pool:
        return [
            "qwen/qwen3-coder:free",
            "deepseek/deepseek-chat-v3-0324:free",
            "meta-llama/llama-3.3-70b-instruct:free",
        ]

    if prefer_coder:
        coder_first = [m for m in pool if any(k in m.lower() for k in ["coder", "code", "qwen", "deepseek"])]
        others = [m for m in pool if m not in coder_first]
        return coder_first + others

    return pool


def get_active_groq_models() -> list:
    """يجلب الموديلات المتاحة فعلياً على مفتاح Groq الحالي"""
    now = time.time()
    if _cached_groq_models["models"] and (now - _cached_groq_models["ts"] < 600):
        return list(_cached_groq_models["models"])

    if not GROQ_API_KEY:
        return []

    try:
        res = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            timeout=20,
        )
        if res.status_code == 200:
            items = res.json().get("data", [])
            ids = [item.get("id") for item in items if item.get("id")]
            # ترتيب يفضل موديلات 70b أو gpt-oss أو llama الحديثة ثم الأسرع
            preferred = []
            for key_word in ["70b", "gpt-oss", "llama-3", "qwen", "gemma", "8b"]:
                for mid in ids:
                    if key_word in mid.lower() and mid not in preferred:
                        preferred.append(mid)
            for mid in ids:
                if mid not in preferred:
                    preferred.append(mid)
            _cached_groq_models["models"] = preferred
            _cached_groq_models["ts"] = now
            return preferred
    except Exception:
        pass

    return ["llama-3.1-8b-instant"]


def sync_to_supabase(table: str, payload: dict):
    if not SUPABASE_URL or not SUPABASE_KEY:
        return
    try:
        url = f"{SUPABASE_URL}/rest/v1/{table}"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }
        requests.post(url, headers=headers, json=payload, timeout=10)
    except Exception:
        pass


def is_coding_prompt(prompt: str) -> bool:
    code_keywords = [
        "كود", "برمجة", "بايثون", "python", "sql", "html", "css", "js",
        "javascript", "flutter", "dart", "api", "script", "error", "bug",
        "function", "class", "docker", "supabase", "مشروع", "دالة", "سكربت",
    ]
    p_lower = prompt.lower()
    return any(word in p_lower for word in code_keywords)


def call_groq(model_id: str, messages: list, is_code: bool = False) -> tuple:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY غير موجود في ملف .env")

    candidates = get_active_groq_models()
    if model_id and not model_id.startswith("auto_groq"):
        if model_id not in candidates:
            candidates = [model_id] + candidates

    if not candidates:
        raise RuntimeError("لم يتم العثور على موديلات متاحة في Groq")

    last_err = ""
    for mid in candidates[:6]:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": mid,
            "messages": messages,
            "temperature": 0.15 if is_code else 0.6,
            "max_tokens": 8000,
        }
        res = requests.post(url, headers=headers, json=payload, timeout=60)
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"], f"Groq ({mid})"
        last_err = f"Groq [{res.status_code}] ({mid}): {res.text[:200]}"

    raise RuntimeError(last_err)


def call_openrouter(model_id: str, messages: list, is_code: bool = False, prefer_coder: bool = False) -> tuple:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY غير موجود في ملف .env")

    candidates = get_free_openrouter_models(prefer_coder=prefer_coder or is_code)
    if model_id and not model_id.startswith("auto_free"):
        if model_id not in candidates:
            candidates = [model_id] + candidates

    last_err = ""
    for mid in candidates[:8]:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://jarvis-cloud.onrender.com",
            "X-Title": "JARVIS Cloud Hub",
        }
        payload = {
            "model": mid,
            "messages": messages,
            "temperature": 0.15 if is_code else 0.6,
            "max_tokens": 8192,
        }
        res = requests.post(url, headers=headers, json=payload, timeout=90)
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"], f"OpenRouter ({mid})"
        last_err = f"OpenRouter [{res.status_code}] ({mid}): {res.text[:200]}"

    raise RuntimeError(last_err)


@app.route("/")
def index():
    return render_template("index.html", models=MODELS)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_message = (data.get("message") or "").strip()
    selected_key = data.get("model", "auto")
    history = data.get("history") or []

    if not user_message:
        return jsonify({"error": "الرسالة فارغة"}), 400

    coding_mode = is_coding_prompt(user_message)
    if selected_key == "auto" or selected_key not in MODELS:
        active_key = "coder_qwen" if coding_mode else "fast_llama70b"
    else:
        active_key = selected_key

    model_info = MODELS[active_key]
    is_code = coding_mode or active_key in ["coder_qwen", "deepseek_v3"]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history[-8:]:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    start_time = time.time()

    try:
        if model_info["provider"] == "groq":
            reply, used_model_name = call_groq(model_info["model_id"], messages, is_code)
        else:
            reply, used_model_name = call_openrouter(
                model_info["model_id"], messages, is_code, prefer_coder=(active_key == "coder_qwen")
            )
    except Exception as primary_err:
        err_detail = str(primary_err)
        try:
            if model_info["provider"] == "openrouter" and GROQ_API_KEY:
                reply, actual_m = call_groq("auto_groq", messages, is_code)
                used_model_name = f"⚡ {actual_m} (Fallback)"
            elif model_info["provider"] == "groq" and OPENROUTER_API_KEY:
                reply, actual_m = call_openrouter("auto_free_chat", messages, is_code, prefer_coder=is_code)
                used_model_name = f"💻 {actual_m} (Fallback)"
            else:
                return jsonify({"error": err_detail}), 500
        except Exception as fallback_err:
            return jsonify({"error": f"{err_detail} | Fallback: {str(fallback_err)}"}), 500

    latency = round(time.time() - start_time, 2)

    sync_to_supabase("jarvis_logs", {
        "model_used": used_model_name,
        "prompt": user_message[:1000],
        "reply": reply[:4000],
        "latency_sec": latency,
    })

    return jsonify({
        "reply": reply,
        "model_used": used_model_name,
        "latency": f"{latency}s",
    })


@app.route("/api/telemetry", methods=["GET", "POST"])
def telemetry():
    if request.method == "POST":
        data = request.json or {}
        secret = data.get("secret") or request.headers.get("X-Jarvis-Secret", "")
        if secret != JARVIS_SECRET:
            return jsonify({"error": "Unauthorized"}), 401

        device = data.get("device", "Unknown Device")
        battery = data.get("battery", "--")
        network = data.get("network", "Connected")
        now_str = datetime.now().strftime("%I:%M %p")

        battery_str = f"{battery}%" if not str(battery).endswith("%") else str(battery)
        device_telemetry[device] = {
            "battery": battery_str,
            "network": network,
            "last_seen": now_str,
            "status": "online",
        }

        sync_to_supabase("device_telemetry", {
            "device": device,
            "battery": battery_str,
            "network": network,
            "last_seen": now_str,
        })

        return jsonify({"status": "updated", "device": device, "time": now_str})

    return jsonify(device_telemetry)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=True)