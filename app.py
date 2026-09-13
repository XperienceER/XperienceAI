import os
import json
import time
import threading
from collections import defaultdict
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, origins=["http://127.0.0.1:5000", "http://localhost:5000"])

# ==================== SECURITY HEADERS ====================
@app.after_request
def set_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.tailwindcss.com https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com https://fonts.googleapis.com; font-src 'self' https://cdnjs.cloudflare.com https://fonts.gstatic.com; img-src 'self' data: https:; connect-src 'self'"
    return response

# ==================== RATE LIMITING ====================
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 20     # requests per window
_rate_limit_db = defaultdict(list)
_rate_lock = threading.Lock()

def check_rate_limit(ip):
    now = time.time()
    with _rate_lock:
        # ลบ IP ที่หมด window แล้วออก กัน dict โตไม่จำกัด (memory leak)
        expired = [k for k, v in _rate_limit_db.items() if not any(t > now - RATE_LIMIT_WINDOW for t in v)]
        for k in expired:
            del _rate_limit_db[k]
        _rate_limit_db[ip] = [t for t in _rate_limit_db[ip] if now - t < RATE_LIMIT_WINDOW]
        if len(_rate_limit_db[ip]) >= RATE_LIMIT_MAX:
            return False
        _rate_limit_db[ip].append(now)
        return True

# ==================== CONFIGURATION ====================
_api_keys_raw = os.environ.get("OPENROUTER_API_KEYS", "")
API_KEYS = [k.strip() for k in _api_keys_raw.split(",") if k.strip()] if _api_keys_raw else []

if not API_KEYS:
    API_KEYS = ["sk-or-v1-PLACEHOLDER"]
    print("[WARN] ไม่พบคีย์ใน .env หรือระบบ - ใช้ placeholder key")
else:
    print(f"[INFO] โหลดสำเร็จ: พบ API Key ทั้งหมด {len(API_KEYS)} คีย์")

current_key_index = 0
_key_lock = threading.Lock()
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# ==================== NVIDIA NIM (โมเดล thinking ตอบช้า) ====================
# คีย์อยู่ใน .env เท่านั้น (NVIDIA_API_KEY=...) — ห้าม hardcode ในไฟล์นี้
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_CHAT_URL = NVIDIA_BASE_URL.rstrip("/") + "/chat/completions"

REASONING_EFFORTS = {"low", "medium", "high", "max"}

ALLOWED_MODELS = [
    "inclusionai/ling-3.0-flash-fin:free",
    "dots-studio/dots-3-note-preview:free",
    "nvidia/nemotron-3.5-lightning:free",
    "poolside/laguna-s-2.1:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
]

DAILY_TOKEN_LIMIT = 10000
COOLDOWN_HOURS = 12
TOKEN_DB_FILE = "token_db.json"
MAX_MESSAGES = 50
MAX_PERSONA_LENGTH = 2000
MAX_REQUEST_BYTES = 1_000_000  # 1MB

# RLock เพราะ load_token_data() เรียก save_token_data() ซ้อนกันได้ (กัน deadlock ตอนไฟล์ยังไม่มี)
_db_lock = threading.RLock()

# ==================== USER DATA (ชื่อ + ประวัติคำถาม + โทเค่นที่ใช้) ====================
USER_DB_FILE = "user_data.json"
MAX_USER_NAME_LENGTH = 50
MAX_HISTORY_ENTRIES = 100
MAX_SAVED_QUESTION_CHARS = 500

_user_lock = threading.RLock()

def _default_user_data():
    return {"name": None, "total_tokens": 0, "history": []}

def _valid_user_data(data):
    return (
        isinstance(data, dict)
        and (data.get("name") is None or isinstance(data.get("name"), str))
        and isinstance(data.get("total_tokens"), int)
        and isinstance(data.get("history"), list)
    )

def load_user_data():
    with _user_lock:
        if not os.path.exists(USER_DB_FILE):
            data = _default_user_data()
            save_user_data(data)
            return data
        try:
            with open(USER_DB_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not _valid_user_data(data):
                return _default_user_data()
            return data
        except Exception:
            return _default_user_data()

def save_user_data(data):
    with _user_lock:
        tmp = USER_DB_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        os.replace(tmp, USER_DB_FILE)

def sanitize_name(name):
    """คืนชื่อที่สะอาดแล้ว หรือ None ถ้าใช้ไม่ได้"""
    if not isinstance(name, str):
        return None
    name = "".join(c for c in name.strip() if c.isprintable())
    if not name or len(name) > MAX_USER_NAME_LENGTH:
        return None
    return name

def record_user_question(question, tokens, model):
    """บันทึกคำถาม + โทเค่นสะสม (เรียกหลัง AI ตอบสำเร็จเท่านั้น)"""
    try:
        tokens = int(tokens or 0)
    except (TypeError, ValueError):
        tokens = 0
    if not isinstance(question, str):
        question = ""
    if not isinstance(model, str):
        model = ""
    with _user_lock:
        data = load_user_data()
        data["total_tokens"] = int(data.get("total_tokens", 0)) + max(0, tokens)
        history = data.get("history", [])
        if not isinstance(history, list):
            history = []
        history.append({
            "q": question[:MAX_SAVED_QUESTION_CHARS],
            "tokens": max(0, tokens),
            "model": model[:120],
            "time": time.time(),
        })
        data["history"] = history[-MAX_HISTORY_ENTRIES:]
        save_user_data(data)
        return int(data["total_tokens"])
    return 0

# ==================== TOKEN & QUOTA DATABASE ====================
def load_token_data():
    with _db_lock:
        if not os.path.exists(TOKEN_DB_FILE):
            data = {
                "used_tokens": 0,
                "locked_until": None
            }
            save_token_data(data)
            return data
        try:
            with open(TOKEN_DB_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"used_tokens": 0, "locked_until": None}

def save_token_data(data):
    with _db_lock:
        tmp = TOKEN_DB_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        os.replace(tmp, TOKEN_DB_FILE)

def check_and_update_quota(tokens_to_add=0):
    """
    ตรวจสอบโควตา Token และจัดการ Cooldown 12 ชั่วโมง
    """
    data = load_token_data()
    now = time.time()

    # ตรวจสอบว่าพ้นช่วงติด Cooldown 12 ชั่วโมงหรือยัง
    if data.get("locked_until"):
        if now >= data["locked_until"]:
            # พ้น 12 ชม. แล้ว ทำการ Reset โควตาใหม่
            data["used_tokens"] = 0
            data["locked_until"] = None
        else:
            # ยังติด Cooldown อยู่
            remaining_seconds = int(data["locked_until"] - now)
            return False, data["used_tokens"], remaining_seconds

    # เพิ่มโทเค่นที่ใช้
    data["used_tokens"] += tokens_to_add

    # ถ้าใช้เกินโควตา 10,000 โทเค่น ให้เริ่มล็อก 12 ชั่วโมง
    if data["used_tokens"] >= DAILY_TOKEN_LIMIT:
        data["locked_until"] = now + (COOLDOWN_HOURS * 3600)
        save_token_data(data)
        return False, data["used_tokens"], COOLDOWN_HOURS * 3600

    save_token_data(data)
    return True, data["used_tokens"], 0

# ==================== ROUTES ====================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status", methods=["GET"])
def get_status():
    data = load_token_data()
    now = time.time()
    remaining_cd = 0

    if data.get("locked_until"):
        if now >= data["locked_until"]:
            data["used_tokens"] = 0
            data["locked_until"] = None
            save_token_data(data)
        else:
            remaining_cd = int(data["locked_until"] - now)

    return jsonify({
        "used_tokens": data["used_tokens"],
        "max_tokens": DAILY_TOKEN_LIMIT,
        "is_locked": remaining_cd > 0,
        "cooldown_remaining": remaining_cd
    })

@app.route("/api/user", methods=["GET"])
def get_user():
    data = load_user_data()
    history = data.get("history", [])
    return jsonify({
        "name": data.get("name"),
        "total_tokens": int(data.get("total_tokens", 0)),
        "question_count": len(history) if isinstance(history, list) else 0,
    })

@app.route("/api/user", methods=["POST"])
def set_user():
    client_ip = request.remote_addr or "unknown"
    if not check_rate_limit(client_ip):
        return jsonify({"error": "คำขอมากเกินไป กรุณารอสักครู่"}), 429

    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return jsonify({"error": "Invalid request body"}), 400

    name = sanitize_name(body.get("name", ""))
    if name is None:
        return jsonify({"error": f"ชื่อต้องมี 1-{MAX_USER_NAME_LENGTH} ตัวอักษร"}), 400

    data = load_user_data()
    data["name"] = name
    save_user_data(data)
    return jsonify({"name": name})

@app.route("/api/chat", methods=["POST"])
def chat():
    global current_key_index

    # Rate limit check
    client_ip = request.remote_addr or "unknown"
    if not check_rate_limit(client_ip):
        return jsonify({"error": "คำขอมากเกินไป กรุณารอสักครู่"}), 429

    # Request size check
    if request.content_length and request.content_length > MAX_REQUEST_BYTES:
        return jsonify({"error": "Request body ใหญ่เกินไป"}), 413

    # silent=True กัน 415/400 ตอน client ส่ง body ที่ไม่ใช่ JSON
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return jsonify({"error": "Invalid request body"}), 400
    user_messages = body.get("messages", [])
    model_name = body.get("model", "inclusionai/ling-3.0-flash-fin:free")
    user_persona = body.get("user_persona", "")
    # กัน crash ตอน client ส่ง persona/model ที่ไม่ใช่ string
    if not isinstance(user_persona, str):
        user_persona = str(user_persona) if user_persona is not None else ""
    if not isinstance(model_name, str):
        model_name = ALLOWED_MODELS[0]
    reasoning_effort = body.get("reasoning_effort", "medium")
    if reasoning_effort not in REASONING_EFFORTS:
        reasoning_effort = "medium"

    # === INPUT VALIDATION ===
    if not isinstance(user_messages, list):
        return jsonify({"error": "Invalid messages format"}), 400

    if len(user_messages) == 0:
        return jsonify({"error": "Messages is empty"}), 400

    if len(user_messages) > MAX_MESSAGES:
        return jsonify({"error": f"Messages exceed limit of {MAX_MESSAGES}"}), 400

    if model_name not in ALLOWED_MODELS:
        model_name = ALLOWED_MODELS[0]

    if len(user_persona) > MAX_PERSONA_LENGTH:
        user_persona = user_persona[:MAX_PERSONA_LENGTH]

    user_persona = "".join(c for c in user_persona if c.isprintable() or c in "\n\r\t")

    sanitized_messages = []
    for msg in user_messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role not in ("user", "assistant", "system"):
            role = "user"
        if not isinstance(content, str):
            content = str(content)
        if len(content) > 100000:
            content = content[:100000]
        sanitized_messages.append({"role": role, "content": content})

    user_messages = sanitized_messages

    full_messages = []
    if user_persona:
        full_messages.append({
            "role": "system",
            "content": f"[คำสั่งพิเศษจำลองตัวตน AI สำหรับผู้ใช้รายนี้]:\n{user_persona}"
        })
    full_messages.extend(user_messages)

    estimated_tokens = len(str(full_messages)) // 4 + 512
    allowed, used_tokens, cooldown_remaining = check_and_update_quota(estimated_tokens)
    if not allowed:
        return jsonify({
            "error": f"โทเค่นครบโควตา {DAILY_TOKEN_LIMIT} แล้ว! กำลังติด Cooldown 12 ชม.",
            "cooldown_remaining": cooldown_remaining
        }), 429

    start_time = time.time()

    # ==================== NVIDIA NIM PATH (โมเดล thinking ตอบช้า) ====================
    # แยก provider ด้วย suffix: ลงท้าย :free = OpenRouter, นอกนั้น = NVIDIA NIM
    if not model_name.endswith(":free"):
        if not NVIDIA_API_KEY:
            return jsonify({"error": "ยังไม่ได้ตั้งค่า NVIDIA_API_KEY ในไฟล์ .env"}), 500
        extra_body = None
        if "deepseek-v4-flash" in model_name:
            extra_body = {"chat_template_kwargs": {"thinking": True, "reasoning_effort": reasoning_effort}}
        elif "nemotron-3-ultra" in model_name:
            extra_body = {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": reasoning_effort}}

        nvidia_payload = {
            "model": model_name,
            "messages": full_messages,
            "temperature": 0.7,
            "max_tokens": 4096,
        }
        if extra_body:
            nvidia_payload["extra_body"] = extra_body

        try:
            # โมเดล thinking ตอบช้า เผื่อ timeout 5 นาที
            response = requests.post(
                NVIDIA_CHAT_URL,
                headers={
                    "Authorization": f"Bearer {NVIDIA_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=nvidia_payload,
                timeout=300,
            )
            if response.status_code != 200:
                return jsonify({"error": "เกิดข้อผิดพลาดจากฝั่ง NVIDIA API กรุณาลองใหม่ภายหลัง"}), 502

            elapsed_time = round(time.time() - start_time, 2)
            res_data = response.json()
            message = res_data["choices"][0]["message"]
            content = message.get("content", "") or ""
            if not content.strip():
                reasoning_text = message.get("reasoning_content", "") or ""
                if reasoning_text.strip():
                    content = reasoning_text

            usage = res_data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", len(str(full_messages)) // 4)
            completion_tokens = usage.get("completion_tokens", len(content) // 4)
            total_msg_tokens = prompt_tokens + completion_tokens

            check_and_update_quota(total_msg_tokens - estimated_tokens)
            _ud = load_token_data()
            total_used = int(_ud.get("used_tokens", 0))
            cd = 0

            try:
                record_user_question(
                    full_messages[-1]["content"] if full_messages and full_messages[-1]["role"] == "user" else "",
                    total_msg_tokens,
                    model_name
                )
            except Exception:
                pass

            return jsonify({
                "content": content,
                "tokens_used": total_msg_tokens,
                "total_used": total_used,
                "max_tokens": DAILY_TOKEN_LIMIT,
                "elapsed_time": elapsed_time,
                "cooldown_remaining": cd
            })
        except requests.exceptions.Timeout:
            check_and_update_quota(-estimated_tokens)
            return jsonify({"error": "โมเดล thinking ใช้เวลาคิดนานเกินไป กรุณาลองใหม่"}), 504
        except Exception as e:
            check_and_update_quota(-estimated_tokens)
            return jsonify({"error": "เกิดข้อผิดพลาดจากฝั่ง API กรุณาลองใหม่ภายหลัง"}), 502

    attempts = 0
    max_attempts = len(API_KEYS)
    last_error = "No available key"

    while attempts < max_attempts:
        with _key_lock:
            key_idx = current_key_index

        headers = {
            "Authorization": f"Bearer {API_KEYS[key_idx]}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_name,
            "messages": full_messages,
            "reasoning": {"enabled": True}
        }

        try:
            # โมเดล free/ไกลบ้านตอบช้าได้ เผื่อ timeout 3 นาที
            response = requests.post(API_URL, headers=headers, json=payload, timeout=180)
            
            if response.status_code == 200:
                elapsed_time = round(time.time() - start_time, 2)
                res_data = response.json()
                choice = res_data['choices'][0]['message']
                content = choice.get('content', '')

                usage = res_data.get('usage', {})
                prompt_tokens = usage.get('prompt_tokens', len(str(full_messages)) // 4)
                completion_tokens = usage.get('completion_tokens', len(content) // 4)
                total_msg_tokens = prompt_tokens + completion_tokens

                check_and_update_quota(total_msg_tokens - estimated_tokens)
                _ud = load_token_data()
                total_used = int(_ud.get("used_tokens", 0))
                cd = 0

                try:
                    record_user_question(
                        full_messages[-1]["content"] if full_messages and full_messages[-1]["role"] == "user" else "",
                        total_msg_tokens,
                        model_name
                    )
                except Exception:
                    pass

                return jsonify({
                    "content": content,
                    "tokens_used": total_msg_tokens,
                    "total_used": total_used,
                    "max_tokens": DAILY_TOKEN_LIMIT,
                    "elapsed_time": elapsed_time,
                    "cooldown_remaining": cd
                })

            elif response.status_code in [429, 402, 401]:
                with _key_lock:
                    current_key_index = (current_key_index + 1) % len(API_KEYS)
                attempts += 1
                last_error = f"Key issue (Status {response.status_code})"
            else:
                return jsonify({"error": "เกิดข้อผิดพลาดจากฝั่ง API กรุณาลองใหม่ภายหลัง"}), 502

        except Exception as e:
            last_error = str(e)
            with _key_lock:
                current_key_index = (current_key_index + 1) % len(API_KEYS)
            attempts += 1

    check_and_update_quota(-estimated_tokens)
    _ud = load_token_data()
    total_used = int(_ud.get("used_tokens", 0))
    cd = 0
    print(f"[ERROR] OpenRouter keys failed: {last_error}")
    return jsonify({
        "error": "API Key ใช้งานไม่ได้ทุกตัว กรุณาลองใหม่ภายหลัง",
        "total_used": total_used,
        "cooldown_remaining": cd
    }), 500

if __name__ == "__main__":
    print("AI Chat System Running on http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000)