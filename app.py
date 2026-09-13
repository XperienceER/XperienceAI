import os
import json
import time
import threading
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
from dotenv import load_dotenv  # <-- เพิ่มบรรทัดนี้

# โหลดตัวแปรทั้งหมดจากไฟล์ .env เข้าสู่ระบบ
load_dotenv()

app = Flask(__name__)
CORS(app, origins=["http://127.0.0.1:5000", "http://localhost:5000"])

# ==================== CONFIGURATION ====================
# อ่านค่า OPENROUTER_API_KEYS จากไฟล์ .env
_api_keys_raw = os.environ.get("OPENROUTER_API_KEYS", "")
API_KEYS = [k.strip() for k in _api_keys_raw.split(",") if k.strip()] if _api_keys_raw else []

if not API_KEYS:
    API_KEYS = ["sk-or-v1-PLACEHOLDER"]
    print("[WARN] ไม่พบคีย์ใน .env หรือระบบ - ใช้ placeholder key")
else:
    print(f"[INFO] โหลดสำเร็จ: พบ API Key ทั้งหมด {len(API_KEYS)} คีย์")

current_key_index = 0
API_URL = "https://openrouter.ai/api/v1/chat/completions"

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
MAX_MESSAGES = 50  # จำกัดจำนวนข้อความสูงสุดต่อ request
MAX_PERSONA_LENGTH = 2000

_db_lock = threading.Lock()

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
            json.dump(data, f, indent=4)
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

@app.route("/api/chat", methods=["POST"])
def chat():
    global current_key_index
    body = request.json or {}
    user_messages = body.get("messages", [])
    model_name = body.get("model", "inclusionai/ling-3.0-flash-fin:free")
    user_persona = body.get("user_persona", "")

    # === INPUT VALIDATION ===
    if not isinstance(user_messages, list):
        return jsonify({"error": "Invalid messages format"}), 400

    if len(user_messages) > MAX_MESSAGES:
        return jsonify({"error": f"Messages exceed limit of {MAX_MESSAGES}"}), 400

    # ตรวจสอบ model ต้องอยู่ใน whitelist
    if model_name not in ALLOWED_MODELS:
        model_name = ALLOWED_MODELS[0]

    # จำกัดความยาว persona
    if len(user_persona) > MAX_PERSONA_LENGTH:
        user_persona = user_persona[:MAX_PERSONA_LENGTH]

    # Sanitize user_persona (ลบ control characters)
    user_persona = "".join(c for c in user_persona if c.isprintable() or c in "\n\r\t")

    # ตรวจสอบว่าแต่ละ message มี structure ที่ถูกต้อง
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
        # จำกัดความยาว content แต่ละข้อความ (100K chars ≈ 100K tokens)
        if len(content) > 100000:
            content = content[:100000]
        sanitized_messages.append({"role": role, "content": content})

    user_messages = sanitized_messages

    # 1. เช็กสถานะ Token ก่อนเริ่มส่ง
    allowed, used_tokens, cooldown_remaining = check_and_update_quota(0)
    if not allowed:
        return jsonify({
            "error": f"โทเค่นครบโควตา {DAILY_TOKEN_LIMIT} แล้ว! กำลังติด Cooldown 12 ชม.",
            "cooldown_remaining": cooldown_remaining
        }), 429

    # 2. ปรับแต่ง Persona (Train AI เฉพาะบุคคล) ผ่าน System Prompt
    full_messages = []
    if user_persona:
        full_messages.append({
            "role": "system",
            "content": f"[คำสั่งพิเศษจำลองตัวตน AI สำหรับผู้ใช้รายนี้]:\n{user_persona}"
        })
    full_messages.extend(user_messages)

    # 3. เริ่มส่ง Request พร้อมวัดเวลา Response Time
    start_time = time.time()
    success = False
    attempts = 0
    max_attempts = len(API_KEYS)
    last_error = "No available key"

    while attempts < max_attempts:
        headers = {
            "Authorization": f"Bearer {API_KEYS[current_key_index]}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_name,
            "messages": full_messages,
            "reasoning": {"enabled": True}
        }

        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                elapsed_time = round(time.time() - start_time, 2)
                res_data = response.json()
                choice = res_data['choices'][0]['message']
                content = choice.get('content', '')

                # ดึงจำนวน Token ที่ใช้จริง
                usage = res_data.get('usage', {})
                prompt_tokens = usage.get('prompt_tokens', len(str(full_messages)) // 4)
                completion_tokens = usage.get('completion_tokens', len(content) // 4)
                total_msg_tokens = prompt_tokens + completion_tokens

                # บันทึกหัก Token ลงฐานข้อมูล
                _, total_used, cd = check_and_update_quota(total_msg_tokens)

                return jsonify({
                    "content": content,
                    "tokens_used": total_msg_tokens,
                    "total_used": total_used,
                    "max_tokens": DAILY_TOKEN_LIMIT,
                    "elapsed_time": elapsed_time,
                    "cooldown_remaining": cd
                })

            elif response.status_code in [429, 402, 401]:
                current_key_index = (current_key_index + 1) % len(API_KEYS)
                attempts += 1
                last_error = f"Key issue (Status {response.status_code})"
            else:
                return jsonify({"error": "เกิดข้อผิดพลาดจากฝั่ง API กรุณาลองใหม่ภายหลัง"}), 502

        except Exception as e:
            last_error = str(e)
            current_key_index = (current_key_index + 1) % len(API_KEYS)
            attempts += 1

    return jsonify({"error": f"API Key ใช้งานไม่ได้ทุกตัว: {last_error}"}), 500

if __name__ == "__main__":
    print("AI Chat System Running on http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000)