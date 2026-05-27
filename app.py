#!/usr/bin/env python3
"""
oh-my-comic WebUI
- Port 5001 (mobile): Story text + direction input + creative mode + image upload
- Port 5002 (desktop): Background + character comic strip

Image generation pipeline (controlled by IMAGE_GENERATION_MODE):
  sdxl_only    → SDXL for all images; Qwen Edit only for uploaded-image-bound characters
  sdxl_hybrid  → SDXL for new characters/backgrounds; Qwen Edit for recurring characters
  zimage_hybrid→ ZImage for new characters/backgrounds; Qwen Edit for recurring characters

LLM: llama-server auto-managed (start/stop/prewarm)
Frontend: pure fetch polling, no Socket.IO dependency

Creative mode:
  ON  → llama-server stays running; user writes multiple segments; no images generated
  OFF → llama-server stops; all pending images generated in batch; llama pre-warmed again

Image upload:
  User attaches 1 image + text → llama-server (multimodal) parses image
  → combined input generates story JSON → only background image generated
  → uploaded image saved as character portrait for character_prompts[0].id
  → that character is registered for forced Qwen Edit in future segments
"""

import os
import sys
import json
import time
import re
import threading
import argparse
import logging
import socket
import subprocess
import base64
import shutil
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv
from werkzeug.serving import make_server

load_dotenv()

# ─── Configuration ───────────────────────────────────────────────────────────
LLAMA_SERVER_EXE         = os.getenv("LLAMA_SERVER_EXE",         "")
QWEN_GGUF_MODEL          = os.getenv("QWEN_GGUF_MODEL",          "")
LLAMA_HOST               = os.getenv("LLAMA_HOST",               "127.0.0.1")
LLAMA_PORT               = int(os.getenv("LLAMA_PORT",           "8080"))
LLAMA_CTX_SIZE           = int(os.getenv("LLAMA_CTX_SIZE",       "131072"))
LLAMA_SPEC_TYPE          = os.getenv("LLAMA_SPEC_TYPE",          "ngram-mod")
LLAMA_SPEC_NGRAM_SIZE_N  = int(os.getenv("LLAMA_SPEC_NGRAM_SIZE_N", "12"))
LLAMA_SPEC_NGRAM_SIZE_M  = int(os.getenv("LLAMA_SPEC_NGRAM_SIZE_M", "48"))
LLAMA_KV_UNIFIED         = os.getenv("LLAMA_KV_UNIFIED",  "true").lower() == "true"
LLAMA_KV_OFFLOAD         = os.getenv("LLAMA_KV_OFFLOAD",  "true").lower() == "true"
LLAMA_MLOCK              = os.getenv("LLAMA_MLOCK",        "true").lower() == "true"
LLAMA_NO_MMAP            = os.getenv("LLAMA_NO_MMAP",      "true").lower() == "true"
LLAMA_FLASH_ATTN         = os.getenv("LLAMA_FLASH_ATTN",   "on")
LLAMA_MMPROJ_MODEL       = os.getenv("LLAMA_MMPROJ_MODEL", "")
LLAMA_EXTRA_ARGS         = os.getenv("LLAMA_EXTRA_ARGS",   "")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", f"http://{os.getenv('LLAMA_HOST','127.0.0.1')}:{os.getenv('LLAMA_PORT','8080')}/v1")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY",  "llama-cpp-local")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL",    "qwen3.6-35b")
PROMPT_RATING   = os.getenv("PROMPT_RATING",   "general")

# ─── Image Generation Mode ────────────────────────────────────────────────────
# sdxl_only    = SDXL for everything; Qwen Edit only for uploaded-image-bound chars
# sdxl_hybrid  = SDXL for new chars/bg; Qwen Edit for recurring chars
# zimage_hybrid= ZImage for new chars/bg; Qwen Edit for recurring chars
_img_mode_raw = os.getenv("IMAGE_GENERATION_MODE", "sdxl_only").lower().strip()
if _img_mode_raw not in ("sdxl_only", "sdxl_hybrid", "zimage_hybrid"):
    # Legacy fallback
    if os.getenv("SD_ENABLE_CHARACTER_IMG2IMG", "false").lower() == "true":
        _img_mode_raw = "sdxl_hybrid"
    else:
        _img_mode_raw = "sdxl_only"

IMAGE_GENERATION_MODE        = _img_mode_raw
BASE_IMAGE_ENGINE            = "zimage" if _img_mode_raw == "zimage_hybrid" else "sdxl"
USE_QWEN_EDIT_FOR_RECURRING  = _img_mode_raw in ("sdxl_hybrid", "zimage_hybrid")

# ─── SDXL Configuration ───────────────────────────────────────────────────────
SDXL_MODEL_PATH      = os.getenv("SDXL_MODEL_PATH",      "/path/to/model.safetensors")
SDXL_DTYPE           = os.getenv("SDXL_DTYPE",           "bfloat16")
SDXL_MAX_BATCH_SIZE  = int(os.getenv("SDXL_MAX_BATCH_SIZE",  "4"))
SDXL_STEPS           = int(os.getenv("SDXL_STEPS",           "30"))
SDXL_GUIDANCE_SCALE  = float(os.getenv("SDXL_GUIDANCE_SCALE","6.0"))
SDXL_NEGATIVE_PROMPT = os.getenv(
    "SDXL_NEGATIVE_PROMPT",
    "worst quality, comic, multiple views, bad quality, low quality, lowres, "
    "displeasing, very displeasing, bad anatomy, bad hands, scan artifacts, "
    "monochrome, greyscale, twitter username, jpeg artifacts, 2koma, 4koma, "
    "extra digits, fewer digits, jaggy lines, unclear, signature",
)
# Shared resolution for both SDXL and ZImage
SDXL_CHARACTER_WIDTH   = int(os.getenv("SDXL_CHARACTER_WIDTH",   "768"))
SDXL_CHARACTER_HEIGHT  = int(os.getenv("SDXL_CHARACTER_HEIGHT",  "1024"))
SDXL_BACKGROUND_WIDTH  = int(os.getenv("SDXL_BACKGROUND_WIDTH",  "1280"))
SDXL_BACKGROUND_HEIGHT = int(os.getenv("SDXL_BACKGROUND_HEIGHT", "720"))

# ─── ZImage Configuration ─────────────────────────────────────────────────────
ZIMAGE_MODEL_PATH          = os.getenv("ZIMAGE_MODEL_PATH",          "/path/to/Z-Image-Turbo")
ZIMAGE_DTYPE               = os.getenv("ZIMAGE_DTYPE",               "bfloat16")
ZIMAGE_STEPS               = int(os.getenv("ZIMAGE_STEPS",           "9"))
ZIMAGE_GUIDANCE_SCALE      = float(os.getenv("ZIMAGE_GUIDANCE_SCALE","0.0"))
ZIMAGE_ATTENTION_BACKEND   = os.getenv("ZIMAGE_ATTENTION_BACKEND",   "flash")
ZIMAGE_ENABLE_CPU_OFFLOAD  = os.getenv("ZIMAGE_ENABLE_CPU_OFFLOAD",  "true").lower() == "true"
ZIMAGE_NEGATIVE_PROMPT     = os.getenv(
    "ZIMAGE_NEGATIVE_PROMPT",
    "extra limbs, extra fingers, extra arms, extra legs, blurry skin texture, "
    "plastic skin, waxy face, oversaturated colors, flat lighting, low contrast, "
    "dull skin tone",
)
# subprocess = ZImage runs in a child process; child exits → OS reclaims all RAM (recommended)
# inprocess  = ZImage runs in the same process as app.py (may leave CPU RAM residue)
ZIMAGE_RUN_MODE = os.getenv("ZIMAGE_RUN_MODE", "subprocess").lower().strip()
if ZIMAGE_RUN_MODE not in ("subprocess", "inprocess"):
    ZIMAGE_RUN_MODE = "subprocess"

# ─── Qwen Edit Configuration ──────────────────────────────────────────────────
QWEN_EDIT_DIFFUSION_MODEL_PATH = os.getenv("QWEN_EDIT_DIFFUSION_MODEL_PATH", "/path/to/diffusion.gguf")
QWEN_EDIT_LLM_PATH             = os.getenv("QWEN_EDIT_LLM_PATH",             "/path/to/llm.gguf")
QWEN_EDIT_F2P_LORA_PATH        = os.getenv("QWEN_EDIT_F2P_LORA_PATH",        "/path/to/dir_of_F2P.safetensors")
QWEN_EDIT_VAE_PATH             = os.getenv("QWEN_EDIT_VAE_PATH",             "/path/to/vae.safetensors")
QWEN_EDIT_CLIP_VISION_PATH     = os.getenv("QWEN_EDIT_CLIP_VISION_PATH",     "/path/to/clip.gguf")
QWEN_EDIT_NEGATIVE_PROMPT = os.getenv(
    "QWEN_EDIT_NEGATIVE_PROMPT",
    "deformed, mutated, disfigured, poorly drawn hands, poorly drawn face, extra limbs, "
    "extra fingers, extra arms, extra legs, malformed limbs, fused fingers, too many fingers, "
    "long neck, missing arms, missing legs, bad anatomy, bad proportions, cloned face, "
    "gross proportions, text, error, cropped, worst quality, low quality, "
    "jpeg artifacts, signature, watermark, username, blurry",
)
QWEN_EDIT_WIDTH          = int(os.getenv("QWEN_EDIT_WIDTH",          "768"))
QWEN_EDIT_HEIGHT         = int(os.getenv("QWEN_EDIT_HEIGHT",         "1024"))
QWEN_EDIT_CFG_SCALE      = float(os.getenv("QWEN_EDIT_CFG_SCALE",    "1"))
QWEN_EDIT_SAMPLE_STEPS   = int(os.getenv("QWEN_EDIT_SAMPLE_STEPS",   "8"))
QWEN_EDIT_SAMPLE_METHOD  = os.getenv("QWEN_EDIT_SAMPLE_METHOD",  "euler_a")
QWEN_EDIT_SCHEDULER      = os.getenv("QWEN_EDIT_SCHEDULER",      "simple")
QWEN_EDIT_SEED           = int(os.getenv("QWEN_EDIT_SEED",        "-1"))

STORY_CONTEXT_SEGMENTS = int(os.getenv("STORY_CONTEXT_SEGMENTS", "6"))
MOBILE_PORT  = int(os.getenv("MOBILE_PORT",  "5001"))
DESKTOP_PORT = int(os.getenv("DESKTOP_PORT", "5002"))

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
STATIC_DIR    = BASE_DIR / "static"
GENERATED_DIR = STATIC_DIR / "generated"
PROMPTS_DIR   = BASE_DIR / "prompts"
UPLOADS_DIR   = DATA_DIR / "uploads"

DATA_DIR.mkdir(exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

STORY_FILE      = DATA_DIR / "story.json"
RAW_LLM_FILE    = DATA_DIR / "raw_llm_response.txt"
LAST_ERROR_FILE = DATA_DIR / "last_error.txt"

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Runtime flags ────────────────────────────────────────────────────────────
_mock_mode        = False
_serve_only       = False
_llama_process    = None
_llama_proc_lock  = threading.Lock()
_llama_start_lock = threading.Lock()
_llm_ready        = False

# ─── Shared State ────────────────────────────────────────────────────────────
story_state = {
    "title": "oh-my-comic",
    "current_index": 0,
    "segments": [],
    "mode": "generate",
    "streaming_segment": None,
    # Character IDs whose images were bound from user uploads.
    # These always use Qwen Edit for consistency, regardless of IMAGE_GENERATION_MODE.
    "force_qwen_edit_character_ids": [],
    # Visual profile for each character: first_prompt, engine, source, etc.
    # Used to inject context into LLM prompts so it can write proper edit_prompts.
    "character_profiles": {},
}
state_lock      = threading.Lock()
generation_lock = threading.Lock()
_is_generating  = False
_creative_mode  = False
_is_batch_imaging = False

_status_log  = deque(maxlen=50)
_status_lock = threading.Lock()

# ─── Flask Apps ──────────────────────────────────────────────────────────────
mobile_app = Flask(
    "mobile_app",
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(STATIC_DIR),
)
mobile_app.config["SECRET_KEY"] = "mobile_secret_key_2024"
mobile_app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

desktop_app = Flask(
    "desktop_app",
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(STATIC_DIR),
)
desktop_app.config["SECRET_KEY"] = "desktop_secret_key_2024"

# ─── Utility ─────────────────────────────────────────────────────────────────
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def broadcast_status(message, status_type="info"):
    log.info(f"[STATUS] {message}")
    entry = {"message": message, "type": status_type, "timestamp": time.time()}
    with _status_lock:
        _status_log.append(entry)


def get_latest_status():
    with _status_lock:
        return _status_log[-1] if _status_log else {"message": "就绪", "type": "info", "timestamp": time.time()}


def log_error(message):
    try:
        with open(LAST_ERROR_FILE, "w", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]\n{message}\n")
    except Exception:
        pass

# ─── Story State Management ──────────────────────────────────────────────────
def save_story():
    with state_lock:
        with open(STORY_FILE, "w", encoding="utf-8") as f:
            json.dump(story_state, f, ensure_ascii=False, indent=2)


def load_story():
    global story_state
    if STORY_FILE.exists():
        with open(STORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        with state_lock:
            story_state.update(data)
        log.info(f"Loaded story with {len(story_state['segments'])} segments")
    else:
        log.info("No existing story found, starting fresh")


def get_story_snapshot():
    with state_lock:
        snap = json.loads(json.dumps(story_state))
    snap["is_generating"]    = _is_generating
    snap["is_batch_imaging"] = _is_batch_imaging
    snap["creative_mode"]    = _creative_mode
    snap["llm_ready"]        = _llm_ready
    snap["latest_status"]    = get_latest_status()
    return snap

# ─── Prompt Templates ────────────────────────────────────────────────────────
DEFAULT_SYSTEM_PROMPT = (
    "你是一个互动连环画故事生成器。\n"
    "你需要根据当前故事历史和用户的新指令，生成下一段故事。\n"
    "必须严格返回 JSON 对象，不要返回 Markdown，不要返回解释文字。"
)

DEFAULT_USER_TEMPLATE = """你正在续写一个互动连环画故事。

【已有角色视觉档案】
{{CHARACTER_PROFILES}}

【已有故事历史】
{{HISTORY}}

【用户希望第 {{SEGMENT_ID}} 段的发展】
{{USER_DIRECTION}}

请生成下一段故事，并给出角色图和背景图的 AI 绘图提示词。

严格只返回 JSON 对象，不要 Markdown，不要解释。
JSON 格式必须是：
{
  "text": "下一段故事正文",
  "character_prompts": [
    {
      "id": "deer",
      "prompt": "English text-to-image prompt",
      "edit_prompt": "中文图生图编辑指令"
    }
  ],
  "background_prompt": {
    "id": "river",
    "prompt": "English text-to-image prompt for background"
  }
}

要求：
1. character_prompts 最多 2 个，可以为空数组 []。
2. 同一角色在后续段落中必须复用完全相同的 id。
3. 每个 character_prompts 项必须同时包含 id、prompt、edit_prompt 三个字段。
4. background_prompt 可以为 null。
5. 不要输出其他字段。"""


def load_prompt_template(filename):
    path = PROMPTS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def _select_user_template_filename():
    """Choose the right prompt template based on the current image generation mode."""
    if BASE_IMAGE_ENGINE == "zimage":
        return "story_user_template_zimage.txt"
    return "story_user_template.txt"


def build_history_text():
    segments = story_state.get("segments", [])
    recent = (
        segments[-STORY_CONTEXT_SEGMENTS:]
        if len(segments) > STORY_CONTEXT_SEGMENTS
        else segments
    )
    if not recent:
        return "（暂无故事历史，这是第一段）"
    return "\n\n".join(f"第 {s['id']+1} 段：\n{s['text']}" for s in recent)


def build_character_id_text():
    """Return a list of all character IDs seen so far, to help the LLM reuse them."""
    seen = {}
    for seg in story_state.get("segments", []):
        for cp in seg.get("character_prompts", []):
            cid = cp.get("id")
            if cid and cid not in seen:
                seen[cid] = True
    if not seen:
        return ""
    return "【已有角色 ID（续写时必须复用）】\n" + "、".join(seen.keys())


def build_character_profile_text():
    """
    Build the {{CHARACTER_PROFILES}} block injected into every LLM prompt.
    Tells the LLM which characters already have reference images and what
    their first_prompt looks like, so it can write proper edit_prompts.
    """
    profiles = story_state.get("character_profiles", {})
    if not profiles:
        return "（暂无角色视觉档案，本段所有角色均为新角色）"

    lines = []
    for cid, p in profiles.items():
        source_note = "（用户上传图片）" if p.get("source") == "uploaded" else ""
        lines.append(
            f"- id: {cid}{source_note}\n"
            f"  首次角色 prompt: {p.get('first_prompt', '')}"
        )
    return "\n".join(lines)


def build_user_prompt(direction, segment_id):
    template_file = _select_user_template_filename()
    template = load_prompt_template(template_file) or DEFAULT_USER_TEMPLATE
    char_ids = build_character_id_text()
    history  = build_history_text()
    if char_ids:
        history = char_ids + "\n\n" + history
    return (
        template
        .replace("{{CHARACTER_PROFILES}}", build_character_profile_text())
        .replace("{{HISTORY}}", history)
        .replace("{{USER_DIRECTION}}", direction)
        .replace("{{SEGMENT_ID}}", str(segment_id + 1))
        .replace("{{PROMPT_RATING}}", PROMPT_RATING)
    )

# ─── Character Profile Management ────────────────────────────────────────────
def update_character_profiles(segment_id, character_prompts, source="generated"):
    """
    Register new characters into story_state["character_profiles"].
    Existing characters are NOT overwritten — first_prompt is the canonical reference.
    """
    profiles = story_state.setdefault("character_profiles", {})
    for cp in character_prompts:
        cid = cp.get("id")
        if not cid or cid in profiles:
            continue
        profiles[cid] = {
            "id": cid,
            "first_prompt": cp.get("prompt", ""),
            "first_segment_id": segment_id,
            "engine": BASE_IMAGE_ENGINE,
            "source": source,
        }
        log.info(f"New character profile registered: '{cid}' (engine={BASE_IMAGE_ENGINE})")

# ─── llama-server Management ─────────────────────────────────────────────────
def _is_target_llama_running():
    if not LLAMA_SERVER_EXE:
        return False
    target = str(Path(LLAMA_SERVER_EXE).resolve()).lower()
    try:
        if os.name == "nt":
            result = subprocess.run(
                [
                    "powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_Process | "
                    "Where-Object { $_.ExecutablePath -and "
                    "$_.ExecutablePath.ToLower() -eq '" + target.replace("'", "''") + "' } | "
                    "Select-Object -First 1 -ExpandProperty ProcessId"
                ],
                capture_output=True, text=True, timeout=10,
            )
            return result.stdout.strip().isdigit()
        else:
            exe_name = Path(LLAMA_SERVER_EXE).name
            result = subprocess.run(
                ["pgrep", "-f", exe_name],
                capture_output=True, text=True, timeout=10,
            )
            return result.returncode == 0
    except Exception as e:
        log.warning(f"Process check failed: {e}")
        return False


def validate_no_existing_llama():
    if _mock_mode or not LLAMA_SERVER_EXE:
        return
    if _is_target_llama_running():
        print("\n" + "=" * 60)
        print("  ERROR: target llama-server is already running!")
        print(f"  EXE: {LLAMA_SERVER_EXE}")
        print()
        print("  Please close it first, then run python app.py again.")
        print("=" * 60 + "\n")
        sys.exit(1)


def _llama_api_ready():
    try:
        import urllib.request
        url = f"http://{LLAMA_HOST}:{LLAMA_PORT}/v1/models"
        with urllib.request.urlopen(url, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def start_llama_server():
    global _llama_process, _llm_ready

    if _llama_api_ready():
        log.info("llama-server already ready")
        broadcast_status("Qwen 语言模型已就绪")
        _llm_ready = True
        return

    with _llama_start_lock:
        if _llama_api_ready():
            broadcast_status("Qwen 语言模型已就绪")
            _llm_ready = True
            return

        if not LLAMA_SERVER_EXE or not QWEN_GGUF_MODEL:
            raise RuntimeError(
                "LLAMA_SERVER_EXE and QWEN_GGUF_MODEL must be set in .env to use LLM generation."
            )

        cmd = [
            LLAMA_SERVER_EXE,
            "-m", QWEN_GGUF_MODEL,
            "--host", LLAMA_HOST,
            "--port", str(LLAMA_PORT),
            "-c", str(LLAMA_CTX_SIZE),
            "--spec-type", LLAMA_SPEC_TYPE,
            "--spec-ngram-size-n", str(LLAMA_SPEC_NGRAM_SIZE_N),
            "--spec-ngram-size-m", str(LLAMA_SPEC_NGRAM_SIZE_M),
        ]
        if LLAMA_KV_UNIFIED:
            cmd.append("--kv-unified")
        if LLAMA_KV_OFFLOAD:
            cmd.append("--kv-offload")
        if LLAMA_MLOCK:
            cmd.append("--mlock")
        if LLAMA_NO_MMAP:
            cmd.append("--no-mmap")
        if LLAMA_FLASH_ATTN:
            cmd.extend(["-fa", LLAMA_FLASH_ATTN])
        if LLAMA_MMPROJ_MODEL:
            cmd.extend(["--mmproj", LLAMA_MMPROJ_MODEL])
        if LLAMA_EXTRA_ARGS:
            cmd.extend(LLAMA_EXTRA_ARGS.split())

        log.info(f"Starting llama-server: {' '.join(cmd)}")
        broadcast_status("正在启动 Qwen 语言模型...")
        _llm_ready = False

        with _llama_proc_lock:
            _llama_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        for i in range(60):
            time.sleep(2)
            if _llama_api_ready():
                log.info(f"llama-server ready after {(i+1)*2}s")
                broadcast_status("Qwen 语言模型已就绪")
                _llm_ready = True
                return
            broadcast_status(f"等待 Qwen 模型加载... ({(i+1)*2}s)")

        _llm_ready = False
        raise RuntimeError("llama-server did not become ready within 120 s")


def prewarm_llama():
    if _mock_mode or not LLAMA_SERVER_EXE:
        return

    def _worker():
        try:
            start_llama_server()
        except Exception as e:
            log.warning(f"Pre-warm llama failed: {e}")
            broadcast_status(f"Qwen 预热失败: {e}", "warning")

    threading.Thread(target=_worker, daemon=True, name="llama-prewarm").start()


def stop_llama_server():
    global _llama_process, _llm_ready
    with _llama_proc_lock:
        proc = _llama_process
        _llama_process = None

    _llm_ready = False

    if proc is None:
        return

    log.info("Stopping llama-server...")
    broadcast_status("正在关闭 Qwen 语言模型，释放显存...")
    try:
        proc.terminate()
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    log.info("llama-server stopped")

    for _ in range(10):
        if not _llama_api_ready():
            break
        time.sleep(1)
    time.sleep(1)

# ─── LLM Call ────────────────────────────────────────────────────────────────
def call_qwen(direction, segment_id):
    system_prompt = load_prompt_template("story_system_prompt.txt") or DEFAULT_SYSTEM_PROMPT
    user_prompt   = build_user_prompt(direction, segment_id)

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    broadcast_status("Qwen 正在思考并生成完整 JSON...")

    common_kwargs = dict(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )

    try:
        completion = client.chat.completions.create(
            **common_kwargs,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        log.warning(f"response_format not supported, retrying without it: {e}")
        completion = client.chat.completions.create(**common_kwargs)

    broadcast_status("Qwen 回复已收到，正在解析 JSON...")
    raw = completion.choices[0].message.content or ""

    with open(RAW_LLM_FILE, "w", encoding="utf-8") as f:
        f.write(raw)
    return raw


def call_qwen_vision_parse(image_path: Path, user_text: str) -> str:
    if not LLAMA_MMPROJ_MODEL:
        return "(图片解析不可用：未配置 LLAMA_MMPROJ_MODEL)"

    broadcast_status("正在解析上传图片...")

    suffix = image_path.suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".webp": "image/webp", ".gif": "image/gif"}
    mime = mime_map.get(suffix, "image/png")

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "请详细描述这张图片中的人物外貌、服装、表情、姿势和背景环境。"
                                "如果图片中有文字，也请一并说明。"
                                f"\n\n用户附加说明：{user_text}"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                }
            ],
        )
        result = completion.choices[0].message.content or ""
        broadcast_status("图片解析完成")
        return result
    except Exception as e:
        err_msg = f"图片解析失败: {e}"
        log.error(err_msg)
        log_error(err_msg)
        broadcast_status(err_msg, "warning")
        return f"(图片解析失败: {e})"


def call_qwen_with_image(image_path: Path, user_text: str, segment_id: int) -> str:
    """
    Parse image, combine with user text, then call story generation.
    A one-time single-character constraint is injected (not stored in history).
    """
    image_description = call_qwen_vision_parse(image_path, user_text)

    upload_rule = (
        "\n\n【本轮上传图片强制规则（仅本轮有效，不影响后续段落）】\n"
        "1. 本轮 character_prompts 只能包含 1 个角色，即上传图片中的这名角色。\n"
        "2. 禁止在本轮 character_prompts 中出现已有故事里的其他角色。\n"
        "3. 如果用户文本中提到其他角色，只能作为背景信息，不得为其生成 character_prompts。\n"
        "4. 本轮上传图片将被绑定为 character_prompts[0].id 对应的角色立绘。"
    )

    combined_direction = (
        f"{user_text}\n\n"
        f"【用户上传图片的内容描述】\n{image_description}"
        f"{upload_rule}"
    )
    return call_qwen(combined_direction, segment_id)

# ─── JSON Parsing ────────────────────────────────────────────────────────────
def parse_llm_json(raw):
    for attempt in (
        lambda: json.loads(raw),
        lambda: json.loads(re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL).group(1)),
        lambda: json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0)),
    ):
        try:
            return attempt()
        except Exception:
            pass
    raise ValueError(f"Cannot parse JSON from LLM output: {raw[:300]}")

# ─── Segment Post-processing ─────────────────────────────────────────────────
def sanitize_id(raw_id):
    if not raw_id:
        return "unknown"
    safe = re.sub(r"[^\w]", "_", str(raw_id)).strip("_")
    return safe or "unknown"


def postprocess_segment(raw_json_str, segment_id, user_text="",
                        skip_character_generation=False,
                        uploaded_image_path: Optional[Path] = None):
    try:
        data = parse_llm_json(raw_json_str)
    except ValueError as e:
        err = f"JSON parse error for segment {segment_id}: {e}"
        log.error(err)
        log_error(err)
        data = {"text": f"（第 {segment_id+1} 段生成失败，请重试）",
                "character_prompts": [], "background_prompt": None}

    text      = data.get("text", "")
    raw_chars = data.get("character_prompts", [])
    if not isinstance(raw_chars, list):
        raw_chars = []
    # Image-upload rounds: enforce single character as backend safety net
    raw_chars = raw_chars[:1] if skip_character_generation else raw_chars[:2]

    character_prompts = []
    character_images  = []

    for idx, cp in enumerate(raw_chars):
        if not isinstance(cp, dict):
            continue
        cid        = sanitize_id(cp.get("id", "character"))
        cprompt    = str(cp.get("prompt", ""))
        cedit      = str(cp.get("edit_prompt", ""))
        character_prompts.append({"id": cid, "prompt": cprompt, "edit_prompt": cedit})

        if skip_character_generation and idx == 0 and uploaded_image_path:
            dest_filename = f"segment_{segment_id:03d}_character_{cid}.png"
            dest_path = GENERATED_DIR / dest_filename
            try:
                shutil.copy2(uploaded_image_path, dest_path)
                character_images.append({
                    "id": cid, "status": "done",
                    "url": f"/static/generated/{dest_filename}",
                    "file": dest_filename,
                    "source": "uploaded",
                })
                # Register for forced Qwen Edit
                force_ids = story_state.setdefault("force_qwen_edit_character_ids", [])
                if cid not in force_ids:
                    force_ids.append(cid)
                    log.info(f"Character '{cid}' registered for forced Qwen Edit (uploaded reference)")
                    broadcast_status(
                        f"已将上传图绑定为角色 {cid} 的参考图，后续该角色将强制使用 Qwen Edit。",
                        "success",
                    )
                # Register in character_profiles with source=uploaded
                profiles = story_state.setdefault("character_profiles", {})
                if cid not in profiles:
                    profiles[cid] = {
                        "id": cid,
                        "first_prompt": cprompt or "（用户上传图片）",
                        "first_segment_id": segment_id,
                        "engine": "uploaded",
                        "source": "uploaded",
                    }
            except Exception as e:
                log.error(f"Failed to copy uploaded image: {e}")
                character_images.append({
                    "id": cid, "status": "failed", "url": None,
                    "file": dest_filename,
                })
        elif skip_character_generation and idx == 0 and not uploaded_image_path:
            broadcast_status(
                "本轮未识别到角色 id 或上传图片保存失败，上传图不会作为角色参考图复用。",
                "warning",
            )
        elif skip_character_generation:
            pass
        else:
            character_images.append({
                "id": cid, "status": "pending", "url": None,
                "file": f"segment_{segment_id:03d}_character_{cid}.png",
            })

    # Update character profiles for non-upload rounds
    if not skip_character_generation:
        update_character_profiles(segment_id, character_prompts)

    raw_bg = data.get("background_prompt")
    background_prompt = None
    background_image  = None
    if raw_bg and isinstance(raw_bg, dict):
        bgid     = sanitize_id(raw_bg.get("id", "background"))
        bgprompt = str(raw_bg.get("prompt", ""))
        background_prompt = {"id": bgid, "prompt": bgprompt}
        background_image  = {
            "id": bgid, "status": "pending", "url": None,
            "file": f"segment_{segment_id:03d}_background_{bgid}.png",
        }

    return {
        "id": segment_id,
        "user_text": user_text,
        "text": text,
        "character_prompts": character_prompts,
        "background_prompt": background_prompt,
        "character_images":  character_images,
        "background_image":  background_image,
        "status": "text_ready",
    }

# ─── Image State Helpers ──────────────────────────────────────────────────────
def _update_image_status(seg_id, image_type, item_id, status, url=None):
    with state_lock:
        seg = next((s for s in story_state["segments"] if s["id"] == seg_id), None)
        if not seg:
            return
        if image_type == "background":
            bi = seg.get("background_image")
            if bi:
                bi["status"] = status
                if url:
                    bi["url"] = url
        else:
            for ci in seg.get("character_images", []):
                if ci["id"] == item_id:
                    ci["status"] = status
                    if url:
                        ci["url"] = url
                    break
    save_story()


def find_first_character_image(character_id, before_segment_id):
    for seg in story_state.get("segments", []):
        if seg["id"] >= before_segment_id:
            continue
        for img in seg.get("character_images", []):
            if img.get("id") == character_id and img.get("status") == "done":
                p = GENERATED_DIR / img["file"]
                if p.exists():
                    return p
    return None

# ─── Image Task Dataclass ─────────────────────────────────────────────────────
@dataclass
class ImageTask:
    seg_id:      int
    image_type:  str
    item_id:     str
    prompt:      str
    filename:    str
    width:       int
    height:      int
    ref_path:    Optional[Path] = None
    edit_prompt: str = ""


def build_image_queues(segment):
    seg_id     = segment["id"]
    base_tasks: List[ImageTask] = []
    edit_tasks: List[ImageTask] = []

    if segment.get("background_prompt") and segment.get("background_image"):
        bi = segment["background_image"]
        if bi.get("status") == "pending":
            base_tasks.append(ImageTask(
                seg_id=seg_id, image_type="background",
                item_id=segment["background_prompt"]["id"],
                prompt=segment["background_prompt"]["prompt"],
                filename=bi["file"],
                width=SDXL_BACKGROUND_WIDTH, height=SDXL_BACKGROUND_HEIGHT,
            ))

    force_ids = story_state.get("force_qwen_edit_character_ids", [])

    for cp in segment.get("character_prompts", []):
        cid = cp["id"]
        ci  = next((c for c in segment["character_images"] if c["id"] == cid), None)
        if not ci or ci.get("status") != "pending":
            continue

        use_edit = USE_QWEN_EDIT_FOR_RECURRING or cid in force_ids
        ref = find_first_character_image(cid, seg_id) if use_edit else None

        task = ImageTask(
            seg_id=seg_id, image_type="character", item_id=cid,
            prompt=cp["prompt"],
            edit_prompt=cp.get("edit_prompt", ""),
            filename=ci["file"],
            width=QWEN_EDIT_WIDTH if ref else SDXL_CHARACTER_WIDTH,
            height=QWEN_EDIT_HEIGHT if ref else SDXL_CHARACTER_HEIGHT,
            ref_path=ref,
        )
        (edit_tasks if ref else base_tasks).append(task)

    return base_tasks, edit_tasks


def collect_all_pending_image_queues():
    all_base: List[ImageTask] = []
    all_edit: List[ImageTask] = []
    for seg in story_state.get("segments", []):
        b, e = build_image_queues(seg)
        all_base.extend(b)
        all_edit.extend(e)
    return all_base, all_edit

# ─── SDXL Batch Generation ────────────────────────────────────────────────────
_sdxl_pipe = None


def _load_sdxl():
    global _sdxl_pipe
    if _sdxl_pipe is not None:
        return _sdxl_pipe
    broadcast_status("正在加载 SDXL 模型...")
    import torch
    from diffusers import StableDiffusionXLPipeline
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    pipe = StableDiffusionXLPipeline.from_single_file(
        SDXL_MODEL_PATH, torch_dtype=dtype_map.get(SDXL_DTYPE, torch.bfloat16),
        use_safetensors=True, low_cpu_mem_usage=False,
    )
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        log.warning("xformers not available, skipping")
    pipe.enable_model_cpu_offload()
    _sdxl_pipe = pipe
    log.info("SDXL model loaded")
    return _sdxl_pipe


def _unload_sdxl():
    global _sdxl_pipe
    if _sdxl_pipe is None:
        return
    import torch
    del _sdxl_pipe
    _sdxl_pipe = None
    torch.cuda.empty_cache()
    log.info("SDXL model unloaded")


def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def run_sdxl_queue(tasks: List[ImageTask]):
    if not tasks:
        return
    import torch
    from collections import defaultdict
    pipe = _load_sdxl()
    groups = defaultdict(list)
    for t in tasks:
        groups[(t.width, t.height)].append(t)

    for (w, h), group in groups.items():
        batches = list(_chunks(group, SDXL_MAX_BATCH_SIZE))
        for batch_idx, batch in enumerate(batches):
            broadcast_status(
                f"SDXL 生成中 ({w}×{h})，批次 {batch_idx+1}/{len(batches)}，"
                f"本批 {len(batch)} 张: " + ", ".join(t.item_id for t in batch)
            )
            try:
                with torch.inference_mode():
                    images = pipe(
                        prompt=[t.prompt for t in batch],
                        negative_prompt=[SDXL_NEGATIVE_PROMPT] * len(batch),
                        width=w, height=h,
                        num_inference_steps=SDXL_STEPS,
                        guidance_scale=SDXL_GUIDANCE_SCALE,
                    ).images
                for task, img in zip(batch, images):
                    img.save(GENERATED_DIR / task.filename)
                    url = f"/static/generated/{task.filename}"
                    _update_image_status(task.seg_id, task.image_type, task.item_id, "done", url)
                    log.info(f"SDXL saved: {task.filename}")
            except Exception as e:
                err = f"SDXL batch error: {e}"
                log.error(err)
                log_error(err)
                for task in batch:
                    _update_image_status(task.seg_id, task.image_type, task.item_id, "failed")
    _unload_sdxl()

# ─── ZImage Serial Generation ─────────────────────────────────────────────────
_zimage_pipe = None


def _load_zimage():
    global _zimage_pipe
    if _zimage_pipe is not None:
        return _zimage_pipe
    broadcast_status("正在加载 ZImage 模型（进程内）...")
    import torch
    from diffusers import ZImagePipeline
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    pipe = ZImagePipeline.from_pretrained(
        ZIMAGE_MODEL_PATH,
        torch_dtype=dtype_map.get(ZIMAGE_DTYPE, torch.bfloat16),
        low_cpu_mem_usage=True,
    )
    pipe.to("cuda")
    if ZIMAGE_ATTENTION_BACKEND:
        try:
            pipe.transformer.set_attention_backend(ZIMAGE_ATTENTION_BACKEND)
        except Exception as e:
            log.warning(f"ZImage attention backend '{ZIMAGE_ATTENTION_BACKEND}' failed: {e}")
    if ZIMAGE_ENABLE_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
    _zimage_pipe = pipe
    log.info("ZImage model loaded (inprocess)")
    return _zimage_pipe


def _unload_zimage():
    global _zimage_pipe
    if _zimage_pipe is None:
        return
    import gc
    import torch
    pipe = _zimage_pipe
    try:
        if hasattr(pipe, "maybe_free_model_hooks"):
            pipe.maybe_free_model_hooks()
    except Exception as e:
        log.warning(f"ZImage maybe_free_model_hooks failed: {e}")
    for attr in ("transformer", "vae", "text_encoder", "tokenizer",
                 "scheduler", "image_processor"):
        try:
            if hasattr(pipe, attr):
                setattr(pipe, attr, None)
        except Exception:
            pass
    del pipe
    del _zimage_pipe
    _zimage_pipe = None
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
    log.info("ZImage model unloaded (inprocess)")


def run_zimage_queue_inprocess(tasks: List[ImageTask]):
    """ZImage generates images serially inside the current process."""
    if not tasks:
        return
    try:
        pipe = _load_zimage()
        for idx, task in enumerate(tasks):
            broadcast_status(
                f"ZImage 生成中（进程内）：{task.item_id}，第 {idx+1}/{len(tasks)} 张..."
            )
            try:
                result = pipe(
                    prompt=task.prompt,
                    negative_prompt=ZIMAGE_NEGATIVE_PROMPT,
                    height=task.height,
                    width=task.width,
                    num_inference_steps=ZIMAGE_STEPS,
                    guidance_scale=ZIMAGE_GUIDANCE_SCALE,
                )
                image = result.images[0]
                image.save(GENERATED_DIR / task.filename)
                del image
                del result
                url = f"/static/generated/{task.filename}"
                _update_image_status(task.seg_id, task.image_type, task.item_id, "done", url)
                log.info(f"ZImage saved: {task.filename}")
            except Exception as e:
                err = f"ZImage error ({task.filename}): {e}"
                log.error(err)
                log_error(err)
                _update_image_status(task.seg_id, task.image_type, task.item_id, "failed")
    finally:
        _unload_zimage()


def run_zimage_queue_subprocess(tasks: List[ImageTask]):
    """
    ZImage generates images in a child process.
    When the child exits, the OS reclaims all ZImage CPU RAM and CUDA memory,
    so Qwen Edit (loaded afterwards) is not slowed down by residue.
    """
    if not tasks:
        return

    ts = int(time.time() * 1000)
    tasks_file   = DATA_DIR / f"zimage_tasks_{ts}.json"
    results_file = DATA_DIR / f"zimage_results_{ts}.json"

    payload = {
        "config": {
            "model_path":         ZIMAGE_MODEL_PATH,
            "dtype":              ZIMAGE_DTYPE,
            "steps":              ZIMAGE_STEPS,
            "guidance_scale":     ZIMAGE_GUIDANCE_SCALE,
            "attention_backend":  ZIMAGE_ATTENTION_BACKEND,
            "negative_prompt":    ZIMAGE_NEGATIVE_PROMPT,
            "enable_cpu_offload": ZIMAGE_ENABLE_CPU_OFFLOAD,
            "generated_dir":      str(GENERATED_DIR),
        },
        "tasks": [
            {
                "seg_id":     t.seg_id,
                "image_type": t.image_type,
                "item_id":    t.item_id,
                "prompt":     t.prompt,
                "filename":   t.filename,
                "width":      t.width,
                "height":     t.height,
            }
            for t in tasks
        ],
    }

    with open(tasks_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    broadcast_status(f"ZImage 子进程启动，共 {len(tasks)} 张...")
    log.info(f"ZImage subprocess: tasks={tasks_file}, results={results_file}")

    try:
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "zimage_worker.py"),
             "--tasks",   str(tasks_file),
             "--results", str(results_file)],
            capture_output=False,
            timeout=3600,
        )
        if result.returncode != 0:
            log.error(f"ZImage worker exited with code {result.returncode}")
    except subprocess.TimeoutExpired:
        log.error("ZImage worker timed out after 3600s")
    except Exception as e:
        log.error(f"ZImage worker launch failed: {e}")
        log_error(f"ZImage worker launch failed: {e}")
        for task in tasks:
            _update_image_status(task.seg_id, task.image_type, task.item_id, "failed")
        return

    # Read results
    if results_file.exists():
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for r in data.get("results", []):
                _update_image_status(
                    r["seg_id"], r["image_type"], r["item_id"],
                    r["status"], r.get("url"),
                )
                log.info(f"ZImage result: {r['item_id']} → {r['status']}")
        except Exception as e:
            log.error(f"Failed to read ZImage results: {e}")
    else:
        log.error(f"ZImage results file not found: {results_file}")
        for task in tasks:
            _update_image_status(task.seg_id, task.image_type, task.item_id, "failed")

    # Cleanup temp files
    for p in (tasks_file, results_file):
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass

    broadcast_status(f"ZImage 子进程完成，共 {len(tasks)} 张")


def run_zimage_queue(tasks: List[ImageTask]):
    """Route ZImage tasks to subprocess or inprocess based on ZIMAGE_RUN_MODE."""
    if not tasks:
        return
    if ZIMAGE_RUN_MODE == "subprocess":
        run_zimage_queue_subprocess(tasks)
    else:
        run_zimage_queue_inprocess(tasks)

# ─── Qwen Edit Serial Generation ─────────────────────────────────────────────
_qwen_edit_pipe = None


def _load_qwen_edit():
    global _qwen_edit_pipe
    if _qwen_edit_pipe is not None:
        return _qwen_edit_pipe
    broadcast_status("正在加载 Qwen Edit 模型...")
    from stable_diffusion_cpp import StableDiffusion
    _qwen_edit_pipe = StableDiffusion(
        diffusion_model_path=QWEN_EDIT_DIFFUSION_MODEL_PATH,
        llm_path=QWEN_EDIT_LLM_PATH,
        vae_path=QWEN_EDIT_VAE_PATH,
        clip_vision_path=QWEN_EDIT_CLIP_VISION_PATH,
        lora_model_dir=QWEN_EDIT_F2P_LORA_PATH,
        qwen_image_zero_cond_t=True,
        diffusion_flash_attn=True,
        offload_params_to_cpu=True,
    )
    log.info("Qwen Edit model loaded")
    return _qwen_edit_pipe


def _unload_qwen_edit():
    global _qwen_edit_pipe
    if _qwen_edit_pipe is None:
        return
    import torch
    del _qwen_edit_pipe
    _qwen_edit_pipe = None
    torch.cuda.empty_cache()
    log.info("Qwen Edit model unloaded")


def build_qwen_edit_prompt(task: ImageTask) -> str:
    """
    Build an edit-style prompt for Qwen Edit.
    Prefers task.edit_prompt (LLM-generated editing instruction) over task.prompt.
    Always appends a backend safety clause to preserve character identity.
    """
    edit_text = task.edit_prompt or task.prompt
    return (
        "<lora:F2P:1>, "
        f"{edit_text} "
        "保持参考图中的人物身份、脸型、发型、发色、瞳色、服装主体、配饰和整体角色设计不变。"
    )


def run_qwen_edit_queue(tasks: List[ImageTask]):
    if not tasks:
        return
    sd = _load_qwen_edit()
    for idx, task in enumerate(tasks):
        broadcast_status(
            f"Qwen Edit 生成中 (角色一致性)：{task.item_id}，"
            f"第 {idx+1}/{len(tasks)} 张..."
        )
        try:
            output = sd.generate_image(
                prompt=build_qwen_edit_prompt(task),
                negative_prompt=QWEN_EDIT_NEGATIVE_PROMPT,
                ref_images=str(task.ref_path),
                cfg_scale=QWEN_EDIT_CFG_SCALE, sample_steps=QWEN_EDIT_SAMPLE_STEPS,
                sample_method=QWEN_EDIT_SAMPLE_METHOD, scheduler=QWEN_EDIT_SCHEDULER,
                width=task.width, height=task.height, seed=QWEN_EDIT_SEED,
            )
            output[0].save(GENERATED_DIR / task.filename)
            url = f"/static/generated/{task.filename}"
            _update_image_status(task.seg_id, task.image_type, task.item_id, "done", url)
            log.info(f"Qwen Edit saved: {task.filename}")
        except Exception as e:
            err = f"Qwen Edit error ({task.filename}): {e}"
            log.error(err)
            log_error(err)
            _update_image_status(task.seg_id, task.image_type, task.item_id, "failed")
    _unload_qwen_edit()

# ─── Base Engine Dispatcher ───────────────────────────────────────────────────
def run_base_queue(tasks: List[ImageTask]):
    """Route base (text-to-image) tasks to SDXL or ZImage based on mode."""
    if not tasks:
        return
    if BASE_IMAGE_ENGINE == "zimage":
        run_zimage_queue(tasks)
    else:
        run_sdxl_queue(tasks)

# ─── Batch Imaging ────────────────────────────────────────────────────────────
def run_batch_imaging():
    global _is_batch_imaging
    _is_batch_imaging = True
    batch_start = time.time()
    try:
        base_tasks, edit_tasks = collect_all_pending_image_queues()
        total = len(base_tasks) + len(edit_tasks)
        if total == 0:
            broadcast_status("没有待生成的图片", "info")
            return

        broadcast_status(f"批量生图开始，共 {total} 张待生成...")

        if base_tasks:
            engine_name = "ZImage" if BASE_IMAGE_ENGINE == "zimage" else "SDXL"
            broadcast_status(f"{engine_name} 批量队列：{len(base_tasks)} 张...")
            run_base_queue(base_tasks)

        if edit_tasks:
            broadcast_status(f"Qwen Edit 批量队列：{len(edit_tasks)} 张...")
            run_qwen_edit_queue(edit_tasks)

        elapsed = int(time.time() - batch_start)
        broadcast_status(f"批量生图完成，总用时 {elapsed}s", "success")
    except Exception as e:
        err = f"批量生图出错: {e}"
        log.error(err, exc_info=True)
        log_error(err)
        broadcast_status(err, "error")
    finally:
        _is_batch_imaging = False

# ─── Mock Generation ─────────────────────────────────────────────────────────
MOCK_STORIES = [
    {
        "text": "月光下，小鹿背着蓝色邮包走在银色的河边。河水轻轻流淌，远处传来悠扬的钟声。",
        "character_prompts": [{"id": "deer", "prompt": "1other, full_body, young deer, general, smile, blue mailbag, moonlit riverside, front view, standing", "edit_prompt": "将图中角色调整为站立姿势，背着蓝色邮包，表情愉快，保持人物身份和整体设计不变。"}],
        "background_prompt": {"id": "river", "prompt": "general, moonlit river, silver reflections, misty forest, night, fantasy"},
    },
    {
        "text": "小鹿发现河边停着一艘没有船夫的小船，船头挂着一盏摇曳的灯笼。",
        "character_prompts": [
            {"id": "deer", "prompt": "1other, full_body, young deer, general, curious expression, moonlit night, front view, standing", "edit_prompt": "将图中角色改为好奇地看向河边小船的姿势，身体略微前倾，表情好奇，保持人物身份和整体设计不变。"},
        ],
        "background_prompt": {"id": "dock", "prompt": "general, misty riverside dock, glowing lantern reflections, night scene, fantasy"},
    },
    {
        "text": "船舱里坐着一个会说梦话的机器人，它的眼睛发出柔和的蓝光，嘴里喃喃着星星的名字。",
        "character_prompts": [{"id": "robot", "prompt": "1other, full_body, old robot, general, glowing blue eyes, sitting, murmuring, gentle light, front view", "edit_prompt": "将图中角色改为坐在船舱内喃喃自语的姿势，眼睛发出蓝光，表情迷茫，保持人物身份和整体设计不变。"}],
        "background_prompt": {"id": "cabin", "prompt": "general, cozy wooden boat cabin interior, blue glowing lights, starry sky through window"},
    },
]


def generate_mock_segment(segment_id, direction):
    base = MOCK_STORIES[segment_id % len(MOCK_STORIES)]
    text = (f"（根据你的指引：{direction}）\n{base['text']}"
            if direction and direction.strip() and direction.strip() != "继续故事"
            else base["text"])
    return json.dumps({"text": text,
                       "character_prompts": base["character_prompts"],
                       "background_prompt": base["background_prompt"]},
                      ensure_ascii=False)


def generate_mock_image(task: ImageTask):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        log.warning("Pillow not installed, skipping mock image generation")
        return
    is_bg = task.image_type == "background"
    color = (
        (30 + task.seg_id * 20 % 80, 50 + task.seg_id * 15 % 80, 80 + task.seg_id * 25 % 80)
        if is_bg else
        (80 + task.seg_id * 20 % 80, 30 + task.seg_id * 15 % 80, 50 + task.seg_id * 25 % 80)
    )
    label = f"{'BG' if is_bg else 'CHAR'}: {task.item_id}\nSeg {task.seg_id}"
    img  = Image.new("RGB", (task.width, task.height), color=color)
    draw = ImageDraw.Draw(img)
    for y in range(0, task.height, 40):
        a = int(255 * (1 - y / task.height) * 0.3)
        draw.rectangle([0, y, task.width, y + 20],
                       fill=(min(255, color[0]+a), min(255, color[1]+a), min(255, color[2]+a)))
    try:
        font  = ImageFont.truetype("arial.ttf", 36)
        sfont = ImageFont.truetype("arial.ttf", 24)
    except Exception:
        font = sfont = ImageFont.load_default()
    draw.text((task.width//2, task.height//2 - 40), label,
              fill=(255,255,255), font=font, anchor="mm")
    draw.text((task.width//2, task.height//2 + 40), f"{task.width}x{task.height}",
              fill=(200,200,200), font=sfont, anchor="mm")
    time.sleep(0.5)
    img.save(GENERATED_DIR / task.filename)
    url = f"/static/generated/{task.filename}"
    _update_image_status(task.seg_id, task.image_type, task.item_id, "done", url)
    log.info(f"Mock image saved: {task.filename}")


def run_mock_image_queue(base_tasks, edit_tasks):
    all_tasks = base_tasks + edit_tasks
    for idx, task in enumerate(all_tasks):
        broadcast_status(f"Mock 生图：{task.item_id}，第 {idx+1}/{len(all_tasks)} 张...")
        generate_mock_image(task)

# ─── Core Generation Flow ────────────────────────────────────────────────────
def run_generation(direction, user_text="", uploaded_image_path: Optional[Path] = None):
    global _is_generating

    if not generation_lock.acquire(blocking=False):
        broadcast_status("当前正在生成，请等待完成。", "warning")
        return

    _is_generating = True
    has_upload = uploaded_image_path is not None

    try:
        segment_id = len(story_state["segments"])
        log.info(f"Generating segment {segment_id}, direction: {direction!r}, "
                 f"upload: {uploaded_image_path}")

        # ── Phase 1: LLM text ──────────────────────────────────────────────
        if _mock_mode:
            broadcast_status("Mock 模式：生成故事文本...")
            raw_json = generate_mock_segment(segment_id, direction)
        else:
            try:
                start_llama_server()
            except Exception as e:
                broadcast_status(f"llama-server 启动失败: {e}", "error")
                log_error(f"llama-server start failed: {e}")
                return
            try:
                broadcast_status("正在提交故事请求...")
                if has_upload:
                    raw_json = call_qwen_with_image(uploaded_image_path, direction, segment_id)
                else:
                    raw_json = call_qwen(direction, segment_id)
            except Exception as e:
                broadcast_status(f"Qwen 调用失败: {e}", "error")
                log_error(f"Qwen call failed: {e}")
                return
            finally:
                if not _creative_mode:
                    stop_llama_server()

        broadcast_status("正在解析故事 JSON...")
        segment = postprocess_segment(
            raw_json, segment_id,
            user_text=user_text,
            skip_character_generation=has_upload,
            uploaded_image_path=uploaded_image_path,
        )

        with state_lock:
            story_state["segments"].append(segment)
        save_story()
        broadcast_status("故事文本已生成！")

        # ── Phase 2: Image generation ──────────────────────────────────────
        if _creative_mode:
            broadcast_status("创作模式：图片已加入待生成队列，关闭创作模式后统一生成")
            return

        base_tasks, edit_tasks = build_image_queues(segment)
        log.info(f"Image queues: {len(base_tasks)} base ({BASE_IMAGE_ENGINE}), "
                 f"{len(edit_tasks)} Qwen Edit")

        if _mock_mode:
            run_mock_image_queue(base_tasks, edit_tasks)
        else:
            if base_tasks:
                engine_name = "ZImage" if BASE_IMAGE_ENGINE == "zimage" else "SDXL"
                broadcast_status(f"{engine_name} 队列：{len(base_tasks)} 张图...")
                run_base_queue(base_tasks)
            if edit_tasks:
                broadcast_status(f"Qwen Edit 队列：{len(edit_tasks)} 张图...")
                run_qwen_edit_queue(edit_tasks)

        broadcast_status(f"第 {segment_id+1} 段生成完成！", "success")
        prewarm_llama()

    except Exception as e:
        log.error(f"Generation error: {e}", exc_info=True)
        log_error(f"Generation error: {e}")
        broadcast_status(f"生成出错: {e}", "error")
    finally:
        _is_generating = False
        generation_lock.release()


def run_creative_mode_off():
    global _creative_mode, _is_batch_imaging

    broadcast_status("创作模式已关闭，正在关闭 Qwen 语言模型...")
    if not _mock_mode:
        stop_llama_server()

    def _worker():
        if _mock_mode:
            base_tasks, edit_tasks = collect_all_pending_image_queues()
            run_mock_image_queue(base_tasks, edit_tasks)
            broadcast_status("批量生图完成（Mock 模式）", "success")
        else:
            run_batch_imaging()
        prewarm_llama()

    threading.Thread(target=_worker, daemon=True, name="batch-imaging").start()

# ─── Mobile App Routes ───────────────────────────────────────────────────────
@mobile_app.route("/")
def mobile_index():
    return render_template("mobile.html")


@mobile_app.route("/api/story")
def mobile_api_story():
    return jsonify(get_story_snapshot())


@mobile_app.route("/api/generate", methods=["POST"])
def mobile_api_generate():
    if _serve_only:
        return jsonify({"ok": False, "message": "当前为只读模式，无法生成新内容。"})
    if _is_generating:
        return jsonify({"ok": False, "message": "当前正在生成，请等待完成。"})
    if _is_batch_imaging:
        return jsonify({"ok": False, "message": "当前正在批量生图，请等待完成。"})

    uploaded_image_path = None

    if request.content_type and "multipart/form-data" in request.content_type:
        direction  = (request.form.get("direction") or "继续故事").strip() or "继续故事"
        user_text  = direction
        image_file = request.files.get("image")
        if image_file and image_file.filename:
            ext = Path(image_file.filename).suffix.lower()
            if ext not in ALLOWED_IMAGE_EXTENSIONS:
                return jsonify({"ok": False, "message": f"不支持的图片格式：{ext}"})
            safe_name = f"upload_{int(time.time())}{ext}"
            save_path = UPLOADS_DIR / safe_name
            image_file.save(str(save_path))
            uploaded_image_path = save_path
            log.info(f"Uploaded image saved: {save_path}")
    else:
        data       = request.get_json() or {}
        direction  = (data.get("direction") or "继续故事").strip() or "继续故事"
        user_text  = direction

    threading.Thread(
        target=run_generation,
        args=(direction,),
        kwargs={"user_text": user_text, "uploaded_image_path": uploaded_image_path},
        daemon=True,
    ).start()
    return jsonify({"ok": True, "message": "开始生成..."})


@mobile_app.route("/api/creative-mode", methods=["POST"])
def mobile_api_creative_mode():
    global _creative_mode
    if _serve_only:
        return jsonify({"ok": False, "message": "只读模式下无法切换创作模式。"})
    if _is_generating:
        return jsonify({"ok": False, "message": "当前 LLM 正在生成，请等待本轮完成后再切换。"})
    if _is_batch_imaging:
        return jsonify({"ok": False, "message": "当前正在批量生图，请等待完成后再切换。"})

    data   = request.get_json() or {}
    enable = bool(data.get("enable", False))

    if enable:
        if not _mock_mode and not _llm_ready:
            return jsonify({"ok": False, "message": "Qwen 语言模型尚未就绪，请等待预热完成后再开启创作模式。"})
        _creative_mode = True
        broadcast_status("创作模式已开启：llama-server 保持运行，图片将在关闭创作模式后统一生成")
        return jsonify({"ok": True, "creative_mode": True})
    else:
        _creative_mode = False
        threading.Thread(target=run_creative_mode_off, daemon=True, name="creative-off").start()
        return jsonify({"ok": True, "creative_mode": False, "message": "创作模式已关闭，正在启动批量生图..."})


@mobile_app.route("/api/current-index", methods=["POST"])
def mobile_api_current_index():
    data = request.get_json() or {}
    idx  = int(data.get("current_index", 0))
    with state_lock:
        story_state["current_index"] = idx
    return jsonify({"ok": True})


@mobile_app.route("/api/status")
def mobile_api_status():
    return jsonify({
        "is_generating":    _is_generating,
        "is_batch_imaging": _is_batch_imaging,
        "creative_mode":    _creative_mode,
        "llm_ready":        _llm_ready,
        "mode":             story_state.get("mode", "generate"),
        "segment_count":    len(story_state["segments"]),
        "latest_status":    get_latest_status(),
    })

# ─── Desktop App Routes ──────────────────────────────────────────────────────
@desktop_app.route("/")
def desktop_index():
    return render_template("desktop.html")


@desktop_app.route("/api/story")
def desktop_api_story():
    return jsonify(get_story_snapshot())


@desktop_app.route("/api/status")
def desktop_api_status():
    return jsonify({
        "is_generating":    _is_generating,
        "is_batch_imaging": _is_batch_imaging,
        "creative_mode":    _creative_mode,
        "llm_ready":        _llm_ready,
        "mode":             story_state.get("mode", "generate"),
        "segment_count":    len(story_state["segments"]),
        "latest_status":    get_latest_status(),
    })

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    global _mock_mode, _serve_only

    parser = argparse.ArgumentParser(description="oh-my-comic WebUI")
    parser.add_argument("--mock",       action="store_true",
                        help="Mock mode: no LLM or SD, use placeholder content")
    parser.add_argument("--serve-only", action="store_true",
                        help="Serve existing story.json without generation")
    args = parser.parse_args()

    _mock_mode  = args.mock
    _serve_only = args.serve_only

    load_story()

    if _serve_only:
        story_state["mode"] = "serve-only"
    elif _mock_mode:
        story_state["mode"] = "mock"
    else:
        story_state["mode"] = "generate"

    validate_no_existing_llama()

    local_ip = get_local_ip()

    mode_label = {
        "sdxl_only":     "SDXL only",
        "sdxl_hybrid":   "SDXL + Qwen Edit",
        "zimage_hybrid": "ZImage + Qwen Edit",
    }.get(IMAGE_GENERATION_MODE, IMAGE_GENERATION_MODE)

    print("\n" + "=" * 60)
    print("  oh-my-comic WebUI")
    print("=" * 60)
    print(f"  Mode          : {story_state['mode']}")
    print(f"  Mobile        : http://{local_ip}:{MOBILE_PORT}  (手机访问)")
    print(f"  Desktop       : http://127.0.0.1:{DESKTOP_PORT}  (电脑访问)")
    print(f"  Segments      : {len(story_state['segments'])}")
    print(f"  LLM           : {OPENAI_BASE_URL}")
    print(f"  Prompt rating : {PROMPT_RATING}")
    print(f"  Image mode    : {mode_label}")
    if LLAMA_MMPROJ_MODEL:
        print(f"  Multimodal    : {LLAMA_MMPROJ_MODEL}")
    if _mock_mode:
        print("  [MOCK] No LLM or SD will be used")
    if _serve_only:
        print("  [SERVE-ONLY] Generation disabled")
    print("=" * 60 + "\n")

    desktop_server = make_server("0.0.0.0", DESKTOP_PORT, desktop_app)
    threading.Thread(target=desktop_server.serve_forever, daemon=True).start()
    log.info(f"Desktop server started on port {DESKTOP_PORT}")

    prewarm_llama()

    mobile_server = make_server("0.0.0.0", MOBILE_PORT, mobile_app)
    log.info(f"Mobile server started on port {MOBILE_PORT}")
    mobile_server.serve_forever()


if __name__ == "__main__":
    main()
