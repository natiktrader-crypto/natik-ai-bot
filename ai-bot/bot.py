import os
import json
import time
import logging
import threading
import concurrent.futures
import urllib.parse
from datetime import datetime, timedelta, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler

import telebot
import httpx
import anthropic
from gtts import gTTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ["AI_BOT_TOKEN"]
AZT = timezone(timedelta(hours=4))

bot = telebot.TeleBot(TELEGRAM_TOKEN, num_threads=16)

PREFS_FILE = os.path.join(os.path.dirname(__file__), "ai_prefs.json")
prefs_lock = threading.Lock()

SYSTEM_PROMPT = (
    "Ты умный многофункциональный AI-ассистент. "
    "Отвечай чётко, грамотно, по делу. "
    "Если в контексте есть результаты веб-поиска — используй их для актуального ответа. "
    "Структурируй длинные ответы (списки, заголовки). "
    "Всегда отвечай на языке вопроса пользователя."
)


def _load_prefs():
    try:
        with open(PREFS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

_prefs: dict = _load_prefs()

def _save_prefs():
    try:
        with open(PREFS_FILE, "w", encoding="utf-8") as f:
            json.dump(_prefs, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning(f"[prefs] {e}")

def get_active(uid):
    return _prefs.get(str(uid), {}).get("active", list(PROVIDERS.keys()))

def set_active(uid, active_list):
    with prefs_lock:
        uid = str(uid)
        if uid not in _prefs:
            _prefs[uid] = {}
        _prefs[uid]["active"] = active_list
        _save_prefs()

def get_history(uid):
    return _prefs.get(str(uid), {}).get("history", [])

def set_history(uid, history):
    with prefs_lock:
        uid = str(uid)
        if uid not in _prefs:
            _prefs[uid] = {}
        _prefs[uid]["history"] = history[-20:]
        _save_prefs()


PROVIDERS = {
    "pollinations_gpt": {
        "name": "GPT-4o",
        "icon": "🌸",
        "env": None,
        "no_key": True,
        "url": "https://text.pollinations.ai/openai",
        "model": "openai",
        "vision": True,
        "free_note": "бесплатно, без ключа",
    },
    "pollinations_mistral": {
        "name": "Mistral Large",
        "icon": "🌺",
        "env": None,
        "no_key": True,
        "url": "https://text.pollinations.ai/openai",
        "model": "mistral-large",
        "vision": False,
        "free_note": "бесплатно, без ключа",
    },
    "pollinations_llama": {
        "name": "Llama 3.3 70B",
        "icon": "🦙",
        "env": None,
        "no_key": True,
        "url": "https://text.pollinations.ai/openai",
        "model": "llama",
        "vision": False,
        "free_note": "бесплатно, без ключа",
    },
    "pollinations_deepseek": {
        "name": "DeepSeek R1",
        "icon": "🐋",
        "env": None,
        "no_key": True,
        "url": "https://text.pollinations.ai/openai",
        "model": "deepseek-reasoning",
        "vision": False,
        "free_note": "бесплатно, без ключа",
    },
    "claude": {
        "name": "Claude Sonnet 4.5",
        "icon": "🟤",
        "env": "ANTHROPIC_API_KEY",
        "url": None,
        "model": "claude-sonnet-4-5",
        "vision": True,
        "free_note": "платный, $3/1M tokens",
    },
    "gemini": {
        "name": "Gemini 2.0 Flash",
        "icon": "🔵",
        "env": "GEMINI_API_KEY",
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "model": "gemini-2.0-flash",
        "vision": True,
        "free_note": "1500 req/day бесплатно",
    },
    "groq": {
        "name": "Groq Llama 70B",
        "icon": "🟡",
        "env": "GROQ_API_KEY",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama-3.3-70b-versatile",
        "vision": False,
        "free_note": "30 req/min бесплатно",
    },
    "deepseek": {
        "name": "DeepSeek Chat",
        "icon": "🟢",
        "env": "DEEPSEEK_API_KEY",
        "url": "https://api.deepseek.com/chat/completions",
        "model": "deepseek-chat",
        "vision": False,
        "free_note": "$0.27/1M tokens",
    },
    "openrouter": {
        "name": "OpenRouter Free",
        "icon": "🟣",
        "env": "OPENROUTER_API_KEY",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "vision": False,
        "free_note": "free tier",
    },
    "mistral": {
        "name": "Mistral Small",
        "icon": "🟠",
        "env": "MISTRAL_API_KEY",
        "url": "https://api.mistral.ai/v1/chat/completions",
        "model": "mistral-small-latest",
        "vision": False,
        "free_note": "free experiment plan",
    },
    "cohere": {
        "name": "Cohere Command R+",
        "icon": "🔴",
        "env": "COHERE_API_KEY",
        "url": "https://api.cohere.ai/compatibility/v1/chat/completions",
        "model": "command-r-plus",
        "vision": False,
        "free_note": "free trial",
    },
    "cerebras": {
        "name": "Cerebras Llama 70B",
        "icon": "⚡",
        "env": "CEREBRAS_API_KEY",
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "model": "llama3.1-70b",
        "vision": False,
        "free_note": "free tier",
    },
    "github": {
        "name": "GitHub GPT-4o",
        "icon": "⬛",
        "env": "GITHUB_MODELS_TOKEN",
        "url": "https://models.github.ai/inference/chat/completions",
        "model": "openai/gpt-4o",
        "vision": False,
        "free_note": "free with GitHub account",
    },
    "chatanywhere": {
        "name": "ChatAnywhere GPT-4o-mini",
        "icon": "🆓",
        "env": "CHATANYWHERE_API_KEY",
        "url": "https://api.chatanywhere.tech/v1/chat/completions",
        "model": "gpt-4o-mini",
        "vision": False,
        "free_note": "бесплатно",
    },
}


# ─── WEB SEARCH (DuckDuckGo) ─────────────────────────────────────────────────

def _web_search(query: str, max_results: int = 4) -> str:
    try:
        q = urllib.parse.quote_plus(query)
        r = httpx.get(
            f"https://api.duckduckgo.com/?q={q}&format=json&no_html=1&skip_disambig=1",
            timeout=8,
            follow_redirects=True,
        )
        data = r.json()
        parts = []

        if data.get("AbstractText"):
            parts.append(f"📖 {data['AbstractText']}")

        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                parts.append(f"• {topic['Text'][:200]}")

        if not parts:
            return ""
        return "\n".join(parts)
    except Exception as e:
        log.debug(f"[search] {e}")
        return ""


def _needs_search(question: str) -> bool:
    keywords = [
        "сейчас", "сегодня", "курс", "цена", "погода", "новости", "последн",
        "текущ", "2024", "2025", "2026", "кто такой", "что такое", "когда",
        "где", "как", "latest", "news", "price", "weather", "today", "now",
        "current", "rate", "stock", "крипто", "биткоин", "bitcoin", "eth",
    ]
    q = question.lower()
    return any(k in q for k in keywords)


# ─── TELEGRAM FILE URL ───────────────────────────────────────────────────────

def _get_file_url(file_id: str) -> str | None:
    try:
        fi = bot.get_file(file_id)
        return f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{fi.file_path}"
    except Exception as e:
        log.warning(f"[file_url] {e}")
        return None


def _download_bytes(url: str) -> bytes | None:
    try:
        r = httpx.get(url, timeout=20, follow_redirects=True)
        return r.content
    except Exception as e:
        log.warning(f"[download] {e}")
        return None


# ─── VOICE TRANSCRIPTION (Whisper via OpenRouter) ────────────────────────────

def _transcript_audio(file_id: str) -> str | None:
    """Загружает голосовой файл и отправляет в OpenRouter Whisper для распознавания текста"""
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        log.warning("[whisper] Нет OPENROUTER_API_KEY для транскрипции голосового.")
        return None

    file_url = _get_file_url(file_id)
    if not file_url:
        return None

    audio_bytes = _download_bytes(file_url)
    if not audio_bytes:
        return None

    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        files = {"file": ("voice.ogg", audio_bytes, "audio/ogg")}
        data = {"model": "openai/whisper"}
        
        r = httpx.post("https://openrouter.ai/api/v1/audio/transcriptions", 
                       headers=headers, files=files, data=data, timeout=30)
        
        if r.status_code == 200:
            return r.json().get("text", "").strip()
        else:
            log.warning(f"[whisper] Ошибка HTTP {r.status_code}: {r.text}")
            return None
    except Exception as e:
        log.warning(f"[whisper] Ошибка при транскрипции: {e}")
        return None


# ─── AI PROVIDERS ────────────────────────────────────────────────────────────

def _call_claude(messages: list, timeout: int = 15) -> tuple[str, str | None]:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return "claude", None
    try:
        client = anthropic.Anthropic(api_key=api_key)
        system_msg = SYSTEM_PROMPT
        chat_msgs = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                chat_msgs.append({"role": m["role"], "content": m["content"]})
        response = client.messages.create(
            model=PROVIDERS["claude"]["model"],
            max_tokens=2000,
            system=system_msg,
            messages=chat_msgs,
        )
        return "claude", response.content[0].text.strip()
    except Exception as e:
        log.warning(f"[claude] {e}")
        return "claude", None


def _call_provider(key: str, messages: list, image_url: str | None = None, timeout: int = 15) -> tuple[str, str | None, str | None]:
    if key == "claude":
        k, text = _call_claude(messages, timeout)
        return k, text, (None if text else "нет ключа или ошибка Claude API")

    p = PROVIDERS[key]

    if not p.get("no_key"):
        api_key = os.environ.get(p.get("env", ""), "").strip()
        if not api_key:
            return key, None, "нет ключа в переменных окружения"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if key == "github":
            headers["Accept"] = "application/json"
    else:
        headers = {"Content-Type": "application/json"}

    final_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in messages:
        if m["role"] == "system":
            continue
        final_messages.append(m)

    if image_url and p.get("vision") and final_messages:
        for i in range(len(final_messages) - 1, -1, -1):
            if final_messages[i]["role"] == "user":
                text_content = final_messages[i]["content"]
                if isinstance(text_content, str):
                    final_messages[i]["content"] = [
                        {"type": "text", "text": text_content},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ]
                break

    body = {
        "model": p["model"],
        "messages": final_messages,
        "max_tokens": 2000,
    }
    if p.get("no_key"):
        body["private"] = True
    else:
        body["temperature"] = 0.7

    try:
        r = httpx.post(p["url"], headers=headers, json=body, timeout=timeout)
        if r.status_code not in (200, 201):
            err = f"HTTP {r.status_code}: {r.text[:120]}"
            log.warning(f"[{key}] {err}")
            return key, None, err
        return key, r.json()["choices"][0]["message"]["content"].strip(), None
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:120]}"
        log.warning(f"[{key}] {err}")
        return key, None, err


def _run_parallel(active_keys: list, messages: list, image_url: str | None = None) -> dict[str, tuple[str | None, str | None]]:
    results = {}
    for k in active_keys:
        try:
            k, text, err = _call_provider(k, messages, image_url, timeout=10)
            results[k] = (text, err)
        except Exception as e:
            results[k] = (None, f"Ошибка: {str(e)[:50]}")
    return results


def _fmt_response(key: str, text: str | None, err: str | None = None) -> str:
    p = PROVIDERS[key]
    if text is None:
        reason = f"\n`{err}`" if err else ""
        return f"{p['icon']} *{p['name']}*\n_❌ Нет ответа_{reason}"
    lines = text.split("\n")
    preview = "\n".join(lines[:50])
    if len(lines) > 50:
        preview += "\n_...ответ обрезан..._"
    return f"{p['icon']} *{p['name']}*\n{preview}"


def _is_available(k: str) -> bool:
    p = PROVIDERS[k]
    if p.get("no_key"):
        return True
    env = p.get("env")
    return bool(env and os.environ.get(env, "").strip())


# ─── COMMANDS ────────────────────────────────────────────────────────────────

@bot.message_handler(commands=["start", "help"])
def cmd_help(message):
    uid = message.from_user.id
    active = get_active(uid)
    available_count = sum(1 for k in active if k in PROVIDERS and _is_available(k))
    text = (
        "🤖 *AI-Ассистент — все модели сразу*\n\n"
        "Отправь *текст*, *фото*, *документ* или *голосовое* — отвечают несколько AI.\n\n"
        "🔍 *Автопоиск*: актуальные вопросы сопровождаются поиском в интернете\n"
        "👁 *Анализ фото*: GPT-4o и Claude видят и описывают изображения\n\n"
        "*Команды:*\n"
        "/models — выбрать AI-провайдеры\n"
        "/search [запрос] — поиск в интернете\n"
        "/clear — очистить историю\n"
        "/keys — бесплатные ключи\n"
        "/status — статус провайдеров\n\n"
        f"✅ *Сейчас активны: {available_count} из {len(PROVIDERS)} AI*"
    )
    bot.reply_to(message, text, parse_mode="Markdown")


@bot.message_handler(commands=["search"])
def cmd_search(message):
    query = message.text.replace("/search", "").strip()
    if not query:
        bot.reply_to(message, "Напиши что искать: `/search курс биткоина`", parse_mode="Markdown")
        return
    msg = bot.reply_to(message, f"🔍 Ищу: *{query}*...", parse_mode="Markdown")
    results = _web_search(query, max_results=6)
    if results:
        try:
            bot.edit_message_text(
                f"🔍 *Результаты поиска по «{query}»:*\n\n{results}",
                chat_id=msg.chat.id,
                message_id=msg.message_id,
                parse_mode="Markdown",
            )
        except Exception:
            bot.send_message(message.chat.id, f"🔍 *{query}*\n\n{results}", parse_mode="Markdown")
    else:
        bot.edit_message_text(
            f"🔍 По запросу «{query}» ничего не найдено.\nПопробуй переформулировать.",
            chat_id=msg.chat.id,
            message_id=msg.message_id,
        )


@bot.message_handler(commands=["keys"])
def cmd_keys(message):
    urls = {
        "gemini":       ("aistudio.google.com/apikey", "1500 req/day — БЕСПЛАТНО"),
        "groq":         ("console.groq.com/keys", "30 req/min — БЕСПЛАТНО"),
        "mistral":      ("console.mistral.ai/api-keys", "free experiment plan"),
        "cohere":       ("dashboard.cohere.com/api-keys", "free trial"),
        "cerebras":     ("cloud.cerebras.ai", "free tier"),
        "openrouter":   ("openrouter.ai/keys", "free models"),
        "github":       ("github.com/settings/tokens", "free with GitHub"),
        "chatanywhere": ("github.com/chatanywhere/GPT_API_free", "бесплатно"),
        "deepseek":     ("platform.deepseek.com/api_keys", "$0.27/1M tokens"),
        "claude":       ("console.anthropic.com/settings/keys", "платный"),
    }
    lines = ["🔑 *Где получить бесплатные ключи:*\n"]
    lines.append("*Без ключа (работают сразу):*")
    for key, p in PROVIDERS.items():
        if p.get("no_key"):
            lines.append(f"  {p['icon']} {p['name']} — ✅ работает без ключа")
    lines.append("\n*С бесплатным ключом:*")
    for key, p in PROVIDERS.items():
        if not p.get("no_key") and key in urls:
            url, note = urls[key]
            lines.append(f"  {p['icon']} *{p['name']}* — {note}\n    `{p['env']}`  🔗 {url}\n")
    lines.append("_Ключи добавляй в Render → Environment Variables!_")
    bot.reply_to(message, "\n".join(lines), parse_mode="Markdown", disable_web_page_preview=True)


@bot.message_handler(commands=["status"])
def cmd_status(message):
    lines = ["🔍 *Статус AI провайдеров:*\n"]
    ok, fail = [], []
    for key, p in PROVIDERS.items():
        if _is_available(key):
            ok.append(f"  {p['icon']} {p['name']}: ✅")
        else:
            fail.append(f"  {p['icon']} {p['name']}: ❌ нет ключа (`{p['env']}`)")
    lines += ok + ([""] if fail else []) + fail
    lines.append(f"\n✅ Готово к работе: *{len(ok)}* из *{len(PROVIDERS)}*")
    bot.reply_to(message, "\n".join(lines), parse_mode="Markdown")


@bot.message_handler(commands=["clear"])
def cmd_clear(message):
    set_history(message.from_user.id, [])
    bot.reply_to(message, "🗑 История очищена.", parse_mode="Markdown")


@bot.message_handler(commands=["models"])
def cmd_models(message):
    uid = message.from_user.id
    active = get_active(uid)
    kb = telebot.types.InlineKeyboardMarkup(row_width=1)
    for key, p in PROVIDERS.items():
        is_active = key in active
        has_key = _is_available(key)
        key_status = "✅" if has_key else "🔑"
        check = "☑️" if is_active else "⬜"
        label = f"{check} {p['icon']} {p['name']} {key_status}"
        kb.add(telebot.types.InlineKeyboardButton(label, callback_data=f"toggle:{key}"))
    kb.add(telebot.types.InlineKeyboardButton("✅ Все", callback_data="toggle:_all"))
    kb.add(telebot.types.InlineKeyboardButton("❌ Сбросить", callback_data="toggle:_none"))
    bot.reply_to(message, "⚙️ *Выбери AI-провайдеры:*\n✅=активен  🔑=нужен ключ",
                 reply_markup=kb, parse_mode="Markdown")


@bot.callback_query_handler(func=lambda c: c.data.startswith("toggle:"))
def cb_toggle(call):
    uid = call.from_user.id
    active = list(get_active(uid))
    key = call.data.split(":", 1)[1]
    if key == "_all":
        active = list(PROVIDERS.keys())
    elif key == "_none":
        active = []
    elif key in active:
        active.remove(key)
    else:
        active.append(key)
    set_active(uid, active)
    kb = telebot.types.InlineKeyboardMarkup(row_width=1)
    for k, p in PROVIDERS.items():
        is_active = k in active
        key_status = "✅" if _is_available(k) else "🔑"
        check = "☑️" if is_active else "⬜"
        kb.add(telebot.types.InlineKeyboardButton(
            f"{check} {p['icon']} {p['name']} {key_status}", callback_data=f"toggle:{k}"))
    kb.add(telebot.types.InlineKeyboardButton("✅ Все", callback_data="toggle:_all"))
    kb.add(telebot.types.InlineKeyboardButton("❌ Сбросить", callback_data="toggle:_none"))
    try:
        bot.edit_message_reply_markup(call.message.chat.id, call.message.message_id, reply_markup=kb)
    except Exception:
        pass
    bot.answer_callback_query(call.id)


# ─── CORE QUESTION HANDLER ───────────────────────────────────────────────────

def _ask_all(message, question: str, image_url: str | None = None, extra_context: str = "", is_voice_input: bool = False):
    uid = message.from_user.id
    active = get_active(uid)
    available = [k for k in active if k in PROVIDERS and _is_available(k)]

    if not available:
        bot.reply_to(message,
            "❌ Нет активных AI.\n/keys — бесплатные ключи\n/status — статус",
            parse_mode="Markdown")
        return

    history = get_history(uid)

    search_results = ""
    if not image_url and _needs_search(question):
        search_results = _web_search(question)

    context_parts = []
    if extra_context:
        context_parts.append(extra_context)
    if search_results:
        context_parts.append(f"[Веб-поиск по запросу]:\n{search_results}")

    user_content = question
    if context_parts:
        user_content = "\n\n".join(context_parts) + "\n\n[Вопрос пользователя]: " + question

    history.append({"role": "user", "content": user_content})
    messages = history[-10:]

    count = len(available)
    search_note = " + 🔍 веб-поиск" if search_results else ""
    names = " | ".join(PROVIDERS[k]["icon"] for k in available)
    waiting_msg = bot.reply_to(
        message,
        f"⏳ Спрашиваю *{count} AI* по очереди{search_note}...\n{names}",
        parse_mode="Markdown",
    )

    start = time.time()
    results = _run_parallel(available, messages, image_url)
    elapsed = round(time.time() - start, 1)

    try:
        bot.delete_message(message.chat.id, waiting_msg.message_id)
    except Exception:
        pass

    if search_results:
        try:
            bot.send_message(
                message.chat.id,
                f"🔍 *Найдено в интернете:*\n{search_results[:800]}",
                parse_mode="Markdown",
                reply_to_message_id=message.message_id,
            )
        except Exception:
            pass

    first_successful_text = None
    assistant_parts = []
    
    for key in available:
        text, err = results.get(key, (None, "нет результата"))
        chunk = _fmt_response(key, text, err)
        try:
            bot.send_message(message.chat.id, chunk, parse_mode="Markdown",
                             reply_to_message_id=message.message_id)
        except Exception:
            try:
                bot.send_message(message.chat.id, chunk.replace("*", "").replace("_", "").replace("`", ""))
            except Exception as e:
                log.warning(f"[send] {e}")
        if text:
            assistant_parts.append(f"[{PROVIDERS[key]['name']}]: {text}")
            if not first_successful_text:
                first_successful_text = text

    if assistant_parts:
        history.append({"role": "assistant", "content": "\n---\n".join(assistant_parts)})
        set_history(uid, history)

    # Озвучка ответа, если пользователь отправил голосовое
    if is_voice_input and first_successful_text:
        try:
            # Детектируем примерный язык для озвучки (по умолчанию русский 'ru', если есть латиница - можно расширить)
            tts = gTTS(text=first_successful_text[:500], lang='ru', slow=False)
            filename = f"reply_{message.message_id}.mp3"
            tts.save(filename)
            with open(filename, 'rb') as audio:
                bot.send_voice(message.chat.id, audio, reply_to_message_id=message.message_id)
            os.remove(filename)
        except Exception as voice_err:
            log.warning(f"[TTS Error] {voice_err}")

    answered = sum(1 for k in available if results.get(k, (None, None))[0])
    try:
        bot.send_message(
            message.chat.id,
            f"✅ _Ответили {answered}/{count} AI за {elapsed} сек_ | /clear /models",
            parse_mode="Markdown",
        )
    except Exception:
        pass


# ─── TEXT MESSAGES ───────────────────────────────────────────────────────────

@bot.message_handler(func=lambda m: True, content_types=["text"])
def handle_text(message):
    if message.text.startswith("/"):
        return
    _ask_all(message, message.text.strip())


# ─── VOICE / AUDIO HANDLER ───────────────────────────────────────────────────

@bot.message_handler(content_types=["voice", "audio"])
def handle_voice(message):
    file_id = message.voice.file_id if message.voice else message.audio.file_id
    
    # Отправляем статус, что бот записывает/обрабатывает голосовое
    status_msg = bot.reply_to(message, "🎙 _Слушаю ваше сообщение..._", parse_mode="Markdown")
    
    text_question = _transcript_audio(file_id)
    
    try:
        bot.delete_message(message.chat.id, status_msg.message_id)
    except Exception:
        pass

    if not text_question:
        bot.reply_to(message, "❌ Не удалось распознать речь. Пожалуйста, говорите четче или проверьте ключ `OPENROUTER_API_KEY`.")
        return

    # Отправляем пользователю текст, который мы распознали из его голоса
    bot.reply_to(message, f"🗣 *Вы сказали:* _{text_question}_", parse_mode="Markdown")
    
    # Запускаем основной цикл генерации ответа
    _ask_all(message, text_question, is_voice_input=True)


# ─── PHOTO / IMAGE ───────────────────────────────────────────────────────────

@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    caption = (message.caption or "").strip() or "Опиши это изображение подробно. Что на нём изображено?"
    file_id = message.photo[-1].file_id
    image_url = _get_file_url(file_id)

    if not image_url:
        bot.reply_to(message, "❌ Не удалось получить фото. Попробуй ещё раз.")
        return

    _ask_all(message, caption, image_url=image_url)


# ─── DOCUMENT ────────────────────────────────────────────────────────────────

@bot.message_handler(content_types=["document"])
def handle_document(message):
    doc = message.document
    caption = (message.caption or "").strip()

    if doc.mime_type and doc.mime_type.startswith("image/"):
        image_url = _get_file_url(doc.file_id)
        question = caption or "Опиши это изображение подробно."
        _ask_all(message, question, image_url=image_url)
    else:
        question = caption or f"Пользователь отправил файл «{doc.file_name}». Что ты можешь сказать о таком типе файла?"
        _ask_all(message, question)


# ─── STICKER ─────────────────────────────────────────────────────────────────

@bot.message_handler(content_types=["sticker"])
def handle_sticker(message):
    emoji = message.sticker.emoji or "🙂"
    _ask_all(message, f"Пользователь отправил стикер с эмодзи {emoji}. Отреагируй остроумно и коротко.")


# ─── HEALTHCHECK ─────────────────────────────────────────────────────────────

START_TIME = datetime.now(AZT)

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        uptime = datetime.now(AZT) - START_TIME
        h, rem = divmod(int(uptime.total_seconds()), 3600)
        m, s = divmod(rem, 60)
        providers_ok = sum(1 for k in PROVIDERS if _is_available(k))
        body = (
            f"AI Bot OK | Uptime: {h}h {m}m {s}s | "
            f"Providers: {providers_ok}/{len(PROVIDERS)}\n"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class _ReuseHTTPServer(HTTPServer):
    allow_reuse_address = True


def _run_health():
    port = int(os.environ.get("AI_BOT_PORT", 8001))
    _ReuseHTTPServer(("0.0.0.0", port), HealthHandler).serve_forever()


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("AI_BOT_PORT", 8001))
    threading.Thread(target=_run_health, daemon=True).start()
    log.info(f"[AI-Bot] Запущен | healthcheck: порт {port} | провайдеров: {len(PROVIDERS)}")

    try:
        bot.delete_webhook(drop_pending_updates=True)
        log.info("[AI-Bot] Вебхук сброшен")
    except Exception as e:
        log.warning(f"[AI-Bot] delete_webhook: {e}")
    time.sleep(1)

    bot.infinity_polling(
        timeout=30,
        long_polling_timeout=25,
        logger_level=logging.WARNING,
        restart_on_change=False,
    )
