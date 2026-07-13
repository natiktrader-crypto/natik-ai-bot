import os
import json
import time
import logging
import threading
import asyncio
import concurrent.futures
from datetime import datetime, timedelta, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler

import telebot
import httpx
import anthropic

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ["AI_BOT_TOKEN"]
AZT = timezone(timedelta(hours=4))

bot = telebot.TeleBot(TELEGRAM_TOKEN, num_threads=12)

PREFS_FILE = os.path.join(os.path.dirname(__file__), "ai_prefs.json")
prefs_lock = threading.Lock()

def _load_prefs():
    try:
        with open(PREFS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

_prefs: dict = _load_prefs()

def get_active(uid):
    return _prefs.get(str(uid), {}).get("active", list(PROVIDERS.keys()))

def set_active(uid, active_list):
    with prefs_lock:
        uid = str(uid)
        if uid not in _prefs:
            _prefs[uid] = {}
        _prefs[uid]["active"] = active_list
        try:
            with open(PREFS_FILE, "w", encoding="utf-8") as f:
                json.dump(_prefs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.warning(f"[prefs] {e}")

def get_history(uid):
    return _prefs.get(str(uid), {}).get("history", [])

def set_history(uid, history):
    with prefs_lock:
        uid = str(uid)
        if uid not in _prefs:
            _prefs[uid] = {}
        _prefs[uid]["history"] = history[-20:]
        try:
            with open(PREFS_FILE, "w", encoding="utf-8") as f:
                json.dump(_prefs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.warning(f"[prefs] {e}")


PROVIDERS = {
    "pollinations_gpt": {
        "name": "GPT-4o (бесплатно)",
        "icon": "🌸",
        "env":  None,
        "no_key": True,
        "url":  "https://text.pollinations.ai/openai",
        "model": "openai",
        "free": True,
        "free_note": "бесплатно, без ключа",
    },
    "pollinations_mistral": {
        "name": "Mistral (бесплатно)",
        "icon": "🌺",
        "env":  None,
        "no_key": True,
        "url":  "https://text.pollinations.ai/openai",
        "model": "mistral",
        "free": True,
        "free_note": "бесплатно, без ключа",
    },
    "claude": {
        "name": "Claude Sonnet 4.5",
        "icon": "🟤",
        "env":  "ANTHROPIC_API_KEY",
        "url":  None,
        "model": "claude-sonnet-4-5",
        "free": False,
        "free_note": "платный, $3/1M tokens",
    },
    "gemini": {
        "name": "Gemini 2.0 Flash",
        "icon": "🔵",
        "env":  "GEMINI_API_KEY",
        "url":  "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "model": "gemini-2.0-flash",
        "free": True,
        "free_note": "1500 req/day",
    },
    "groq": {
        "name": "Groq (Llama 3.3 70B)",
        "icon": "🟡",
        "env":  "GROQ_API_KEY",
        "url":  "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama-3.3-70b-versatile",
        "free": True,
        "free_note": "30 req/min",
    },
    "deepseek": {
        "name": "DeepSeek Chat",
        "icon": "🟢",
        "env":  "DEEPSEEK_API_KEY",
        "url":  "https://api.deepseek.com/chat/completions",
        "model": "deepseek-chat",
        "free": False,
        "free_note": "$0.27/1M tokens",
    },
    "openrouter": {
        "name": "OpenRouter (Free)",
        "icon": "🟣",
        "env":  "OPENROUTER_API_KEY",
        "url":  "https://openrouter.ai/api/v1/chat/completions",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "free": True,
        "free_note": "free tier",
    },
    "mistral": {
        "name": "Mistral Small",
        "icon": "🟠",
        "env":  "MISTRAL_API_KEY",
        "url":  "https://api.mistral.ai/v1/chat/completions",
        "model": "mistral-small-latest",
        "free": True,
        "free_note": "free experiment plan",
    },
    "cohere": {
        "name": "Cohere Command R",
        "icon": "🔴",
        "env":  "COHERE_API_KEY",
        "url":  "https://api.cohere.ai/compatibility/v1/chat/completions",
        "model": "command-r-plus",
        "free": True,
        "free_note": "free trial",
    },
    "cerebras": {
        "name": "Cerebras (Llama 3.1 70B)",
        "icon": "⚡",
        "env":  "CEREBRAS_API_KEY",
        "url":  "https://api.cerebras.ai/v1/chat/completions",
        "model": "llama3.1-70b",
        "free": True,
        "free_note": "free tier",
    },
    "github": {
        "name": "GitHub Models (GPT-4o)",
        "icon": "⬛",
        "env":  "GITHUB_MODELS_TOKEN",
        "url":  "https://models.github.ai/inference/chat/completions",
        "model": "openai/gpt-4o",
        "free": True,
        "free_note": "free with GitHub account",
    },
    "chatanywhere": {
        "name": "ChatAnywhere (GPT-4o-mini)",
        "icon": "🆓",
        "env":  "CHATANYWHERE_API_KEY",
        "url":  "https://api.chatanywhere.tech/v1/chat/completions",
        "model": "gpt-4o-mini",
        "free": True,
        "free_note": "бесплатно, github.com/chatanywhere/GPT_API_free",
    },
}


def _call_claude(messages: list, timeout: int = 30) -> tuple[str, str | None]:
    """Call Claude via Anthropic SDK."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return "claude", None
    try:
        client = anthropic.Anthropic(api_key=api_key)
        # Separate system messages if present
        system_msg = None
        chat_msgs = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                chat_msgs.append({"role": m["role"], "content": m["content"]})
        kwargs = {
            "model": PROVIDERS["claude"]["model"],
            "max_tokens": 1500,
            "messages": chat_msgs,
        }
        if system_msg:
            kwargs["system"] = system_msg
        response = client.messages.create(**kwargs)
        text = response.content[0].text.strip()
        return "claude", text
    except Exception as e:
        log.warning(f"[claude] Error: {e}")
        return "claude", None


def _call_provider(key: str, messages: list, timeout: int = 30) -> tuple[str, str | None]:
    """Call one provider. Returns (key, reply_text) or (key, None) on error."""
    if key == "claude":
        return _call_claude(messages, timeout)

    p = PROVIDERS[key]

    # Providers that need a key
    if not p.get("no_key"):
        api_key = os.environ.get(p["env"], "").strip()
        if not api_key:
            return key, None
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if key == "github":
            headers["Accept"] = "application/json"
    else:
        headers = {"Content-Type": "application/json"}

    body = {
        "model": p["model"],
        "messages": messages,
        "max_tokens": 1500,
    }
    # pollinations: add private flag to avoid caching
    if p.get("no_key"):
        body["private"] = True

    if not p.get("no_key"):
        body["temperature"] = 0.7

    try:
        r = httpx.post(p["url"], headers=headers, json=body, timeout=timeout)
        if r.status_code not in (200, 201):
            log.warning(f"[{key}] HTTP {r.status_code}: {r.text[:200]}")
            return key, None
        data = r.json()
        text = data["choices"][0]["message"]["content"].strip()
        return key, text
    except Exception as e:
        log.warning(f"[{key}] Error: {e}")
        return key, None


def _run_parallel(active_keys: list, messages: list) -> dict[str, str | None]:
    """Run all active providers in parallel, return dict key→reply."""
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(active_keys)) as ex:
        futures = {ex.submit(_call_provider, k, messages): k for k in active_keys}
        for fut in concurrent.futures.as_completed(futures, timeout=35):
            k, text = fut.result()
            results[k] = text
    return results


def _fmt_response(key: str, text: str | None) -> str:
    p = PROVIDERS[key]
    icon = p["icon"]
    name = p["name"]
    if text is None:
        return f"{icon} *{name}*\n_❌ Нет ответа (нет ключа или ошибка)_"
    lines = text.split("\n")
    preview = "\n".join(lines[:40])
    if len(lines) > 40:
        preview += "\n_...обрезано..._"
    return f"{icon} *{name}*\n{preview}"


@bot.message_handler(commands=["start", "help"])
def cmd_help(message):
    uid = message.from_user.id
    active = get_active(uid)
    active_names = ", ".join(PROVIDERS[k]["icon"] + PROVIDERS[k]["name"] for k in active if k in PROVIDERS)
    text = (
        "🤖 *AI-бот — все модели сразу*\n\n"
        "Просто напиши любой вопрос — получишь ответы от нескольких AI одновременно.\n\n"
        "*Команды:*\n"
        "/models — выбрать AI-провайдеры\n"
        "/clear — очистить историю чата\n"
        "/keys — как получить бесплатные ключи\n"
        "/status — проверить какие ключи активны\n\n"
        f"*Сейчас активны:* {active_names or 'все'}\n\n"
        "💡 _Вопрос задаётся всем AI параллельно — ответы приходят по мере готовности._"
    )
    bot.reply_to(message, text, parse_mode="Markdown")


@bot.message_handler(commands=["keys"])
def cmd_keys(message):
    lines = ["🔑 *AI провайдеры — где получить ключи:*\n"]
    urls = {
        "claude":       "console.anthropic.com/settings/keys",
        "gemini":       "aistudio.google.com/apikey",
        "groq":         "console.groq.com/keys",
        "deepseek":     "platform.deepseek.com/api_keys",
        "openrouter":   "openrouter.ai/keys",
        "mistral":      "console.mistral.ai/api-keys",
        "cohere":       "dashboard.cohere.com/api-keys",
        "cerebras":     "cloud.cerebras.ai",
        "github":       "github.com/settings/tokens",
        "chatanywhere": "github.com/chatanywhere/GPT_API_free",
    }
    for key, p in PROVIDERS.items():
        note = p.get("free_note", "")
        if p.get("no_key"):
            lines.append(f"{p['icon']} *{p['name']}* — ✅ работает без ключа\n")
        else:
            env = p.get("env","")
            url = urls.get(key, "")
            lines.append(f"{p['icon']} *{p['name']}* — {note}\n`{env}`\n🔗 {url}\n")
    lines.append("_Ключи добавляй через Replit → Secrets (не в чат!)_")
    bot.reply_to(message, "\n".join(lines), parse_mode="Markdown", disable_web_page_preview=True)


@bot.message_handler(commands=["status"])
def cmd_status(message):
    lines = ["🔍 *Статус AI провайдеров:*\n"]
    for key, p in PROVIDERS.items():
        if p.get("no_key"):
            lines.append(f"{p['icon']} {p['name']}: ✅ работает (без ключа)")
        else:
            val = os.environ.get(p.get("env",""), "").strip()
            if val:
                lines.append(f"{p['icon']} {p['name']}: ✅ ключ есть")
            else:
                lines.append(f"{p['icon']} {p['name']}: ❌ нет ключа (`{p['env']}`)")
    bot.reply_to(message, "\n".join(lines), parse_mode="Markdown")


@bot.message_handler(commands=["clear"])
def cmd_clear(message):
    uid = message.from_user.id
    set_history(uid, [])
    bot.reply_to(message, "🗑 История чата очищена.", parse_mode="Markdown")


@bot.message_handler(commands=["models"])
def cmd_models(message):
    uid = message.from_user.id
    active = get_active(uid)
    kb = telebot.types.InlineKeyboardMarkup(row_width=1)
    for key, p in PROVIDERS.items():
        is_active = key in active
        has_key = p.get("no_key") or bool(os.environ.get(p.get("env",""), "").strip())
        key_status = "✅" if has_key else "🔑"
        check = "☑️" if is_active else "⬜"
        label = f"{check} {p['icon']} {p['name']} {key_status}"
        kb.add(telebot.types.InlineKeyboardButton(label, callback_data=f"toggle:{key}"))
    kb.add(telebot.types.InlineKeyboardButton("✅ Выбрать все", callback_data="toggle:_all"))
    kb.add(telebot.types.InlineKeyboardButton("❌ Снять все",   callback_data="toggle:_none"))
    bot.reply_to(
        message,
        "⚙️ *Выбери AI-провайдеры:*\n✅=активен  🔑=нужен ключ",
        reply_markup=kb,
        parse_mode="Markdown",
    )


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
        has_key = p.get("no_key") or bool(os.environ.get(p.get("env",""), "").strip())
        key_status = "✅" if has_key else "🔑"
        check = "☑️" if is_active else "⬜"
        label = f"{check} {p['icon']} {p['name']} {key_status}"
        kb.add(telebot.types.InlineKeyboardButton(label, callback_data=f"toggle:{k}"))
    kb.add(telebot.types.InlineKeyboardButton("✅ Выбрать все", callback_data="toggle:_all"))
    kb.add(telebot.types.InlineKeyboardButton("❌ Снять все",   callback_data="toggle:_none"))

    try:
        bot.edit_message_reply_markup(
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            reply_markup=kb,
        )
    except Exception:
        pass
    bot.answer_callback_query(call.id)


@bot.message_handler(func=lambda m: True, content_types=["text"])
def handle_question(message):
    uid = message.from_user.id
    question = message.text.strip()

    if question.startswith("/"):
        return

    active = get_active(uid)
    def _is_available(k):
        p = PROVIDERS[k]
        if p.get("no_key"):
            return True
        env = p.get("env")
        return bool(env and os.environ.get(env, "").strip())

    available = [k for k in active if k in PROVIDERS and _is_available(k)]

    if not available:
        bot.reply_to(
            message,
            "❌ Нет активных AI.\n\n"
            "/keys — где получить бесплатные ключи\n"
            "/status — проверить статус",
            parse_mode="Markdown",
        )
        return

    history = get_history(uid)
    history.append({"role": "user", "content": question})
    messages = history[-10:]

    count = len(available)
    names = " | ".join(PROVIDERS[k]["icon"] + PROVIDERS[k]["name"] for k in available)
    waiting_msg = bot.reply_to(
        message,
        f"⏳ Спрашиваю *{count}* AI параллельно...\n_{names}_",
        parse_mode="Markdown",
    )

    start = time.time()

    results = _run_parallel(available, messages)

    elapsed = round(time.time() - start, 1)

    try:
        bot.delete_message(message.chat.id, waiting_msg.message_id)
    except Exception:
        pass

    assistant_parts = []
    for key in available:
        text = results.get(key)
        chunk = _fmt_response(key, text)

        try:
            bot.send_message(
                message.chat.id,
                chunk,
                parse_mode="Markdown",
                reply_to_message_id=message.message_id,
            )
        except Exception:
            try:
                bot.send_message(message.chat.id, chunk.replace("*", "").replace("_", ""))
            except Exception as e:
                log.warning(f"[send] {e}")

        if text:
            assistant_parts.append(f"[{PROVIDERS[key]['name']}]: {text}")

    if assistant_parts:
        combined = "\n---\n".join(assistant_parts)
        history.append({"role": "assistant", "content": combined})
        set_history(uid, history)

    try:
        bot.send_message(
            message.chat.id,
            f"✅ _Ответили за {elapsed} сек. | /clear — очистить историю | /models — сменить AI_",
            parse_mode="Markdown",
        )
    except Exception:
        pass


START_TIME = datetime.now(AZT)

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        uptime = datetime.now(AZT) - START_TIME
        h, rem = divmod(int(uptime.total_seconds()), 3600)
        m, s   = divmod(rem, 60)
        body = f"AI Bot OK | Uptime: {h}h {m}m {s}s\n".encode()
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


if __name__ == "__main__":
    port = int(os.environ.get("AI_BOT_PORT", 8001))
    t = threading.Thread(target=_run_health, daemon=True)
    t.start()
    log.info(f"[AI-Bot] Запущен. Healthcheck: порт {port}")

    # Сбрасываем вебхук и все конфликты предыдущих инстанций
    try:
        bot.delete_webhook(drop_pending_updates=True)
        log.info("[AI-Bot] Вебхук сброшен, конфликты очищены")
    except Exception as e:
        log.warning(f"[AI-Bot] delete_webhook: {e}")
    time.sleep(1)

    bot.infinity_polling(timeout=30, long_polling_timeout=25, logger_level=logging.WARNING,
                         restart_on_change=False)
