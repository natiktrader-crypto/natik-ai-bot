# 🤖 natik-ai-bot

Multi-AI Telegram bot — задаёшь вопрос один раз, отвечают несколько AI **параллельно**.

## Поддерживаемые AI

| Иконка | Провайдер | Ключ нужен? |
|--------|-----------|-------------|
| 🌸 | GPT-4o (pollinations.ai) | ❌ Нет! |
| 🌺 | Mistral (pollinations.ai) | ❌ Нет! |
| 🟤 | Claude Sonnet 4.5 | ✅ ANTHROPIC_API_KEY |
| 🔵 | Gemini 2.0 Flash | ✅ GEMINI_API_KEY |
| 🟡 | Groq Llama 3.3 70B | ✅ GROQ_API_KEY |
| 🟢 | DeepSeek Chat | ✅ DEEPSEEK_API_KEY |
| 🟣 | OpenRouter (Free) | ✅ OPENROUTER_API_KEY |
| 🟠 | Mistral Small | ✅ MISTRAL_API_KEY |
| 🔴 | Cohere Command R | ✅ COHERE_API_KEY |
| ⚡ | Cerebras Llama 70B | ✅ CEREBRAS_API_KEY |
| ⬛ | GitHub Models GPT-4o | ✅ GITHUB_MODELS_TOKEN |
| 🆓 | ChatAnywhere GPT-4o-mini | ✅ CHATANYWHERE_API_KEY |

## Команды бота

- `/start` — начало работы
- `/models` — выбрать AI-провайдеры (включить/выключить)
- `/status` — проверить какие ключи активны
- `/keys` — где получить бесплатные ключи
- `/clear` — очистить историю чата

## Деплой на Render

1. Fork этот репо
2. Зайди на [render.com](https://render.com) → New → Background Worker
3. Подключи GitHub, выбери этот репо
4. `render.yaml` автоматически настроит всё
5. Добавь env var: минимум `AI_BOT_TOKEN`

## Локальный запуск

```bash
pip install pytelegrambotapi httpx anthropic
export AI_BOT_TOKEN=your_telegram_bot_token
python ai-bot/bot.py
```

## Особенности

- 🚀 Все AI отвечают **параллельно** (не по очереди)
- 💾 История чата сохраняется (последние 10 сообщений)
- ⚙️ Каждый пользователь выбирает свои AI через `/models`
- 🏥 Healthcheck сервер на порту 8001 (для Render)
