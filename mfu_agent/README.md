# MFU Health Agent

AI-агент мониторинга парка многофункциональных устройств (МФУ).
Рассчитывает интегральный индекс здоровья (1–100) для каждого устройства
на основе событий ошибок, ресурсов и сервисной документации.
Результат — HTML/PDF-отчёт с объяснением факторов снижения и рекомендациями,
плюс интерактивный чат с RAG-агентом по данным текущего отчёта.

## Требования

- Python >= 3.11
- Docker (для Qdrant)
- Локальный `llama-server` с OpenAI-совместимым API (порт 8000) **или**
  GigaChat-аккаунт (ключ в `.env` → `GIGACHAT_AUTH_KEY`)

## Запуск панели управления

Если зависимости уже установлены — одна команда поднимает всё:

```bash
cd ~/агенты/ии\ индекс\ здоровья-2/mfu_agent && make start
```

Что делает `make start`:
1. Поднимает Qdrant в Docker (`docker compose up -d` на портах 6343/6344).
2. Запускает `llama-server` на `localhost:8000` через [scripts/start_llama.sh](scripts/start_llama.sh),
   если он ещё не отвечает. По умолчанию грузит первую `.gguf` из
   `/home/albert/models/`. Чтобы выбрать конкретную модель:
   ```bash
   MFU_LOCAL_MODEL=Qwen2.5-14B-Instruct-Q8_0.gguf make start
   ```
   Логи llama-server: `/tmp/llama-server.log`.
3. Запускает Streamlit на порту **8504**.

Панель откроется в браузере по адресу **http://localhost:8504**.

Остановить всё (Streamlit + llama-server + Qdrant):

```bash
make stop
```

### Первая установка

Только в самый первый раз (создаёт `.venv`, ставит зависимости, поднимает Qdrant):

```bash
cd ~/агенты/ии\ индекс\ здоровья-2/mfu_agent && make install
```

После этого для ежедневного запуска достаточно `make start`.

### Страницы панели управления

| Страница | Назначение |
|----------|-----------|
| 1 — Загрузка данных | Загрузка CSV/XLSX/JSON мониторинга, запуск фонового анализа |
| 2 — Dashboard | Результаты анализа парка, deep-LLM по устройствам, анализ массовых ошибок |
| 3 — Weights | Настройка весов формулы здоровья, пересчёт индексов |
| 4 — LLM Chat | Чат с ИИ-агентом по текущему отчёту и сервисной документации |
| 5 — Error Codes | Управление паттернами ошибок (yaml) |
| 5 — RAG Admin | Управление RAG-базой: коллекции, индексация, поиск, eval |

### LLM-провайдеры

В сайдбаре каждой страницы — селектор провайдера:

- **Локальная** — `llama-server` на `localhost:8000`. В раскрывающемся блоке
  «🔀 Переключить локальную модель» — список `.gguf` из
  `/home/albert/models/` (путь настраивается через env `MFU_LOCAL_MODELS_DIR`).
  При нажатии «Применить»:
  1. Останавливается старый `llama-server` (SIGTERM → SIGKILL через 5 с).
  2. Ожидание освобождения TCP-порта и VRAM.
  3. Запуск нового `llama-server` с теми же флагами, но новым `-m`. stdout/stderr
     пишутся в `/tmp/llama-server.log` (путь меняется через `MFU_LLAMA_LOG`).
  4. При провале нового процесса — **автоматический откат на старую модель**
     и сообщение с хвостом лога.
- **GigaChat** — облако Сбера. Ключ `GIGACHAT_AUTH_KEY` в `.env`
  (см. `.env.example`).

### Отдельные команды

```bash
make install        # установка зависимостей + запуск Qdrant
make start          # Qdrant + llama-server + Streamlit (всё)
make stop           # остановить всё
make up             # только Qdrant
make down           # остановить Qdrant
make llama          # запустить только llama-server (idempotent)
make run            # запустить только Streamlit
make test           # прогон тестов (pytest)
make lint           # ruff + mypy
make format         # автоформатирование кода
```

## Архитектура

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — краткий обзор
- [docs/ARCHITECTURE_FULL.md](docs/ARCHITECTURE_FULL.md) — подробное описание

### Ключевые компоненты

| Компонент | Путь | Назначение |
|---|---|---|
| Агент (plan → tool → reflection) | `agent/core.py` | Оркестрация batch-расчёта и chat-режима |
| 13 tool'ов | `agent/tools/impl.py` | RAG, события, ресурсы, расчёт индекса, сводки отчёта |
| RAG-поиск | `rag/search.py` | Qdrant + BGE-M3 embedder + BGE-reranker, hybrid retrieval |
| Формула здоровья | `calculator/` | Детерминированный расчёт по WeightsProfile |
| LLM-клиент | `llm/client.py` | Автодетект стратегии tool-calling: native / guided_json / **ReAct** |
| Переключатель локальных моделей | `ui/local_model_manager.py` | Управление `llama-server` из UI: stop → port/VRAM-wait → start → откат |
| Сборка отчёта | `reporting/report_builder.py` | FleetSummary, DeviceReport, executive_summary через LLM |

### Чат с агентом

Page `4_LLM_Chat`:

- Guardrail — чат недоступен, пока не сформирован отчёт (`st.warning` →
  «Загрузите данные на странице Загрузка данных»).
- Каждый запрос пересобирает `ToolDependencies` с актуальным `Report` и
  `mass_error_analyses` из session, так что три дополнительных tool'а
  отвечают по **живым данным** текущего отчёта:
  - `get_current_report_summary` — fleet_summary + executive_summary +
    счётчики зон.
  - `list_red_zone_devices(limit, sort_by)` — устройства в красной зоне
    с моделью, локацией, топ-3 факторами.
  - `list_mass_errors(limit, severity)` — массовые коды с severity из
    Qdrant-коллекции `error_codes`, фильтрация `critical/high/medium/low`.
- Сериализация отчёта в system-prompt — короткая сводка ≤ 6 000 символов,
  детали берутся через tool-вызовы (не перегружает контекст на 16К-моделях).
- ReAct-парсер — для local GGUF без native function-calling. Толерантен к
  форматам «Действие: X Параметры: {...}» в одну или несколько строк,
  с/без кавычек на имени tool'а, с вложенным JSON.
- Strip reasoning-артефактов (`<think>`, ChatML `<|im_end|>`) —
  автоматически в chat-режиме. Nemotron-модели распознаются как reasoning.

## Конфигурация

| Файл | Что настраивает |
|---|---|
| `configs/llm_endpoints.yaml` | Endpoint'ы: Локальная, GigaChat. Модели, auth, TLS |
| `configs/agent_config.yaml` | Лимиты loop'а, chat vs batch-params, memory |
| `configs/rag_config.yaml` | Qdrant хост/порт, embedding и reranker модели |
| `configs/report_config.yaml` | Верстка отчёта, пороги зон |
| `configs/weights/default.yaml` | Весовой профиль формулы здоровья |
| `configs/field_synonyms.yaml` | Маппинг колонок входных CSV/XLSX |
| `configs/model_aliases.yaml` | Нормализация названий МФУ (Kyocera, Xerox, HP…) |
| `configs/error_code_patterns.yaml` | Паттерны распознавания кодов ошибок |
| `.env` | Секреты: `GIGACHAT_AUTH_KEY` и т.п. (см. `.env.example`) |

## Данные, хранимые локально

- `storage/qdrant/` — индексы Qdrant (монтируется в контейнер)
- `storage/uploads/` — входные файлы парка
- `storage/reports/` — сгенерированные HTML/PDF
- `storage/ingestion_checkpoints/` — чекпоинты фонового анализа

Все `storage/` поддиректории в `.gitignore` — не коммитятся.
