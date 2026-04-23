# Архитектура `mfu_agent` — полный технический справочник

Глубокое описание подсистем, контрактов и данных для разработчика, который должен ориентироваться в коде за 20 минут. Это дополнение к короткому [ARCHITECTURE.md](ARCHITECTURE.md) с mermaid-диаграммами и pipeline по этапам.

---

## 1. Что это и зачем

**`mfu_agent`** — Streamlit-приложение для мониторинга парка МФУ (многофункциональных устройств). Принимает выгрузку Zabbix (или любой CSV/XLSX/JSON), для каждого устройства считает **интегральный индекс здоровья `H ∈ [1..100]`**, классифицирует устройства по зонам (green/yellow/red) и генерирует HTML/PDF-отчёт с объяснением причин снижения и рекомендациями.

Ядро расчёта — детерминированная чистая функция; LLM используется как **дополнительный** слой для классификации severity, глубокого анализа red-зоны, анализа массовых ошибок и chat-режима.

### Технологический стек

| Слой | Технология |
|---|---|
| Python | >= 3.11 |
| UI | Streamlit 1.38+ (multi-page app) |
| LLM-клиент | `openai` 1.40+ (OpenAI-совместимый, работает с локальным vLLM / llama-server / Ollama) |
| Промпты и контракты | Pydantic 2 + `response_schema` |
| RAG | Qdrant 1.11, `BAAI/bge-m3` embeddings, `BAAI/bge-reranker-v2-m3`, гибридный dense+sparse |
| Парсинг | pandas, openpyxl, chardet, python-dateutil |
| Отчёт | Jinja2 + WeasyPrint (HTML→PDF), PyMuPDF |
| Тесты | pytest 8, 35 файлов, 741+ тестов |
| Качество | ruff, mypy |

---

## 2. Высокоуровневая архитектура: пять треков

| Трек | Подсистема | Директория / ключевые файлы | Чем отвечает |
|---|---|---|---|
| **A** | Health Index | `tools/calculator.py`, `configs/weights/*.yaml` | Детерминированный расчёт `H = 100 − Σ(S·R·C·A)` |
| **B** | Agent | `agent/core.py`, `agent/tools/impl.py`, `agent/prompts/*.md`, `llm/client.py` | LLM-оркестровка: batch-lite, deep-single-shot, mass-error, chat |
| **C** | Reporting | `reporting/report_builder.py`, `reporting/html_renderer.py` | Сборка `Report`, HTML/PDF-рендер, executive summary |
| **D** | RAG | `rag/ingestion.py`, `rag/search.py`, `configs/rag_config.yaml` | Индексация и гибридный поиск по service manuals + error codes |
| **E** | Ingestion | `data_io/parsers.py`, `zabbix_transform.py`, `field_mapper.py`, `normalizer.py`, `factor_store.py` | Файл → нормализованные события/ресурсы → иммутабельный `FactorStore` |

### Поток данных в одном предложении

> Пользователь загружает Zabbix-файл → парсер → `zabbix_transform` → `FieldMapper` → `Normalizer` строит `NormalizedEvent` + `ResourceSnapshot` → `FactorStore.freeze()` → `run_batch_lite` для каждого устройства вычисляет `H` через чистый `calculate_health_index` → результат в `session_state` → Dashboard рисует зоны и по кнопкам запускает LLM-анализ массовых ошибок и red-устройств → Weights-страница пересчитывает `H` с новыми весами без повторного прогона агента.

---

## 3. Структура пакета

```
mfu_agent/
├── app.py                         # entrypoint Streamlit, st.navigation
├── pyproject.toml                 # Python >= 3.11, ruff + mypy
├── Makefile                       # install / up / run / test / lint / format
├── docker-compose.yml             # Qdrant (порты 6343→6333, 6344→6334)
├── README.md
│
├── agent/
│   ├── core.py                    # Agent: run_batch_lite, analyze_mass_error,
│   │                              #        analyze_device_deep, run_chat
│   ├── memory.py                  # LearnedPattern persistence
│   ├── tools/
│   │   ├── impl.py                # 10 tools (см. §7)
│   │   └── registry.py            # реестр для LLM tool-calling
│   └── prompts/
│       ├── classify_severity.md   # severity для одного error_code
│       ├── device_deep_analysis.md# глубокий анализ red-устройства
│       ├── executive_summary.md   # summary для отчёта
│       ├── field_mapping.md       # LLM-fallback для FieldMapper
│       ├── identify_model.md      # определение модели устройства
│       ├── mass_error_analysis.md # анализ массовой ошибки
│       ├── reflection.md          # self-check агента
│       ├── system_batch.md        # системный промпт batch-режима
│       └── system_chat.md         # системный промпт chat-режима
│
├── data_io/
│   ├── parsers.py                 # CSV/TSV/JSON/JSONL/XLSX → DataFrame
│   ├── preamble.py                # strip SQL-комментариев в CSV
│   ├── zabbix_transform.py        # long-format → metadata/resources/events
│   ├── field_mapper.py            # 4-уровневое распознавание колонок
│   ├── normalizer.py              # ingest_file() — единая точка ingestion
│   ├── factor_store.py            # immutable хранилище событий/ресурсов
│   └── models.py                  # Pydantic-контракты (35+ классов)
│
├── tools/
│   └── calculator.py              # чистая calculate_health_index (S·R·C·A)
│
├── llm/
│   └── client.py                  # OpenAI-совместимый клиент,
│                                  # strip_reasoning_artifacts (<think>…</think>)
│
├── rag/
│   ├── ingestion.py               # индексация PDF/YAML в Qdrant
│   ├── search.py                  # hybrid dense+sparse, reranker
│   └── reindex_error_codes.py
│
├── error_codes/
│   ├── schema.py                  # Pydantic ErrorCode, ModelErrorCodes
│   ├── loader.py                  # load per-model YAML
│   ├── parsers.py                 # CSV/XLSX/YAML импорт
│   ├── writer.py                  # save + backup в _trash/
│   ├── consistency.py             # детект конфликтов
│   └── aliases.py                 # model_aliases.yaml CRUD
│
├── reporting/
│   ├── report_builder.py          # Report из HealthResults
│   └── html_renderer.py           # Jinja2 + WeasyPrint (HTML/PDF)
│
├── pages/
│   ├── 1_Загрузка_данных.py       # file uploader + background worker
│   ├── 2_Dashboard.py             # метрики, зоны, mass/deep кнопки
│   ├── 3_Weights.py               # слайдеры весов, пересчёт H
│   ├── 4_LLM_Chat.py              # чат с агентом по результатам
│   ├── 5_Error_Codes.py           # CRUD справочника кодов ошибок
│   └── 5_RAG_Admin.py             # управление коллекциями Qdrant
│
├── state/
│   ├── session.py                 # session_state обёртки
│   ├── singletons.py              # factor_store / llm / rag кэш
│   ├── _analysis_bg.py            # module-state для ingestion worker
│   ├── _deep_bg.py                # module-state для deep/mass workers
│   └── _rag_bg.py                 # module-state для RAG reindex worker
│
├── configs/
│   ├── agent_config.yaml          # max_attempts, reflection, memory, llm knobs
│   ├── weights/default.yaml       # WeightsProfile (S, R, C, A, zones, conf)
│   ├── model_aliases.yaml         # canonical model names + aliases
│   ├── field_synonyms.yaml        # маппинг колонок → канонические поля
│   ├── llm_endpoints.yaml         # URL/model/timeouts для vLLM
│   ├── rag_config.yaml            # embeddings, reranker, hybrid, chunking
│   ├── report_config.yaml         # параметры отчёта
│   ├── component_vocabulary.yaml  # список компонентов (print/quality/...)
│   ├── error_code_patterns.yaml   # regex для нормализации
│   ├── xerox_display_codes.yaml   # Xerox error registry (conf=0.9)
│   ├── xerox_severity_map.yaml    # Xerox severity overrides
│   ├── error_codes/{vendor}/{model_slug}.yaml
│   ├── schema_profiles/           # сохранённые профили FieldMapper
│   └── _backups/                  # автобэкапы model_aliases.yaml
│
├── scripts/
│   └── migrate_error_codes.py
│
├── storage/
│   └── qdrant/                    # persistent volume для Qdrant
│
├── tests/                         # 35 файлов, 741+ тестов
│   ├── e2e/                       # end-to-end сценарии
│   ├── fixtures/                  # тестовые данные
│   ├── artifacts/                 # артефакты тестов
│   └── test_*.py                  # юниты по каждой подсистеме
│
└── docs/
    ├── ARCHITECTURE.md            # короткий обзор + mermaid
    └── ARCHITECTURE_FULL.md       # этот файл
```

---

## 4. Track A — Формула индекса здоровья

**Файл:** [tools/calculator.py](../tools/calculator.py) — чистая функция `calculate_health_index(factors, context, weights_profile) → HealthResult`.

### Формула

```
H = max(1, 100 − Σ(S · R · C · A))
```

| Коэффициент | Смысл | Диапазон | Формула |
|---|---|---|---|
| **S** | severity-вес для уровня ошибки | 0–60 | из `WeightsProfile.severity` |
| **R** | repeatability — сглаженный рост от числа повторений | 1.0–5.0 | `min(R_max, 1 + log_base(n))` |
| **C** | context — множитель за состояние ресурсов | 1.0–1.5 | `ContextConfig.modifiers` (toner_low, drum_high, fuser_high, mileage_over_interval) |
| **A** | age decay — возрастной спад свежести события | 0.0–1.0 | `exp(−days / τ)`, τ=14 дней |

### WeightsProfile (из [configs/weights/default.yaml](../configs/weights/default.yaml))

```yaml
severity:
  critical: 60.0    # S
  high: 20.0
  medium: 10.0
  low: 3.0
  info: 0.0

repeatability:
  base: 2           # логарифм
  max_value: 5.0    # R_max
  window_days: 14

context:
  modifiers:
    toner_low:          { threshold: 10, multiplier: 1.5, applies_to: [print, quality] }
    drum_high:          { threshold: 90, multiplier: 1.3, applies_to: [quality] }
    fuser_high:         { threshold: 90, multiplier: 1.5, applies_to: [fusing, jam] }
    mileage_over_interval: { multiplier: 1.2, applies_to: ['*'] }
  max_value: 1.5    # C_max

age:
  tau_days: 14
  window_days: 30

zones:
  green_threshold: 75     # H ≥ 75 → green
  red_threshold: 40       # H < 40 → red, иначе yellow

critical_per_day_limit: 1
silent_device_mode: data_quality  # {optimistic, data_quality, carry_forward}
```

### Confidence

```
confidence = max(min_value, Π penalties)
min_value = 0.2
```

Penalties (мультипликативные) задаются в `ConfidencePenalties`:

| Флаг | Penalty |
|---|---|
| `rag_not_found` | 0.7 |
| `missing_resources` | 0.85 |
| `missing_model` | 0.6 |
| `abnormal_daily_jump` | 0.8 |
| `anomalous_event_count` | 0.7 |
| `no_events_and_no_resources` | 0.9 |

Confidence-зона: `high ≥ 0.7`, `medium ≥ 0.5`, иначе `low`.

---

## 5. Track B — Agent

**Файл:** [agent/core.py](../agent/core.py) — класс `Agent` с четырьмя публичными методами.

### Режимы исполнения

| Метод | Когда | Вход | Выход | LLM-вызовы |
|---|---|---|---|---|
| `run_batch_lite` | Этап 4 ingestion | `factor_store`, `device_id`, `weights_profile` | `HealthResult` | 1 на уникальный код (classify) |
| `analyze_device_deep` | По кнопке red-зоны | `device_id` | `DeepDeviceAnalysis` | 1 (single-shot с `response_schema`) |
| `analyze_mass_error` | По кнопке mass-error | `error_code`, `description` | `MassErrorAnalysis` | 1 (single-shot) |
| `run_chat` | Страница 4_LLM_Chat | `user_query`, `ChatContext` | stream токенов + tool-calls | N (tool loop) |

### Конфигурация агента ([configs/agent_config.yaml](../configs/agent_config.yaml))

```yaml
agent:
  max_attempts_per_device: 2
  max_tool_calls_per_attempt: 15
  max_llm_calls_per_attempt: 20
  trace_retention_days: 30

reflection:
  enabled: true
  apply_in_chat_mode: false

memory:
  enabled: true
  max_patterns_per_model: 50
  pattern_min_evidence_devices: 2

llm:
  batch_mode:         { temperature: 0.2, top_p: 0.9, max_tokens: 2000 }
  chat_mode:          { temperature: 0.4, top_p: 0.9, max_tokens: 3000 }
  reflection:         { temperature: 0.1, top_p: 1.0, max_tokens:  500 }
  classify_severity:  { temperature: 0.0, top_p: 1.0, max_tokens:  300 }
```

### Trace

Каждый запуск агента пишет `Trace` (в [data_io/models.py](../data_io/models.py)):

```python
Trace(session_id, mode, device_id?, started_at, ended_at, steps[...],
      final_result, total_tool_calls, total_llm_calls, total_tokens,
      attempts=1..2, flagged_for_review, rag_hits[...])

TraceStep(step_number, type, thought?, tool_name?, tool_args?,
          tool_result_summary?, duration_ms, tokens_used?)
```

Step-типы: `plan | llm_call | tool_call | reflection | memory_save`.

### Reflection (self-check)

После основного прогона агент делает до **2 попыток**: если reflection-LLM возвращает `SUSPICIOUS` / `NEEDS_REVISION` — вторая попытка. `flagged_for_review=true` при устойчивой проблеме.

Выход reflection:
```python
ReflectionResult(verdict: approved|needs_revision|suspicious,
                 issues: [{issue, severity}],
                 recommended_action: accept|recalculate|flag_for_review)
```

### LLM-клиент ([llm/client.py](../llm/client.py))

- OpenAI-совместимый (`openai>=1.40`).
- `strip_reasoning_artifacts` — выкидывает `<think>…</think>` от reasoning-моделей (Qwen, Nemotron).
- `_robust_json_parse` в [agent/core.py](../agent/core.py#L54) — извлекает JSON из ответа с мусором, чинит обрезанные строки и недозакрытые скобки.

Endpoint-конфиг ([configs/llm_endpoints.yaml](../configs/llm_endpoints.yaml)):
```yaml
endpoints:
  default:
    url: "http://localhost:8000/v1"
    api_key: "dummy-for-local"
    model: "nvidia_NVIDIA-Nemotron-Nano-12B-v2-Q8_0.gguf"
    timeout_seconds: 180.0
    max_retries_network: 3
    max_retries_invalid: 2
```

---

## 6. Классификация severity — каскад

`classify_error_severity` пробует источники в порядке падающей уверенности:

1. **Xerox registry** (`configs/xerox_display_codes.yaml`, `xerox_severity_map.yaml`) — regex `XX-YYY-ZZ`, confidence = **0.9**.
2. **Heuristic** — по ключевым словам в описании (`jam`, `toner`, `fuser`, `drum`).
3. **RAG** — поиск в Qdrant коллекции `error_codes` → возвращает `severity` из документа.
4. **LLM fallback** — single-shot вызов с промптом [classify_severity.md](../agent/prompts/classify_severity.md) + `response_schema`.
5. **`_CLASSIFY_FALLBACK`** — `medium / 0.3` как крайний случай.

---

## 7. Tool Registry

[agent/tools/registry.py](../agent/tools/registry.py) реализует протокол `Tool` и диспетчер. Все инструменты возвращают `ToolResult(success, data, error?)`.

Зарегистрированные tools ([agent/tools/impl.py](../agent/tools/impl.py)):

| Tool | Назначение |
|---|---|
| `SearchServiceDocsTool` | поиск по RAG (service_manuals + error_codes) |
| `ClassifyErrorSeverityTool` | severity каскад (см. §6) |
| `GetDeviceEventsTool` | события по устройству с фильтрами `window_days`, `error_code` |
| `GetDeviceResourcesTool` | ResourceSnapshot по устройству (самый свежий) |
| `CountErrorRepetitionsTool` | n повторений за `window_days` |
| `CalculateHealthIndexTool` | обёртка над `tools/calculator.calculate_health_index` |
| `GetFleetStatisticsTool` | FleetSummary (avg, median, zone_counts) |
| `FindSimilarDevicesTool` | similar по dimension: errors / model / location / error_and_model |
| `GetDeviceHistoryTool` | HistoryPoint[] для sparkline |
| `GetLearnedPatternsTool` | сохранённые агентом LearnedPattern-ы |

---

## 8. Track E — Ingestion pipeline

### 4-уровневый FieldMapper ([data_io/field_mapper.py](../data_io/field_mapper.py))

```
1. Saved profile           (configs/schema_profiles/*.yaml — ранее принятый маппинг)
2. Synonyms yaml           (configs/field_synonyms.yaml — ["host_name", "hostname", "device"])
3. Heuristic by content    (regex над выборкой строк — Xerox код, IP, timestamp)
4. LLM fallback            (промпт agent/prompts/field_mapping.md, response_schema)
```

### Normalizer ([data_io/normalizer.py](../data_io/normalizer.py))

Для каждой строки после маппинга:

1. `parse_timestamp` — Unix-секунды + 8 форматов даты + dateutil-fallback.
2. `normalize_error_code` — regex Xerox `\b[0-9]{2}-[0-9]{3}-[0-9]{2}\b`, снимает префиксы `Error:`, `Код:`.
3. `canonicalize_model` — через [configs/model_aliases.yaml](../configs/model_aliases.yaml) (алиасы → канон).
4. `detect_resource_unit` — percent / fraction / raw → приводит к **%**.

Строка с `error_code` или `description` → `NormalizedEvent`.
Строка с `toner | drum | fuser | mileage` → `ResourceSnapshot` (по устройству сохраняется самый свежий timestamp).

### FactorStore ([data_io/factor_store.py](../data_io/factor_store.py))

```python
fs = FactorStore()
fs.add_events([NormalizedEvent, ...])
fs.set_resources(device_id, ResourceSnapshot)
fs.set_device_metadata(device_id, DeviceMetadata)
fs.set_fleet_meta(FleetMeta)
fs.set_reference_time(datetime.now(UTC))
fs.freeze()                      # immutable — read-only
```

После `freeze()` воркеры читают параллельно без блокировок. Запись после freeze → исключение.

---

## 9. Track D — RAG

### Конфиг ([configs/rag_config.yaml](../configs/rag_config.yaml))

```yaml
embeddings:
  model: "BAAI/bge-m3"
  device: "cuda"
  batch_size: 32
  max_length: 1024
  fp16: true

reranker:
  model: "BAAI/bge-reranker-v2-m3"
  device: "cuda"
  top_n_input: 30
  top_n_output: 8

hybrid_search:
  use_qdrant_fusion: true
  rrf_k: 60
  dense_weight: 1.0
  sparse_weight: 1.0
  top_k_per_branch: 30

chunking:
  service_manuals:  { strategy: hierarchical, max_tokens: 800, min: 200, overlap: 100, bookmarks: true }
  error_codes:      { strategy: per_record, record_separator: "^Код\\s+[A-Z]\\d+" }
  internal_guides:  { strategy: recursive, max_tokens: 500, overlap: 80 }
  parts_catalog:    { strategy: per_record }
```

### Коллекции Qdrant

| Коллекция | Источник | Стратегия chunking |
|---|---|---|
| `service_manuals` | PDF сервис-мануалы | hierarchical по bookmarks |
| `error_codes` | YAML справочники кодов | per_record по regex `^Код X\d+` |
| `internal_guides` | внутренние guides | recursive |
| `parts_catalog` | каталог запчастей | per_record |

### Гибридный поиск ([rag/search.py](../rag/search.py))

Dense (bge-m3) + sparse (SPLADE/BM25) → Qdrant Hybrid Fusion (RRF, k=60) → bge-reranker-v2-m3 → top-8.

---

## 10. Модели данных (Pydantic) — ключевые классы

Из [data_io/models.py](../data_io/models.py) — около 35 классов. Самые важные:

### Track A
- `SeverityLevel` — enum: `Critical|High|Medium|Low|Info`
- `HealthZone` — enum: `green|yellow|red`
- `ConfidenceZone` — enum: `high|medium|low`
- `Factor` — один фактор расчёта (code, severity, S, R, C, A, n_repetitions, age_days)
- `FactorContribution` — вклад фактора в итог (label, penalty, S, R, C, A, source)
- `HealthResult` — результат на устройство (device_id, health_index, confidence, zone, factor_contributions, calculation_snapshot, calculated_at)
- `WeightsProfile` — полный профиль весов с вложенными `SeverityWeights`, `RepeatabilityConfig`, `ContextConfig`, `AgeConfig`, `ConfidenceConfig`, `ZoneThresholds`

### Track B
- `Trace`, `TraceStep` — трассировка исполнения
- `ReflectionResult` — вердикт self-check
- `DeepDeviceAnalysis` — результат глубокого анализа устройства (health_index_llm, root_cause, recommended_action, explanation, related_codes)
- `MassErrorAnalysis` — результат анализа массовой ошибки (is_systemic, what_is_this, why_this_pattern, business_impact, immediate_action, long_term_action, indicators_to_watch)
- `LearnedPattern` — паттерн из памяти
- `BatchContext`, `ChatContext` — контейнеры для режимов

### Track E
- `NormalizedEvent` — event (device_id, timestamp, error_code, description, model?)
- `ResourceSnapshot` — ресурсы (toner, drum, fuser, mileage, service_interval, timestamp)
- `DeviceMetadata` — модель, локация, теги
- `FleetMeta` — общие метаданные выгрузки

### Track C
- `FleetSummary`, `DeviceReport`, `CalculationSnapshot`, `Report` — компоненты отчёта
- `CalculationSnapshot` — **замороженные** параметры расчёта (для воспроизводимости)

---

## 11. Streamlit — страницы и state

### Страницы ([pages/](../pages))

| Файл | Назначение |
|---|---|
| [1_Загрузка_данных.py](../pages/1_Загрузка_данных.py) | `st.file_uploader` (≤200 МБ), preamble-strip, фоновый `_run_analysis_worker` |
| [2_Dashboard.py](../pages/2_Dashboard.py) | метрики, зоны, таблица; кнопки **Mass error** и **Red zone** — стартуют фоновые воркеры |
| [3_Weights.py](../pages/3_Weights.py) | слайдеры → пересчёт `H` из **raw_factors** (без LLM) |
| [4_LLM_Chat.py](../pages/4_LLM_Chat.py) | чат-режим агента со stream и tool-calls |
| [5_Error_Codes.py](../pages/5_Error_Codes.py) | CRUD справочников per-vendor/per-model |
| [5_RAG_Admin.py](../pages/5_RAG_Admin.py) | коллекции Qdrant, индексация, eval |

### State

- [state/session.py](../state/session.py) — сеттеры/геттеры `st.session_state`:
  `current_health_results`, `baseline_health_results`, `raw_factors`, `current_factor_store`, `active_weights_profile`.
- [state/singletons.py](../state/singletons.py) — `@st.cache_resource` для `LLMClient`, `FactorStore`, RAG singletons.
- [state/_analysis_bg.py](../state/_analysis_bg.py), [_deep_bg.py](../state/_deep_bg.py), [_rag_bg.py](../state/_rag_bg.py) — **module-level state** для фоновых thread-ов (Streamlit `session_state` недоступен вне UI-thread'а; воркеры пишут сюда progress).

---

## 12. Инварианты системы

| # | Инвариант | Следствие |
|---|---|---|
| 1 | `FactorStore` иммутабелен после `freeze()` | Воркеры читают параллельно без блокировок |
| 2 | `calculate_health_index` — чистая функция | Один путь для первичного batch и пересчёта Weights — воспроизводимость |
| 3 | `raw_factors` сохраняются на Этапе 4 в session_state | Пересчёт весов без повторного прогона LLM |
| 4 | Фоновые воркеры пишут в module-state, не в `st.session_state` | UI-поток читает progress из модуля |
| 5 | **Latest-timestamp-per-code** группировка | Один `Factor` на уникальный код, `A` = свежесть самого нового |
| 6 | Two-tier LLM: lite batch → deep single-shot | LLM для red-зоны вызывается отложенно, по кнопке |
| 7 | Severity многоуровневая (registry → heuristic → RAG → LLM) | Быстрый путь для известных кодов, дорогой LLM только для новых |
| 8 | `strip_reasoning_artifacts` везде на выходе LLM | `<think>…</think>` от reasoning-моделей не просачиваются в UI |
| 9 | `CalculationSnapshot` замораживает параметры | Отчёт воспроизводим |

---

## 13. Запуск

### Первая установка

```bash
cd mfu_agent
make install        # .venv + pip install -e .[dev] + docker compose up -d (Qdrant)
```

### Ежедневное

```bash
make up             # только Qdrant
make run            # Streamlit на порту 8504 --headless
```

По умолчанию Streamlit открывается на `http://localhost:8501` (в README) либо `8504` (в Makefile `run`). Qdrant — `localhost:6343` (маппится с контейнерного 6333).

### Переменные окружения

LLM endpoint зашит в [configs/llm_endpoints.yaml](../configs/llm_endpoints.yaml), по умолчанию `http://localhost:8000/v1`. Подразумевается локальный vLLM / llama-server / Ollama с OpenAI-совместимым API.

### Makefile-цели

| Цель | Что делает |
|---|---|
| `install` | venv + pip install editable + docker compose up |
| `up` / `down` | запуск/остановка Qdrant |
| `run` | `streamlit run app.py --server.port 8504 --server.headless true` |
| `test` | `pytest -v` |
| `lint` | `ruff check .` + `mypy .` |
| `format` | `ruff format .` |

---

## 14. Тесты

**35 файлов**, 741+ тест-кейсов. Покрытие по подсистемам:

| Файл | Покрывает |
|---|---|
| `test_calculator.py`, `test_calculator_*.py` (6 файлов) | формула H, determinism, bounds, contributions, zones, confidence, perf |
| `test_normalizer.py`, `test_parsers*.py` | Track E парсеры и нормализация |
| `test_field_mapper.py` | FieldMapper 4 уровня |
| `test_factor_store.py` | immutability, read-after-freeze |
| `test_agent_core.py`, `test_agent_tools.py`, `test_agent_p0.py` | Track B |
| `test_deep_analysis.py`, `test_bug_hunt.py` | deep/mass LLM сценарии |
| `test_rag_p0.py` | Track D |
| `test_error_codes.py` | справочник кодов |
| `test_report_builder.py` | Track C |
| `test_contracts_p0.py`, `test_nf_p0.py` | контракты, функциональные требования |
| `test_p0_ingestion.py`, `test_csv_preamble_consistency.py` | E2E ingestion |
| `test_derived_state_reset.py` | правильный reset session_state |
| `e2e/` | end-to-end сценарии |

Запуск: `.venv/bin/pytest -v` или `make test`.

---

## 15. Последние изменения (5 коммитов)

| SHA | Сообщение | Смысл |
|---|---|---|
| `35a5090` | fix(models): handle Streamlit hot-reload class identity for WeightsProfile | Pydantic видел два разных класса `WeightsProfile` после Streamlit hot-reload — упала isinstance-проверка |
| `03159db` | fix(models): prevent Pydantic re-validation of WeightsProfile in BatchContext | `@model_validator(mode='before')` приводит вложенный WP к dict, чтобы Pydantic его не валидировал повторно |
| `72e55d6` | fix(agent): fallback RAG search without model filter when no results | Если фильтр по модели дал пусто — повторный поиск без фильтра |
| `1fd2575` | fix(agent): call LLM for error classification when error_description available | Если есть описание, стоит дёрнуть LLM вместо жёсткого fallback |
| `fedba71` | fix: Xerox error codes, executive summary artifacts, reasoning model detection | Подчистка артефактов reasoning-моделей в summary |

---

## 16. Точки расширения

Куда смотреть при типовых задачах:

| Задача | Где менять |
|---|---|
| Добавить новый severity-источник (новый вендор) | `agent/tools/impl.py::ClassifyErrorSeverityTool`, конфиг-regex в `configs/error_code_patterns.yaml` |
| Поменять формулу H | `tools/calculator.py` — чистая функция + добавить тест в `test_calculator_*` |
| Добавить новый модификатор контекста | `configs/weights/default.yaml::context.modifiers` + логика в calculator |
| Новая коллекция RAG | `configs/rag_config.yaml::chunking.<name>` + `rag/ingestion.py` + `rag/search.py` |
| Новый tool для агента | `agent/tools/impl.py` (класс + `name/schema/execute`) → регистрация в `ToolRegistry` |
| Новая Streamlit-страница | `pages/<number>_<name>.py` |
| Новый формат файла | `data_io/parsers.py::read_*` + enum `FileFormat` |
| Новый промпт | `agent/prompts/<name>.md` + загрузка через `_load_prompt()` в `agent/core.py` |

---

## 17. Где искать при проблемах

| Симптом | Первое место посмотреть |
|---|---|
| "FieldMapper не распознал колонку" | `configs/field_synonyms.yaml`, затем `agent/prompts/field_mapping.md` |
| Health index = 1 на всех | `tools/calculator.py` + `WeightsProfile` values |
| Qdrant не отвечает | `docker compose ps`, порт 6343, `storage/qdrant/` volume |
| LLM таймаутит | `configs/llm_endpoints.yaml::timeout_seconds`, `max_retries_network` |
| `<think>` в UI | `strip_reasoning_artifacts` в `llm/client.py` |
| Reflection зациклилась | `configs/agent_config.yaml::agent.max_attempts_per_device` |
| Веса не пересчитываются | `raw_factors` в `session_state`, Этап 4 ingestion |
| Pydantic orderly fails на hot-reload | `@model_validator(mode='before')` в `BatchContext` (см. коммит `03159db`) |
