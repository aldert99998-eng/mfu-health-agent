# Архитектура ИИ-агента мониторинга здоровья МФУ

## Назначение документа

Это **сводный архитектурный документ**, обобщающий решения из треков A–E. Предназначен для:

1. **Прикрепления в начало каждой LLM-сессии** в VS Code — даёт модели общую картину проекта и позволяет ориентироваться в связях между модулями.
2. **Быстрого ревью** новыми участниками — один файл, где видно всё устройство системы.
3. **Принятия решений при изменениях** — когда нужно понять, какие модули затронет изменение в одном месте.

Документ **не заменяет** детальные треки, а служит картой к ним. За подробностями — в соответствующий track_*.md.

---

## 1. Цель системы

Автоматизированный мониторинг парка многофункциональных устройств (МФУ) с расчётом **интегрального индекса здоровья** по шкале 1–100 для каждого устройства.

**Входные данные:** произвольные табличные выгрузки из систем мониторинга (CSV, JSON, Excel) с событиями ошибок, счётчиками и ресурсами.

**Выходные данные:** HTML + PDF отчёт с индексом по каждому устройству, объяснением факторов снижения, массовыми проблемами и рекомендациями от агента.

**Ключевой принцип:** агент не принимает действий — только анализирует и предоставляет информацию. Менеджер принимает решения сам.

---

## 2. Пять треков — карта ответственности

| Трек | Что описывает | Основной вопрос |
|---|---|---|
| A | Формула индекса здоровья | Как из факторов получается число? |
| B | Агент и его инструменты | Как агент рассуждает и принимает решения? |
| C | Отчёт (HTML + PDF) | Как результат показывается пользователю? |
| D | RAG-база документации | Где агент берёт знания о критичности ошибок? |
| E | Ingestion входных файлов | Как произвольный файл превращается в понятные данные? |

**Принцип разделения:** каждый трек — автономный слой с чётким интерфейсом. Изменение внутри трека не должно ломать остальные — это обеспечивается за счёт явных контрактов (раздел 6).

---

## 3. Высокоуровневая архитектура

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ПОЛЬЗОВАТЕЛЬСКИЙ ИНТЕРФЕЙС                        │
│                             (Streamlit)                                  │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ 1. Дашборд   │  │ 2. Веса      │  │ 3. LLM/Чат   │  │ 4. RAG-админ │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           SHARED STATE                                   │
│                                                                          │
│   session_state: factor_store, weights, report, chat_history            │
│   singletons (cache_resource): RAGEngine, LLMClient, Agent              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       СЛОЙ БИЗНЕС-ЛОГИКИ                                 │
│                                                                          │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐    │
│  │   Agent (трек B) │   │ Calculator (A)   │   │ ReportBuilder(C) │    │
│  │                  │   │                  │   │                  │    │
│  │  • Plan          │   │ • Формула H      │   │ • HTML + PDF     │    │
│  │  • Tools         │   │ • Confidence     │   │ • Executive sum  │    │
│  │  • Reflection    │   │ • Зоны           │   │ • Top patterns   │    │
│  │  • Memory        │   │ Чистая функция   │   │ Jinja2+WeasyPrint│    │
│  └──────────────────┘   └──────────────────┘   └──────────────────┘    │
│           │                      ▲                      ▲                │
│           │                      │                      │                │
│           └──────────────────────┼──────────────────────┘                │
│                                  │                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        СЛОЙ ДАННЫХ И ЗНАНИЙ                              │
│                                                                          │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐    │
│  │ Factor-Store (E) │   │   RAG (трек D)   │   │  LLM Client (B)  │    │
│  │                  │   │                  │   │                  │    │
│  │ • Events         │   │ • Qdrant         │   │ • OpenAI-compat  │    │
│  │ • Resources      │   │ • BGE-M3         │   │ • Tool strategy  │    │
│  │ • DeviceMeta     │   │ • Reranker       │   │   autodetect     │    │
│  │ • Immutable      │   │ • Hybrid search  │   │ • Switchable     │    │
│  │   после freeze   │   │ • Eval           │   │   endpoints      │    │
│  └──────────────────┘   └──────────────────┘   └──────────────────┘    │
│           ▲                                              │               │
│           │                                              │               │
└───────────┼──────────────────────────────────────────────┼──────────────┘
            │                                              │
            │                                              ▼
┌───────────┼──────────────────────────────────────┐   ┌──────────────────┐
│           │      INGESTION (трек E)              │   │ Локальная LLM    │
│           │                                      │   │ (внешняя)        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────┐ │   │ OpenAI-compat    │
│  │ Парсинг      │ │ Маппинг      │ │Нормализ  │ │   │ endpoint         │
│  │ CSV/JSON/XLSX│ │3 ступени+UI  │ │+Pydantic │ │   └──────────────────┘
│  └──────────────┘ └──────────────┘ └──────────┘ │
└──────────────────────────────────────────────────┘
            ▲
            │
┌───────────┼────────────────────────────────────────────┐
│   Входной файл мониторинга (CSV/JSON/Excel)            │
└────────────────────────────────────────────────────────┘
```

---

## 4. Структура проекта (дерево файлов)

```
mfu_agent/
│
├── app.py                          # Streamlit entry point
├── pyproject.toml
├── docker-compose.yml
├── Makefile
├── README.md
├── .env.example
├── .gitignore
│
├── pages/                          # Streamlit страницы
│   ├── 1_Dashboard.py
│   ├── 2_Weights.py
│   ├── 3_LLM_Chat.py
│   └── 4_RAG_Admin.py
│
├── agent/                          # ТРЕК B
│   ├── core.py                     # Agent class (batch + chat)
│   ├── memory.py                   # MemoryManager для learned patterns
│   ├── tools/
│   │   ├── registry.py             # ToolRegistry
│   │   └── impl.py                 # 9 tool-функций
│   └── prompts/
│       ├── system_batch.md
│       ├── system_chat.md
│       ├── reflection.md
│       ├── classify_severity.md
│       ├── identify_model.md
│       ├── field_mapping.md
│       └── executive_summary.md
│
├── tools/                          # ТРЕК A
│   └── calculator.py               # Детерминированная формула
│
├── rag/                            # ТРЕК D
│   ├── qdrant_client.py            # QdrantManager
│   ├── embeddings.py               # BGEEmbedder
│   ├── reranker.py                 # BGEReranker
│   ├── search.py                   # HybridSearcher
│   ├── ingestion.py                # Document pipeline
│   ├── evaluation.py               # Eval + метрики
│   └── __init__.py                 # RAGEngine (фасад)
│
├── data_io/                        # ТРЕК E
│   ├── models.py                   # ВСЕ Pydantic-модели проекта
│   ├── parsers.py                  # Format detection + parsers
│   ├── field_mapper.py             # 3-ступенчатый маппинг
│   ├── normalizer.py               # Валидация и нормализация
│   ├── factor_store.py             # In-memory хранилище
│   └── ingestion_pipeline.py       # Координатор
│
├── reporting/                      # ТРЕК C
│   ├── report_builder.py           # Report assembly
│   ├── pdf_generator.py            # WeasyPrint wrapper
│   ├── templates/
│   │   └── report.jinja2
│   ├── styles/
│   │   ├── report.css
│   │   └── report_print.css
│   └── assets/
│       └── fonts/
│
├── llm/                            # ТРЕК B (инфраструктура)
│   ├── client.py                   # LLMClient + autodetect
│   └── endpoints.py                # EndpointManager
│
├── config/                         # Конфиги
│   ├── loader.py                   # ConfigManager
│   └── weights_manager.py          # WeightsManager + профили
│
├── state/                          # Streamlit-обёртки
│   ├── singletons.py               # cache_resource singletons
│   └── session.py                  # session_state helpers
│
├── configs/                        # Данные конфигурации (YAML)
│   ├── weights/
│   │   ├── default.yaml
│   │   └── profiles/*.yaml
│   ├── agent_config.yaml
│   ├── report_config.yaml
│   ├── rag_config.yaml
│   ├── ingestion_config.yaml
│   ├── llm_endpoints.yaml
│   ├── eval_dataset.yaml
│   ├── field_synonyms.yaml
│   ├── model_aliases.yaml
│   ├── component_vocabulary.yaml
│   ├── error_code_patterns.yaml
│   └── schema_profiles/*.yaml
│
├── storage/                        # Runtime-данные
│   ├── qdrant/                     # Volume для Qdrant
│   ├── uploads/                    # Загруженные файлы
│   ├── reports/                    # Сгенерированные отчёты
│   ├── eval_history/
│   └── ingestion_checkpoints/
│
├── tests/                          # Тесты
│   ├── test_calculator.py          # Трек A
│   ├── test_agent.py               # Трек B
│   ├── test_report.py              # Трек C
│   ├── test_rag.py                 # Трек D
│   ├── test_ingestion.py           # Трек E
│   ├── e2e/
│   │   ├── test_full_batch.py
│   │   ├── test_rag_eval.py
│   │   └── test_report_rendering.py
│   ├── visual/
│   │   └── test_report_rendering.py
│   └── fixtures/
│
└── docs/                           # Документация
    ├── architecture_overview.md    # Этот документ
    ├── track_A_health_index_formula.md
    ├── track_B_agent_prompts_and_loop.md
    ├── track_C_report_layout.md
    ├── track_D_rag_base.md
    ├── track_E_ingestion.md
    └── vscode_llm_playbook.md
```

---

## 5. Агентный поток — две основные цепочки

### 5.1 Batch-режим — расчёт индексов из файла

```
┌──────────────────────────────────────────────────────────────────┐
│ Пользователь загружает файл через страницу Dashboard/Загрузка    │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ IngestionPipeline (E):                                           │
│   parse → map → normalize → factor_store.freeze()                │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Для каждого device_id из factor_store:                           │
│                                                                   │
│   Agent.run_batch(device_id, context):                           │
│     │                                                             │
│     ├─> LLM: Plan                                                │
│     │                                                             │
│     ├─> Tool: get_device_events            → events              │
│     ├─> Tool: get_device_resources         → resources           │
│     │                                                             │
│     ├─> Для каждого уникального error_code:                      │
│     │     Tool: classify_error_severity   → {severity, ...}      │
│     │       │                                                     │
│     │       └─> RAG.search (D): поиск по коду + модели           │
│     │                                                             │
│     │     Tool: count_error_repetitions    → n                   │
│     │                                                             │
│     ├─> Tool: calculate_health_index (A)  → HealthResult         │
│     │                                                             │
│     └─> LLM: Reflection                                          │
│           └─> verdict: approved | needs_revision                 │
│                                                                   │
│   [При needs_revision — до 2 попыток с замечаниями]              │
│                                                                   │
│   Сохранение trace в session_state                               │
│   Сохранение памяти (learned_patterns) если есть                 │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Все HealthResult собраны. ReportBuilder.build():                 │
│   • fleet_summary (из результатов)                               │
│   • executive_summary (LLM-вызов)                                │
│   • top_patterns (find_similar_devices)                          │
│   • per-device cards                                             │
│   • calculation_snapshot                                         │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Рендер HTML в UI + кнопка "Скачать PDF"                          │
└──────────────────────────────────────────────────────────────────┘
```

**Ожидаемые характеристики:**
- 100 устройств × 3–5 tool-вызовов × ~2 сек = 5–10 минут на локальной LLM.
- 3–8 LLM-вызовов на устройство (план, классификации, reflection).
- Пользователь видит прогресс-бар с текущим шагом.

### 5.2 Chat-режим — диалог с агентом

```
┌──────────────────────────────────────────────────────────────────┐
│ Пользователь задаёт вопрос на странице 3                         │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Agent.run_chat(user_message, context):                           │
│                                                                   │
│   context = {                                                    │
│     current_report, conversation_history,                        │
│     factor_store, fleet_stats                                    │
│   }                                                              │
│                                                                   │
│   LLM: anal    izer                                              │
│     │                                                             │
│     ├─> Если вопрос требует данных:                              │
│     │     Tool: get_device_events / get_fleet_statistics / ...   │
│     │                                                             │
│     ├─> Если вопрос технический:                                 │
│     │     Tool: search_service_docs → RAG                        │
│     │                                                             │
│     └─> Сбор финального текстового ответа                        │
│                                                                   │
│   [Reflection НЕ запускается — в chat не нужен]                  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Ответ в UI + expandable trace (для инженера)                     │
└──────────────────────────────────────────────────────────────────┘
```

**Типичные сценарии:**

| Вопрос | Инструменты | LLM-вызовов |
|---|---|---|
| «Что такое индекс здоровья?» | — | 1 |
| «Почему MFP-042 в красной зоне?» | get_device_events, get_device_history | 2–3 |
| «Сколько устройств Kyocera в красной зоне?» | get_fleet_statistics с фильтром | 2 |
| «Как исправить C6000?» | search_service_docs | 2 |
| «Покажи устройства с той же проблемой, что у MFP-042» | find_similar_devices | 2 |

---

## 6. Межмодульные контракты

Эти структуры данных — **интерфейсы** между модулями. Изменение любой из них затрагивает несколько треков и требует аккуратности.

### 6.1 `Factor` — контракт A ↔ B ↔ E

```python
@dataclass(frozen=True)
class Factor:
    error_code: str
    severity_level: SeverityLevel       # из RAG (через classify_error_severity)
    S: float                            # из weights[severity_level]
    n_repetitions: int                  # из factor_store
    R: float                            # рассчитан из n (Calculator)
    C: float                            # рассчитан из modifiers
    A: float                            # рассчитан из age_days
    event_timestamp: datetime           # из factor_store
    age_days: int                       # рассчитан
    applicable_modifiers: list[str]     # из правил C
    source: str                         # из RAG (ссылка на документ)
    confidence_flags: list[str]         # для confidence
```

**Кто создаёт:** Agent (собирает из tools).
**Кто потребляет:** Calculator.
**Кто владеет определением:** трек A (`data_io/models.py`).

### 6.2 `HealthResult` — контракт A ↔ C

```python
@dataclass(frozen=True)
class HealthResult:
    device_id: str
    health_index: int                   # 1..100
    confidence: float                   # 0.2..1.0
    zone: Zone                          # green | yellow | red
    confidence_zone: ConfidenceZone
    factor_contributions: list[FactorContribution]
    confidence_reasons: list[str]
    calculation_snapshot: CalculationSnapshot
    calculated_at: datetime
    reflection_notes: str | None
    flag_for_review: bool
```

**Кто создаёт:** Calculator + Agent (reflection дополняет).
**Кто потребляет:** ReportBuilder.

### 6.3 `FactorStore` — контракт E ↔ B

```python
class FactorStore:
    # Для ingestion
    def add_events(device_id: str, events: list[NormalizedEvent]) -> None
    def set_resources(device_id: str, snapshot: ResourceSnapshot) -> None
    def set_device_metadata(device_id: str, meta: DeviceMetadata) -> None
    def freeze() -> None
    
    # Для агента (через tools)
    def get_events(device_id: str, window_days: int) -> list[NormalizedEvent]
    def get_resources(device_id: str) -> ResourceSnapshot | None
    def count_repetitions(device_id: str, error_code: str, window_days: int) -> int
    def list_devices() -> list[str]
    def get_device_metadata(device_id: str) -> DeviceMetadata
```

**Иммутабельность после freeze()** — гарантия того, что при смене весов через UI factor-store не повреждается. Пересчёт на новых весах работает поверх того же store.

### 6.4 `SearchResult` — контракт D ↔ B

```python
@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    document_id: str
    text: str
    score: float                        # финальный rerank score
    dense_score: float | None
    sparse_score: float | None
    payload: dict                       # весь payload Qdrant
    
    @property
    def source_ref(self) -> str:
        """Человекочитаемая ссылка (doc_title, section, page)."""
```

**Кто создаёт:** HybridSearcher.
**Кто потребляет:** tools `search_service_docs` и `classify_error_severity`.

### 6.5 `Trace` — контракт B ↔ C

```python
@dataclass
class Trace:
    session_id: str
    mode: Literal['batch', 'chat']
    device_id: str | None
    user_query: str | None
    started_at: datetime
    ended_at: datetime | None
    steps: list[TraceStep]
    final_result: dict
    total_tool_calls: int
    total_llm_calls: int
    total_tokens: int
    attempts: int
    flagged_for_review: bool
```

**Кто создаёт:** Agent.
**Кто потребляет:** ReportBuilder (секция 7 отчёта), UI (страница 3 — трассировка).

### 6.6 `WeightsProfile` — контракт A ↔ C ↔ UI

Единый профиль, с которым работают:
- Calculator (использует значения для формулы).
- ReportBuilder (кладёт snapshot в отчёт).
- Страница 2 UI (отображает и редактирует).

Определение — см. трек A, раздел «Конфигурационные параметры».

### 6.7 `Report` — контракт C ↔ UI

Финальная структура отчёта. Все поля подробно — в треке C, раздел «Data model».

---

## 7. Словари-мосты между треками

Эти YAML-файлы — **единый источник правды** для нескольких треков одновременно. Расхождения между ними ломают работу.

### 7.1 `component_vocabulary.yaml` — мост A ↔ D

Канонические имена узлов МФУ. Используется:
- В формуле C (правила `applies_to_components` в weights.yaml).
- При индексации RAG (поле `components` в payload).
- При классификации ошибок (`affected_components` из `classify_error_severity`).

```yaml
canonical_components:
  fuser:
    aliases: ["фьюзер", "fuser", "heat roller", "блок закрепления"]
    category: fusing
  drum:
    aliases: ["барабан", "drum", "photo conductor", "фоторецептор"]
    category: imaging
  scanner:
    aliases: ["сканер", "scanner", "ADF"]
    category: scanning
```

Если в RAG индексирован чанк с упоминанием «фьюзера», он помечается `components: ["fuser"]`. Если формула C имеет правило `applies_to_components: [fusing]`, а компонент `fuser` относится к категории `fusing` — правило применится.

### 7.2 `model_aliases.yaml` — мост D ↔ E

Канонические имена моделей МФУ. Используется:
- В ingestion (нормализация поля `model`).
- В RAG (payload `models`).
- В Agent при вызове `search_service_docs(model=...)`.

Без этого словаря агент не сможет находить документацию: в данных модель как `"3253ci"`, а в индексированной документации как `"Kyocera TASKalfa 3253ci"`.

### 7.3 `error_code_patterns.yaml` — мост D ↔ E

Регулярки для извлечения кодов ошибок. Используется:
- В ingestion (нормализация `error_code`).
- В RAG (извлечение `error_codes` при индексации).

```yaml
patterns:
  - name: "Kyocera C-codes"
    regex: "\\bC\\d{4,5}\\b"
  - name: "HP E-codes"
    regex: "\\bE\\d{3}\\b"
  - name: "Service codes"
    regex: "\\bSC\\d{3}\\b"
```

### 7.4 `field_synonyms.yaml` — внутренний словарь E

Синонимы имён колонок для маппинга полей. Используется только в треке E. Но пополнение — живое, по мере появления новых источников данных.

---

## 8. LLM-вызовы в системе

Система делает LLM-вызовы **не только в агенте**. Вот все места, где они происходят.

| Место | Где | Частота | Prompt | Tool strategy |
|---|---|---|---|---|
| Planning и tool use | `agent/core.py: agent_loop` | Много раз на устройство | `system_batch.md` / `system_chat.md` | native / guided / react |
| Self-check | `agent/core.py: run_reflection` | 1 раз на устройство | `reflection.md` | guided (JSON response) |
| Severity classification | `tools/impl.py: ClassifyErrorSeverityTool` | 1 раз на уникальный код | `classify_severity.md` | guided |
| Field mapping | `data_io/field_mapper.py: LLMMatcher` | 1 раз на файл | `field_mapping.md` | guided |
| Identify model | `rag/ingestion.py` | 1 раз на документ | `identify_model.md` | guided |
| Content-type classification | `rag/ingestion.py` | Батчами на документ | inline prompt | guided |
| Executive summary | `reporting/report_builder.py` | 1 раз на отчёт | `executive_summary.md` | —  (text output) |
| Agent recommendation | `reporting/report_builder.py` | 1 раз на устройство | inline prompt | — (text) |

**Единый LLMClient** (трек B) используется везде. Параметры генерации настраиваются через `llm_config` для разных use-case (Agent batch температура 0.2, classify — 0.0, executive summary — 0.4).

**Оптимизация:** результаты `classify_error_severity` и `identify_model` кэшируются (один и тот же код ошибки на той же модели → один ответ).

---

## 9. Закрытие методики оценки агентности

Соответствие 7 лучам методики — **ключевая цель** проекта. Ниже — матрица, какой модуль закрывает какой луч.

| Луч | Вес | Уровень | Где закрывается |
|---|---|---|---|
| Role | 1 | 2 | `system_batch.md` + динамические дополнения по контексту устройства (роль адаптируется) |
| Reasoning | 3 | 2 | `agent/core.py: agent_loop` — план не детерминирован, меняется по ходу tool-вызовов |
| Reflection | 2 | 2 | `agent/core.py: run_reflection` — оценка всей цепочки, может триггерить перезапуск |
| Memory | 2 | 2 | `agent/memory.py` — агент сам решает, что сохранить в `learned_patterns` |
| Domain knowledge | 1 | 2 | `rag/` — сквозной RAG по нескольким коллекциям (service_manuals, internal_guides и т.д.) |
| Autonomy | 3 | 1 | Агент не спрашивает подтверждений на промежуточных шагах. Итоговое решение — за менеджером |
| Action | 2 | 2 | `agent/tools/registry.py` — структурированные вызовы через function calling / guided JSON |

**Ожидаемая сумма:** ≈ 81 балл.

**Путь до 100:** поднять Autonomy до 2 (агент сам помечает критичные МФУ как `flag_for_review`) и Action до 3 (возможность агента формировать новые SQL-запросы к БД — уже для прод-версии).

---

## 10. Deployment и окружение

### 10.1 Локальная разработка

```
Пользователь
    │
    ├─> Streamlit app (localhost:8501)
    │         │
    │         ├─> RAGEngine
    │         │    └─> Qdrant (localhost:6333 — Docker)
    │         │
    │         └─> LLMClient
    │              └─> Локальная LLM (localhost:8000 — vLLM/Ollama/etc.)
    │
    └─> Docker Compose: qdrant
```

Все компоненты работают локально. Docker нужен только для Qdrant.

### 10.2 Требования к ресурсам

| Компонент | RAM | Диск |
|---|---|---|
| Python процесс Streamlit | 500 МБ | — |
| BGE-M3 (embedder) | ~2.5 ГБ | 2.3 ГБ |
| BGE-reranker-v2-m3 | ~2.5 ГБ | 2.2 ГБ |
| Qdrant (50К чанков) | 2 ГБ | 1 ГБ |
| Локальная LLM | зависит от модели (7B — ~14 ГБ на CPU fp32 или ~7 ГБ на GPU fp16) | зависит |

**Итого без LLM:** 7–8 ГБ RAM. С LLM: 15–20 ГБ RAM (CPU) или 10 ГБ VRAM (GPU) + 7–8 ГБ RAM.

### 10.3 Переменные окружения

```
LLM_BASE_URL=http://localhost:8000/v1
LLM_API_KEY=dummy-for-local
LLM_MODEL=qwen2.5-7b-instruct

QDRANT_HOST=localhost
QDRANT_PORT=6333

LOG_LEVEL=INFO
APP_ENV=development
```

---

## 11. Порядок изменений — кто кого затрагивает

Частый вопрос при разработке: «если я поменяю X, что ещё нужно проверить?»

### 11.1 Изменение формулы индекса (трек A)

**Затрагивает:**
- `configs/weights/default.yaml` — добавить/изменить параметр.
- `tests/test_calculator.py` — обновить тесты.
- `pages/2_Weights.py` — добавить слайдер в UI.
- `reporting/templates/report.jinja2` — секция 8 «Конфигурация расчёта».

**Не затрагивает:** B, D, E.

### 11.2 Добавление нового инструмента агента (трек B)

**Затрагивает:**
- `agent/tools/impl.py` — реализация.
- `agent/tools/registry.py` — регистрация.
- `agent/prompts/system_batch.md` или `system_chat.md` — упоминание инструмента.

**Не затрагивает:** A, C, D, E — но если инструмент требует новых данных, возможно E.

### 11.3 Изменение структуры Report (трек C)

**Затрагивает:**
- `data_io/models.py` — структура `Report`.
- `reporting/report_builder.py` — заполнение.
- `reporting/templates/report.jinja2` — рендер.
- `reporting/styles/report.css` — если новая секция.

**Не затрагивает:** A, B, D, E.

### 11.4 Добавление коллекции в RAG (трек D)

**Затрагивает:**
- `configs/rag_config.yaml` — описание коллекции.
- `rag/ingestion.py` — возможно новая стратегия чанкинга.
- `pages/4_RAG_Admin.py` — UI управления.
- `configs/eval_dataset.yaml` — тестовые запросы для новой коллекции.

**Не затрагивает:** A, C, E. Агент (B) не меняется — он использует `search_service_docs` с параметром `collection`.

### 11.5 Добавление нового формата входного файла (трек E)

**Затрагивает:**
- `data_io/parsers.py` — новый парсер.
- `configs/ingestion_config.yaml` — список поддерживаемых форматов.
- `tests/test_ingestion.py` — тесты нового формата.

**Не затрагивает:** A, B, C, D — контракт `FactorStore` неизменен.

### 11.6 Добавление нового словарного элемента (component, model, код)

**Затрагивает все треки, где словарь используется.** Именно поэтому словари — в отдельных YAML-файлах и явно помечены как «мосты между треками» в разделе 7.

---

## 12. Критические риски системы

Из всех треков собраны **топ-10 рисков**, которые требуют особого внимания при разработке.

| # | Риск | Где описан | Защита |
|---|---|---|---|
| 1 | Несогласованность имён компонентов между RAG и формулой C | D, риск 11 | Единый `component_vocabulary.yaml` |
| 2 | Несогласованность имён моделей между ingestion и RAG | D, риск 4 / E, риск 12 | Единый `model_aliases.yaml` |
| 3 | Critical ×1.5×2 = индекс всегда 1 (ограничение потолка C) | A, риск не явный | Потолок C ≤ 1.5, правило «1 Critical в день» |
| 4 | Бесконечный цикл tool-вызовов агента | B, риск 1 | MAX_TOOL_CALLS, MAX_LLM_CALLS + forced termination |
| 5 | LLM вернула невалидный JSON | B, риск 3 | Guided JSON output + retry |
| 6 | Tool-strategy автодетекция ошибается | B, риск 5 | Fallback chain (native → guided → react) |
| 7 | Переполнение контекста LLM | B, риск 6 | Компрессия старых tool results |
| 8 | PDF не собирается из-за шрифтов | C, риск 1 | Шрифты в Docker-образе, WeasyPrint-deps |
| 9 | Большой файл не помещается в память | E, риск 1 | Жёсткий лимит max_size_mb |
| 10 | Ложное срабатывание профиля маппинга | E, риск 5 | Плашка «применён профиль X», возможность пересмотра |

---

## 13. Чек-лист перед сборкой

Перед финальной сборкой проекта убедиться, что:

**Конфигурация:**
- [ ] Все YAML-файлы в `configs/` валидируются через Pydantic.
- [ ] Дефолтные значения из track_*.md скопированы в YAML без расхождений.
- [ ] Словари-мосты (`component_vocabulary`, `model_aliases`) заполнены под целевые модели МФУ.

**Модели данных:**
- [ ] Все dataclass/pydantic-модели в `data_io/models.py`.
- [ ] Поля `frozen=True` для всех «snapshot»-структур (HealthResult, Factor, SearchResult).
- [ ] `__all__` экспортирует нужные типы.

**Агент:**
- [ ] 9 tool-функций реализованы и зарегистрированы.
- [ ] Все 5 промптов в `agent/prompts/` соответствуют track_B.
- [ ] Self-check включён в batch, выключен в chat.
- [ ] Memory пишется после approved reflection.

**RAG:**
- [ ] 4 коллекции в Qdrant создаются на старте.
- [ ] Payload-индексы созданы.
- [ ] BGE-M3 и reranker скачиваются при первом запуске.
- [ ] Eval-датасет содержит минимум 50 запросов с разметкой.

**UI:**
- [ ] Все 4 страницы работают.
- [ ] Singletons через `@st.cache_resource`.
- [ ] Session state обёрнут в `state/session.py`.
- [ ] Graceful handling пустых состояний.

**Тесты:**
- [ ] Unit-тесты по всем 5 трекам зелёные.
- [ ] E2E-тесты проходят на fixture-данных.
- [ ] Coverage ≥ 80% на критичных модулях (calculator, agent, rag).
- [ ] Visual regression tests для отчёта зелёные.

**Операционное:**
- [ ] Docker-compose запускает Qdrant.
- [ ] `make install` устанавливает всё без ошибок.
- [ ] `make run` открывает Streamlit.
- [ ] README содержит quickstart на 3 команды.

---

## 14. Глоссарий

| Термин | Значение |
|---|---|
| **Индекс здоровья (H)** | Число 1–100, интегральная оценка состояния МФУ |
| **Confidence** | 0.2–1.0, уверенность агента в корректности индекса |
| **Factor** | Один «штрафной фактор» для формулы: (S, R, C, A) |
| **Factor-store** | In-memory хранилище нормализованных данных, интерфейс для агента |
| **Trace** | Запись всех шагов одной цепочки работы агента |
| **Zone** | Цветовая зона индекса: green (≥75) / yellow (40–74) / red (<40) |
| **Pattern** | Сгруппированная проблема: массовая / локационная / одиночная |
| **Profile** | Именованный набор параметров (веса или маппинг колонок) |
| **Signature** | Хэш набора колонок для автомэппинга профиля |
| **Collection** | Именованный индекс в Qdrant |
| **Chunk** | Фрагмент документа, помещаемый в коллекцию |
| **Hybrid search** | Поиск dense + sparse с RRF-слиянием |
| **Reranker** | Cross-encoder, переранжирующий кандидатов из hybrid |
| **Reflection** | Самопроверка агента отдельным LLM-вызовом |
| **Tool strategy** | Способ вызова tools: native / guided JSON / ReAct |

---

## 15. Итог

Система состоит из 5 функциональных треков, связанных через **явные контракты** (6 ключевых структур данных) и **словари-мосты** (4 YAML-файла единой правды). Архитектура допускает развитие каждого слоя независимо, при условии соблюдения контрактов.

Главный принцип, пронизывающий всю систему — **детерминированное ядро плюс ИИ-оболочка**. Математика индекса (трек A) — чистая функция. Агент (трек B) подготавливает входы для этой функции, вызывая инструменты. Отчёт (трек C) — рендер выхода. Это разделение делает систему объяснимой для менеджеров и тестируемой для инженеров.

Ожидаемая оценка по методике агентности: **≈ 81 балл из 100**, с опцией подъёма до 100 во второй очереди.

Для старта реализации — использовать `vscode_llm_playbook.md`, идти последовательно от фазы 0 к фазе 8.
