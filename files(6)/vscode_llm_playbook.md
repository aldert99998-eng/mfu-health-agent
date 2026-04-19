# Playbook для VS Code — промпты и контексты по этапам разработки

## Назначение документа

Это **рабочая инструкция для LLM-ассистента в VS Code** (Cursor, Copilot Chat, Cline, Claude Code, Continue и т.д.). Для каждого этапа разработки указано:

- Какие файлы прикрепить в контекст.
- Какой промпт использовать.
- Какие файлы НЕ прикреплять (чтобы не забить контекст).
- Что проверить после генерации.

## Как пользоваться

1. **Выбрать этап** из оглавления ниже.
2. **Прикрепить указанные файлы** в chat (функция @-attach или drag-and-drop в контекст).
3. **Скопировать промпт** и отправить.
4. **Проверить результат** по критериям из раздела «Проверка».
5. Переходить к следующему этапу.

**Принцип:** один промпт = один результат (модуль, набор функций, набор тестов). Не пытаться сгенерировать пол-проекта за один запрос.

## Оглавление этапов

### Фаза 0 — Подготовка
- Этап 0.1 — Инициализация проекта
- Этап 0.2 — Базовая инфраструктура (docker-compose, зависимости)

### Фаза 1 — Фундамент
- Этап 1.1 — Pydantic-модели и dataclasses (скелет типов)
- Этап 1.2 — Конфиги и их загрузка
- Этап 1.3 — Factor-store (in-memory хранилище данных)

### Фаза 2 — Ядро трека A
- Этап 2.1 — Health calculator (чистая функция формулы)
- Этап 2.2 — Тесты калькулятора
- Этап 2.3 — Управление профилями весов

### Фаза 3 — Ingestion (трек E)
- Этап 3.1 — Детекция формата и парсинг
- Этап 3.2 — Семантический маппинг полей
- Этап 3.3 — Нормализация и валидация

### Фаза 4 — RAG (трек D)
- Этап 4.1 — Qdrant-клиент и коллекции
- Этап 4.2 — Эмбеддинги (BGE-M3)
- Этап 4.3 — Ingestion-pipeline для документов
- Этап 4.4 — Hybrid search + reranker
- Этап 4.5 — Evaluation и метрики

### Фаза 5 — Агент (трек B)
- Этап 5.1 — LLM-клиент с автодетекцией возможностей
- Этап 5.2 — Tool registry и tool-функции
- Этап 5.3 — Agent loop (batch-режим)
- Этап 5.4 — Self-check (reflection)
- Этап 5.5 — Agent loop (chat-режим)
- Этап 5.6 — Trace и сохранение памяти

### Фаза 6 — Отчёт (трек C)
- Этап 6.1 — Data model отчёта и ReportBuilder
- Этап 6.2 — Jinja2-шаблон и CSS
- Этап 6.3 — PDF-генерация через WeasyPrint

### Фаза 7 — Streamlit UI
- Этап 7.1 — Shared state и singletons
- Этап 7.2 — Страница 1 — Дашборд
- Этап 7.3 — Страница 2 — Веса
- Этап 7.4 — Страница 3 — LLM и чат
- Этап 7.5 — Страница 4 — RAG-админ

### Фаза 8 — Интеграция и e2e-тесты
- Этап 8.1 — Полный сценарий batch-расчёта
- Этап 8.2 — Eval RAG на реальных данных
- Этап 8.3 — Визуальные тесты отчёта

---

## Универсальная инструкция (системный промпт)

**Рекомендуется добавить в настройки ассистента** как system prompt или rules. Если нет такой возможности — вставлять в начало каждого чата.

```
Ты — Python-разработчик, работающий над ИИ-агентом мониторинга
парка МФУ. Проект описан в комплекте документов track_*.md и
architecture_overview.md.

Правила работы:

1. Прежде чем писать код — прочитай прикреплённые к чату
   документы. Если в них есть ответ на вопрос, используй его.
   Не додумывай. Если не хватает информации — спрашивай.

2. Следуй структуре проекта из architecture_overview.md. Не
   создавай новые директории без необходимости.

3. Для всех dataclass'ов и конфигов используй Pydantic v2.
   Для типов — стандартный typing + pydantic.BaseModel.

4. Каждая функция должна иметь:
   - Type hints для всех параметров и возврата.
   - Docstring в стиле Google (Args, Returns, Raises).
   - Обработку исключений с информативными сообщениями.

5. Код пишется на Python 3.11+. Используй современный синтаксис:
   - match/case вместо if/elif/else где уместно
   - X | Y вместо Union[X, Y]
   - list[X] вместо List[X]

6. Не добавляй эмодзи в код. Не добавляй ASCII-арт.
   Комментарии — только по делу, объясняют «почему», а не «что».

7. После генерации модуля — предложи к нему тесты pytest,
   основываясь на тест-кейсах из прикреплённых track_*.md.

8. Если увидел противоречие между документами — остановись и
   укажи на него, не выбирай сам.
```

---

## Фаза 0 — Подготовка

### Этап 0.1 — Инициализация проекта

**Цель:** создать структуру директорий, `pyproject.toml`, `.gitignore`, `README.md`.

**Прикрепить:**
- `architecture_overview.md`
- `README.md` (если уже есть заготовка)

**НЕ прикреплять:** track_*.md (на этом этапе детали не нужны).

**Промпт:**
```
На основе architecture_overview.md создай:

1. Структуру директорий проекта (командой mkdir -p) согласно
   разделу «Структура проекта».

2. pyproject.toml с:
   - Python >= 3.11
   - Зависимостями: streamlit, pydantic v2, qdrant-client,
     sentence-transformers, openai, PyYAML, jinja2, weasyprint,
     pymupdf, pandas, tiktoken, pytest, pytest-asyncio, python-dateutil
   - dev-зависимостями: ruff, mypy, pytest-cov
   - Конфигурацией ruff (линтер) и mypy

3. .gitignore для Python-проекта с дополнениями для нашего
   случая: storage/qdrant/, storage/uploads/, .env, *.log,
   storage/reports/, models/, __pycache__, .venv

4. README.md с:
   - Кратким описанием проекта
   - Requirements (Python, Docker)
   - Quickstart (3 команды)
   - Ссылками на track_*.md

Не создавай код модулей — только структуру и конфиги.
```

**Проверка:**
- `tree -L 2` показывает все директории из architecture_overview.md.
- `pip install -e .` проходит без ошибок.
- `.gitignore` содержит все критичные паттерны.

---

### Этап 0.2 — Базовая инфраструктура

**Цель:** `docker-compose.yml` для Qdrant, `.env.example`, скрипты запуска.

**Прикрепить:**
- `architecture_overview.md`
- `track_D_rag_base.md` (раздел «Dependencies» и «Размер ресурсов»)

**Промпт:**
```
Создай инфраструктуру для локального запуска:

1. docker-compose.yml с сервисом Qdrant:
   - Образ qdrant/qdrant:v1.11.0
   - Порты 6333 (HTTP) и 6334 (gRPC)
   - Volume для персистентности в storage/qdrant/
   - Healthcheck

2. .env.example с переменными:
   - LLM_BASE_URL (пример: http://localhost:8000/v1)
   - LLM_API_KEY (может быть dummy для локальной модели)
   - LLM_MODEL (имя модели)
   - QDRANT_HOST, QDRANT_PORT
   - LOG_LEVEL

3. Makefile с командами:
   - make install — установка зависимостей
   - make up — docker-compose up -d
   - make down — stop
   - make run — запуск streamlit
   - make test — pytest
   - make lint — ruff check + mypy
   - make format — ruff format

Ориентируйся на минимальный demo-stack, без production-плюшек
(мониторинг, backup и т.п.).
```

**Проверка:**
- `docker-compose up -d` запускает Qd- `tree -L 2` показывает все директории из architecture_overview.md.
- `pip install -e .` проходит без ошибок.
- `.gitignore` содержит все критичные паттерны.rant, `curl localhost:6333/` возвращает JSON.
- `make install` работает.
- `.env.example` содержит все нужные переменные, в `.env` реальные значения (не коммитится).

---

## Фаза 1 — Фундамент

### Этап 1.1 — Pydantic-модели и dataclasses

**Цель:** все типы данных проекта — в одном месте, до того как пишется логика. Это каркас, от которого дальше всё пляшет.

**Прикрепить:**
- `track_A_health_index_formula.md`
- `track_B_agent_prompts_and_loop.md`
- `track_C_report_layout.md`
- `track_E_ingestion.md`

**НЕ прикреплять:** track_D (типы RAG создадим отдельно на этапе 4).

**Промпт:**
```
На основе прикреплённых документов создай единый модуль с
моделями данных в data_io/models.py.

Включи ВСЕ dataclass/pydantic модели, которые упомянуты в
разделах «Интерфейсы модулей» каждого трека:

Из track_A:
- Factor
- ConfidenceFactors
- HealthResult
- FactorContribution
- WeightsProfile (с вложенными моделями для severity, context и т.д.)

Из track_B:
- TraceStep, Trace
- BatchContext, ChatContext
- LearnedPattern

Из track_C:
- Report, FleetSummary, PatternGroup
- DeviceReport, ResourceState, DocReference
- CalculationSnapshot, AgentTraceSummary, RAGSummary

Из track_E:
- NormalizedEvent
- ResourceSnapshot
- SchemaProfile
- IngestionResult

Требования:
- Используй Pydantic v2 BaseModel для всех моделей.
- model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)
- Валидаторы для критичных полей (например, health_index: int,
  ge=1, le=100).
- Enum-поля (zone, severity_level, confidence_zone, pattern_type)
  через StrEnum.
- Docstring у каждого класса — одно предложение.

В конце файла добавь __all__ со списком экспортируемых типов.

Не реализуй методы кроме валидаторов и property. Вся логика
будет в других модулях.
```

**Проверка:**
- `mypy data_io/models.py` проходит без ошибок.
- Каждая модель имеет пример в docstring или в отдельном файле `tests/fixtures/models_examples.py`.
- Валидаторы ловят некорректные значения (быстрый тест: создать модель с `health_index=150`, должна упасть).

---

### Этап 1.2 — Конфиги и их загрузка

**Цель:** `ConfigManager`, который загружает все YAML'ы проекта, валидирует их и даёт типобезопасный доступ.

**Прикрепить:**
- `track_A_health_index_formula.md` (раздел «Конфигурационные параметры»)
- `track_B_agent_prompts_and_loop.md` (раздел «Конфигурационные параметры»)
- `track_C_report_layout.md` (раздел «Конфигурационные параметры»)
- `track_D_rag_base.md` (раздел «Конфигурационные параметры»)
- `data_io/models.py` (созданный на предыдущем этапе)

**Промпт:**
```
Создай систему загрузки конфигов в config/loader.py.

Требования:

1. Для каждого YAML-файла из track_*.md — соответствующая
   Pydantic-модель:
   - WeightsProfile (configs/weights/*.yaml)
   - AgentConfig (configs/agent_config.yaml)
   - ReportConfig (configs/report_config.yaml)
   - RAGConfig (configs/rag_config.yaml)
   - LLMEndpoint (элементы configs/llm_endpoints.yaml)
   - SchemaProfile (configs/schema_profiles/*.yaml)

2. Класс ConfigManager:
   - Singleton (но без global state — через @lru_cache на уровне
     фабричной функции get_config_manager()).
   - Методы: load_weights(profile_name), load_agent_config(),
     load_report_config(), load_rag_config(), list_profiles(),
     save_weights_profile().
   - Чтение YAML через PyYAML с safe_load.
   - Валидация каждого YAML через соответствующую модель.
   - Ошибки загрузки — с понятным сообщением о том, что не так.

3. Создай файлы дефолтов:
   - configs/weights/default.yaml — значения из track_A
   - configs/agent_config.yaml — значения из track_B
   - configs/report_config.yaml — значения из track_C
   - configs/rag_config.yaml — значения из track_D

Значения в YAML должны ТОЧНО соответствовать дефолтам из
соответствующих track_*.md — это наш ground truth.

Добавь класс ConfigValidationError с информативными
сообщениями для пользователя.
```

**Проверка:**
- `python -c "from config.loader import get_config_manager; get_config_manager().load_weights('default')"` — работает.
- Намеренно сломанный YAML даёт внятную ошибку с указанием поля.
- Значения в YAML совпадают с числами из track_*.md.

---

### Этап 1.3 — Factor-store

**Цель:** in-memory хранилище нормализованных данных. Интерфейс, через который агент и калькулятор получают данные об устройствах.

**Прикрепить:**
- `track_A_health_index_formula.md` (раздел про иммутабельность factor-store)
- `track_B_agent_prompts_and_loop.md` (tools `get_device_events`, `get_device_resources`)
- `track_E_ingestion.md`
- `data_io/models.py`

**Промпт:**
```
Создай модуль data_io/factor_store.py.

Класс FactorStore — in-memory хранилище с интерфейсом, через
который агент обращается к данным устройств.

Требования:

1. Хранит:
   - events: dict[device_id, list[NormalizedEvent]]
   - resources: dict[device_id, ResourceSnapshot]
   - device_metadata: dict[device_id, DeviceMetadata]
   - fleet_meta: общая метаинформация (хэш файла, timestamp
     загрузки, название профиля маппинга)

2. Методы-геттеры (соответствуют tool-функциям агента):
   - get_events(device_id, window_days=30) → list
   - get_resources(device_id) → ResourceSnapshot | None
   - count_repetitions(device_id, error_code, window_days=14) → int
   - list_devices() → list[str]
   - get_device_metadata(device_id) → DeviceMetadata

3. Методы для ingestion (только из трека E):
   - add_events(device_id, events)
   - set_resources(device_id, snapshot)
   - set_device_metadata(...)

4. После завершения ingestion вызывается freeze() — делает
   store иммутабельным (внутренние словари заменяются на
   frozen-варианты). После freeze попытка add_* возвращает
   FactorStoreFrozenError.

5. Метод to_dict() для сериализации, from_dict() для загрузки.
   Нужно для снапшотов отчётов.

6. Оконные запросы (events в window_days): используй
   datetime.now(timezone.utc) как точку отсчёта по умолчанию,
   но дай возможность переопределить (для тестов и
   воспроизводимости).

Не используй асинхронность — store синхронный. Параллелизм
на уровне агента будет добавляться позже (не в демо).
```

**Проверка:**
- Тесты: заполнить store, freeze, попытаться добавить → ошибка.
- Тесты: оконные запросы работают корректно на граничных значениях (ровно 30 дней).
- Сериализация to_dict/from_dict — roundtrip без потерь.

---

## Фаза 2 — Ядро трека A

### Этап 2.1 — Health calculator

**Цель:** чистая функция расчёта индекса. Никаких LLM, никаких вызовов вовне.

**Прикрепить:**
- `track_A_health_index_formula.md`
- `data_io/models.py`
- `config/loader.py`

**Промпт:**
```
Реализуй tools/calculator.py — детерминированный калькулятор
индекса здоровья.

Основная функция:

def calculate_health_index(
    factors: list[Factor],
    confidence_factors: ConfidenceFactors,
    weights: WeightsProfile,
    silent_device_mode: str = "data_quality"
) -> HealthResult:
    ...

Реализация строго по разделам track_A:
- Формула H из раздела «1. Формула индекса здоровья»
- R из раздела «3. Коэффициент повторяемости»
- C из раздела «4. Контекстный модификатор» (с правилами
  применения по типу ошибки)
- A из раздела «5. Коэффициент давности»
- Правило одного Critical в день (раздел 2)
- Формула confidence (раздел 6)
- Режимы silent device (раздел 7)
- Определение zone по порогам (раздел 8)

Важные детали:

1. Функция — ЧИСТАЯ. Получает всё на вход, возвращает результат.
   Никаких глобальных state'ов, никаких обращений к LLM/RAG.

2. Каждый фактор в HealthResult.factor_contributions должен
   содержать все компоненты (S, R, C, A) для объяснения в
   отчёте.

3. Добавь вспомогательные функции:
   - compute_R(n_repetitions, base, max_value) → float
   - compute_C(applicable_modifiers, weights, max_value) → float
   - compute_A(age_days, tau_days) → float
   - select_one_critical_per_day(factors) → list[Factor]

4. Все числа — Decimal или float? Используй float, но финальный
   health_index округляется до int через round().

5. Обработка edge cases:
   - Пустой factors + silent_device_mode='optimistic' → H=100, Conf=1.0
   - Пустой factors + silent_device_mode='data_quality' → H=100, Conf=0.9
   - Сумма penalty > 99 → H=1 (min)
   - Произведение confidence < 0.2 → Conf=0.2 (min)

Тесты позже. Сейчас — только код модуля.
```

**Проверка:**
- Прогнать вручную сценарии из track_A раздела «Проверим на реалистичных сценариях»:
  - Здоровое устройство → H=100
  - 3 замятия за неделю → H≈74
  - Critical + fuser 95% → H=10 (с новой калибровкой)
  - Плохие данные → Conf ≈ 0.29

---

### Этап 2.2 — Тесты калькулятора

**Цель:** полный pytest-файл, покрывающий все тест-кейсы из track_A.

**Прикрепить:**
- `track_A_health_index_formula.md` (раздел «Тест-кейсы»)
- `tools/calculator.py`
- `data_io/models.py`

**Промпт:**
```
Создай tests/test_calculator.py.

Реализуй все тест-кейсы TC-A-1 до TC-A-20 из track_A.md.

Требования:

1. Используй pytest.
2. Для каждого тест-кейса — отдельная функция test_tc_a_N_short_name.
3. В docstring теста — полное описание TC из документа.
4. Используй pytest.fixture для общих объектов:
   - weights_default (дефолтный WeightsProfile)
   - confidence_empty (пустой ConfidenceFactors)
5. Параметризуй тесты где уместно (например, TC-A-2 до TC-A-5
   проверяют одну и ту же логику с разными входами).
6. Используй math.isclose для сравнения float.
7. Для TC-A-18 (снапшот) — создай golden-файл
   tests/fixtures/golden_health_result.json.

Не используй моки. Калькулятор — чистая функция, моки не
нужны.

Запуск: pytest tests/test_calculator.py -v
Целевое покрытие: 100% строк calculator.py.
```

**Проверка:**
- `pytest tests/test_calculator.py -v` — все 20 тестов зелёные.
- `pytest --cov=tools.calculator` — покрытие ≥ 95%.
- Тесты падают, если намеренно сломать одну строчку в формуле (это проверка, что тесты не формальные).

---

### Этап 2.3 — Управление профилями весов

**Цель:** CRUD профилей весов с историей изменений.

**Прикрепить:**
- `track_A_health_index_formula.md`
- `config/loader.py`
- `data_io/models.py`

**Промпт:**
```
Создай модуль config/weights_manager.py.

Класс WeightsManager с методами:

1. list_profiles() → list[ProfileMeta]
   Возвращает список всех профилей в configs/weights/profiles/
   с метаданными (имя, дата создания, автор, дата изменения).

2. load_profile(name: str) → WeightsProfile
   Загружает профиль, валидирует через Pydantic.

3. save_profile(profile: WeightsProfile, author: str) → None
   Сохраняет в configs/weights/profiles/<name>.yaml.
   Автоматически дописывает запись в configs/weights/history.log
   с timestamp, автором, именем профиля, хэшем параметров.

4. compare_profiles(name_a: str, name_b: str) → list[Diff]
   Возвращает структурированный diff двух профилей:
   какие параметры изменились, с каких на какие значения.

5. reset_to_default() → WeightsProfile
   Возвращает дефолтный профиль (копию из configs/weights/default.yaml).

Класс ProfileMeta (Pydantic):
- name: str
- created_at: datetime
- modified_at: datetime
- author: str | None
- params_hash: str  # SHA-256 от канонического JSON параметров

История изменений — append-only, не редактируется.

Используй файловую блокировку (fcntl на Linux) при записи
профилей, чтобы два параллельных сохранения не испортили
историю. На macOS/Windows — fallback на advisory-locks.

ВАЖНО: history.log должен быть в формате JSONL — одна строка
= один JSON-объект. Это упростит чтение при большом количестве
записей.
```

**Проверка:**
- Создать профиль, изменить, прочитать историю — все записи в JSONL.
- compare_profiles показывает diff понятно.
- Попытка сохранить профиль с именем, содержащим `/` или `..` — ошибка (защита от path traversal).

---

## Фаза 3 — Ingestion (трек E)

### Этап 3.1 — Детекция формата и парсинг

**Прикрепить:**
- `track_E_ingestion.md` (раздел «Уровень 1»)
- `data_io/models.py`

**Промпт:**
```
Создай модуль data_io/parsers.py.

Реализуй:

1. class FormatDetector:
   - detect_format(file_path: Path) → FileFormat
     (enum: CSV, JSON, JSONL, XLSX, UNKNOWN)
   - Определяет по расширению + magic bytes + первым байтам.

2. class FileParser (абстрактный):
   - parse(file_path: Path) → ParsedData
     ParsedData = DataFrame + metadata (format, encoding, rows, cols).

3. Конкретные реализации:
   - CSVParser: через pandas + chardet для кодировки +
     csv.Sniffer для разделителя.
   - JSONParser: поддержка трёх структур (array, wrapped, JSONL),
     auto-detect. Вложенные объекты уплощаются через
     pandas.json_normalize.
   - XLSXParser: через openpyxl (только чтение). Первый лист
     по умолчанию.

4. Функция parse_file(file_path) → ParsedData — диспетчер по
   формату.

Обработка ошибок:
- UnsupportedFormatError — формат не распознан.
- EncodingError — не удалось определить кодировку.
- EmptyFileError — файл пустой.
- MalformedFileError — структура нарушена (битый JSON, разорванный CSV).

В каждой ошибке — конкретное сообщение на русском, что пошло
не так и что можно сделать (например, «проверьте, что файл
сохранён в UTF-8»).

Логируй все операции на уровне INFO (ibabel: размер, строки,
колонки после парсинга).
```

**Проверка:**
- Тест на всех типах файлов: cp1251-CSV с `;`, UTF-8 CSV с `,`, wrapped JSON, JSONL, XLSX.
- Битый JSON → MalformedFileError с указанием строки.

---

### Этап 3.2 — Семантический маппинг полей

**Прикрепить:**
- `track_E_ingestion.md` (раздел «Уровень 2»)
- `data_io/parsers.py`
- `data_io/models.py`
- `llm/client.py` (если уже готов; если нет — пометь TODO и верни stub-интерфейс)

**Промпт:**
```
Создай модуль data_io/field_mapper.py.

Реализуй четырёхступенчатый маппинг из track_E раздел 3.

1. class SynonymMatcher:
   - match(column_names: list[str]) → dict[col, target_field]
   - Использует configs/field_synonyms.yaml (создай этот
     файл с содержимым из track_E).
   - Нормализация перед сравнением: lowercase, замена
     пробелов/дефисов на `_`, транслитерация диакритики.

2. class ContentMatcher:
   - match(df: DataFrame, unmapped_cols: list[str]) →
     dict[col, target_field]
   - Проверяет содержимое колонок по правилам из track_E:
     * Regex для кодов ошибок [CJEF]\d{3,5}
     * Успешный парсинг dateutil → timestamp
     * Диапазоны 0-100 + имя содержит 'toner' → toner_level
     * Высокая уникальность → device_id
     * и т.д.

3. class LLMMatcher:
   - match(df: DataFrame, unmapped_cols: list[str]) →
     dict[col, target_field]
   - Принимает список колонок + 5-10 примеров значений каждой.
   - Делает ОДИН LLM-вызов с промптом из track_E.
   - Парсит JSON-ответ через Pydantic.
   - Если LLM недоступна — возвращает пустой маппинг,
     остальные колонки идут на ручное подтверждение.

4. class FieldMapper (координатор):
   - map(df: DataFrame) → MappingResult
     MappingResult содержит:
     - auto_mapping: dict[col, target_field]
     - confidence: dict[col, 'synonym' | 'content' | 'llm' | 'unknown']
     - unmapped: list[col]
   - Применяет три ступени последовательно.

5. Профили маппинга:
   - save_profile(mapping: MappingResult, df_columns: list[str],
     profile_name: str)
   - try_apply_profile(df: DataFrame) → MappingResult | None
     Считает signature (хэш нормализованных имён колонок),
     ищет совпадение в configs/schema_profiles/.

Важно: у класса LLMMatcher интерфейс должен позволить работать
БЕЗ LLM-клиента (для тестов). Через Protocol/ABC с mock'ом.

Промпт для LLMMatcher — вынеси в отдельный файл
agent/prompts/field_mapping.md. Там же — JSON-схема ответа
(для guided JSON output).
```

**Проверка:**
- Тестовые датасеты с разными именами колонок → корректный маппинг.
- LLMMatcher отрабатывает с заглушкой (ручной JSON-ответ в тесте).
- Профили сохраняются и применяются — повторный файл того же формата маппится без LLM.

---

### Этап 3.3 — Нормализация и валидация

**Прикрепить:**
- `track_E_ingestion.md` (раздел «Уровень 3»)
- `data_io/field_mapper.py`
- `data_io/factor_store.py`
- `data_io/models.py`

**Промпт:**
```
Создай data_io/normalizer.py.

Класс Normalizer с методом:

def normalize(
    df: DataFrame,
    mapping: dict[str, str],
    resource_unit_hints: dict | None = None
) -> NormalizationResult:

NormalizationResult содержит:
- valid_events: list[NormalizedEvent]
- valid_resources: dict[device_id, ResourceSnapshot]
- invalid_records: list[InvalidRecord] (с причиной)
- stats: dict (total, valid, invalid, warnings)

Нормализация по track_E раздел 4:

1. Парсинг timestamp — через dateutil.parser с fallback на
   известные форматы (dd.mm.yyyy, Unix timestamp).

2. Нормализация error_code — верхний регистр, удаление
   пробелов, дефисов, префиксов («Error», «Err»).

3. Нормализация model через configs/model_aliases.yaml.
   Если модель — алиас, заменяется на каноническую.

4. Автоопределение единиц ресурсов (raw vs percent):
   - Все значения в [0, 100] → percent.
   - Все в [0, 1] → fraction × 100.
   - Иначе → raw, флажок в ResourceSnapshot.unit_raw=True.

5. Pydantic-валидация каждой записи через NormalizedEvent.
   Ошибки — в invalid_records с текстом.

6. Разделение на события и ресурсы:
   - Запись с error_code → event.
   - Запись без error_code, но с ресурсами → resource snapshot.
   - Для device — последний snapshot по timestamp.

7. Заполнение factor_store:
   store.add_events(device_id, events)
   store.set_resources(device_id, snapshot)
   В конце — store.freeze()

Интеграционная функция:

def ingest_file(file_path: Path, factor_store: FactorStore) ->
    IngestionResult:
    # parse → map → normalize → fill store
    ...

IngestionResult возвращает summary для UI:
- total_records, valid, invalid
- mapping_used (profile name если применялся)
- warnings

В UI это показывается пользователю на шаге 4 wizard'а.

Все ошибки нормализации — с внятным сообщением, содержащим
номер строки и колонку.
```

**Проверка:**
- E2E тест с fixture-файлом: parse → map → normalize → factor_store.list_devices() возвращает правильные ID.
- Битые timestamps не падают, идут в invalid_records.
- Автоопределение единиц корректно (тест с fraction 0.45 vs percent 45).

---

## Фаза 4 — RAG (трек D)

### Этап 4.1 — Qdrant-клиент и коллекции

**Прикрепить:**
- `track_D_rag_base.md` (разделы 1, 2, 3)
- `config/loader.py`

**Промпт:**
```
Создай модуль rag/qdrant_client.py.

Требования:

1. class QdrantManager — обёртка над qdrant_client.QdrantClient.
   - Конструктор принимает RAGConfig.
   - Инициализирует client (prefer_grpc=True).

2. Метод ensure_collection(collection_name: str) → None:
   - Если коллекции нет — создаёт с:
     * named vectors: dense (size из конфига, distance=COSINE,
       HNSW параметры из конфига)
     * sparse vectors: sparse (modifier=IDF)
   - Создаёт payload-индексы для полей:
     models, vendor, error_codes, content_type, components,
     document_id (все keyword).

3. Метод ensure_all_collections() → None:
   - Применяет ensure_collection к каждой коллекции из RAGConfig.

4. Метод drop_collection(name) → None — с подтверждением
   через параметр confirm=True (иначе ошибка).

5. Метод collection_info(name) → CollectionInfo:
   - Количество point'ов, размер, статус индексации.

6. Метод healthcheck() → bool — проверка, что Qdrant доступен.

Используй qdrant_client.models для типобезопасности.

Обработка ошибок:
- QdrantUnavailableError — при падении соединения.
- CollectionExistsError — если пытаются создать существующую
  без флага recreate.

Логируй все операции на уровне INFO.
```

**Проверка:**
- `python -c "from rag.qdrant_client import QdrantManager; qm = QdrantManager(...); qm.ensure_all_collections()"` — в Qdrant появляются 4 коллекции.
- Просмотр через Qdrant UI (`http://localhost:6333/dashboard`) показывает корректную структуру.

---

### Этап 4.2 — Эмбеддинги

**Прикрепить:**
- `track_D_rag_base.md` (раздел 5)
- `config/loader.py`

**Промпт:**
```
Создай модуль rag/embeddings.py.

Класс BGEEmbedder:

1. Конструктор: принимает RAGConfig.embeddings.
   Загружает модель BAAI/bge-m3 через sentence-transformers.
   Определяет device (cpu/cuda) из конфига.
   При первом запуске модель кэшируется в models/.

2. Метод encode(texts: list[str], return_sparse: bool = True)
   → EmbeddingResult:
   EmbeddingResult:
   - dense: np.ndarray[n, 1024]
   - sparse: list[SparseVector] | None
   Батчами по config.batch_size.

3. Метод encode_query(query: str, return_sparse: bool = True)
   → tuple[np.ndarray, SparseVector | None]:
   Удобный метод для одного query.

4. Нормализация dense-векторов если config.normalize=True.

5. Метод embedding_version() → str:
   Возвращает «bge-m3@<version>» для записи в payload.
   Версия читается из model_card (sentence-transformers v.3.x).

Используй FlagEmbedding (не sentence-transformers) если
нужно sparse output — BGE-M3 через FlagEmbedding даёт и
dense и sparse одновременно.

Зависимости: pip install FlagEmbedding

Обработка ошибок:
- ModelLoadError — не удалось скачать/загрузить модель.
- OOMError — недостаточно памяти (с подсказкой уменьшить
  batch_size).

При первой загрузке модели — логируй прогресс скачивания.
```

**Проверка:**
- `encode(["тестовый текст"])` возвращает dense shape (1, 1024) + sparse.
- Векторы нормализованы (норма ≈ 1).

---

### Этап 4.3 — Ingestion pipeline

**Прикрепить:**
- `track_D_rag_base.md` (разделы 4, 8, 9, 12)
- `rag/qdrant_client.py`
- `rag/embeddings.py`

**Промпт:**
```
Создай rag/ingestion.py.

7-этапный pipeline из track_D раздела 8.

Основная функция:

def index_document(
    file_path: Path,
    collection: str,
    metadata_override: dict | None = None,
    progress_callback: Callable[[str, float], None] | None = None
) -> IndexingResult:
    ...

Внутренние этапы (каждый — отдельная функция):

1. parse_document(file_path) → ParsedDocument:
   Использует pymupdf (fitz). Сохраняет hash файла,
   извлекает bookmarks, текст постранично.

2. preprocess_text(doc) → CleanedDocument:
   Чистка OCR-артефактов, нормализация пробелов.

3. identify_model_and_vendor(doc) → ModelMetadata:
   LLM-вызов с промптом из agent/prompts/identify_model.md
   (создай этот файл). Guided JSON output.
   Fallback: извлечение из имени файла регуляркой.

4. chunk_document(doc, collection, config) → list[Chunk]:
   Применяет стратегию из RAGConfig.chunking[collection]:
   - hierarchical (для service_manuals)
   - per_record (для error_codes, parts_catalog)
   - recursive (для internal_guides)
   Использует tiktoken для подсчёта токенов.

5. enrich_metadata(chunks, model_meta) → list[EnrichedChunk]:
   - Извлечение error_codes регулярками
     (configs/error_code_patterns.yaml).
   - Извлечение components по словарю
     (configs/component_vocabulary.yaml).
   - Классификация content_type через LLM батчами по 15.
   - Применение PII-фильтра.

6. compute_embeddings(chunks) → list[EmbeddedChunk]:
   BGEEmbedder.encode() батчами.

7. upsert_to_qdrant(embedded_chunks, collection):
   - Удаление старых чанков с таким же document_id.
   - Upsert батчами по 100.

Progress callback вызывается на каждом этапе с именем и
процентом от 0 до 1.

Checkpoint-файлы: после каждого этапа сохраняй промежуточный
результат в storage/ingestion_checkpoints/<file_hash>_<step>.pkl.
При перезапуске — читай последний checkpoint, продолжай.

IndexingResult:
- document_id
- chunks_count
- errors (если были)
- duration_seconds
- metadata_identified (модель, вендор)
- embedding_version
```

**Проверка:**
- Индексация тестового PDF (short_test_manual.pdf в fixtures) проходит за < 2 минут.
- В Qdrant появились чанки с правильным payload.
- Повторный запуск — либо skip (если файл не изменился), либо перезапись.

---

### Этап 4.4 — Hybrid search + reranker

**Прикрепить:**
- `track_D_rag_base.md` (разделы 6, 7)
- `rag/qdrant_client.py`
- `rag/embeddings.py`

**Промпт:**
```
Создай rag/search.py и rag/reranker.py.

### rag/reranker.py

class BGEReranker:
- Конструктор: RAGConfig.reranker.
- Модель BAAI/bge-reranker-v2-m3 через FlagEmbedding.
- Метод rerank(query: str, candidates: list[Chunk], top_n: int)
  → list[ScoredChunk]: cross-encoder батчами, сортировка по
  score убыванием.

### rag/search.py

class HybridSearcher:
- Конструктор: QdrantManager, BGEEmbedder, BGEReranker, RAGConfig.

Метод search:

def search(
    query: str,
    collection: str,
    filters: dict | None = None,  # model, content_type и т.д.
    top_k: int = 8,
    use_reranker: bool = True
) -> list[SearchResult]:

Логика:

1. Построить Qdrant Filter из dict:
   - filters['model'] → match any из payload['models'].
   - filters['content_type'] → match payload['content_type'].
   - и т.д.

2. Получить dense + sparse векторы через embedder.encode_query.

3. Если Qdrant >= 1.10 — использовать Query API с Fusion:
   client.query_points(
       collection_name=collection,
       prefetch=[
           Prefetch(query=dense_vec, using="dense", limit=30),
           Prefetch(query=sparse_vec, using="sparse", limit=30,
                    query_filter=qfilter),
       ],
       query=FusionQuery(fusion=Fusion.RRF),
       limit=30,
       query_filter=qfilter
   )
   
   Иначе — два параллельных поиска + rrf_merge вручную
   (реализуй функцию rrf_merge).

4. Если use_reranker=True — reranker на top-30, выход top_n_output.

5. Вернуть list[SearchResult] с полями:
   chunk_id, document_id, text, score,
   dense_score, sparse_score, payload.

Проверь версию Qdrant на старте:
def _detect_qdrant_version(client) → tuple[int, int, int]

Логируй каждый запрос на DEBUG, с временем каждого этапа.

Если модель в фильтре не найдена в словаре алиасов — warning
в логах, поиск идёт без фильтра модели (лучше плохой результат,
чем пустой).
```

**Проверка:**
- Тесты на fixture-коллекции с 50 чанками: запрос по известному коду находит правильный чанк в top-3.
- Фильтр по модели ограничивает результаты только этой моделью.
- С reranker и без — разные порядки в top-8.

---

### Этап 4.5 — Evaluation и метрики

**Прикрепить:**
- `track_D_rag_base.md` (раздел 11)
- `rag/search.py`
- `data_io/models.py`

**Промпт:**
```
Создай rag/evaluation.py.

class RAGEvaluator:

1. Конструктор: HybridSearcher, путь к eval_dataset.yaml.

2. Метод run_eval() → EvalReport:
   - Загружает eval-датасет из YAML.
   - Для каждого query — запускает search с параметрами из
     датасета.
   - Вычисляет метрики:
     * Recall@5 (hit rate — есть ли expected_chunks в top-5)
     * Recall@10
     * MRR (1 / rank первого релевантного)
     * Precision@5
     * nDCG@10
   - Group by scenario — per-scenario метрики.

3. Метод save_report(report: EvalReport) → None:
   Сохраняет в storage/eval_history/<timestamp>.json.

4. Метод get_history(last_n: int = 10) → list[EvalReport]:
   Загружает последние N отчётов для графика.

5. Метод check_thresholds(report: EvalReport) → dict:
   Сравнивает с acceptance_thresholds из RAGConfig.
   Возвращает {metric: {value, threshold, passed}}.

6. Метод delta_vs_previous(report: EvalReport) → dict | None:
   Если есть предыдущий отчёт — вычисляет дельту по каждой
   метрике.

Pydantic-модели:
- EvalQuery (из YAML)
- QueryEvalResult (результат по одному query)
- EvalReport (агрегат)

Важно: eval должен быть детерминированным. Если параметры RAG
не менялись и датасет не менялся — метрики должны совпадать
между запусками.
```

**Проверка:**
- Запустить eval на пустой коллекции → все метрики 0, report сохранён.
- Загрузить документ, проиндексировать, прогнать eval → метрики > 0.
- Два последовательных прогона eval на одинаковой базе дают идентичные метрики.

---

## Фаза 5 — Агент (трек B)

### Этап 5.1 — LLM-клиент с автодетекцией

**Прикрепить:**
- `track_B_agent_prompts_and_loop.md` (раздел 10 про fallback)
- `config/loader.py`

**Промпт:**
```
Создай llm/client.py.

class LLMClient:

Обёртка над OpenAI Python SDK (используется для любого
OpenAI-совместимого endpoint'а).

1. Конструктор: LLMEndpointConfig (url, api_key, model, params).

2. Автодетекция tool_strategy — метод detect_capabilities():
   a) Попробовать chat.completions.create с tools=[...]
      (одна фейковая схема). Если получили tool_call в
      response — strategy = "native".
   b) Иначе — попробовать с response_format=json_schema или
      extra_body={"guided_json": {...}}. Если работает —
      strategy = "guided_json".
   c) Иначе — strategy = "react".
   Результат кэшируется в памяти и сохраняется в
   LLMEndpointConfig.tool_strategy (persistent).

3. Основной метод generate:

def generate(
    messages: list[Message],
    tools: list[ToolSchema] | None = None,
    response_schema: dict | None = None,
    params: GenerationParams | None = None
) -> LLMResponse:

LLMResponse:
- content: str (текстовый ответ)
- tool_calls: list[ToolCall] | None
- finish_reason: str
- usage: TokenUsage

В зависимости от tool_strategy метод по-разному формирует
запрос:
- native: tools передаются как-есть, tool_calls из response.
- guided_json: подмешивает в last message инструкцию «верни
  JSON по схеме», парсит текст как JSON.
- react: подмешивает инструкцию «используй формат Мысль/Действие/
  Параметры», парсит регулярками.

4. Метод для health-check: ping() → bool + латентность.

5. Обработка ошибок:
   - ConnectionError (недоступен endpoint)
   - TimeoutError
   - InvalidResponseError (модель вернула не то)
   - RateLimitError

Retry policy через tenacity:
- Сетевые ошибки: 3 попытки с exponential backoff.
- InvalidResponseError: 2 попытки с уточнением промпта.

LLMClient должен быть re-entrant (можно вызывать generate из
разных threads для chat).
```

**Проверка:**
- Ping на работающий endpoint → True, латентность < 500 мс.
- detect_capabilities корректно определяет стратегию для известной модели.
- С поддельным endpoint'ом (timeout) — ConnectionError с внятным сообщением.

---

### Этап 5.2 — Tool registry и tool-функции

**Прикрепить:**
- `track_B_agent_prompts_and_loop.md` (раздел 2)
- `tools/calculator.py`
- `rag/search.py`
- `data_io/factor_store.py`
- `llm/client.py`

**Промпт:**
```
Создай:
- agent/tools/registry.py
- agent/tools/impl.py

### registry.py

class ToolRegistry:
- register(tool: Tool) → None
- get_schema(name) → dict (JSON Schema из track_B раздел 2)
- get_all_schemas() → list[dict]
- execute(name: str, args: dict) → ToolResult

class Tool (Protocol):
- name: str
- schema: dict
- execute(args: dict) → ToolResult

class ToolResult:
- success: bool
- data: Any (JSON-сериализуемое)
- error: str | None

### impl.py

Реализуй ВСЕ 9 tool-функций из track_B раздел 2:

1. SearchServiceDocsTool — обёртка над HybridSearcher.search().
2. ClassifyErrorSeverityTool — композитный: делает search,
   потом LLM-классификатор с промптом из
   agent/prompts/classify_severity.md.
3. GetDeviceEventsTool — обёртка над FactorStore.get_events().
4. GetDeviceResourcesTool — обёртка над FactorStore.get_resources().
5. CountErrorRepetitionsTool — FactorStore.count_repetitions().
6. CalculateHealthIndexTool — обёртка над calculate_health_index().
7. GetFleetStatisticsTool — новая функция, собирает статистику
   по всему factor_store + результатам предыдущих расчётов.
8. FindSimilarDevicesTool — кластеризация по событиям/модели/
   локации. Реализация: наивное KNN по бинарным векторам
   «есть эта ошибка — нет». Для демо достаточно.
9. GetDeviceHistoryTool — история результатов расчёта для
   устройства (из in-memory кэша).

Каждая tool-функция:
- Pydantic-валидация args на входе.
- Обработка ошибок FactorStoreError, SearchError и т.д.
- Возврат ToolResult с data или error.

Промпты для композитных tools (classify_severity) — в
agent/prompts/*.md.

ClassifyErrorSeverityTool возвращает (при успехе):
{
  "severity": "Critical",
  "confidence": 0.9,
  "affected_components": [...],
  "source": "...",
  "reasoning": "..."
}

Если RAG не нашёл — дефолт Medium с confidence=0.3, source=null.

Регистрация: создай функцию register_all_tools(registry,
dependencies) которая регистрирует все 9.
```

**Проверка:**
- Для каждого tool — unit-тест с моком зависимостей.
- JSON-схемы всех 9 tools валидируются через jsonschema.
- execute несуществующего tool → UnknownToolError.

---

### Этап 5.3 — Agent loop (batch)

**Прикрепить:**
- `track_B_agent_prompts_and_loop.md` (разделы 3, 8, 11)
- `agent/tools/registry.py`
- `llm/client.py`

**Промпт:**
```
Создай:
- agent/prompts/system_batch.md (содержимое из track_B раздел 3)
- agent/prompts/reflection.md (из track_B раздел 6)
- agent/core.py — основной класс Agent

class Agent:

Конструктор: LLMClient, ToolRegistry, RAGEngine, FactorStore,
AgentConfig.

Метод run_batch:

def run_batch(
    self,
    device_id: str,
    context: BatchContext
) -> tuple[HealthResult, Trace]:

Логика из track_B раздела 8 (псевдокод):

```
attempt = 0
revision_notes = None
while attempt < MAX_ATTEMPTS:
    attempt += 1
    messages = build_messages(
        system_prompt=load('system_batch.md'),
        role_extensions=get_role_extensions(device_id, context),
        learned_patterns=get_learned_patterns(context),
        user_task=f'Рассчитай индекс для {device_id}',
        revision_notes=revision_notes
    )
    result, trace = agent_loop(messages)
    reflection = run_reflection(result, trace, context)
    if reflection.verdict == 'approved':
        save_memory(result.memory_to_save)
        return result, trace
    elif reflection.verdict == 'needs_revision' and attempt < MAX_ATTEMPTS:
        revision_notes = reflection.issues
    else:
        result.flag_for_review = True
        return result, trace
```

Метод agent_loop:

```
while True:
    if len(trace.llm_calls) > MAX_LLM_CALLS:
        raise MaxCallsExceeded
    response = llm_client.generate(messages, tools=registry.get_all_schemas())
    trace.add_llm_call(response)
    if response.tool_calls:
        for tc in response.tool_calls:
            result = registry.execute(tc.name, tc.args)
            trace.add_tool_call(tc, result)
            messages.append(ToolMessage(tool_call_id=tc.id, content=result))
        continue
    return parse_final_result(response.content), trace
```

Важные детали:

1. parse_final_result ожидает JSON со структурой из
   system_batch.md. Если не JSON — пытается извлечь через
   guided JSON output (ещё один LLM-вызов с явной схемой).

2. Role extensions из track_B раздел 5 — отдельная функция,
   читает device_metadata и добавляет блок «[Дополнение к роли]».

3. Self-check — отдельный метод run_reflection. Использует
   reflection.md, temperature=0.1.

4. Trace собирается на каждом шаге (plan / llm_call / tool_call
   / reflection). Каждый шаг — TraceStep с duration_ms.

5. Константы MAX_ATTEMPTS=2, MAX_LLM_CALLS=20, MAX_TOOL_CALLS=15
   — из AgentConfig.

6. Compression длинных контекстов:
   Если total_tokens_used в messages > 80% от context window,
   старые tool_results заменяются на summary (track_B Риск 6).
```

**Проверка:**
- E2E тест: factor_store с одним устройством → run_batch → HealthResult.
- Trace содержит plan, llm_call, tool_call, reflection — все 4 типа шагов.
- Намеренно сломанный factor_store триггерит flag_for_review после 2 попыток.

---

### Этап 5.4 — Self-check

Уже частично покрыт в 5.3 — здесь отдельные тесты и полировка.

**Прикрепить:**
- `track_B_agent_prompts_and_loop.md` (раздел 6)
- `agent/core.py`
- `agent/prompts/reflection.md`

**Промпт:**
```
Уточни реализацию run_reflection в agent/core.py.

Требования:

1. Reflection — ОТДЕЛЬНЫЙ LLM-вызов с temperature=0.1,
   max_tokens=500.

2. Промпт собирается из:
   - reflection.md (статика)
   - Краткое резюме входных данных устройства.
   - Факторы из результата.
   - История индексов (если есть, через GetDeviceHistoryTool).
   - Статистика по парку (если есть, через GetFleetStatisticsTool).

3. Ожидаемый JSON-ответ валидируется через ReflectionResult:

class ReflectionResult(BaseModel):
    verdict: Literal['approved', 'needs_revision', 'suspicious']
    issues: list[ReflectionIssue]
    recommended_action: Literal['accept', 'recalculate', 'flag_for_review']

class ReflectionIssue(BaseModel):
    issue: str
    severity: Literal['high', 'medium', 'low']

4. Если LLM возвращает невалидный JSON — retry 2 раза с
   явным напоминанием схемы. После двух неудач —
   verdict='approved' (дефолт = не блокируем пользователя).

5. Логируй результаты reflection на INFO, issues — на WARNING.

Добавь тесты:
- Mock LLMClient возвращает verdict='approved' → основной
  цикл завершается.
- Mock возвращает 'needs_revision' на первой попытке, 'approved'
  на второй → attempts=2.
- Mock возвращает 'needs_revision' дважды → flag_for_review=True.
```

**Проверка:**
- Тесты пройдены.
- `flagged_for_review_rate` на fixture-данных < 10% (если больше — reflection слишком строгий).

---

### Этап 5.5 — Agent loop (chat)

**Прикрепить:**
- `track_B_agent_prompts_and_loop.md` (разделы 1, 3, 4)
- `agent/core.py`

**Промпт:**
```
Добавь в agent/core.py метод run_chat:

def run_chat(
    self,
    user_message: str,
    context: ChatContext
) -> tuple[str, Trace]:

Логика:

1. Строит messages из:
   - system_chat.md
   - Контекст последнего отчёта (сериализованный Report).
   - История диалога из ChatContext.conversation_history.
   - User message.

2. agent_loop тот же, что и в batch, но:
   - Reflection НЕ вызывается (track_B раздел 1).
   - Финальный результат — текст, не JSON.
   - MAX_TOOL_CALLS=10 (меньше, чем в batch).

3. Возвращает (text_answer, trace).

4. Сериализация Report в системный промпт — через to_dict() +
   json.dumps. Размер ограничен 8К токенов; если больше —
   только summary (fleet_summary + топ-10 устройств из таблицы).

Создай agent/prompts/system_chat.md (содержимое из track_B
раздел 4).

Тесты:
- «Что такое индекс здоровья?» → 0 tool calls, текстовый ответ.
- «Почему индекс MFP-042 низкий?» → ≥1 tool call (get_device_events
  или similar).
- «Создай тикет» → агент отказывается (system prompt запрещает).
```

**Проверка:**
- Три сценария из тестов работают.
- Trace содержит mode='chat', не содержит reflection-шагов.

---

### Этап 5.6 — Trace и память

**Прикрепить:**
- `track_B_agent_prompts_and_loop.md` (разделы 7, 9)
- `agent/core.py`

**Промпт:**
```
1. Уточни класс Trace в data_io/models.py (если ещё не реализован
   полностью из track_B раздел 9):
   - Сериализация через to_json() / from_json().
   - Метод summary() — краткая сводка для отчёта.

2. Создай agent/memory.py:

class MemoryManager:
- Демо-версия: хранит паттерны в памяти процесса (dict).
- Метод save_pattern(pattern: LearnedPattern) → bool:
  - Проверяет минимум evidence_devices (из config).
  - Проверяет дубликаты — если паттерн с таким observation и
    scope уже есть, обновляет (merging evidence).
- Метод get_patterns(model: str | None) → list[LearnedPattern]:
  Возвращает применимые паттерны (с scope matching).
- Метод to_dict() / from_dict() — для сохранения между
  перезапусками Streamlit (в session_state).

3. В agent/core.py:
- В run_batch после approved reflection — вызов
  memory_manager.save_pattern() для каждого элемента в
  result.memory_to_save.
- В build_messages — preamble с паттернами от
  memory_manager.get_patterns(model).

4. В реестр добавь 10-й tool:
   GetLearnedPatternsTool:
   - Args: {model: str | null}
   - Data: list[LearnedPattern]
```

**Проверка:**
- Паттерн с evidence=1 → не сохраняется (порог 2).
- Паттерн с evidence=2 → сохраняется, виден в последующих run_batch для той же модели.
- to_dict/from_dict — roundtrip.

---

## Фаза 6 — Отчёт (трек C)

### Этап 6.1 — ReportBuilder

**Прикрепить:**
- `track_C_report_layout.md`
- `agent/core.py`
- `data_io/models.py`
- `tools/calculator.py`

**Промпт:**
```
Создай reporting/report_builder.py.

class ReportBuilder:

Конструктор: Agent (для генерации summary), ReportConfig.

Метод build:

def build(
    self,
    health_results: list[HealthResult],
    factor_store: FactorStore,
    calculation_snapshot: CalculationSnapshot,
    source_file_info: SourceFileInfo,
    include_agent_trace: bool = False
) -> Report:

Шаги:

1. Сформировать FleetSummary:
   - avg_index, median
   - zone_counts (по порогам из config)
   - avg_confidence
   - delta_vs_previous (из storage/previous_reports.json
     если есть)

2. Сгенерировать executive_summary — через agent.run_chat с
   промптом из agent/prompts/executive_summary.md (создай его).

3. Найти top_patterns через agent.tools['find_similar_devices']:
   - Массовые (3+ устройств одной модели с одной ошибкой)
   - Локационные (кластеризация по локации)
   - Критические (H<20 или Conf<0.4)
   Отсортировать, взять топ-5.

4. Сформировать index_distribution (10 бинов).

5. Сформировать category_breakdown по правилам из config:
   - Если >1 модели → по моделям.
   - Иначе если >1 локации → по локациям.
   - Иначе — по confidence-зонам.

6. Для каждого HealthResult → DeviceReport:
   - Заполнить factor_contributions.
   - Получить ResourceState из factor_store.
   - index_history из storage/history/<device_id>.json.
   - agent_recommendation — либо из reflection_notes, либо
     сгенерировать short LLM-вызовом.

7. Если include_agent_trace:
   - Выбрать 3 устройства (worst, best, flagged).
   - Собрать AgentTraceSummary из их trace'ов.

8. Заполнить calculation_snapshot всем содержимым системы.

Методы рендера:

def render_html(report: Report) -> str
def render_pdf(report: Report) -> bytes

Используют Jinja2 + WeasyPrint. Шаблон и стили — отдельный
модуль (этап 6.2).
```

**Проверка:**
- build с fixture-data возвращает Report с заполненными полями.
- Числа в fleet_summary консистентны (avg, median верные).
- top_patterns не больше 5 элементов.

---

### Этап 6.2 — Jinja2-шаблон и CSS

**Прикрепить:**
- `track_C_report_layout.md`
- `reporting/report_builder.py`
- `data_io/models.py`

**Промпт:**
```
Создай:
- reporting/templates/report.jinja2
- reporting/styles/report.css
- reporting/styles/report_print.css

### report.jinja2

Полный шаблон по track_C раздел 2 (все 8 секций).

Принципы:

1. Секции разделены <section id="section-X" class="section">.
2. Условия {% if interactive %} для переключений:
   - <details> vs развёрнутые блоки.
   - Plotly vs inline SVG (предрассчитанный).
   - Sortable table vs plain.
3. Каждое устройство имеет id="device-{{ d.device_id }}"
   для якорных ссылок.
4. Все данные экранируются (autoescape=True). Безопасный
   вывод только для SVG графиков (|safe).
5. Строгий режим Jinja (StrictUndefined) — любая забытая
   переменная падает громко.

### report.css

Стили из track_C раздел 5:
- Типографика: Inter/PT Sans sans-serif.
- Цветовая схема для зон.
- Индикаторные полоски (bar-fill pattern).
- Grid-layout для 4 метрик сверху.
- Responsive breakpoints (desktop > 1200px, tablet 700-1200px).

### report_print.css

Только для PDF:
- @page с размерами A4 + header/footer.
- page-break правила.
- Скрытие .search-bar, .filter-chips и т.д.
- Повтор <thead> при переносе таблицы.
- Принудительные размеры для WeasyPrint.

### JS (inline в <script>)

Скопируй в конец шаблона (только если interactive):
- sortTable(header) для клика по заголовкам.
- filterTable(query) для поиска.
- filterByZone(zone) для чипов.

Без внешних зависимостей. Plotly подключается через CDN если
embed_offline=false, иначе inline bundle (загружается при
сборке шаблона).
```

**Проверка:**
- Рендер fixture-отчёта в HTML открывается в браузере — все секции видны.
- Сортировка таблицы работает.
- JS-ошибок в консоли нет.

---

### Этап 6.3 — PDF-генерация

**Прикрепить:**
- `track_C_report_layout.md` (раздел 4)
- `reporting/report_builder.py`
- `reporting/templates/report.jinja2`

**Промпт:**
```
Создай reporting/pdf_generator.py.

class PDFGenerator:

Конструктор: ReportConfig.

def generate(report: Report) -> bytes:
    # 1. Рендерит шаблон с interactive=False.
    # 2. WeasyPrint: HTML(string=html).write_pdf(
    #        stylesheets=[CSS(report.css), CSS(report_print.css)]
    #    )
    # 3. Возвращает PDF bytes.

Важно:

1. Подключи шрифты через @font-face в report_print.css.
   Файлы .ttf — в reporting/assets/fonts/.
   Укажи base_url=Path("reporting/") в HTML() для resolve.

2. При ошибке WeasyPrint — PDFGenerationError с понятным
   сообщением (обычно это проблема со шрифтами или с парсингом
   CSS).

3. Валидация PDF после генерации:
   - Открыть через pypdf → не падает.
   - Проверить, что количество страниц > 0.
   - Проверить, что в тексте есть «Отчёт о здоровье».

4. Добавь docker-dependency комментарий в README:
   Для работы WeasyPrint в Linux нужны пакеты:
   apt install libpango-1.0-0 libpangoft2-1.0-0
   libcairo2 libgdk-pixbuf2.0-0
```

**Проверка:**
- PDF на fixture-отчёте генерируется за < 5 секунд.
- Кириллица в PDF читается.
- Headers/footers на месте.

---

## Фаза 7 — Streamlit UI

### Этап 7.1 — Shared state и singletons

**Прикрепить:**
- Комментарий архитектуры Streamlit из общих обсуждений
- `config/loader.py`
- `rag/search.py`
- `llm/client.py`
- `agent/core.py`

**Промпт:**
```
Создай state/singletons.py и state/session.py.

### singletons.py

Используй @st.cache_resource для долгоживущих объектов:

@st.cache_resource
def get_rag_engine() -> RAGEngine:
    config = get_config_manager().load_rag_config()
    return RAGEngine(
        qdrant=QdrantManager(config),
        embedder=BGEEmbedder(config.embeddings),
        reranker=BGEReranker(config.reranker),
    )

@st.cache_resource
def get_llm_client(endpoint_name: str) -> LLMClient:
    endpoint = get_config_manager().load_llm_endpoint(endpoint_name)
    return LLMClient(endpoint)

@st.cache_resource
def get_agent(endpoint_name: str) -> Agent:
    return Agent(
        llm_client=get_llm_client(endpoint_name),
        registry=get_tool_registry(),
        rag_engine=get_rag_engine(),
        factor_store=None,  # устанавливается в session
        config=get_config_manager().load_agent_config(),
    )

Важно: смена endpoint'а должна инвалидировать get_llm_client
и get_agent. Используй st.cache_resource.clear() или параметр
endpoint_name в ключе кэша.

### session.py

Обёртки над st.session_state:

def get_current_factor_store() -> FactorStore | None
def set_current_factor_store(fs: FactorStore)

def get_current_health_results() -> list[HealthResult]
def set_current_health_results(...)

def get_current_report() -> Report | None
def set_current_report(...)

def get_active_weights_profile() -> WeightsProfile
def set_active_weights_profile(...)

def get_active_llm_endpoint() -> str
def set_active_llm_endpoint(name)

def get_chat_history() -> list[Message]
def append_chat_message(role, content)

def clear_all():
    ...  # сброс session_state

Все функции типизированы, документированы. Ключи в
session_state — строковые константы в начале файла.
```

**Проверка:**
- Streamlit приложение стартует.
- `st.cache_resource` работает — модели не перезагружаются при перерисовке.

---

### Этапы 7.2–7.5 — Страницы

Для каждой страницы — отдельный промпт с прикреплением:
- `architecture_overview.md` (раздел про эту страницу)
- `state/singletons.py`
- `state/session.py`
- Соответствующий track_*.md (A для весов, D для RAG, B для LLM-чата)

Детализация каждой страницы — большая задача, но техника та же:

**Этап 7.2 — Дашборд (pages/1_Dashboard.py):**
Промпт: «На основе architecture_overview раздела про страницу 1 + данных из session_state.current_report реализуй все 5 блоков дашборда из описания. Используй streamlit-elements или нативные st.metric/st.dataframe. Обработка пустого состояния (нет отчёта) — предложить загрузить файл».

**Этап 7.3 — Веса (pages/2_Weights.py):**
Промпт: «Реализуй страницу управления весами из track_A. Все слайдеры живут в session_state.weights_profile_draft. Применение — кнопка, которая запускает пересчёт health_results на основе draft. Live-сводка «было/стало» через streamlit columns».

**Этап 7.4 — LLM/чат (pages/3_LLM_Chat.py):**
Промпт: «Реализуй страницу с тремя блоками: выбор endpoint, чат, трассировка. Чат — через st.chat_message / st.chat_input. Трассировка — expandable JSON под каждым ответом».

**Этап 7.5 — RAG (pages/4_RAG_Admin.py):**
Промпт: «Реализуй админку RAG из track_D. Секции: Коллекции, Загрузка, Поиск (test), Метрики, Операции. Все долгие операции через st.status с прогрессом».

---

## Фаза 8 — Интеграция и e2e-тесты

### Этап 8.1 — E2E batch-расчёт

**Прикрепить:**
- Все track_*.md
- `architecture_overview.md`

**Промпт:**
```
Создай tests/e2e/test_full_batch.py.

Сценарий:

1. Запускается fixture-окружение:
   - Qdrant через testcontainers.
   - Mock LLMClient с заготовленными ответами.
   - Fixture-коллекция из 3 documents (50 чанков).

2. Тестовый файл fixture: 10 устройств, разные состояния.

3. Полный pipeline:
   - ingest_file → factor_store filled
   - для каждого device → agent.run_batch → HealthResult
   - build_report → Report
   - render_pdf → bytes

4. Ассершены:
   - 10 HealthResult с валидными индексами.
   - Нет дубликатов.
   - Distribution: хотя бы одно устройство в каждой из
     трёх зон.
   - PDF открывается, содержит все 10 devices.
   - Executive summary — непустая строка.

5. Измерение производительности:
   - Полный E2E на 10 устройствах должен завершиться < 2 минут
     с mock-LLM.

Запускай через pytest -m e2e (pytest.ini: markers = e2e).
```

---

### Этап 8.2 — Eval RAG

**Промпт:**
```
Создай tests/e2e/test_rag_eval.py.

Прогон eval-датасета на fixture-коллекции с проверкой
порогов приёмки (Recall@5 ≥ 0.7, MRR ≥ 0.5).

Если метрики падают ниже порогов — тест красный. Это гейт
перед релизом: RAG должен быть достаточно хорош, прежде
чем пускать его в прод.
```

---

### Этап 8.3 — Визуальные тесты отчёта

**Промпт:**
```
Создай tests/visual/test_report_rendering.py.

1. Рендер fixture-отчёта в PNG (через Playwright или imgkit).
2. Сравнение с golden-файлом через PIL (пиксельная разница).
3. Threshold: различия < 1% принимаются, > 1% — тест падает,
   требуется ручная ревизия и обновление golden.

Позволит ловить случайные поломки вёрстки при изменении CSS.
```

---

## Общие советы

### Если LLM начинает галлюцинировать структуру файлов

- Прикрепить `architecture_overview.md` с деревом проекта.
- В начале промпта написать: «Работаем в файле X. Не создавай другие файлы».

### Если LLM пишет Python 3.9-style (List[X] вместо list[X])

- В system prompt зафиксировать: «Python 3.11+, современный синтаксис».
- После генерации прогнать `ruff check --fix` — некоторые проблемы ruff исправит автоматически.

### Если тесты от LLM формальные (mock возвращает X — проверь что X)

- Прикрепить track_*.md с разделом тест-кейсов.
- В промпте просить: «Тест должен упасть, если я намеренно сломаю одну строку в коде. Не пиши тесты, которые проверяют только mock».

### Если LLM создаёт слишком большие классы (god objects)

- В начале каждого промпта: «Single Responsibility Principle. Если класс > 200 строк — разбивай».

### Когда проект разросся и контекст стал > 100К токенов

- Перейти на специализированные ассистенты: Claude Code (читает весь проект), Cursor с indexing.
- Для простых правок — prompting с прикреплением только изменяемых файлов.

---

## Итоговая рекомендация по порядку этапов

Строго последовательно от фазы 0 к 8. Внутри фазы — тоже последовательно. Параллелить можно только фазы 4 (RAG) и 5 (Агент) — у них слабая взаимосвязь, первое пересечение на этапе 5.2 (tool registry).

Оценка времени (опытный разработчик + LLM-ассистент):
- Фазы 0–2: 1–2 дня.
- Фаза 3: 1–2 дня.
- Фаза 4: 2–3 дня.
- Фаза 5: 2–3 дня.
- Фаза 6: 1–2 дня.
- Фаза 7: 2–3 дня.
- Фаза 8: 1 день.

Итого: 10–16 дней разработки.
