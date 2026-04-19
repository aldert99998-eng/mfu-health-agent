# Задача для LLM: починить executive summary и пайплайн severity классификации

**Инструкции для LLM:**
Прочитай всю эту задачу целиком, потом составь план, потом чини. Не торопись с правками кода — сначала изучи текущее состояние репозитория. После каждой правки — короткая проверка, что не сломал смежное.

---

## Контекст проблемы

Агент прогнали на реальных данных парка МФУ (68 устройств, выгрузка из Zabbix, коды ошибок Xerox формата `XX-YYY-ZZ`). На выходе получили executive summary следующего вида:

```
<think> Хорошо, мне нужно составить executive summary...
Проверю структуру: три абзаца с общей оценкой...
Первый абзац должен отразить общую ситуацию.
[ещё ~500 слов рассуждений]
</think>

{ "action": "final_answer", "final_answer": {
  "summary": "Парк состоит из 68 устройств с средним индексом здоровья 61.8 (медиана 50.0). 16 устройств находятся в зелёной зоне, 52 — в жёлтой, красных зон нет. Уровень уверенности расчётов составляет 36%.",
  "risk_areas": "Большинство устройств (76%) находятся в жёлтой зоне... Средний индекс ниже порога для жёлтой зоны (предположительно 70+)...",
  "recommendations": "Сначала стоит провести анализ устройств в жёлтой зоне..."
}}
```

Это **неправильно на нескольких уровнях одновременно**. Ниже — список проблем и что с ними делать.

---

## Проблема №1 — Сырой `<think>` блок и JSON в финальном выводе

**Приоритет:** P0 (критично, пользователь видит это)

**Где:**
- `agent/prompts/executive_summary.md` — промт не запрещает служебный вывод
- `reporting/report_builder.py` — код не фильтрует ответ LLM
- `llm/client.py` — возможно нет пост-обработки думающих моделей

**Что сделать:**

### 1.1. Переписать промт `agent/prompts/executive_summary.md`

Замени весь файл на:

```markdown
Ты пишешь краткое резюме состояния парка МФУ для менеджера IT-отдела.

═══════════════════════════════════════════════════════════════════
ЖЁСТКИЕ ТРЕБОВАНИЯ К ФОРМАТУ ОТВЕТА
═══════════════════════════════════════════════════════════════════

Ответ — ТОЛЬКО три абзаца обычного русского текста, 2-4 предложения в каждом.

КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО:
- Теги <think>, </think> и любые другие XML/HTML теги
- JSON-структуры любого вида
- Блоки ```markdown, ```json, ```text
- Фразы "проверю структуру", "нужно упомянуть", "первый абзац должен"
- Рассуждения о том, как ты формируешь ответ
- Слова "предположительно", "возможно" применительно к числам и порогам
- Маркированные/нумерованные списки
- Заголовки (## ###)

Сразу начинай с первого абзаца. Не здоровайся, не объясняй структуру.

═══════════════════════════════════════════════════════════════════
СТРУКТУРА ТРЁХ АБЗАЦЕВ
═══════════════════════════════════════════════════════════════════

Абзац 1 — Общая картина:
Всего устройств в парке, средний и медианный индекс здоровья,
распределение по зонам (зелёная/жёлтая/красная), средний confidence.

Абзац 2 — Зоны риска:
Топ-3 проблемных устройства по device_id с их индексами и моделями.
Топ-3 самые частые коды ошибок с кратким описанием.
Если есть массовые паттерны (одна проблема на 10+ устройствах) — упомянуть.

Абзац 3 — Рекомендации:
Конкретные действия по самым частым проблемам (замена расходников,
техобслуживание конкретных моделей). Без общих фраз про "дополнительный
анализ" и "калибровку систем". Рекомендации должны быть actionable.

═══════════════════════════════════════════════════════════════════
ОПРЕДЕЛЕНИЯ (используй ТОЛЬКО эти значения, не придумывай)
═══════════════════════════════════════════════════════════════════

Зоны индекса здоровья:
- Зелёная: health_index >= 75
- Жёлтая: 40 <= health_index < 75
- Красная: health_index < 40

Confidence — уверенность расчёта (0.2..1.0). Ниже 0.5 = низкая уверенность.

═══════════════════════════════════════════════════════════════════
ВХОДНЫЕ ДАННЫЕ
═══════════════════════════════════════════════════════════════════

Fleet summary:
{fleet_summary_json}

Топ-5 проблемных устройств:
{worst_devices_json}

Топ-5 кодов ошибок:
{top_error_codes_json}

Массовые паттерны:
{top_patterns_json}

═══════════════════════════════════════════════════════════════════

Теперь напиши три абзаца. Сразу, без преамбулы.
```

### 1.2. Добавить пост-обработку в `llm/client.py`

Найди метод генерации (может называться `generate`, `complete`, `chat`) и добавь на выходе:

```python
import re

class LLMClient:
    # ... существующий код ...

    @staticmethod
    def strip_reasoning_artifacts(text: str) -> str:
        """Удаляет артефакты reasoning-моделей (Qwen, DeepSeek-R1) и
        JSON-обёртки, если модель ошибочно их добавила для текстового режима.
        """
        # 1. Удалить <think>...</think> блоки (возможно вложенные, DOTALL)
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL | re.IGNORECASE)
        # На случай незакрытого тега — обрезать всё до первого </think>
        if '<think>' in text.lower():
            parts = re.split(r'</think>', text, flags=re.IGNORECASE, maxsplit=1)
            text = parts[-1] if len(parts) > 1 else ''

        # 2. Удалить обёртку типа {"action": "final_answer", "final_answer": {...}}
        #    Если весь ответ — это JSON с полем final_answer, извлечь содержимое
        m = re.search(
            r'\{\s*"action"\s*:\s*"final_answer".*?"final_answer"\s*:\s*(\{.*\}|".*?")\s*\}',
            text, flags=re.DOTALL
        )
        if m:
            try:
                import json
                obj = json.loads(m.group(0))
                fa = obj.get('final_answer', '')
                if isinstance(fa, dict):
                    # Склеиваем summary + risk_areas + recommendations
                    text = '\n\n'.join(
                        str(fa.get(k, '')).strip()
                        for k in ('summary', 'risk_areas', 'recommendations')
                        if fa.get(k)
                    )
                elif isinstance(fa, str):
                    text = fa
            except (json.JSONDecodeError, AttributeError):
                pass

        # 3. Убрать обёртку ```…```
        text = re.sub(r'^```(?:json|markdown|text)?\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s*```\s*$', '', text, flags=re.MULTILINE)

        return text.strip()
```

И вызывай её перед возвратом текста из функций, которые должны вернуть plain text (executive_summary, agent recommendations). Не трогай функции, где нужен именно JSON (tool calls) — там strip сломает формат.

### 1.3. Применить обработку в `reporting/report_builder.py`

В методе, который генерирует executive_summary:

```python
def _generate_executive_summary(self, fleet_summary, top_patterns, worst_devices):
    prompt = load_prompt("executive_summary.md").format(
        fleet_summary_json=json.dumps(fleet_summary.dict(), ensure_ascii=False, indent=2),
        worst_devices_json=json.dumps([d.dict() for d in worst_devices[:5]], ensure_ascii=False, indent=2),
        top_error_codes_json=json.dumps(self._top_error_codes(5), ensure_ascii=False),
        top_patterns_json=json.dumps([p.dict() for p in top_patterns[:3]], ensure_ascii=False, indent=2),
    )

    raw_response = self.llm.generate(
        prompt=prompt,
        tools=None,              # КРИТИЧНО: никаких tools для текстового вывода
        temperature=0.4,
        max_tokens=800,
    )

    # Очищаем артефакты
    clean = LLMClient.strip_reasoning_artifacts(raw_response)

    # Валидация: если после очистки пусто — fallback
    if not clean or len(clean) < 50:
        logger.warning("Executive summary после очистки слишком короткий, fallback на шаблон")
        clean = self._fallback_executive_summary(fleet_summary, top_patterns, worst_devices)

    return clean
```

---

## Проблема №2 — Агент использует tool format для чистого текста

**Приоритет:** P0

**Симптом:** В выводе `{"action": "final_answer", ...}` — это формат ReAct/function calling, но для executive summary tools вообще не нужны.

**Где:**
- `llm/client.py` — tool strategy autodetect
- `reporting/report_builder.py` — вызов LLM для summary

**Что сделать:**

### 2.1. Проверить вызов LLM для executive_summary

В `report_builder.py` при генерации summary параметр `tools` должен быть `None` (см. 1.3 выше). Если там сейчас передаётся список инструментов — убери.

### 2.2. Добавить явный режим text в LLMClient

В `llm/client.py`:

```python
class ToolStrategy(str, Enum):
    NATIVE = "native"
    GUIDED = "guided"
    REACT = "react"
    TEXT = "text"          # ← добавить

    @classmethod
    def for_text_generation(cls) -> "ToolStrategy":
        """Для задач без tool use — просто генерация текста."""
        return cls.TEXT


def generate(self, prompt: str, tools: list | None = None,
             tool_strategy: ToolStrategy | None = None, ...):
    # Если tools не переданы — не выбираем стратегию, это чистый текст
    if tools is None or tool_strategy == ToolStrategy.TEXT:
        return self._generate_plain_text(prompt, ...)
    # ... дальше существующая логика для tool use
```

И в `_generate_plain_text` — обычный `chat.completions.create` без `tools=` и без `response_format=`.

---

## Проблема №3 — 0 устройств в красной зоне

**Приоритет:** P0

**Симптом:** В отчёте «красных зон нет», хотя в парке есть устройства с 400+ событиями ошибок. Этого не может быть при правильной классификации.

**Причины (проверить в порядке вероятности):**

### 3.1. Xerox-коды не распознаются — ДОБАВИТЬ В `configs/error_code_patterns.yaml`

Коды формата `XX-YYY-ZZ` (например, `75-530-00`, `72-535-00`, `07-535-00`) — это Xerox service codes. Их нужно добавить в regex, иначе `error_code` остаётся сырым текстом, RAG не находит документации, severity = minor.

Отредактируй файл `configs/error_code_patterns.yaml`:

```yaml
patterns:
  - name: "Xerox service codes (XX-YYY-ZZ)"
    regex: '\b\d{2}-\d{3}-\d{2}\b'
    vendors: ["Xerox"]
    canonical_format: "XX-YYY-ZZ"

  - name: "Kyocera C-codes"
    regex: '\bC\d{4,5}\b'
    vendors: ["Kyocera"]

  - name: "HP E-codes"
    regex: '\bE\d{3}\b'
    vendors: ["HP"]

  - name: "Service codes (generic)"
    regex: '\bSC\d{3,4}\b'
    vendors: ["Ricoh", "Lanier"]

  - name: "Lexmark codes"
    regex: '\b\d{3}\.\d{2}\b'
    vendors: ["Lexmark"]
```

### 3.2. Проверить, что normalizer применяет regex

В `data_io/normalizer.py` (или аналогичном) должна быть логика:

```python
def normalize_error_code(raw: str, patterns: list[ErrorCodePattern]) -> tuple[str, list[str]]:
    """
    Возвращает (canonical_code, flags).
    Если ни один regex не сработал — flags=['unknown_error_code_format'], code=raw.
    """
    for p in patterns:
        m = re.search(p.regex, raw)
        if m:
            return m.group(0), []
    return raw, ['unknown_error_code_format']
```

Проверить, что эта функция реально вызывается при нормализации каждого события в `ingestion_pipeline.py`.

### 3.3. Проверить RAG — есть ли Xerox-документация

Запусти в Python-консоли (через `streamlit run` или отдельным скриптом):

```python
from rag import RAGEngine
rag = RAGEngine()

# Должны быть результаты с высоким score
results = rag.search(
    query="75-530-00 Xerox toner cartridge replacement",
    collection="service_manuals",
    top_k=5,
)
for r in results:
    print(f"score={r.score:.3f}  source={r.source_ref}  text={r.text[:100]}")
```

Если `len(results) == 0` или все scores < 0.4 — **в RAG не загружена документация по Xerox**. Это причина всех «minor severity» и низкого confidence.

**Что сделать:** добавить fallback в `classify_error_severity` tool. В `agent/tools/impl.py`:

```python
class ClassifyErrorSeverityTool:
    # heuristic fallback, когда RAG пустой или low score
    XEROX_CRITICAL_CODES = {
        '75-530-00',  # toner замена
        '09-594-00', '09-604-00', '09-605-00',  # серия 09 — критичные аппаратные
    }
    XEROX_MAJOR_CODES = {
        '72-535-00',  # jam
        '73-530-00', '73-535-00',
        '17-562-00', '17-570-00',
    }

    def _heuristic_severity(self, error_code: str) -> tuple[SeverityLevel, list[str]]:
        """Запасной план, когда RAG не дал уверенного ответа."""
        if error_code in self.XEROX_CRITICAL_CODES:
            return SeverityLevel.CRITICAL, ['heuristic_fallback']
        if error_code in self.XEROX_MAJOR_CODES:
            return SeverityLevel.MAJOR, ['heuristic_fallback']
        return SeverityLevel.MINOR, ['heuristic_fallback', 'unknown_code']

    def execute(self, error_code: str, model: str) -> dict:
        rag_results = self.rag.search(
            query=f"{error_code} {model} severity",
            collection="service_manuals",
            top_k=3,
        )

        if rag_results and rag_results[0].score > 0.6:
            # Используем RAG через LLM classify
            return self._classify_via_llm(error_code, model, rag_results)
        else:
            # Fallback
            severity, flags = self._heuristic_severity(error_code)
            return {
                "severity": severity,
                "affected_components": [],
                "rationale": f"Heuristic fallback: no reliable RAG match (top score: {rag_results[0].score if rag_results else 0:.2f})",
                "source_refs": [],
                "confidence_flags": flags,
            }
```

---

## Проблема №4 — Confidence 0.36 (почти минимум)

**Приоритет:** P0 (причина та же, что у №3)

**Симптом:** `avg_confidence = 0.36`, а по архитектуре минимум — 0.2. Значит почти все факторы помечены `confidence_flags`.

**Что сделать:**

### 4.1. Если проблема №3 починена — confidence должен подняться сам

После добавления Xerox regex и heuristic fallback флаги `unknown_error_code_format` и `low_rag_score` исчезнут у большинства факторов.

### 4.2. Залоггировать распределение флагов для диагностики

В `tools/calculator.py` или где считается confidence:

```python
import logging
logger = logging.getLogger(__name__)

def _compute_confidence(factors: list[Factor]) -> tuple[float, list[str]]:
    # ... существующий расчёт ...

    # Добавь диагностический лог
    flag_counter = Counter(flag for f in factors for flag in f.confidence_flags)
    logger.info(f"Confidence flags distribution: {dict(flag_counter)}")
    # ...
```

После прогона посмотри в логе — какие флаги доминируют. Это покажет, что именно проседает.

---

## Проблема №5 — Summary галлюцинирует пороги зон

**Приоритет:** P1

**Симптом:** В тексте «порог для жёлтой зоны (предположительно 70+)» — агент не знает реальные пороги.

**Что сделать:**

Правка уже есть в 1.1 — новый промт содержит явные пороги. Проверь, что промт реально подхватывается. Если используется кэш промтов — очисти:

```bash
rm -rf ~/.cache/mfu_agent/prompts_cache  # или где у вас кэш
```

Или добавь в код:

```python
# В reporting/report_builder.py
def load_prompt(name: str) -> str:
    """Загружает промт без кэширования (в dev-режиме)."""
    path = PROMPTS_DIR / name
    return path.read_text(encoding='utf-8')
```

---

## Проблема №6 — Возможно запущена reasoning-модель вместо instruct

**Приоритет:** P1

**Симптом:** `<think>` блок — это признак Qwen/QwQ-32B, DeepSeek-R1 или другой reasoning-модели.

**Что сделать:**

### 6.1. Проверить реально запущенную модель

```bash
# Если vLLM / OpenAI-compatible server:
curl http://localhost:8000/v1/models | python3 -m json.tool

# Если Ollama:
ollama list

# Что указано в .env
cat .env | grep LLM_MODEL
```

### 6.2. Если модель reasoning — принять решение

**Вариант A (проще):** переключиться на instruct-модель.

В `.env`:
```
LLM_MODEL=qwen2.5-7b-instruct
```
(или `llama-3.1-8b-instruct`, `mistral-7b-instruct-v0.3`)

**Вариант B:** оставить reasoning-модель, но всегда чистить `<think>` (это уже сделано в 1.2 через `strip_reasoning_artifacts`).

### 6.3. Добавить определение модели в логи при старте

В `llm/client.py` — при инициализации:

```python
class LLMClient:
    REASONING_PATTERNS = ['qwq', 'r1', 'deepseek-r1', 'thinking']

    def __init__(self, config: LLMConfig):
        self.config = config
        self.is_reasoning_model = any(
            p in config.model.lower() for p in self.REASONING_PATTERNS
        )
        logger.info(
            f"LLM initialized: model={config.model}, "
            f"reasoning={self.is_reasoning_model}"
        )
```

---

## Проблема №7 — Summary без конкретики

**Приоритет:** P1

**Симптом:** «Сначала стоит провести анализ устройств в жёлтой зоне» — общая фраза без device_id, без кодов, без чисел.

**Что сделать:**

### 7.1. Расширить контекст, передаваемый в LLM для summary

В `reporting/report_builder.py` метод `_generate_executive_summary`:

```python
def _generate_executive_summary(self, fleet_summary, results: list[HealthResult]):
    # Топ-5 худших устройств
    worst = sorted(results, key=lambda r: r.health_index)[:5]
    worst_data = [
        {
            "device_id": r.device_id,
            "model": r.device_metadata.model if r.device_metadata else "unknown",
            "health_index": r.health_index,
            "zone": r.zone.value,
            "top_factor": r.factor_contributions[0].error_code if r.factor_contributions else None,
        }
        for r in worst
    ]

    # Топ-5 кодов ошибок по парку
    from collections import Counter
    code_counter = Counter()
    for r in results:
        for fc in r.factor_contributions:
            code_counter[fc.error_code] += 1
    top_codes = [
        {"code": code, "count": cnt, "affected_devices": self._count_devices_with_code(results, code)}
        for code, cnt in code_counter.most_common(5)
    ]

    prompt = load_prompt("executive_summary.md").format(
        fleet_summary_json=json.dumps(fleet_summary.dict(), ensure_ascii=False, indent=2, default=str),
        worst_devices_json=json.dumps(worst_data, ensure_ascii=False, indent=2),
        top_error_codes_json=json.dumps(top_codes, ensure_ascii=False, indent=2),
        top_patterns_json=json.dumps(
            [p.dict() for p in self.top_patterns[:3]],
            ensure_ascii=False, indent=2, default=str
        ),
    )

    raw = self.llm.generate(prompt=prompt, tools=None, temperature=0.4, max_tokens=800)
    return LLMClient.strip_reasoning_artifacts(raw)
```

---

## План работ — рекомендуемый порядок

```
День 1 — P0:
  1. Починить error_code_patterns.yaml (проблема 3.1)  — 15 мин
  2. Проверить RAG через консоль (проблема 3.3)        — 15 мин
  3. Добавить heuristic fallback (проблема 3.3)        — 30 мин
  4. Переписать executive_summary.md промт (1.1)       — 15 мин
  5. Добавить strip_reasoning_artifacts (1.2)          — 30 мин
  6. Применить его в report_builder (1.3)              — 15 мин
  7. Проверить tools=None в вызове summary (2.1)       — 10 мин
  8. Расширить контекст для summary (7.1)              — 30 мин

После — прогон на tests/fixtures/03_valid_large.csv, 
сверить с ожиданием из README_fixtures.md.

День 2 — P1:
  9. Определение reasoning-модели и логирование (6)    — 30 мин
  10. Fallback-шаблон executive_summary (1.3)          — 30 мин
  11. Тесты на strip_reasoning_artifacts               — 30 мин
```

---

## Критерии приёмки после правок

После всех изменений прогон на `tests/fixtures/03_valid_large.csv` должен дать:

**Технически:**
- [ ] В executive_summary НЕТ `<think>`, `</think>`, `{"action":`
- [ ] Текст summary — 3 абзаца обычного русского текста
- [ ] Пороги зон в тексте называются правильно (75 / 40), нет слова «предположительно»

**Содержательно:**
- [ ] Устройств в красной зоне **больше 0** (ожидаем 5–15 штук при 43 устройствах с ошибками)
- [ ] Средний confidence **>= 0.6** (сейчас 0.36)
- [ ] В summary упомянуты конкретные device_id (MFP-00379, MFP-00381, MFP-00150 и т.п.)
- [ ] В summary упомянуты конкретные коды (`75-530-00`, `72-535-00`, `07-535-00`)
- [ ] В summary упомянута массовая проблема замены тонера

**Формальные тесты:**
- [ ] `pytest tests/test_ingestion.py -k xerox_codes` — зелёный
- [ ] `pytest tests/test_agent.py -k severity_fallback` — зелёный
- [ ] `pytest tests/test_report.py -k executive_summary_no_artifacts` — зелёный
- [ ] Прогон на `tests/fixtures/04_single_critical.csv` даёт `zone=red` и `confidence >= 0.7`

---

## Напиши новые тесты одновременно с фиксами

Для каждой правки добавь тест. Минимум:

```python
# tests/test_llm_client.py
def test_strip_think_block():
    raw = "<think>рассуждения</think>\n\nФинальный текст."
    assert LLMClient.strip_reasoning_artifacts(raw) == "Финальный текст."

def test_strip_action_final_answer_json():
    raw = '{"action": "final_answer", "final_answer": {"summary": "Текст.", "risk_areas": "Риски.", "recommendations": "Рек."}}'
    result = LLMClient.strip_reasoning_artifacts(raw)
    assert "Текст." in result
    assert "Риски." in result
    assert '"action"' not in result
    assert '{' not in result

def test_strip_combined():
    raw = "<think>Долгие размышления</think>\n```json\n{\"action\": \"final_answer\", \"final_answer\": \"Готовый текст.\"}\n```"
    assert LLMClient.strip_reasoning_artifacts(raw) == "Готовый текст."

# tests/test_normalizer.py
@pytest.mark.parametrize("raw,expected", [
    ("75-530-00", "75-530-00"),
    ("Error 75-530-00 occurred", "75-530-00"),
    ("errdisp-[07-535-00 Tray empty]", "07-535-00"),
    ("C6000", "C6000"),
    ("E102", "E102"),
])
def test_xerox_and_other_codes_extracted(raw, expected):
    code, flags = normalize_error_code(raw, patterns=load_patterns())
    assert code == expected
    assert 'unknown_error_code_format' not in flags

def test_unknown_code_flagged():
    code, flags = normalize_error_code("random text no codes", patterns=load_patterns())
    assert 'unknown_error_code_format' in flags

# tests/test_agent_tools.py
def test_heuristic_severity_for_known_critical_code():
    tool = ClassifyErrorSeverityTool(rag=EmptyRAG())  # RAG пустой → fallback
    result = tool.execute(error_code="75-530-00", model="Xerox AltaLink B8045")
    assert result["severity"] == SeverityLevel.CRITICAL
    assert "heuristic_fallback" in result["confidence_flags"]

# tests/test_report.py
def test_executive_summary_no_artifacts(mock_llm_returning_thinking):
    # mock LLM возвращает <think>...</think>JSON
    builder = ReportBuilder(llm=mock_llm_returning_thinking, ...)
    report = builder.build(results=SAMPLE_RESULTS, ...)
    assert "<think>" not in report.executive_summary
    assert '{"action"' not in report.executive_summary
    assert len(report.executive_summary) > 100
```

---

## Порядок действий для LLM в VS Code

1. **Прочитай этот файл целиком.** Не начинай с правок.
2. **Изучи структуру проекта:** `ls -R agent/ reporting/ data_io/ configs/ llm/`
3. **Посмотри текущее состояние** файлов, которые будешь менять:
   - `agent/prompts/executive_summary.md`
   - `reporting/report_builder.py` (функция генерации summary)
   - `llm/client.py` (метод generate)
   - `configs/error_code_patterns.yaml`
   - `data_io/normalizer.py` (normalize_error_code)
   - `agent/tools/impl.py` (ClassifyErrorSeverityTool)
4. **Составь и покажи мне план правок** в формате:
   ```
   Файл X:
     - изменение 1 (строки Y-Z): краткое описание
     - изменение 2 (строки A-B): ...
   Новые файлы: ...
   Новые тесты: ...
   ```
5. **Жди моего «поехали»**, прежде чем править код.
6. **Фикси по порядку плана работ** (P0 сначала).
7. **После каждого блока правок** (например, после всех правок по проблеме №3) — запусти `pytest` по новым тестам. Если зелёно — двигайся дальше. Если красно — чини до зелёного, не накатывай следующий блок.
8. **В конце** — прогон на `tests/fixtures/03_valid_large.csv` через UI (если доступен Playwright MCP) или через скрипт. Сверить с «Критерии приёмки».
9. **Коммиты по одному блоку правок** (не все в один коммит), сообщения в стиле:
   - `fix(config): add Xerox XX-YYY-ZZ error code pattern`
   - `fix(llm): strip <think> and action/final_answer artifacts`
   - `fix(report): no tools for executive summary, expand context`
   - `feat(agent): heuristic severity fallback for known Xerox codes`

---

## Что НЕ делать

- Не меняй формулу индекса здоровья (трек A) — она работает правильно.
- Не трогай RAG-ingestion — это отдельная большая задача.
- Не переписывай calculator — проблема не в нём.
- Не добавляй новые LLM-вызовы без необходимости — существующие должны заработать.

---

**Начни с плана. Жди подтверждения «поехали» перед первой правкой кода.**
