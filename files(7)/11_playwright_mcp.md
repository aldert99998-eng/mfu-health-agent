# Дополнение 11 — Playwright MCP для UI-тестирования

**Назначение:** расширение основного QA-промта (`10_qa_agent_prompts.md`) для автоматизации UI-кейсов через Playwright MCP-сервер в Claude Code.

**Предусловие:** `claude mcp list` показывает `playwright: ✓ Connected`.

**Покрывает тест-планы:**
- `06_ui_streamlit.md` — почти весь (Streamlit — это браузерный UI)
- `07_e2e_integration.md` — UI-часть сквозных сценариев
- `01_track_E_ingestion.md` раздел 6 — загрузка файла, переключение вкладок, F5

---

## 1. Доступные Playwright-инструменты

После подключения MCP у агента появляются инструменты (точные имена могут отличаться — сверяйся с `list available mcp tools`):

| Инструмент | Что делает |
|---|---|
| `browser_navigate(url)` | Открыть URL |
| `browser_snapshot()` | Снимок DOM с accessibility-деревом (основной способ «видеть» страницу) |
| `browser_click(element, ref)` | Клик по элементу (по ref из snapshot) |
| `browser_type(element, ref, text)` | Ввести текст |
| `browser_file_upload(paths)` | Загрузить файл в input[type=file] |
| `browser_screenshot()` | PNG-снимок (для визуальных сверок) |
| `browser_wait_for(text/element)` | Ожидание условия |
| `browser_press_key(key)` | Нажать клавишу (`F5`, `Enter`, `Escape`, `Tab`) |
| `browser_evaluate(js)` | Выполнить JS в странице (для assertions) |
| `browser_console_messages()` | Лог консоли браузера |
| `browser_network_requests()` | Сетевые запросы (проверка, что XHR не отменён) |
| `browser_tabs()` / `browser_tab_new()` / `browser_tab_select()` | Управление вкладками браузера |

**Ключевое отличие от обычного Playwright:** Playwright MCP работает с **accessibility-деревом**, а не со скриншотами. Агент получает структурированный DOM с ref'ами и кликает по ним. Это быстрее и детерминированнее, чем image recognition.

---

## 2. Базовый шаблон промта для UI-прогона

Добавь к основному QA-промту следующий блок:

```
═══════════════════════════════════════════════════════════════════
UI-ТЕСТИРОВАНИЕ ЧЕРЕЗ PLAYWRIGHT MCP
═══════════════════════════════════════════════════════════════════

Если TC из раздела UI (06_ui_streamlit, 07_e2e, 01 раздел 6):

1. ПОДНИМИ ПРИЛОЖЕНИЕ (если ещё не запущено):
   Проверь, доступен ли http://localhost:8501 через browser_navigate.
   Если нет — сообщи пользователю: "Запусти 'make run' или 'streamlit
   run app.py' в отдельном терминале и скажи готово".

2. РАБОТА С DOM:
   Всегда сначала browser_snapshot() чтобы увидеть структуру.
   Потом browser_click / browser_type с ref из snapshot.
   НЕ пытайся кликать по координатам — используй ref.

3. ОЖИДАНИЯ:
   Streamlit перерисовывает страницу на каждое действие.
   После клика/ввода — browser_wait_for нужного текста или элемента.
   Таймаут по умолчанию: 30 сек для операций, 5 мин для batch.

4. ЗАГРУЗКА ФАЙЛОВ:
   Используй browser_file_upload с абсолютным путём к fixture.
   Fixtures лежат в tests/fixtures/.

5. СКРИНШОТЫ:
   Для TC с визуальным контролем — browser_screenshot()
   ДО действия и ПОСЛЕ. Сохраняй в tests/artifacts/screenshots/
   с именем <tc_id>_<step>.png.

6. ПРОВЕРКА КОНСОЛИ:
   После каждого TC — browser_console_messages() для отлова JS-ошибок.
   Errors в консоли = TC помечен WARNING даже при функциональном PASS.

7. СЕТЕВЫЕ ПРОВЕРКИ (критично для TC-E-080, TC-E-081):
   browser_network_requests() чтобы убедиться, что XHR загрузки
   не был отменён (aborted) при переключении вкладок.

═══════════════════════════════════════════════════════════════════
```

---

## 3. Паттерны для ключевых UI-кейсов

### Паттерн A — Загрузка файла (TC-E-021, TC-UI-021)

```
ШАГИ АГЕНТА:
1. browser_navigate("http://localhost:8501")
2. browser_snapshot() — найти страницу Dashboard
3. Если не на Dashboard — browser_click по ссылке "Dashboard"
4. browser_wait_for("Загрузите файл")
5. browser_file_upload(["/абсолютный/путь/tests/fixtures/fleet_50_standard.csv"])
6. browser_wait_for("обработано") — сообщение об успехе ingestion
7. browser_snapshot() — проверить, что виден summary с "50 устройств"
8. browser_console_messages() — проверить, что нет JS-ошибок

ASSERTIONS:
• Summary содержит ожидаемое число строк
• Нет ошибок в консоли
• В network_requests есть успешный POST с файлом
```

### Паттерн B — Переключение вкладок во время batch (TC-E-080, TC-UI-023)

**Это ключевой кейс, специально под него Playwright и нужен.**

```
ШАГИ АГЕНТА:
1. Загрузить большой файл (TD-03 ~25 МБ) через паттерн A
2. browser_click по кнопке "Запустить анализ"
3. browser_wait_for(прогресс > 20%)
4. Зафиксировать время старта и номер XHR-запроса через
   browser_network_requests()
5. browser_click по ссылке "2. Веса" в боковой панели
6. browser_wait_for загрузки страницы Weights
7. Подождать 10 сек (browser_evaluate("await new Promise(r => setTimeout(r, 10000))"))
8. browser_click по "3. LLM/Чат"
9. browser_wait_for
10. browser_click по "1. Dashboard" — возврат
11. browser_snapshot() — увидеть текущий прогресс
12. browser_network_requests() — убедиться, что исходный XHR
    со статусом != 'cancelled'/'aborted'
13. browser_wait_for("Анализ завершён") — до MAX 10 минут

ASSERTIONS:
• Прогресс на Dashboard >= прогресса до переключения (не откат)
• XHR не отменён
• По завершении число устройств в отчёте = числу в файле
• Нет дублей в БД / factor_store

ПРИЗНАКИ БАГА (FAIL для этого TC):
• На возврате прогресс "замер" на старом значении и не двигается
• В consoleMessages есть "Request aborted" / "XHR cancelled"
• В network_requests статус XHR = 'cancelled'
• Количество импортированных устройств < ожидаемого
```

### Паттерн C — Reload во время batch (TC-E-081, TC-UI-024)

```
ШАГИ АГЕНТА:
1. Загрузить файл, запустить batch (паттерн A + клик на "Запустить")
2. browser_wait_for(прогресс 30-50%)
3. Зафиксировать job_id из DOM (если отображается) или из console
4. browser_press_key("F5")
5. browser_wait_for загрузки Streamlit после reload
6. browser_snapshot() — проверить, что виден активный batch
7. Сравнить текущий job_id с зафиксированным — ДОЛЖНЫ СОВПАДАТЬ
8. browser_wait_for("Анализ завершён")
9. Скачать / открыть отчёт
10. Проверить количество устройств

ASSERTIONS:
• job_id до и после reload — идентичен
• Итоговое число устройств = файлу (нет потерь, нет дублей)
• Прогресс после reload = или > прогресса до reload
• На странице есть индикация "загрузка продолжается"

ПРИЗНАКИ БАГА:
• После reload UI показывает "нет активных задач" (задача потерялась)
• Новый job_id (задача перезапустилась с нуля)
• Дубли в итоговых результатах (двойная обработка)
```

### Паттерн D — Изменение весов на странице Weights (TC-UI-031, TC-UI-037)

```
ШАГИ АГЕНТА:
1. browser_navigate на страницу "2. Веса"
2. browser_snapshot() — найти слайдеры S_minor, S_major, S_critical
3. Для каждого слайдера:
   • browser_click / browser_type новое значение
   • browser_wait_for обновления
4. browser_click "Пересчитать"
5. browser_wait_for обновлённого отчёта
6. Сверить, что health_index для reference-устройства изменился
   в ожидаемую сторону

ASSERTIONS:
• Новое значение index соответствует ручному расчёту по новой формуле
• factor_store не пересобирался (нет повторного LLM-вызова severity)
• UI отражает новый WeightsProfile в calculation_snapshot
```

### Паттерн E — Chat с агентом (TC-UI-040, TC-E2E-020)

```
ШАГИ АГЕНТА:
1. Предусловие: batch на fleet_50 уже выполнен в той же сессии
2. browser_navigate "3. LLM/Чат"
3. browser_snapshot() — найти input
4. browser_type в input: "Почему MFP-017 в красной зоне?"
5. browser_press_key("Enter") или клик по "Отправить"
6. browser_wait_for ответа (может быть 30+ сек)
7. browser_snapshot() — забрать текст ответа
8. Раскрыть expandable "Trace":
   • browser_click по "Показать трассировку"
   • browser_snapshot()

ASSERTIONS:
• Ответ содержит "MFP-017"
• Ответ ссылается на конкретные error_codes из файла
• В trace видны tool calls: get_device_events, classify_error_severity
• Нет галлюцинаций (упомянутых устройств / кодов нет в данных)

ДЕТЕКТОР ГАЛЛЮЦИНАЦИЙ:
• Извлечь из ответа все упомянутые device_id и error_code
• Сверить с factor_store (через API или прямой query)
• Несоответствие = FAIL
```

### Паттерн F — Визуальный regression отчёта (TC-C-070, TC-C-071)

```
ШАГИ АГЕНТА:
1. Прогнать TC-E2E-002 (golden reference 20 устройств)
2. Открыть сгенерированный отчёт в UI
3. browser_screenshot() — сохранить как tests/artifacts/screenshots/report_actual.png
4. Сравнить с tests/fixtures/screenshots/report_baseline.png через
   browser_evaluate с pixelmatch или отдать Python-скрипту для diff

ASSERTIONS:
• pixel-diff < threshold (например, 2%)
• Структурно все секции на месте (проверить snapshot-ом DOM)

ПЕРВЫЙ ПРОГОН:
• Если baseline отсутствует — создать его, пометить TC как
  BASELINE_CREATED, не PASS/FAIL
```

### Паттерн G — Drag-and-drop (TC-E-086)

```
Streamlit file_uploader поддерживает и клик, и drag-and-drop.
Через Playwright MCP drag-and-drop сложнее — используй browser_evaluate
с DataTransfer API:

1. browser_evaluate(`
   const dt = new DataTransfer();
   const file = await fetch('/путь к файлу').then(r => r.blob());
   dt.items.add(new File([file], 'test.csv'));
   const dropzone = document.querySelector('[data-testid="stFileUploaderDropzone"]');
   dropzone.dispatchEvent(new DragEvent('drop', {dataTransfer: dt, bubbles: true}));
`)
2. browser_wait_for ingestion
3. Проверить, что результат такой же, как через browser_file_upload
```

---

## 4. Готовые промты для запуска

### Промт 1: «Прогони UI-smoke-suite»

```
Используй Playwright MCP для прогона критичных UI-кейсов:

1. TC-UI-001 (запуск)
2. TC-UI-021 (загрузка CSV)
3. TC-E-080 / TC-UI-023 (переключение вкладок во время batch) — ОСОБО ВАЖНО
4. TC-E-081 / TC-UI-024 (F5 во время batch) — ОСОБО ВАЖНО
5. TC-UI-030 (страница Weights отображается)
6. TC-UI-040 (chat отвечает)
7. TC-E2E-020 (chat про конкретное устройство после batch)

Предусловие: Streamlit запущен на localhost:8501, Qdrant поднят,
LLM доступна.

Используй паттерны из 11_playwright_mcp.md (паттерны A-E).

Для каждого TC:
• до действия — browser_snapshot
• после — browser_snapshot + browser_console_messages
• скриншот в tests/artifacts/screenshots/<tc_id>.png

Формат отчёта — как в основном QA-промте (10_qa_agent_prompts.md).
Время: ожидаемо ~45 минут.

Если Streamlit не запущен — СТОП, спроси пользователя.
```

### Промт 2: «Прогоночный цикл переключений»

Для агрессивной проверки стабильности (TC-E-013 стресс-тест):

```
Используя Playwright, выполни стресс-сценарий TC-E-013:

1. Запустить batch на fleet_1000 (fixture).
2. В цикле 20 раз в случайные моменты:
   • Переключение между 4 вкладками Streamlit
   • F5 reload
   • Клик по случайному слайдеру на странице Weights
3. Дождаться завершения batch.

Каждые 30 сек — browser_snapshot + фиксация прогресса.

Assertions:
• Batch завершился (не завис, не упал)
• Число устройств в отчёте = ожидаемому
• Нет aborted XHR
• Нет duplicate записей в factor_store

Если хоть один assertion fail — FAIL с детальным логом действий.
```

### Промт 3: «Генерация Playwright-тестов под pytest»

```
Из test_plans/06_ui_streamlit.md сгенерируй pytest-файлы, которые
используют playwright-python (не MCP) для автономного запуска в CI.

Структура:
tests/ui/
├── conftest.py           # фикстуры: streamlit_url, browser, page
├── test_dashboard.py     # TC-UI-020..028
├── test_weights.py       # TC-UI-030..037
├── test_chat.py          # TC-UI-040..048
├── test_rag_admin.py     # TC-UI-050..056
└── test_stability.py     # TC-UI-023, 024, 031 — критичные

Правила:
• @pytest.mark.p0 / p1 / p2 / p3 для приоритизации
• фикстура streamlit_app: запускает процесс, yield URL, убивает
• используй page.expect_response() для проверки XHR
• скриншоты через page.screenshot() в tests/artifacts/
• для TC-UI-023 / TC-UI-024 — obязательно проверка job_id

В конце — инструкция как запустить: pytest tests/ui/ -m p0
```

---

## 5. Специфика Streamlit

Streamlit имеет особенности, которые надо учесть в промте:

```
═══════════════════════════════════════════════════════════════════
STREAMLIT-СПЕЦИФИКА
═══════════════════════════════════════════════════════════════════

1. RERUN МОДЕЛЬ:
   Streamlit перезапускает весь скрипт на каждое действие widget.
   После клика/ввода ВСЕГДА делай browser_wait_for нужного элемента.
   Не предполагай синхронного обновления.

2. STABLE SELECTORS:
   Streamlit генерирует id'шники с хешами — они МЕНЯЮТСЯ между
   запусками. Используй:
   • data-testid атрибуты (если разработчик их добавил)
   • текст кнопок и лейблов (stable)
   • aria-label
   НЕ используй CSS-селекторы с хеш-id.

3. FILE UPLOADER:
   Streamlit file_uploader создаёт скрытый input[type=file].
   Используй browser_file_upload с ref этого input.
   После загрузки — wait_for сообщения "File uploaded" или аналога.

4. SPINNER / PROGRESS:
   Streamlit показывает спиннер через st.spinner() и progress-bar
   через st.progress(). В DOM это элементы с классами
   stSpinner / stProgress. Их наличие — признак "идёт обработка".

5. SESSION STATE:
   Недоступен напрямую из browser. Проверять через UI-реакцию
   (виден ли отчёт, выбран ли profile).

6. MULTIPAGE NAVIGATION:
   Страницы в боковой панели — это <a> с текстом названия.
   browser_click по тексту "Dashboard", "Веса" и т.д.

7. RERUN НА F5:
   F5 в Streamlit — это полный reload страницы. Session state
   сбрасывается, но cache_resource (singletons) остаётся.
   Это и есть суть TC-UI-024: проверить, что ВАЖНЫЕ данные
   (factor_store) в cache_resource и переживают reload.
═══════════════════════════════════════════════════════════════════
```

---

## 6. Антипаттерны именно для UI-тестирования

```
ЧТО АГЕНТ НЕ ДОЛЖЕН ДЕЛАТЬ:

❌ Клик по координатам (x, y) — координаты плавают, ref стабилен.
❌ time.sleep(10) в ожидании — используй browser_wait_for.
❌ Тесты, зависящие от конкретного текста LLM — LLM недетерминирована.
   Проверяй структуру ответа (упомянут device_id, есть ссылки),
   а не точный текст.
❌ Игнорирование console_messages — JS-ошибки это уже FAIL.
❌ Один огромный тест на 50 действий — разбивай на атомарные TC.
❌ Предположения "обычно работает" — всегда snapshot + assertion.
❌ Запуск UI-тестов без явной проверки, что Streamlit поднят.
```

---

## 7. Практика: что сделать прямо сейчас

1. **Убедиться, что Streamlit запускается:**
```bash
cd /путь/к/mfu_agent
make run
# в другом терминале проверь: curl http://localhost:8501
```

2. **Проверить fixtures:**
```bash
ls tests/fixtures/
# должны быть: fleet_50_standard.csv, golden_fleet_20.csv и т.д.
```

3. **Открыть Claude Code в папке проекта:**
```bash
claude
```

4. **Прикрепить в первом сообщении:**
   - `docs/architecture_overview.md`
   - `docs/test_plans/00_INDEX.md`
   - `docs/test_plans/10_qa_agent_prompts.md`
   - `docs/test_plans/11_playwright_mcp.md` (этот файл)
   - Целевой тест-план (например, `06_ui_streamlit.md`)

5. **Вставить запрос** — один из 3 готовых промтов выше, например:

> Используй основной QA-промт из 10_qa_agent_prompts.md в режиме
> MODE 3 EXECUTE. Применяй паттерны и правила из 11_playwright_mcp.md.
> Прогон: UI-smoke-suite (промт 1 из файла 11).
> Начинай с плана, жди моего "поехали".

---

## 8. Расширение: запись baseline для visual regression

```
После первого успешного полного прогона golden fleet сохрани:

1. Скриншот HTML-отчёта → tests/fixtures/screenshots/report_baseline.png
2. Скриншоты каждой страницы Streamlit в пустом и рабочем состоянии
   → tests/fixtures/screenshots/page_<N>_<state>.png
3. Структурный снимок отчёта (DOM без данных) → JSON

Эти файлы — эталон для будущих регрессионных прогонов.
Меняй их только осознанно с пометкой в commit message.
```

---

## 9. Резюме

С подключённым Playwright MCP ваш QA-агент получает возможность:

- **Реально гонять UI-кейсы** из планов 06 и 07, а не только читать код
- **Воспроизводить кейсы с переключением вкладок и F5** — самые критичные для стабильности
- **Делать визуальный regression** через скриншоты
- **Ловить JS-ошибки** через консоль браузера
- **Проверять сеть** (aborted XHR — главный признак бага в TC-E-080/081)

Всё это в рамках одной Claude Code сессии, без переключения на Selenium/Cypress. Агент сам поднимает браузер, сам ходит по страницам, сам собирает отчёт.

**Рекомендуемая последовательность внедрения:**

1. **День 1** — прогнать промт 1 (UI-smoke), убедиться, что базовый сценарий работает.
2. **День 2** — промт 2 (стресс переключений) для поиска плавающих багов.
3. **Неделя 2** — промт 3 (генерация pytest) для постановки в CI.
4. **Релиз-цикл** — UI-smoke перед каждым релизом как часть GO/NO-GO.
