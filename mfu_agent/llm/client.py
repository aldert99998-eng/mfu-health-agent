"""LLM client — Track B, Level 1.

Thread-safe wrapper around the OpenAI Python SDK for any
OpenAI-compatible endpoint.  Supports three tool strategies
(native / guided_json / react) with automatic detection and
fallback.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import TYPE_CHECKING, Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
)
from openai import (
    RateLimitError as OpenAIRateLimitError,
)
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .auth import TokenProvider, build_token_provider
from .http import build_http_client

if TYPE_CHECKING:
    from config.loader import LLMEndpointConfig, LLMGenerationParams

logger = logging.getLogger(__name__)

# ── Strategy constants ────────────────────────────────────────────────────────

STRATEGY_NATIVE = "native"
STRATEGY_GUIDED_JSON = "guided_json"
STRATEGY_REACT = "react"

_VALID_STRATEGIES = {STRATEGY_NATIVE, STRATEGY_GUIDED_JSON, STRATEGY_REACT}

# ── Custom errors ─────────────────────────────────────────────────────────────


class LLMConnectionError(Exception):
    """Endpoint unreachable."""


class LLMTimeoutError(Exception):
    """Request timed out."""


class InvalidResponseError(Exception):
    """Model returned an unparseable or structurally invalid response."""


class LLMRateLimitError(Exception):
    """Rate-limited by the endpoint."""


# ── Response models ───────────────────────────────────────────────────────────


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ToolCall(BaseModel):
    id: str = ""
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    content: str = ""
    tool_calls: list[ToolCall] | None = None
    finish_reason: str = ""
    usage: TokenUsage = Field(default_factory=TokenUsage)


# ── Ping result ───────────────────────────────────────────────────────────────


class PingResult(BaseModel):
    ok: bool
    latency_ms: float = 0.0
    error: str = ""


# ── Probe tool schema for detection ──────────────────────────────────────────

_PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "_probe_tool_support",
        "description": "Probe function for tool support detection.",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {"type": "string"},
            },
            "required": ["value"],
        },
    },
}

_PROBE_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": 'Call the _probe_tool_support tool with value="test".'},
]

_GUIDED_JSON_PROBE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string"},
        "value": {"type": "string"},
    },
    "required": ["action", "value"],
}

# ── ReAct parsing ─────────────────────────────────────────────────────────────

# Locate the tool name ("Действие: <identifier>") and the start of the JSON
# block ("Параметры: {"). Whitespace between the two is any run of spaces,
# tabs, or newlines — some local models emit them on one line without a
# newline. The JSON body is extracted with a real JSON decoder (see
# _parse_react_tool_calls) rather than a regex, to handle nested objects.
_REACT_ACTION_RE = re.compile(
    r"Действие:\s*[`\"']?([A-Za-z_][A-Za-z0-9_]*)[`\"']?\s+Параметры:\s*",
    re.IGNORECASE,
)
_REACT_ACTION_EN_RE = re.compile(
    r"Action:\s*[`\"']?([A-Za-z_][A-Za-z0-9_]*)[`\"']?\s+Parameters:\s*",
    re.IGNORECASE,
)

_REACT_INSTRUCTION_RU = (
    "\n\nЕсли тебе нужно вызвать инструмент, используй строго такой формат:\n"
    "Мысль: <что ты хочешь сделать>\n"
    "Действие: <имя_инструмента>\n"
    "Параметры: <JSON с аргументами>\n\n"
    "Если инструмент не нужен, просто ответь текстом."
)

_GUIDED_JSON_INSTRUCTION = (
    "\n\nОтветь строго в формате JSON по следующей схеме:\n{schema}\n"
    "Ничего кроме JSON не пиши."
)


# ── Reasoning model detection ────────────────────────────────────────────────

REASONING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"deepseek.*r1", re.IGNORECASE),
    re.compile(r"qwen.*qwq", re.IGNORECASE),
    re.compile(r"qwq", re.IGNORECASE),
    re.compile(r"\br1\b", re.IGNORECASE),
    re.compile(r"o[134]-", re.IGNORECASE),
    re.compile(r"reasoning", re.IGNORECASE),
    re.compile(r"nemotron", re.IGNORECASE),
]


def is_reasoning_model(model_name: str) -> bool:
    return any(p.search(model_name) for p in REASONING_PATTERNS)


# ── LLMClient ─────────────────────────────────────────────────────────────────


class LLMClient:
    """Thread-safe OpenAI-compatible LLM client with tool-strategy autodetection.

    Parameters
    ----------
    config:
        ``LLMEndpointConfig`` with endpoint URL, API key, model, and params.
    """

    def __init__(self, config: LLMEndpointConfig) -> None:
        self._config = config
        self._model = config.model
        self._lock = threading.Lock()

        # Threading-local guard preventing 401-retry recursion storms.
        self._retry_state = threading.local()

        # Provider-agnostic auth layer.  For local endpoints this is a
        # no-op and behaviour is byte-identical to the pre-multi-provider
        # code path.  For GigaChat the http_client's event_hook rewrites
        # Authorization with a fresh Bearer on every request.
        self._token_provider: TokenProvider = build_token_provider(config)
        self._http_client = build_http_client(config, self._token_provider)

        self._client = OpenAI(
            base_url=config.url,
            api_key=config.api_key or "placeholder",
            timeout=config.timeout_seconds,
            max_retries=0,
            http_client=self._http_client,
        )

        self._tool_strategy: str | None = (
            config.tool_strategy if config.tool_strategy in _VALID_STRATEGIES else None
        )

        self._is_reasoning = is_reasoning_model(self._model)

        logger.info(
            "LLMClient инициализирован: %s model=%s strategy=%s auth=%s",
            config.url,
            config.model,
            self._tool_strategy or "auto",
            (config.auth.type if config.auth else "static"),
        )
        if self._is_reasoning:
            logger.warning(
                "Обнаружена reasoning-модель (%s). Ответы могут содержать <think> блоки — "
                "они будут автоматически удалены через strip_reasoning_artifacts().",
                self._model,
            )

    @property
    def tool_strategy(self) -> str:
        """Current tool strategy (may trigger autodetection)."""
        if self._tool_strategy is None:
            self.detect_capabilities()
        return self._tool_strategy  # type: ignore[return-value]

    # ── detect_capabilities ───────────────────────────────────────────────

    def detect_capabilities(self) -> str:
        """Probe the endpoint and cache the best tool strategy.

        Returns the detected strategy string.
        """
        with self._lock:
            if self._tool_strategy is not None:
                return self._tool_strategy

            logger.info("Автодетекция tool_strategy для %s…", self._model)

            if self._probe_native_tools():
                self._tool_strategy = STRATEGY_NATIVE
            elif self._probe_guided_json():
                self._tool_strategy = STRATEGY_GUIDED_JSON
            else:
                self._tool_strategy = STRATEGY_REACT

            self._config.tool_strategy = self._tool_strategy
            logger.info("tool_strategy определена: %s", self._tool_strategy)
            return self._tool_strategy

    # ── generate ──────────────────────────────────────────────────────────

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_schema: dict[str, Any] | None = None,
        params: LLMGenerationParams | None = None,
    ) -> LLMResponse:
        """Send a chat completion request using the active tool strategy.

        Parameters
        ----------
        messages:
            OpenAI-format message list.
        tools:
            Tool schemas (OpenAI tools format).
        response_schema:
            JSON schema for structured output (used in guided_json).
        params:
            Generation parameters override.

        Returns
        -------
        LLMResponse
        """
        strategy = self.tool_strategy

        if strategy == STRATEGY_NATIVE:
            return self._generate_native(messages, tools=tools, params=params)
        if strategy == STRATEGY_GUIDED_JSON:
            return self._generate_guided_json(
                messages, tools=tools, response_schema=response_schema, params=params,
            )
        return self._generate_react(messages, tools=tools, params=params)

    # ── ping ──────────────────────────────────────────────────────────────

    def ping(self) -> PingResult:
        """Health-check: send a tiny request and measure latency."""
        t0 = time.perf_counter()
        try:
            self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            latency = (time.perf_counter() - t0) * 1000
            return PingResult(ok=True, latency_ms=round(latency, 1))
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            return PingResult(
                ok=False,
                latency_ms=round(latency, 1),
                error=str(exc)[:200],
            )

    def list_available_models(self) -> list[str]:
        """Query /v1/models — returns actually loaded models, not YAML hint."""
        try:
            resp = self._client.models.list()
        except Exception:
            return []
        return [m.id for m in getattr(resp, "data", []) if getattr(m, "id", None)]

    # ── Native tools strategy ─────────────────────────────────────────────

    def _generate_native(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        params: LLMGenerationParams | None = None,
    ) -> LLMResponse:
        kwargs = self._base_kwargs(params)
        kwargs["messages"] = messages
        if tools:
            kwargs["tools"] = tools

        raw = self._call_api(**kwargs)
        choice = raw.choices[0]

        tool_calls: list[ToolCall] | None = None
        if choice.message.tool_calls:
            tool_calls = []
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {"_raw": tc.function.arguments}
                tool_calls.append(
                    ToolCall(
                        id=tc.id or "",
                        name=tc.function.name,
                        arguments=args,
                    ),
                )

        return LLMResponse(
            content=choice.message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "",
            usage=self._extract_usage(raw),
        )

    # ── Guided JSON strategy ──────────────────────────────────────────────

    def _generate_guided_json(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_schema: dict[str, Any] | None = None,
        params: LLMGenerationParams | None = None,
    ) -> LLMResponse:
        msgs = list(messages)
        schema = response_schema
        if tools and not schema:
            schema = self._tools_to_action_schema(tools)

        if schema:
            instruction = _GUIDED_JSON_INSTRUCTION.replace(
                "{schema}", json.dumps(schema, ensure_ascii=False, indent=2),
            )
            msgs = self._append_to_last_user(msgs, instruction)

        kwargs = self._base_kwargs(params)
        kwargs["messages"] = msgs

        raw = self._call_api(**kwargs)
        choice = raw.choices[0]
        text = choice.message.content or ""

        tool_calls = self._parse_json_tool_calls(text) if tools else None

        return LLMResponse(
            content=text,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "",
            usage=self._extract_usage(raw),
        )

    # ── ReAct strategy ────────────────────────────────────────────────────

    def _generate_react(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        params: LLMGenerationParams | None = None,
    ) -> LLMResponse:
        msgs = list(messages)
        if tools:
            tool_list = ", ".join(
                t.get("function", {}).get("name", "?") for t in tools
            )
            instruction = (
                f"\n\nДоступные инструменты: {tool_list}"
                + _REACT_INSTRUCTION_RU
            )
            msgs = self._append_to_last_user(msgs, instruction)

        kwargs = self._base_kwargs(params)
        kwargs["messages"] = msgs

        raw = self._call_api(**kwargs)
        choice = raw.choices[0]
        text = choice.message.content or ""

        tool_calls = self._parse_react_tool_calls(text) if tools else None

        return LLMResponse(
            content=text,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "",
            usage=self._extract_usage(raw),
        )

    # ── API call with retry ───────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type((LLMConnectionError, LLMTimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def _call_api(self, **kwargs: Any) -> Any:
        try:
            return self._client.chat.completions.create(**kwargs)
        except APIConnectionError as exc:
            raise LLMConnectionError(
                f"Не удалось подключиться к {self._config.url}: {exc}",
            ) from exc
        except APITimeoutError as exc:
            raise LLMTimeoutError(
                f"Таймаут запроса к {self._config.url}: {exc}",
            ) from exc
        except OpenAIRateLimitError as exc:
            raise LLMRateLimitError(
                f"Rate limit от {self._config.url}: {exc}",
            ) from exc
        except APIStatusError as exc:
            # 401 → token likely stale (e.g. GigaChat clock skew).  Drop the
            # cached token, retry once inline, then surface the error.
            if exc.status_code == 401 and not getattr(
                self._retry_state, "retrying_401", False
            ):
                logger.warning(
                    "401 от %s — инвалидирую token и повторяю запрос один раз",
                    self._config.url,
                )
                self._token_provider.invalidate()
                self._retry_state.retrying_401 = True
                try:
                    return self._client.chat.completions.create(**kwargs)
                except APIStatusError as exc2:
                    raise LLMConnectionError(
                        f"HTTP {exc2.status_code} от {self._config.url} "
                        f"(после повтора с обновлённым token): {exc2}",
                    ) from exc2
                finally:
                    self._retry_state.retrying_401 = False
            raise LLMConnectionError(
                f"HTTP ошибка {exc.status_code} от {self._config.url}: {exc}",
            ) from exc

    # ── Probes for autodetection ──────────────────────────────────────────

    def _probe_native_tools(self) -> bool:
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=_PROBE_MESSAGES,  # type: ignore[arg-type]
                tools=[_PROBE_TOOL],  # type: ignore[list-item]
                max_tokens=100,
            )
            choice = resp.choices[0]
            if choice.message.tool_calls:
                logger.debug("Native tools: поддерживается")
                return True
        except Exception:
            logger.debug("Native tools: не поддерживается", exc_info=True)
        return False

    def _probe_guided_json(self) -> bool:
        probe_msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": (
                    'Return JSON: {"action": "test", "value": "ok"}. '
                    "Nothing else."
                ),
            },
        ]
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=probe_msgs,  # type: ignore[arg-type]
                max_tokens=60,
                extra_body={"guided_json": json.dumps(_GUIDED_JSON_PROBE_SCHEMA)},
            )
            text = resp.choices[0].message.content or ""
            json.loads(text)
            logger.debug("Guided JSON: поддерживается (extra_body)")
            return True
        except Exception:
            pass

        try:
            resp = self._client.chat.completions.create(  # type: ignore[call-overload]
                model=self._model,
                messages=probe_msgs,
                max_tokens=60,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content or ""
            json.loads(text)
            logger.debug("Guided JSON: поддерживается (response_format)")
            return True
        except Exception:
            pass

        logger.debug("Guided JSON: не поддерживается")
        return False

    # ── Parsing helpers ───────────────────────────────────────────────────

    @staticmethod
    def _parse_json_tool_calls(text: str) -> list[ToolCall] | None:
        """Try to extract tool call from JSON text returned by guided_json."""
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        if data.get("action") == "tool_call" and isinstance(data.get("tool_call"), dict):
            tc = data["tool_call"]
            return [
                ToolCall(
                    name=tc.get("name", ""),
                    arguments=tc.get("args", tc.get("arguments", {})),
                ),
            ]

        if "name" in data and "arguments" in data:
            return [
                ToolCall(name=data["name"], arguments=data["arguments"]),
            ]

        return None

    @staticmethod
    def _parse_react_tool_calls(text: str) -> list[ToolCall] | None:
        """Parse ReAct-style Мысль/Действие/Параметры output."""
        for pattern in (_REACT_ACTION_RE, _REACT_ACTION_EN_RE):
            m = pattern.search(text)
            if not m:
                continue
            name = m.group(1).strip()
            # Parse the JSON that starts at m.end() — use raw_decode so
            # nested objects (e.g. {"filters": {"a": 1}}) are handled and
            # trailing prose ("Мысль: ...") is ignored.
            tail = text[m.end():].lstrip()
            if not tail.startswith("{"):
                return [ToolCall(name=name, arguments={"_raw": tail[:200]})]
            try:
                obj, _ = json.JSONDecoder().raw_decode(tail)
                args = obj if isinstance(obj, dict) else {"_raw": str(obj)[:200]}
            except json.JSONDecodeError:
                args = {"_raw": tail[:200]}
            return [ToolCall(name=name, arguments=args)]
        return None

    @staticmethod
    def _tools_to_action_schema(tools: list[dict[str, Any]]) -> dict[str, Any]:
        """Build a JSON schema that wraps tool_call selection."""
        tool_names = [t.get("function", {}).get("name", "") for t in tools]
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["tool_call", "final_answer"],
                },
                "tool_call": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": tool_names},
                        "args": {"type": "object"},
                    },
                },
                "final_answer": {"type": "object"},
            },
            "required": ["action"],
        }

    @staticmethod
    def _append_to_last_user(
        messages: list[dict[str, Any]],
        suffix: str,
    ) -> list[dict[str, Any]]:
        """Return a copy with suffix appended to the last user message."""
        msgs = [dict(m) for m in messages]
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "user":
                msgs[i]["content"] = (msgs[i].get("content") or "") + suffix
                break
        return msgs

    def _base_kwargs(self, params: LLMGenerationParams | None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": self._model}
        if params:
            kwargs["temperature"] = params.temperature
            kwargs["top_p"] = params.top_p
            kwargs["max_tokens"] = params.max_tokens
        return kwargs

    @staticmethod
    def strip_reasoning_artifacts(text: str) -> str:
        """Remove reasoning model artifacts (<think> blocks, JSON wrappers).

        Safe to call on any LLM output that should be plain text.
        Do NOT use on responses expected to contain JSON tool calls.
        """
        # 0. Cut at the first ChatML turn boundary — if the model keeps
        #    talking past <|im_end|> or opens a new <|im_start|> turn,
        #    drop everything from that point on (including follow-up
        #    "<|im_start|>assistant <think>…" leaks).
        cuts = [text.find(tok) for tok in ("<|im_end|>", "<|im_start|>")]
        cuts = [c for c in cuts if c >= 0]
        if cuts:
            text = text[: min(cuts)]

        # 1. Remove <think>...</think> blocks (possibly nested, DOTALL)
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
        # Handle unclosed <think> — drop everything before </think>
        if "<think>" in text.lower():
            parts = re.split(r"</think>", text, flags=re.IGNORECASE, maxsplit=1)
            text = parts[-1] if len(parts) > 1 else ""

        # 2. Unwrap {"action": "final_answer", "final_answer": ...}
        m = re.search(
            r'\{\s*"action"\s*:\s*"final_answer".*?"final_answer"\s*:\s*(\{.*\}|".*?")\s*\}',
            text,
            flags=re.DOTALL,
        )
        if m:
            try:
                obj = json.loads(m.group(0))
                fa = obj.get("final_answer", "")
                if isinstance(fa, dict):
                    text = "\n\n".join(
                        str(fa.get(k, "")).strip()
                        for k in ("summary", "risk_areas", "recommendations")
                        if fa.get(k)
                    )
                elif isinstance(fa, str):
                    text = fa
            except (json.JSONDecodeError, AttributeError):
                pass

        # 3. Strip ```...``` wrappers
        text = re.sub(r"^```(?:json|markdown|text)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```\s*$", "", text, flags=re.MULTILINE)

        return text.strip()

    @staticmethod
    def _extract_usage(raw: Any) -> TokenUsage:
        if raw.usage:
            return TokenUsage(
                prompt_tokens=raw.usage.prompt_tokens or 0,
                completion_tokens=raw.usage.completion_tokens or 0,
                total_tokens=raw.usage.total_tokens or 0,
            )
        return TokenUsage()
