"""Document ingestion pipeline — Track D, Level 3.

7-stage pipeline: parse → preprocess → identify_model → chunk →
enrich → embed → upsert.  Checkpointed and resumable.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import pickle
import re
import time
import unicodedata
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import tiktoken
import yaml

from rag.embeddings import BGEEmbedder
from rag.qdrant_client import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, QdrantManager

if TYPE_CHECKING:
    import numpy as np
    from qdrant_client.http.models import SparseVector

    from config.loader import (
        ChunkingStrategyConfig,
        PIIFilterConfig,
        RAGConfig,
    )

logger = logging.getLogger(__name__)

CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent / "storage" / "ingestion_checkpoints"
PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "agent" / "prompts" / "identify_model.md"
)
ERROR_PATTERNS_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "error_code_patterns.yaml"
)
COMPONENT_VOCAB_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "component_vocabulary.yaml"
)
MODEL_ALIASES_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "model_aliases.yaml"
)

_TOKENIZER: tiktoken.Encoding | None = None

PIPELINE_STEPS = [
    "parse",
    "preprocess",
    "identify_model",
    "chunk",
    "enrich",
    "embed",
    "upsert",
]

_OCR_ARTIFACTS_RE = re.compile(r"[^\S\n]{2,}|[\x00-\x08\x0b\x0c\x0e-\x1f]")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_BROKEN_SENTENCE_RE = re.compile(r"(\w)-\n(\w)")

ProgressCallback = Callable[[str, float], None]


# ── Custom errors ───────────────────────────────────────────────────────────


class IngestionError(Exception):
    """Generic pipeline error."""


class ParseError(IngestionError):
    """PDF/document parsing failed."""


class ChunkingError(IngestionError):
    """Chunking stage failed."""


# ── Data models ─────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Result of stage 1 — raw document content."""

    file_path: Path
    file_hash: str
    file_name: str
    pages: list[str]
    bookmarks: list[tuple[int, str, int]]
    total_pages: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CleanedDocument:
    """Result of stage 2 — preprocessed text."""

    pages: list[str]
    full_text: str
    file_hash: str
    file_name: str
    bookmarks: list[tuple[int, str, int]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Result of stage 3 — identified model/vendor."""

    vendor: str | None = None
    model: str | None = None
    model_family: str | None = None
    source: str = "unknown"


@dataclass(slots=True)
class Chunk:
    """Result of stage 4 — single text chunk."""

    chunk_id: str
    document_id: str
    text: str
    page_number: int | None = None
    section: str | None = None
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EnrichedChunk:
    """Result of stage 5 — chunk with enriched metadata."""

    chunk_id: str
    document_id: str
    text: str
    page_number: int | None = None
    section: str | None = None
    token_count: int = 0
    error_codes: list[str] = field(default_factory=list)
    components: list[str] = field(default_factory=list)
    content_type: str = "reference"
    vendor: str | None = None
    model: str | None = None
    models: list[str] = field(default_factory=list)
    model_family: str | None = None
    language: str = "ru"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EmbeddedChunk:
    """Result of stage 6 — chunk with vectors."""

    chunk_id: str
    document_id: str
    text: str
    dense_vector: np.ndarray
    sparse_vector: SparseVector | None = None
    page_number: int | None = None
    section: str | None = None
    token_count: int = 0
    error_codes: list[str] = field(default_factory=list)
    components: list[str] = field(default_factory=list)
    content_type: str = "reference"
    vendor: str | None = None
    model: str | None = None
    models: list[str] = field(default_factory=list)
    model_family: str | None = None
    language: str = "ru"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IndexingResult:
    """Final result of the ingestion pipeline."""

    document_id: str
    chunks_count: int
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    metadata_identified: ModelMetadata | None = None
    embedding_version: str = ""


# ── LLM protocol ───────────────────────────────────────────────────────────


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


# ── Tokenizer helper ───────────────────────────────────────────────────────


def _get_tokenizer() -> tiktoken.Encoding:
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = tiktoken.get_encoding("cl100k_base")
    return _TOKENIZER


def count_tokens(text: str) -> int:
    return len(_get_tokenizer().encode(text))


# ── Checkpoint helpers ──────────────────────────────────────────────────────


def _checkpoint_path(file_hash: str, step: str) -> Path:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINTS_DIR / f"{file_hash}_{step}.pkl"


def _save_checkpoint(file_hash: str, step: str, data: Any) -> None:
    path = _checkpoint_path(file_hash, step)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.debug("Checkpoint сохранён: %s", path.name)


def _load_checkpoint(file_hash: str, step: str) -> Any | None:
    path = _checkpoint_path(file_hash, step)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        logger.warning("Не удалось загрузить checkpoint %s", path.name)
        return None


def _find_last_checkpoint(file_hash: str) -> tuple[str | None, Any | None]:
    for step in reversed(PIPELINE_STEPS[:-1]):
        data = _load_checkpoint(file_hash, step)
        if data is not None:
            return step, data
    return None, None


def _clear_checkpoints(file_hash: str) -> None:
    for step in PIPELINE_STEPS:
        path = _checkpoint_path(file_hash, step)
        if path.exists():
            path.unlink()


# ── Stage 1: Parse ──────────────────────────────────────────────────────────


_LARGE_PDF_THRESHOLD = 20 * 1024 * 1024  # 20 MB


def parse_document(file_path: Path) -> ParsedDocument:
    """Stage 1: parse a PDF file.

    Uses pdftotext (poppler) for large files (>20MB) because pymupdf
    hangs on very large PDFs (1000+ pages, 200K+ objects).
    Falls back to pymupdf for smaller files and bookmark extraction.
    """
    file_bytes = file_path.read_bytes()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    file_size = len(file_bytes)

    if file_size > _LARGE_PDF_THRESHOLD:
        return _parse_with_pdftotext(file_path, file_hash, file_bytes)
    return _parse_with_pymupdf(file_path, file_hash)


def _parse_with_pdftotext(
    file_path: Path, file_hash: str, file_bytes: bytes
) -> ParsedDocument:
    import subprocess
    import tempfile

    logger.info("Большой PDF (%d МБ), используем pdftotext", len(file_bytes) // (1024 * 1024))

    try:
        proc = subprocess.run(
            ["pdftotext", "-layout", str(file_path), "-"],
            capture_output=True, timeout=300,
        )
        if proc.returncode != 0:
            raise ParseError(f"pdftotext ошибка: {proc.stderr.decode(errors='replace')}")
        full_text = proc.stdout.decode("utf-8", errors="replace")
    except FileNotFoundError:
        raise ParseError("pdftotext не установлен (apt install poppler-utils)")
    except subprocess.TimeoutExpired:
        raise ParseError(f"pdftotext: таймаут на файле {file_path.name}")

    raw_pages = full_text.split("\f")
    pages = [p for p in raw_pages if p.strip()]
    total_pages = len(pages)

    bookmarks: list[tuple[int, str, int]] = []

    logger.info(
        "Парсинг (pdftotext): %s — %d страниц, %d закладок, hash=%s",
        file_path.name, total_pages, len(bookmarks), file_hash[:12],
    )

    return ParsedDocument(
        file_path=file_path,
        file_hash=file_hash,
        file_name=file_path.name,
        pages=pages,
        bookmarks=bookmarks,
        total_pages=total_pages,
    )


def _parse_with_pymupdf(file_path: Path, file_hash: str) -> ParsedDocument:
    import fitz

    try:
        doc = fitz.open(file_path)
    except Exception as exc:
        raise ParseError(f"Не удалось открыть PDF {file_path.name}: {exc}") from exc

    pages: list[str] = []
    for page in doc:
        pages.append(page.get_text("text"))

    bookmarks: list[tuple[int, str, int]] = []
    toc = doc.get_toc(simple=True)
    for level, title, page_num in toc:
        bookmarks.append((level, title, page_num))

    total_pages = len(doc)
    doc.close()

    logger.info(
        "Парсинг (pymupdf): %s — %d страниц, %d закладок, hash=%s",
        file_path.name, total_pages, len(bookmarks), file_hash[:12],
    )

    return ParsedDocument(
        file_path=file_path,
        file_hash=file_hash,
        file_name=file_path.name,
        pages=pages,
        bookmarks=bookmarks,
        total_pages=total_pages,
    )


# ── Stage 2: Preprocess ────────────────────────────────────────────────────


def preprocess_text(doc: ParsedDocument) -> CleanedDocument:
    """Stage 2: clean OCR artifacts, normalize whitespace, join broken sentences."""
    cleaned_pages: list[str] = []
    for page_text in doc.pages:
        text = _BROKEN_SENTENCE_RE.sub(r"\1\2", page_text)
        text = _OCR_ARTIFACTS_RE.sub(" ", text)
        text = _MULTI_NEWLINE_RE.sub("\n\n", text)
        text = unicodedata.normalize("NFC", text)
        text = text.strip()
        cleaned_pages.append(text)

    full_text = "\n\n".join(cleaned_pages)

    logger.info("Предобработка: %d страниц, %d символов итого", len(cleaned_pages), len(full_text))

    return CleanedDocument(
        pages=cleaned_pages,
        full_text=full_text,
        file_hash=doc.file_hash,
        file_name=doc.file_name,
        bookmarks=doc.bookmarks,
        metadata=doc.metadata,
    )


# ── Stage 3: Identify model and vendor ──────────────────────────────────────


def _load_model_aliases() -> dict[str, list[str]]:
    if not MODEL_ALIASES_PATH.exists():
        return {}
    with open(MODEL_ALIASES_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _extract_model_from_filename(file_name: str) -> ModelMetadata | None:
    aliases = _load_model_aliases()
    name_lower = file_name.lower()
    for canonical, alias_list in aliases.items():
        for alias in alias_list:
            if alias.lower() in name_lower:
                parts = canonical.split(maxsplit=1)
                vendor = parts[0] if parts else None
                return ModelMetadata(
                    vendor=vendor,
                    model=canonical,
                    source="filename",
                )
    return None


def _extract_model_from_text(text: str) -> ModelMetadata | None:
    """Try to find a known model name in the text (first page content)."""
    aliases = _load_model_aliases()
    text_lower = text.lower()
    for canonical, alias_list in aliases.items():
        if canonical.lower() in text_lower:
            parts = canonical.split(maxsplit=1)
            return ModelMetadata(vendor=parts[0] if parts else None, model=canonical, source="text")
        for alias in alias_list:
            if alias.lower() in text_lower:
                parts = canonical.split(maxsplit=1)
                return ModelMetadata(vendor=parts[0] if parts else None, model=canonical, source="text")
    return None


def identify_model_and_vendor(
    doc: CleanedDocument,
    llm_client: LLMClient | None = None,
) -> ModelMetadata:
    """Stage 3: identify model/vendor via LLM, fallback to filename regex."""
    if llm_client and PROMPT_PATH.exists():
        try:
            template = PROMPT_PATH.read_text(encoding="utf-8")
            first_page = doc.pages[0] if doc.pages else ""
            prompt = template.replace("{first_page_text}", first_page[:3000])
            prompt = prompt.replace("{file_name}", doc.file_name)
            raw = llm_client.complete(prompt)
            parsed = _parse_model_json(raw)
            if parsed and (parsed.vendor or parsed.model):
                logger.info("Модель определена LLM: vendor=%s, model=%s", parsed.vendor, parsed.model)
                return parsed
        except Exception:
            logger.warning("LLM определение модели не удалось, используем fallback")

    from_filename = _extract_model_from_filename(doc.file_name)
    if from_filename:
        logger.info("Модель определена из имени файла: %s", from_filename.model)
        return from_filename

    first_page = doc.pages[0] if doc.pages else ""
    from_text = _extract_model_from_text(first_page)
    if from_text:
        logger.info("Модель определена из текста: %s", from_text.model)
        return from_text

    logger.info("Модель не определена")
    return ModelMetadata(source="none")


def _parse_model_json(raw: str) -> ModelMetadata | None:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        data = json.loads(text)
        return ModelMetadata(
            vendor=data.get("vendor"),
            model=data.get("model"),
            model_family=data.get("model_family"),
            source="llm",
        )
    except (json.JSONDecodeError, AttributeError):
        return None


# ── Stage 4: Chunking ──────────────────────────────────────────────────────


def chunk_document(
    doc: CleanedDocument,
    collection: str,
    config: RAGConfig,
) -> list[Chunk]:
    """Stage 4: chunk document according to collection strategy."""
    chunk_cfg = config.chunking.get(collection)
    if not chunk_cfg:
        logger.warning("Нет конфигурации чанкинга для %s, используем recursive", collection)
        from config.loader import ChunkingStrategyConfig
        chunk_cfg = ChunkingStrategyConfig(strategy="recursive", max_tokens=500, overlap_tokens=80)

    document_id = _make_document_id(doc.file_name, doc.file_hash)
    strategy = chunk_cfg.strategy

    if strategy == "hierarchical":
        chunks = _chunk_hierarchical(doc, document_id, chunk_cfg)
    elif strategy == "per_record":
        chunks = _chunk_per_record(doc, document_id, chunk_cfg)
    elif strategy == "recursive":
        chunks = _chunk_recursive(doc, document_id, chunk_cfg)
    else:
        raise ChunkingError(f"Неизвестная стратегия чанкинга: {strategy}")

    for chunk in chunks:
        chunk.token_count = count_tokens(chunk.text)

    logger.info(
        "Чанкинг [%s]: стратегия=%s, чанков=%d",
        collection, strategy, len(chunks),
    )
    return chunks


def _make_document_id(file_name: str, file_hash: str) -> str:
    stem = Path(file_name).stem
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", stem)[:60]
    return f"{safe}_{file_hash[:8]}"


def _chunk_hierarchical(
    doc: CleanedDocument,
    document_id: str,
    cfg: ChunkingStrategyConfig,
) -> list[Chunk]:
    max_tokens = cfg.max_tokens or 800
    overlap_tokens = cfg.overlap_tokens or 100

    sections = _split_by_bookmarks(doc) if cfg.use_bookmarks and doc.bookmarks else _split_by_headings(doc)

    if not sections:
        sections = [(_SectionInfo("Document", None), doc.full_text)]

    chunks: list[Chunk] = []
    for section_info, text in sections:
        tokens = count_tokens(text)
        if tokens <= max_tokens:
            chunks.append(Chunk(
                chunk_id=f"{document_id}_c{len(chunks)}",
                document_id=document_id,
                text=text,
                page_number=section_info.page,
                section=section_info.title,
            ))
        else:
            sub_chunks = _recursive_split(text, max_tokens, overlap_tokens)
            for sub in sub_chunks:
                chunks.append(Chunk(
                    chunk_id=f"{document_id}_c{len(chunks)}",
                    document_id=document_id,
                    text=sub,
                    page_number=section_info.page,
                    section=section_info.title,
                ))

    return chunks


@dataclass
class _SectionInfo:
    title: str
    page: int | None


def _split_by_bookmarks(
    doc: CleanedDocument,
) -> list[tuple[_SectionInfo, str]]:
    if not doc.bookmarks:
        return []

    relevant = [(lvl, title, pg) for lvl, title, pg in doc.bookmarks if lvl <= 3]
    if not relevant:
        return []

    sections: list[tuple[_SectionInfo, str]] = []
    for i, (_, title, page_num) in enumerate(relevant):
        page_idx = max(0, page_num - 1)
        next_page_idx = (
            max(0, relevant[i + 1][2] - 1) if i + 1 < len(relevant) else len(doc.pages)
        )

        text_parts = doc.pages[page_idx:next_page_idx]
        text = "\n\n".join(text_parts).strip()
        if text:
            sections.append((_SectionInfo(title, page_num), text))

    return sections


def _split_by_headings(doc: CleanedDocument) -> list[tuple[_SectionInfo, str]]:
    heading_re = re.compile(r"^(#{1,3}\s+.+|[A-ZА-ЯЁ][A-ZА-ЯЁ\s]{5,}$)", re.MULTILINE)
    parts = heading_re.split(doc.full_text)

    sections: list[tuple[_SectionInfo, str]] = []
    current_title = "Introduction"
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if heading_re.fullmatch(part):
            current_title = part.strip("#").strip()
        else:
            sections.append((_SectionInfo(current_title, None), part))

    return sections


def _chunk_per_record(
    doc: CleanedDocument,
    document_id: str,
    cfg: ChunkingStrategyConfig,
) -> list[Chunk]:
    separator = cfg.record_separator
    if separator:
        pattern = re.compile(separator, re.MULTILINE)
        parts = pattern.split(doc.full_text)
    else:
        parts = doc.full_text.split("\n\n")

    chunks: list[Chunk] = []
    for part in parts:
        text = part.strip()
        if not text:
            continue
        chunks.append(Chunk(
            chunk_id=f"{document_id}_c{len(chunks)}",
            document_id=document_id,
            text=text,
        ))

    return chunks


def _chunk_recursive(
    doc: CleanedDocument,
    document_id: str,
    cfg: ChunkingStrategyConfig,
) -> list[Chunk]:
    max_tokens = cfg.max_tokens or 500
    overlap_tokens = cfg.overlap_tokens or 80

    sub_chunks = _recursive_split(doc.full_text, max_tokens, overlap_tokens)

    chunks: list[Chunk] = []
    for text in sub_chunks:
        chunks.append(Chunk(
            chunk_id=f"{document_id}_c{len(chunks)}",
            document_id=document_id,
            text=text,
        ))

    return chunks


def _recursive_split(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    tokenizer = _get_tokenizer()
    tokens = tokenizer.encode(text)

    if len(tokens) <= max_tokens:
        return [text]

    separators = ["\n\n", "\n", ". ", " "]
    for sep in separators:
        parts = text.split(sep)
        if len(parts) > 1:
            break
    else:
        parts = [text]

    result: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for part in parts:
        part_tokens = len(tokenizer.encode(part))

        if current_tokens + part_tokens > max_tokens and current:
            result.append(sep.join(current))
            overlap_text = sep.join(current)
            overlap_enc = tokenizer.encode(overlap_text)
            if len(overlap_enc) > overlap_tokens:
                tail = tokenizer.decode(overlap_enc[-overlap_tokens:])
                current = [tail]
                current_tokens = overlap_tokens
            else:
                current_tokens = len(overlap_enc)
        current.append(part)
        current_tokens += part_tokens

    if current:
        result.append(sep.join(current))

    return result


# ── Stage 5: Enrich metadata ───────────────────────────────────────────────


def _load_error_patterns(vendor: str | None = None) -> list[re.Pattern[str]]:
    if not ERROR_PATTERNS_PATH.exists():
        return [re.compile(r"[CJEFS]\d{3,5}"), re.compile(r"SC\d{3}")]

    with open(ERROR_PATTERNS_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return [re.compile(r"[CJEFS]\d{3,5}")]

    patterns: list[re.Pattern[str]] = []
    vendor_key = vendor.lower() if vendor else "generic"

    for key in (vendor_key, "generic"):
        raw_patterns = data.get(key, [])
        for p in raw_patterns:
            with contextlib.suppress(re.error):
                patterns.append(re.compile(p))

    return patterns or [re.compile(r"[CJEFS]\d{3,5}")]


def _load_component_vocab() -> dict[str, list[str]]:
    if not COMPONENT_VOCAB_PATH.exists():
        return {}
    with open(COMPONENT_VOCAB_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _extract_error_codes(text: str, patterns: list[re.Pattern[str]]) -> list[str]:
    codes: set[str] = set()
    for pattern in patterns:
        for match in pattern.finditer(text):
            codes.add(match.group().upper())
    return sorted(codes)


def _extract_components(text: str, vocab: dict[str, list[str]]) -> list[str]:
    text_lower = text.lower()
    found: set[str] = set()
    for canonical, synonyms in vocab.items():
        for syn in synonyms:
            if syn.lower() in text_lower:
                found.add(canonical)
                break
        if canonical.lower() in text_lower:
            found.add(canonical)
    return sorted(found)


def _apply_pii_filter(text: str, pii_config: PIIFilterConfig) -> str:
    if not pii_config.enabled:
        return text
    for pii_pattern in pii_config.patterns:
        with contextlib.suppress(re.error):
            text = re.sub(pii_pattern.pattern, pii_pattern.replacement, text)
    return text


def _classify_content_type_batch(
    texts: list[str],
    llm_client: LLMClient | None,
    fallback: str = "reference",
) -> list[str]:
    if not llm_client:
        return [fallback] * len(texts)

    prompt_parts = [
        "Для каждого чанка ниже отнеси его к одной из категорий: "
        "symptom | cause | procedure | specification | reference.\n"
        "Верни JSON массив classifications — строки категорий, по одной на чанк.\n\n"
    ]
    for i, text in enumerate(texts):
        snippet = text[:500]
        prompt_parts.append(f"Чанк {i + 1}:\n{snippet}\n")

    prompt = "\n".join(prompt_parts)

    try:
        raw = llm_client.complete(prompt)
        return _parse_classification_response(raw, len(texts), fallback)
    except Exception:
        logger.warning("LLM классификация не удалась, используем fallback=%s", fallback)
        return [fallback] * len(texts)


def _parse_classification_response(
    raw: str, expected: int, fallback: str
) -> list[str]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines)

    valid_types = {"symptom", "cause", "procedure", "specification", "reference"}

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("classifications", [])
        if isinstance(data, list):
            result = []
            for item in data:
                val = str(item).lower().strip()
                result.append(val if val in valid_types else fallback)
            while len(result) < expected:
                result.append(fallback)
            return result[:expected]
    except (json.JSONDecodeError, AttributeError):
        pass

    return [fallback] * expected


def enrich_metadata(
    chunks: list[Chunk],
    model_meta: ModelMetadata,
    config: RAGConfig,
    llm_client: LLMClient | None = None,
) -> list[EnrichedChunk]:
    """Stage 5: enrich chunks with error codes, components, content_type, PII filter."""
    error_patterns = _load_error_patterns(model_meta.vendor)
    component_vocab = _load_component_vocab()
    pii_config = config.pii_filter

    enrichment_cfg = config.ingestion.llm_enrichment
    batch_size = enrichment_cfg.classify_batch_size
    fallback_type = enrichment_cfg.fallback_content_type
    use_llm = enrichment_cfg.enabled and llm_client is not None

    models_list: list[str] = []
    if model_meta.model:
        models_list.append(model_meta.model)

    content_types: list[str] = []
    if use_llm:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_types = _classify_content_type_batch(
                [c.text for c in batch], llm_client, fallback_type,
            )
            content_types.extend(batch_types)
    else:
        content_types = [fallback_type] * len(chunks)

    enriched: list[EnrichedChunk] = []
    for idx, chunk in enumerate(chunks):
        filtered_text = _apply_pii_filter(chunk.text, pii_config)
        error_codes = _extract_error_codes(chunk.text, error_patterns)
        components = _extract_components(chunk.text, component_vocab)

        enriched.append(EnrichedChunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            text=filtered_text,
            page_number=chunk.page_number,
            section=chunk.section,
            token_count=chunk.token_count,
            error_codes=error_codes,
            components=components,
            content_type=content_types[idx] if idx < len(content_types) else fallback_type,
            vendor=model_meta.vendor,
            model=model_meta.model,
            models=models_list,
            model_family=model_meta.model_family,
            metadata=chunk.metadata,
        ))

    logger.info(
        "Обогащение: %d чанков, error_codes найдены в %d, components в %d",
        len(enriched),
        sum(1 for e in enriched if e.error_codes),
        sum(1 for e in enriched if e.components),
    )
    return enriched


# ── Stage 6: Compute embeddings ─────────────────────────────────────────────


def compute_embeddings(
    chunks: list[EnrichedChunk],
    embedder: BGEEmbedder,
) -> list[EmbeddedChunk]:
    """Stage 6: produce dense + sparse vectors via BGEEmbedder."""
    if not chunks:
        return []

    texts = [c.text for c in chunks]
    result = embedder.encode(texts, return_sparse=True)

    embedded: list[EmbeddedChunk] = []
    for idx, chunk in enumerate(chunks):
        sparse_vec = result.sparse[idx] if result.sparse else None
        embedded.append(EmbeddedChunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            text=chunk.text,
            dense_vector=result.dense[idx],
            sparse_vector=sparse_vec,
            page_number=chunk.page_number,
            section=chunk.section,
            token_count=chunk.token_count,
            error_codes=chunk.error_codes,
            components=chunk.components,
            content_type=chunk.content_type,
            vendor=chunk.vendor,
            model=chunk.model,
            models=chunk.models,
            model_family=chunk.model_family,
            language=chunk.language,
            metadata=chunk.metadata,
        ))

    logger.info("Эмбеддинги: %d чанков закодировано", len(embedded))
    return embedded


# ── Stage 7: Upsert to Qdrant ──────────────────────────────────────────────


def upsert_to_qdrant(
    embedded_chunks: list[EmbeddedChunk],
    collection: str,
    qdrant_manager: QdrantManager,
    embedder: BGEEmbedder,
    batch_size: int = 100,
) -> None:
    """Stage 7: delete old chunks for this document, then upsert new ones."""
    from qdrant_client.http.models import FieldCondition, Filter, MatchValue, PointStruct

    if not embedded_chunks:
        return

    client = qdrant_manager.rest_client
    document_id = embedded_chunks[0].document_id

    try:
        client.delete(
            collection_name=collection,
            points_selector=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
        )
        logger.info("Удалены старые чанки document_id=%s из %s", document_id, collection)
    except Exception:
        logger.debug("Не удалось удалить старые чанки (возможно, их нет)")

    now_iso = datetime.now(tz=UTC).isoformat()
    embed_version = embedder.embedding_version()

    for i in range(0, len(embedded_chunks), batch_size):
        batch = embedded_chunks[i : i + batch_size]
        points = []

        for chunk in batch:
            payload: dict[str, Any] = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "collection": collection,
                "vendor": chunk.vendor,
                "models": chunk.models,
                "model": chunk.model,
                "model_family": chunk.model_family,
                "doc_title": chunk.document_id,
                "section": chunk.section,
                "page_number": chunk.page_number,
                "content_type": chunk.content_type,
                "error_codes": chunk.error_codes,
                "components": chunk.components,
                "text": chunk.text,
                "language": chunk.language,
                "indexed_at": now_iso,
                "chunk_size_tokens": chunk.token_count,
                "embedding_model": embed_version,
                "source_hash": chunk.metadata.get("source_hash", ""),
            }

            vectors: dict[str, Any] = {
                DENSE_VECTOR_NAME: chunk.dense_vector.tolist(),
            }
            if chunk.sparse_vector is not None and chunk.sparse_vector.indices:
                vectors[SPARSE_VECTOR_NAME] = chunk.sparse_vector

            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            points.append(PointStruct(
                id=point_id,
                vector=vectors,
                payload=payload,
            ))

        client.upsert(collection_name=collection, points=points)
        logger.debug("Upsert батч %d-%d в %s", i, i + len(batch), collection)

    logger.info("Upsert завершён: %d чанков в %s", len(embedded_chunks), collection)


# ── Main pipeline ───────────────────────────────────────────────────────────


def index_document(
    file_path: Path,
    collection: str,
    qdrant_manager: QdrantManager,
    embedder: BGEEmbedder,
    config: RAGConfig,
    *,
    llm_client: LLMClient | None = None,
    metadata_override: dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> IndexingResult:
    """Run the full 7-stage ingestion pipeline for a single document.

    Checkpoints are saved after each stage for resumability.
    """
    start = time.monotonic()
    errors: list[str] = []

    def _progress(step: str, pct: float) -> None:
        if progress_callback:
            progress_callback(step, pct)

    # Check if file has already been indexed (same hash)
    file_bytes = file_path.read_bytes()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    last_step, last_data = _find_last_checkpoint(file_hash)
    step_idx = PIPELINE_STEPS.index(last_step) + 1 if last_step else 0

    # Stage 1: Parse
    if step_idx <= 0:
        _progress("parse", 0.0)
        try:
            parsed = parse_document(file_path)
        except ParseError as exc:
            return IndexingResult(
                document_id="",
                chunks_count=0,
                errors=[str(exc)],
                duration_seconds=time.monotonic() - start,
            )
        _save_checkpoint(file_hash, "parse", parsed)
        _progress("parse", 1.0)
    else:
        parsed = last_data if last_step == "parse" else _load_checkpoint(file_hash, "parse")  # type: ignore[assignment]

    # Stage 2: Preprocess
    if step_idx <= 1:
        _progress("preprocess", 0.0)
        cleaned = preprocess_text(parsed)
        _save_checkpoint(file_hash, "preprocess", cleaned)
        _progress("preprocess", 1.0)
    else:
        cleaned = last_data if last_step == "preprocess" else _load_checkpoint(file_hash, "preprocess")  # type: ignore[assignment]

    # Stage 3: Identify model
    if step_idx <= 2:
        _progress("identify_model", 0.0)
        model_meta = identify_model_and_vendor(cleaned, llm_client)
        if metadata_override:
            model_meta = ModelMetadata(
                vendor=metadata_override.get("vendor", model_meta.vendor),
                model=metadata_override.get("model", model_meta.model),
                model_family=metadata_override.get("model_family", model_meta.model_family),
                source="override",
            )
        _save_checkpoint(file_hash, "identify_model", model_meta)
        _progress("identify_model", 1.0)
    else:
        model_meta = (
            last_data if last_step == "identify_model"  # type: ignore[assignment]
            else _load_checkpoint(file_hash, "identify_model")
        )

    # Stage 4: Chunk
    if step_idx <= 3:
        _progress("chunk", 0.0)
        chunks = chunk_document(cleaned, collection, config)
        _save_checkpoint(file_hash, "chunk", chunks)
        _progress("chunk", 1.0)
    else:
        chunks = last_data if last_step == "chunk" else _load_checkpoint(file_hash, "chunk")  # type: ignore[assignment]

    # Stage 5: Enrich
    if step_idx <= 4:
        _progress("enrich", 0.0)
        enriched = enrich_metadata(chunks, model_meta, config, llm_client)
        for ec in enriched:
            ec.metadata["source_hash"] = file_hash
        _save_checkpoint(file_hash, "enrich", enriched)
        _progress("enrich", 1.0)
    else:
        enriched = last_data if last_step == "enrich" else _load_checkpoint(file_hash, "enrich")  # type: ignore[assignment]

    # Stage 6: Embed
    if step_idx <= 5:
        _progress("embed", 0.0)
        embedded = compute_embeddings(enriched, embedder)
        _save_checkpoint(file_hash, "embed", embedded)
        _progress("embed", 1.0)
    else:
        embedded = last_data if last_step == "embed" else _load_checkpoint(file_hash, "embed")  # type: ignore[assignment]

    # Stage 7: Upsert
    _progress("upsert", 0.0)
    try:
        upsert_to_qdrant(
            embedded,
            collection,
            qdrant_manager,
            embedder,
            batch_size=config.ingestion.upsert_batch_size,
        )
    except Exception as exc:
        errors.append(f"Ошибка upsert: {exc}")
        logger.error("Upsert ошибка: %s", exc)
    _progress("upsert", 1.0)

    _clear_checkpoints(file_hash)

    document_id = embedded[0].document_id if embedded else _make_document_id(file_path.name, file_hash)
    duration = time.monotonic() - start

    logger.info(
        "Индексация завершена: %s → %s, %d чанков, %.1f сек",
        file_path.name, collection, len(embedded), duration,
    )

    return IndexingResult(
        document_id=document_id,
        chunks_count=len(embedded),
        errors=errors,
        duration_seconds=duration,
        metadata_identified=model_meta,
        embedding_version=embedder.embedding_version(),
    )
