"""Per-device-model error code dictionary management."""

from .aliases import (
    AliasConflict,
    add_alias,
    get_canonical_for_alias,
    list_aliases_for_canonical,
    remove_alias,
)
from .consistency import Conflict, find_conflicts, sync_to_all_models
from .loader import (
    ERROR_CODES_ROOT,
    TRASH_DIR,
    invalidate_cache,
    list_models,
    list_vendors,
    load_codes,
    model_slug,
    models_sharing_slug,
    vendor_slug,
)
from .parsers import ParseError, parse_csv, parse_file, parse_xlsx, parse_yaml
from .schema import SUPPORTED_VENDORS, ErrorCode, ModelErrorCodes, SeverityLiteral
from .writer import delete, save

__all__ = [
    "SUPPORTED_VENDORS",
    "SeverityLiteral",
    "ErrorCode",
    "ModelErrorCodes",
    "Conflict",
    "AliasConflict",
    "add_alias",
    "remove_alias",
    "get_canonical_for_alias",
    "list_aliases_for_canonical",
    "ERROR_CODES_ROOT",
    "TRASH_DIR",
    "ParseError",
    "delete",
    "find_conflicts",
    "invalidate_cache",
    "list_models",
    "list_vendors",
    "load_codes",
    "model_slug",
    "models_sharing_slug",
    "parse_csv",
    "parse_file",
    "parse_xlsx",
    "parse_yaml",
    "save",
    "sync_to_all_models",
    "vendor_slug",
]
