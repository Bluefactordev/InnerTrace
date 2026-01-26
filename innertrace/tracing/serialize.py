"""Safe serialization utilities for function tracing.

This module provides safe serialization of function arguments and return values
with automatic redaction, size limiting, and graceful degradation.
"""

from typing import Any, List, Optional
from .redact import redact_payload, truncate_preview


def serialize_safe(
    obj: Any,
    max_repr: int = 500,
    max_items: int = 50,
    redact_keys: Optional[List[str]] = None
) -> Any:
    """
    Safely serialize an object to a JSON-compatible format.

    This function handles:
    - Basic JSON types (str, int, float, bool, None)
    - Collections (list, tuple, dict) with size limiting
    - Non-serializable objects (fallback to repr())
    - Secret redaction for sensitive keys
    - Graceful error handling (never raises)

    Args:
        obj: Object to serialize
        max_repr: Maximum length for repr() strings
        max_items: Maximum number of items in lists/dicts
        redact_keys: Additional keys to redact (beyond standard secrets)

    Returns:
        JSON-serializable representation of obj
    """
    try:
        return _serialize_recursive(obj, max_repr, max_items, redact_keys, depth=0, max_depth=10)
    except Exception:
        # Final safety net - should never reach here due to try-except in recursive calls
        return "<serialization error>"


def _serialize_recursive(
    obj: Any,
    max_repr: int,
    max_items: int,
    redact_keys: Optional[List[str]],
    depth: int,
    max_depth: int
) -> Any:
    """
    Recursively serialize an object with depth limiting.

    Args:
        obj: Object to serialize
        max_repr: Maximum repr length
        max_items: Maximum collection size
        redact_keys: Keys to redact
        depth: Current recursion depth
        max_depth: Maximum recursion depth

    Returns:
        Serialized object
    """
    # Depth limit to prevent infinite recursion
    if depth > max_depth:
        return "<max depth exceeded>"

    # Handle None
    if obj is None:
        return None

    # Handle basic JSON types
    if isinstance(obj, (bool, int, float)):
        return obj

    if isinstance(obj, str):
        # Truncate long strings
        return truncate_preview(obj, max_repr)

    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        try:
            serialized_list = []
            for i, item in enumerate(obj):
                if i >= max_items:
                    remaining = len(obj) - max_items
                    serialized_list.append(f"...(+{remaining} more items)")
                    break
                serialized_list.append(_serialize_recursive(item, max_repr, max_items, redact_keys, depth + 1, max_depth))
            return serialized_list
        except Exception:
            return f"<list serialization error: {truncate_preview(repr(obj), max_repr)}>"

    # Handle dictionaries
    if isinstance(obj, dict):
        try:
            serialized_dict = {}
            item_count = 0
            for key, value in obj.items():
                if item_count >= max_items:
                    remaining = len(obj) - max_items
                    serialized_dict["..."] = f"(+{remaining} more keys)"
                    break

                # Convert key to string if needed
                key_str = str(key) if not isinstance(key, str) else key

                # Check if this key should be redacted
                key_lower = key_str.lower().replace("-", "_").replace(" ", "_")
                should_redact = False

                # Use existing redaction logic from redact.py
                if redact_keys:
                    for redact_key in redact_keys:
                        if redact_key.lower() in key_lower:
                            should_redact = True
                            break

                if should_redact:
                    serialized_dict[key_str] = "***REDACTED***"
                else:
                    serialized_dict[key_str] = _serialize_recursive(value, max_repr, max_items, redact_keys, depth + 1, max_depth)

                item_count += 1

            # Apply standard redaction from redact.py
            return redact_payload(serialized_dict, depth=depth, max_depth=max_depth)
        except Exception:
            return f"<dict serialization error: {truncate_preview(repr(obj), max_repr)}>"

    # Handle non-serializable objects - fallback to repr()
    try:
        obj_repr = repr(obj)
        return truncate_preview(obj_repr, max_repr)
    except Exception:
        # Even repr() failed - ultimate fallback
        return f"<object of type {type(obj).__name__}>"


def serialize_args(
    args: tuple,
    kwargs: dict,
    max_repr: int = 500,
    max_items: int = 50,
    redact_keys: Optional[List[str]] = None
) -> dict:
    """
    Serialize function arguments and keyword arguments.

    Args:
        args: Positional arguments tuple
        kwargs: Keyword arguments dict
        max_repr: Maximum repr length
        max_items: Maximum collection size
        redact_keys: Keys to redact

    Returns:
        Dict with 'args' and 'kwargs' keys containing serialized values
    """
    try:
        return {
            "args": serialize_safe(args, max_repr, max_items, redact_keys),
            "kwargs": serialize_safe(kwargs, max_repr, max_items, redact_keys)
        }
    except Exception:
        return {
            "args": "<serialization error>",
            "kwargs": "<serialization error>"
        }
