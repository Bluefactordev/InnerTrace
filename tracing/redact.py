"""Secret redaction utilities for trace payloads."""

import re
from typing import Any, Dict, List


# Secret key patterns (case-insensitive)
SECRET_KEYS = {
    "api_key",
    "apikey",
    "api-key",
    "token",
    "access_token",
    "refresh_token",
    "authorization",
    "auth",
    "password",
    "passwd",
    "pwd",
    "secret",
    "private_key",
    "privatekey",
    "aws_secret_access_key",
    "aws_access_key_id",
}

# Patterns in string values that look like secrets
SECRET_VALUE_PATTERNS = [
    re.compile(r'Bearer\s+[A-Za-z0-9\-._~+/]+=*', re.IGNORECASE),
    re.compile(r'sk-[A-Za-z0-9]{32,}'),  # OpenAI-style keys
    re.compile(r'xoxb-[A-Za-z0-9\-]+'),  # Slack tokens
]


def redact_payload(payload: Any, depth: int = 0, max_depth: int = 10) -> Any:
    """
    Recursively redact secrets from payload.

    Args:
        payload: Any JSON-serializable value
        depth: Current recursion depth
        max_depth: Maximum recursion depth

    Returns:
        Redacted copy of payload
    """
    if depth > max_depth:
        return payload

    if isinstance(payload, dict):
        return {
            key: _redact_value(key, value, depth)
            for key, value in payload.items()
        }
    elif isinstance(payload, list):
        return [redact_payload(item, depth + 1, max_depth) for item in payload]
    else:
        return payload


def _redact_value(key: str, value: Any, depth: int) -> Any:
    """Redact a single key-value pair."""
    # Check if key matches secret patterns
    key_lower = key.lower().replace("-", "_").replace(" ", "_")
    if any(secret_key in key_lower for secret_key in SECRET_KEYS):
        return "***REDACTED***"

    # Recursively handle nested structures
    if isinstance(value, dict):
        return redact_payload(value, depth + 1)
    elif isinstance(value, list):
        return redact_payload(value, depth + 1)
    elif isinstance(value, str):
        # Check if value matches secret patterns
        for pattern in SECRET_VALUE_PATTERNS:
            if pattern.search(value):
                return "***REDACTED***"
        return value
    else:
        return value


def truncate_preview(text: str, max_len: int = 500) -> str:
    """
    Truncate text for preview with ellipsis.

    Args:
        text: Text to truncate
        max_len: Maximum length

    Returns:
        Truncated text
    """
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."
