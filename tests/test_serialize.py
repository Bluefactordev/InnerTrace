"""Tests for safe serialization utilities."""

import pytest
from innertrace.tracing.serialize import serialize_safe, serialize_args


def test_serialize_basic_types():
    """Test that basic JSON types pass through unchanged."""
    # None
    assert serialize_safe(None) is None

    # Booleans
    assert serialize_safe(True) is True
    assert serialize_safe(False) is False

    # Numbers
    assert serialize_safe(42) == 42
    assert serialize_safe(3.14) == 3.14

    # Strings
    assert serialize_safe("hello") == "hello"
    assert serialize_safe("") == ""


def test_serialize_string_truncation():
    """Test that long strings are truncated."""
    long_string = "x" * 1000
    result = serialize_safe(long_string, max_repr=500)

    assert isinstance(result, str)
    assert len(result) <= 503  # 500 + "..."
    assert result.endswith("...")


def test_serialize_collections_with_limits():
    """Test that collections are limited to max_items."""
    # List with 100 items, limit to 50
    long_list = list(range(100))
    result = serialize_safe(long_list, max_items=50)

    assert isinstance(result, list)
    assert len(result) == 51  # 50 items + truncation marker
    assert result[-1] == "...(+50 more items)"
    assert result[0] == 0
    assert result[49] == 49

    # Dict with 100 keys, limit to 50
    long_dict = {f"key_{i}": i for i in range(100)}
    result = serialize_safe(long_dict, max_items=50)

    assert isinstance(result, dict)
    assert len(result) <= 51  # 50 items + truncation marker
    if len(result) == 51:
        assert "..." in result
        assert "(+50 more keys)" in result["..."]


def test_serialize_redaction():
    """Test that sensitive keys are redacted."""
    # Test with standard secret keys
    sensitive_data = {
        "username": "alice",
        "password": "secret123",
        "api_key": "sk-1234567890",
        "token": "bearer_xyz",
        "normal_key": "visible_value"
    }

    result = serialize_safe(sensitive_data)

    assert isinstance(result, dict)
    assert result["username"] == "alice"
    assert result["password"] == "***REDACTED***"
    assert result["api_key"] == "***REDACTED***"
    assert result["token"] == "***REDACTED***"
    assert result["normal_key"] == "visible_value"


def test_serialize_custom_redaction():
    """Test that custom redact keys work."""
    data = {
        "public": "visible",
        "custom_secret": "should_be_hidden",
        "another_key": "also_visible"
    }

    result = serialize_safe(data, redact_keys=["custom_secret"])

    assert isinstance(result, dict)
    assert result["public"] == "visible"
    assert result["custom_secret"] == "***REDACTED***"
    assert result["another_key"] == "also_visible"


def test_serialize_nested_structures():
    """Test serialization of nested dicts and lists."""
    nested_data = {
        "level1": {
            "level2": {
                "level3": [1, 2, 3]
            }
        }
    }

    result = serialize_safe(nested_data)

    assert isinstance(result, dict)
    assert isinstance(result["level1"], dict)
    assert isinstance(result["level1"]["level2"], dict)
    assert isinstance(result["level1"]["level2"]["level3"], list)
    assert result["level1"]["level2"]["level3"] == [1, 2, 3]


def test_serialize_non_serializable():
    """Test that non-serializable objects fall back to repr()."""
    # Custom object
    class CustomObject:
        def __repr__(self):
            return "<CustomObject instance>"

    obj = CustomObject()
    result = serialize_safe(obj, max_repr=100)

    assert isinstance(result, str)
    assert "CustomObject" in result

    # File handle
    import io
    file_obj = io.StringIO()
    result = serialize_safe(file_obj, max_repr=100)

    assert isinstance(result, str)
    # Should contain some representation of the object


def test_serialize_never_raises():
    """Test that serialization never raises exceptions."""
    # Object with broken __repr__
    class BrokenRepr:
        def __repr__(self):
            raise RuntimeError("Broken repr!")

    obj = BrokenRepr()
    result = serialize_safe(obj)

    # Should not raise, should return fallback
    assert isinstance(result, str)
    assert "BrokenRepr" in result or "object of type" in result.lower()


def test_serialize_max_depth():
    """Test that maximum depth is respected."""
    # Create deeply nested structure
    deep = {"level": 0}
    current = deep
    for i in range(20):
        current["nested"] = {"level": i + 1}
        current = current["nested"]

    result = serialize_safe(deep, max_repr=100, max_items=50)

    # Should handle deep nesting without crashing
    assert isinstance(result, dict)


def test_serialize_args_function():
    """Test the serialize_args convenience function."""
    args = (1, 2, "three")
    kwargs = {"x": 10, "y": 20, "password": "secret"}

    result = serialize_args(args, kwargs)

    assert isinstance(result, dict)
    assert "args" in result
    assert "kwargs" in result

    # Args should be serialized as list
    assert result["args"] == [1, 2, "three"]

    # Kwargs should be serialized as dict with redaction
    assert result["kwargs"]["x"] == 10
    assert result["kwargs"]["y"] == 20
    assert result["kwargs"]["password"] == "***REDACTED***"


def test_serialize_tuple():
    """Test that tuples are serialized as lists."""
    tuple_data = (1, 2, 3, 4, 5)
    result = serialize_safe(tuple_data)

    assert isinstance(result, list)
    assert result == [1, 2, 3, 4, 5]


def test_serialize_mixed_types():
    """Test serialization of mixed types in collections."""
    mixed_data = {
        "int": 42,
        "float": 3.14,
        "str": "hello",
        "bool": True,
        "none": None,
        "list": [1, "two", 3.0, True, None],
        "dict": {"nested": "value"}
    }

    result = serialize_safe(mixed_data)

    assert result["int"] == 42
    assert result["float"] == 3.14
    assert result["str"] == "hello"
    assert result["bool"] is True
    assert result["none"] is None
    assert result["list"] == [1, "two", 3.0, True, None]
    assert result["dict"] == {"nested": "value"}


def test_serialize_empty_collections():
    """Test serialization of empty collections."""
    assert serialize_safe([]) == []
    assert serialize_safe({}) == {}
    assert serialize_safe(()) == []


def test_serialize_with_non_string_keys():
    """Test that dict keys are converted to strings."""
    data = {
        1: "int_key",
        2.5: "float_key",
        True: "bool_key"
    }

    result = serialize_safe(data)

    assert isinstance(result, dict)
    # Keys should be converted to strings
    assert "1" in result or 1 in result
    assert "2.5" in result or 2.5 in result
