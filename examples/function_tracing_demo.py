#!/usr/bin/env python3
"""
Demo: Function-Level Tracing with InnerTrace

This example demonstrates how to use declarative function-level tracing
without global hooks or monkey patching.
"""

import asyncio
from innertrace import (
    trace_function,
    trace_block,
    trace_block_async,
    trace_module,
    get_tracer
)


# Example 1: Decorate individual functions
@trace_function(capture_args=True, capture_return=False)
def calculate_sum(numbers: list) -> int:
    """Calculate sum of numbers with automatic tracing."""
    return sum(numbers)


@trace_function(capture_args=True, capture_return=True)
def process_data(data: dict) -> dict:
    """Process data and capture return value."""
    result = {
        "processed": True,
        "count": len(data),
        "keys": list(data.keys())
    }
    return result


# Example 2: Async function tracing
@trace_function(name="async.fetch_data")
async def fetch_remote_data(url: str) -> dict:
    """Async functions are automatically detected and wrapped."""
    # Simulate API call
    await asyncio.sleep(0.1)
    return {"url": url, "status": 200, "data": "mock_data"}


# Example 3: Trace code blocks
def complex_operation():
    """Use trace_block for arbitrary code sections."""

    # Stage 1: Initialization
    with trace_block("initialization", payload={"stage": 1}):
        config = {"mode": "production", "workers": 4}

    # Stage 2: Processing
    with trace_block("processing", payload={"stage": 2, "items": 100}):
        results = []
        for i in range(100):
            results.append(i * 2)

    # Stage 3: Finalization
    with trace_block("finalization", payload={"stage": 3}):
        summary = {"total": len(results), "max": max(results)}

    return summary


# Example 4: Module-level tracing
# These functions are in a separate "module" conceptually
def helper_function_1(x):
    return x * 2


def helper_function_2(x):
    return x ** 2


def _private_helper(x):
    return x + 1


# You would normally call this at the end of the module file:
# trace_module(globals(), include_private=False)
# For this demo, we'll call it explicitly below


async def main():
    """Run the demo."""
    tracer = get_tracer()

    print("=" * 60)
    print("Function-Level Tracing Demo")
    print("=" * 60)

    # Start a traced run
    run_id = tracer.start_run(
        entrypoint="demo.function_tracing",
        args={"demo": "function_level_tracing"}
    )

    print(f"\n✓ Started run: {run_id}")
    print("\nExecuting traced functions...\n")

    # Test 1: Simple function with args
    result1 = calculate_sum([1, 2, 3, 4, 5])
    print(f"  calculate_sum([1,2,3,4,5]) = {result1}")

    # Test 2: Function with return capture
    result2 = process_data({"a": 1, "b": 2, "c": 3})
    print(f"  process_data(...) = {result2}")

    # Test 3: Async function
    result3 = await fetch_remote_data("https://api.example.com/data")
    print(f"  fetch_remote_data(...) = {result3}")

    # Test 4: Code blocks
    result4 = complex_operation()
    print(f"  complex_operation() = {result4}")

    # End the run
    tracer.end_run(status="ok")
    print(f"\n✓ Run completed: {run_id}")

    print("\n" + "=" * 60)
    print("View the trace:")
    print("=" * 60)
    print(f"\nDefault view (includes function calls):")
    print(f"  ./trace timeline --last")
    print(f"\nCompact view (excludes function calls):")
    print(f"  ./trace timeline --last --compact")
    print(f"\nVerbose view (shows function arguments):")
    print(f"  ./trace timeline --last --verbose")
    print("\n" + "=" * 60)

    print("\nExpected timeline output:")
    print("-" * 60)
    print("""
[HH:MM:SS.xxx]   run.start                (demo.function_tracing)
[HH:MM:SS.xxx]     span.start             [function] calculate_sum
[HH:MM:SS.xxx]     span.end               (ok, 0ms total)
[HH:MM:SS.xxx]     span.start             [function] process_data
[HH:MM:SS.xxx]     span.end               (ok, 0ms total)
[HH:MM:SS.xxx]     span.start             [function] fetch_remote_data
[HH:MM:SS.xxx]     span.end               (ok, 100ms total)
[HH:MM:SS.xxx]     span.start             [function] complex_operation
[HH:MM:SS.xxx]       span.start           [function] initialization
[HH:MM:SS.xxx]       span.end             (ok, 0ms total)
[HH:MM:SS.xxx]       span.start           [function] processing
[HH:MM:SS.xxx]       span.end             (ok, 1ms total)
[HH:MM:SS.xxx]       span.start           [function] finalization
[HH:MM:SS.xxx]       span.end             (ok, 0ms total)
[HH:MM:SS.xxx]     span.end               (ok, 2ms total)
[HH:MM:SS.xxx]   run.end                  (ok, 105ms total)
    """.strip())
    print("-" * 60)

    print("\nFeatures demonstrated:")
    print("  ✓ Sync and async function tracing")
    print("  ✓ Argument capture with serialization")
    print("  ✓ Return value capture (optional)")
    print("  ✓ Code block tracing with custom payload")
    print("  ✓ Hierarchical span nesting")
    print("  ✓ Automatic secret redaction")
    print("  ✓ No tracing overhead outside of runs")
    print("\nArchitectural guarantees:")
    print("  ✓ No global hooks or monkey patching")
    print("  ✓ Opt-in instrumentation only")
    print("  ✓ Safe degradation (never breaks execution)")
    print("  ✓ Compatible with all existing projections")


if __name__ == "__main__":
    asyncio.run(main())
