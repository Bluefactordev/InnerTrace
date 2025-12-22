"""
Example usage of the tracing system.

This demonstrates a complete traced run with LLM calls, tool calls, and router decisions.
"""

import asyncio
import time
from tracing import get_tracer
from tracing.integration import (
    start_traced_run,
    end_traced_run,
    trace_router_decision,
)
from tracing.tracer import (
    emit_llm_call_start,
    emit_llm_call_end,
    emit_tool_call_start,
    emit_tool_call_end,
    emit_sandbox_exec_start,
    emit_sandbox_exec_end,
)


async def simulate_llm_call(prompt: str, model: str = "gpt-4"):
    """Simulate an LLM call with tracing."""
    tracer = get_tracer()

    # Emit LLM call start
    emit_llm_call_start(
        tracer,
        model=model,
        prompt=prompt,
        params={"temperature": 0.7, "max_tokens": 1000},
        purpose="example_generation"
    )

    # Simulate LLM processing
    await asyncio.sleep(0.2)

    # Simulate response
    response = f"This is a simulated response to: {prompt[:50]}..."

    # Emit LLM call end
    emit_llm_call_end(
        tracer,
        response=response,
        usage={"input_tokens": 100, "output_tokens": 50},
        finish_reason="stop"
    )

    return response


async def simulate_tool_call(tool_name: str, args: dict):
    """Simulate a tool call with tracing."""
    tracer = get_tracer()

    # Emit tool call start
    emit_tool_call_start(tracer, tool_name, args)

    start_time = time.time()

    # Simulate tool execution
    await asyncio.sleep(0.15)

    result = {
        "status": "success",
        "data": f"Tool {tool_name} executed with args: {args}"
    }

    latency_ms = int((time.time() - start_time) * 1000)

    # Emit tool call end
    emit_tool_call_end(tracer, tool_name, result, "ok", latency_ms)

    return result


async def simulate_sandbox_execution(code: str):
    """Simulate sandbox execution with tracing."""
    tracer = get_tracer()

    sandbox_info = {"kind": "python", "image": "python:3.11"}

    # Emit sandbox exec start
    emit_sandbox_exec_start(tracer, sandbox_info, code)

    # Simulate execution
    await asyncio.sleep(0.1)

    stdout = "Hello from sandbox!\nExecution completed."
    result = {"exit_code": 0, "output": stdout}

    # Emit sandbox exec end
    emit_sandbox_exec_end(
        tracer,
        status="ok",
        stdout=stdout,
        stderr=None,
        result=result
    )

    return result


async def example_workflow():
    """
    Example workflow demonstrating all tracing features.
    """
    # Start a traced run
    run_id = start_traced_run(
        entrypoint="example.workflow",
        args={"query": "What is the capital of France?", "depth": 2},
        env={"api_key": "sk-secret123"}  # Will be redacted
    )

    print(f"\n🚀 Started traced run: {run_id}\n")

    start_time = time.time()

    try:
        tracer = get_tracer()

        # Create main processing span
        with tracer.span("main_processing", "example.workflow", "function", tags=["demo", "example"]):

            # 1. Router decision
            print("📍 Step 1: Router decision")
            trace_router_decision(
                rule="complexity_based",
                candidates=["simple", "deep"],
                chosen="deep",
                why="Query requires multiple steps and reasoning"
            )

            # 2. Planning span with LLM call
            with tracer.span("planning", "planner", "agent", tags=["planning"]):
                print("🧠 Step 2: Planning with LLM")
                plan = await simulate_llm_call(
                    "Create a plan to answer: What is the capital of France?",
                    model="gpt-4"
                )
                print(f"   Plan created: {plan[:60]}...")

            # 3. Tool execution span
            with tracer.span("tool_execution", "executor", "tool", tags=["tools"]):
                print("🔧 Step 3: Executing tools")

                # Search tool
                search_result = await simulate_tool_call(
                    "web_search",
                    {"query": "capital of France", "num_results": 3}
                )
                print(f"   Search completed")

                # Knowledge base tool
                kb_result = await simulate_tool_call(
                    "knowledge_base",
                    {"query": "France capital", "top_k": 5}
                )
                print(f"   Knowledge base queried")

            # 4. Code execution span
            with tracer.span("code_execution", "sandbox", "sandbox", tags=["python"]):
                print("🐍 Step 4: Code execution in sandbox")
                code = """
def get_capital():
    return "Paris"

print(get_capital())
"""
                code_result = await simulate_sandbox_execution(code)
                print(f"   Code executed successfully")

            # 5. Final synthesis with LLM
            with tracer.span("synthesis", "synthesizer", "agent", tags=["synthesis"]):
                print("✨ Step 5: Final synthesis")
                final_response = await simulate_llm_call(
                    "Based on the search results and code execution, provide final answer",
                    model="gpt-4"
                )
                print(f"   Final response: {final_response[:60]}...")

        # End run successfully
        latency_ms = int((time.time() - start_time) * 1000)
        end_traced_run("ok", latency_ms)

        print(f"\n✅ Run completed successfully in {latency_ms}ms")
        print(f"   Run ID: {run_id}")
        print(f"\n📊 View trace with:")
        print(f"   ./trace view run --run-id {run_id}")
        print(f"\n📁 Events written to: traces/events.jsonl")
        print(f"📦 Blobs stored in: traces/blobs/sha256/")

    except Exception as e:
        # End run with error
        latency_ms = int((time.time() - start_time) * 1000)
        end_traced_run("error", latency_ms)
        print(f"\n❌ Run failed: {e}")
        raise


async def example_with_error():
    """
    Example demonstrating exception tracing.
    """
    run_id = start_traced_run(
        entrypoint="example.error",
        args={"test": "error_handling"}
    )

    print(f"\n🚀 Started traced run (with error): {run_id}\n")

    start_time = time.time()

    try:
        tracer = get_tracer()

        with tracer.span("error_demo", "example", "function"):
            print("⚠️  Simulating an error...")

            # Do some work before error
            await simulate_llm_call("This will succeed")

            # Simulate an error
            raise ValueError("This is a simulated error for demonstration")

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        end_traced_run("error", latency_ms)

        print(f"\n❌ Run failed as expected: {e}")
        print(f"   Run ID: {run_id}")
        print(f"\n📊 View failure context with:")
        print(f"   ./trace view failure --run-id {run_id}")


async def main():
    """Run all examples."""
    print("=" * 60)
    print("LLM Execution Observability - Example Demonstration")
    print("=" * 60)

    # Example 1: Successful workflow
    print("\n" + "=" * 60)
    print("Example 1: Successful Workflow")
    print("=" * 60)
    await example_workflow()

    # Wait a bit
    await asyncio.sleep(1)

    # Example 2: Error handling
    print("\n" + "=" * 60)
    print("Example 2: Error Handling")
    print("=" * 60)
    await example_with_error()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. List runs:     ./trace ls-runs")
    print("2. View run:      ./trace view run --run-id <run_id>")
    print("3. View failure:  ./trace view failure --run-id <run_id>")
    print("4. View events:   cat traces/events.jsonl | python -m json.tool")
    print("5. List blobs:    ls -lh traces/blobs/sha256/")
    print()


if __name__ == "__main__":
    asyncio.run(main())
