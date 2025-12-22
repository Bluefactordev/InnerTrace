"""CLI for trace viewing and analysis."""

import argparse
import json
import os
import sys
from pathlib import Path

from .blob_store import BlobStore
from .synthesis import get_synthesis_provider, TruncationProvider
from .projections import (
    compact_run_view,
    failure_context_view,
    find_last_run,
    llm_call_view,
    list_runs,
    timeline_view,
    tool_chain_view,
)


def load_config():
    """Load configuration from tracing/config.json."""
    config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config.json: {e}", file=sys.stderr)
    return None


def load_env_file():
    """Load environment variables from tracing/.env if present."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        try:
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        # Only set if not already in environment
                        if key not in os.environ:
                            os.environ[key] = value
        except Exception as e:
            print(f"Warning: Could not load .env file: {e}", file=sys.stderr)


def cmd_ls_runs(args):
    """List recent runs."""
    runs = list_runs(args.events_path, limit=args.limit)

    if not runs:
        print("No runs found.")
        return

    print(f"{'Run ID':<40} {'Entrypoint':<30} {'Status':<10} {'Start Time':<20}")
    print("-" * 100)

    for run in runs:
        run_id = run["run_id"][:38]
        entrypoint = (run.get("entrypoint") or "unknown")[:28]
        status = run.get("status") or "running"
        start_ts = run.get("start_ts", 0)

        # Format timestamp
        if start_ts:
            from datetime import datetime
            start_str = datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d %H:%M:%S")
        else:
            start_str = "unknown"

        print(f"{run_id:<40} {entrypoint:<30} {status:<10} {start_str:<20}")


def cmd_view_run(args):
    """View compact run view."""
    # Resolve run_id from filters if needed
    run_id = args.run_id
    
    if not run_id:
        # Check if any filter is specified
        if not (args.last or args.last_error or args.last_ok):
            print("Error: Either --run-id or one of --last/--last-error/--last-ok must be specified.", file=sys.stderr)
            sys.exit(1)
        
        status = None
        if args.last_error:
            status = "error"
        elif args.last_ok:
            status = "ok"
        # args.last means any status (status=None)
        
        run_id = find_last_run(
            events_path=args.events_path,
            status=status,
            entrypoint=args.endpoint
        )
        
        if not run_id:
            print("No matching run found.", file=sys.stderr)
            sys.exit(1)
        
        print(f"Using run_id: {run_id[:8]}...", file=sys.stderr)
    
    view = compact_run_view(run_id, args.events_path)
    print(json.dumps(view, indent=2))


def cmd_view_failure(args):
    """View failure context."""
    # Resolve run_id from filters if needed
    run_id = args.run_id
    
    if not run_id:
        # Check if any filter is specified
        if not (args.last or args.last_ok):
            # Default to last error if nothing specified
            args.last = True
        
        status = "error"  # Default to error for failure view
        if args.last_ok:
            status = "ok"
        elif args.last:
            status = "error"  # For failure view, --last means last error
        
        run_id = find_last_run(
            events_path=args.events_path,
            status=status,
            entrypoint=args.endpoint
        )
        
        if not run_id:
            print("No matching run found.", file=sys.stderr)
            sys.exit(1)
        
        print(f"Using run_id: {run_id[:8]}...", file=sys.stderr)
    
    view = failure_context_view(run_id, n=args.n, events_path=args.events_path)
    print(json.dumps(view, indent=2))


def cmd_view_tool_chain(args):
    """View tool chain."""
    view = tool_chain_view(args.span_id, args.events_path)
    print(json.dumps(view, indent=2))


def cmd_view_llm(args):
    """View LLM call."""
    view = llm_call_view(args.span_id, args.events_path)
    print(json.dumps(view, indent=2))


def cmd_blob(args):
    """View blob content."""
    blob_store = BlobStore(args.blobs_path)

    try:
        content = blob_store.get(args.ref)
        print(content)
    except FileNotFoundError:
        print(f"Blob not found: {args.ref}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_timeline(args):
    """View timeline with human-readable timestamps and delta times."""
    # Load .env file if present
    load_env_file()
    
    # Apply defaults: --last is True by default
    # Since argparse with action="store_true" sets False if not present, we need to set defaults
    if not args.run_id and not args.last_error and not args.last_ok:
        args.last = True
    
    # Resolve run_id from filters if needed
    run_id = args.run_id
    
    if not run_id:
        # Check if any filter is specified (--last is now default True)
        if not (args.last or args.last_error or args.last_ok):
            print("Error: Either --run-id or one of --last/--last-error/--last-ok must be specified.", file=sys.stderr)
            sys.exit(1)
        
        # Try to find run based on filters
        status = None
        if args.last_error:
            status = "error"
        elif args.last_ok:
            status = "ok"
        # args.last means any status (status=None)
        
        run_id = find_last_run(
            events_path=args.events_path,
            status=status,
            entrypoint=args.endpoint
        )
        
        if not run_id:
            print("No matching run found.", file=sys.stderr)
            sys.exit(1)
        
        print(f"Using run_id: {run_id[:8]}...", file=sys.stderr)
    
    # Configure synthesis provider from configuration and environment variables
    synthesis_provider = None
    if args.synthesize:
        # Load config.json
        config = load_config()
        
        # Determine quality level (default to high if not specified)
        quality = getattr(args, "quality", "high")
        if quality not in ["low", "high"]:
            print(f"Warning: Invalid quality '{quality}', using 'high'", file=sys.stderr)
            quality = "high"
        
        # Read provider type from env (default: openai if API key is set, else truncation)
        provider_type = os.getenv("BF_TRACE_SYNTHESIS_PROVIDER", "truncation")
        
        # If API key is set, default to openai provider
        api_key = os.getenv("BF_TRACE_SYNTHESIS_API_KEY")
        if api_key and provider_type == "truncation":
            provider_type = "openai"
        
        if not api_key:
            print(
                f"Warning: No BF_TRACE_SYNTHESIS_API_KEY found. Using truncation provider (no LLM synthesis).\n"
                f"To enable synthesis, create tracing/.env with BF_TRACE_SYNTHESIS_API_KEY=sk-your-key",
                file=sys.stderr,
            )

        if provider_type == "openai":
            # Get base URL from env or config
            base_url = os.getenv("BF_TRACE_SYNTHESIS_BASE_URL")
            if not base_url and config:
                base_url = config.get("base_url")

            if not base_url:
                print(
                    "Error: BF_TRACE_SYNTHESIS_BASE_URL not configured.\n"
                    "Set it in tracing/.env or tracing/config.json (\"base_url\" field).\n"
                    "Example: BF_TRACE_SYNTHESIS_BASE_URL=http://localhost:5099/v1",
                    file=sys.stderr
                )
                sys.exit(1)
            
            # Get model from config based on quality, or from env, or default
            model = None
            if config and "models" in config and quality in config["models"]:
                model = config["models"][quality].get("model")
            
            if not model:
                model = os.getenv("BF_TRACE_SYNTHESIS_MODEL")
            
            if not model:
                # Fallback defaults (use internal models)
                model = "vllm/qwen3-30b-a3b-thinking-2507-awq-4bit" if quality == "high" else "vllm/google/gemma-3-270m-it"

            if not api_key:
                print(
                    "Error: --synthesize with openai provider requires BF_TRACE_SYNTHESIS_API_KEY environment variable.\n"
                    "Set it in tracing/.env or as environment variable.",
                    file=sys.stderr,
                )
                sys.exit(1)

            # Get timeout and max_concurrent from config or use defaults
            timeout = 120.0  # Default 120 seconds (conservative timeout for synthesis)
            max_concurrent = 5  # Default 5 concurrent requests (reduced to avoid server overload)
            
            if config:
                timeout = config.get("synthesis_timeout", timeout)
                max_concurrent = config.get("synthesis_max_concurrent", max_concurrent)
            
            print(f"Using model '{model}' for quality '{quality}' (timeout={timeout}s, max_concurrent={max_concurrent})", file=sys.stderr)
            synthesis_provider = get_synthesis_provider(
                "openai", 
                base_url=base_url, 
                api_key=api_key, 
                model=model,
                timeout=timeout,
                max_concurrent=max_concurrent
            )
        else:
            # Default to truncation provider
            synthesis_provider = TruncationProvider()

    lines = timeline_view(
        run_id, args.events_path, compact=args.compact, synthesize=args.synthesize, synthesis_provider=synthesis_provider
    )

    for line in lines:
        print(line)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CLI for trace viewing and analysis",
        prog="trace"
    )

    parser.add_argument(
        "--events-path",
        default="traces/events.jsonl",
        help="Path to events file (default: traces/events.jsonl)"
    )

    parser.add_argument(
        "--blobs-path",
        default="traces/blobs",
        help="Path to blobs directory (default: traces/blobs)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ls-runs command
    parser_ls = subparsers.add_parser("ls-runs", help="List recent runs")
    parser_ls.add_argument("--limit", type=int, default=20, help="Maximum number of runs (default: 20)")
    parser_ls.set_defaults(func=cmd_ls_runs)

    # view command with subcommands
    parser_view = subparsers.add_parser("view", help="View trace details")
    view_subparsers = parser_view.add_subparsers(dest="view_command", help="View type")

    # view run
    parser_view_run = view_subparsers.add_parser("run", help="View compact run view")
    parser_view_run.add_argument("--run-id", help="Run ID (optional if using --last)")
    parser_view_run.add_argument("--last", action="store_true", help="Use most recent run")
    parser_view_run.add_argument("--last-error", action="store_true", help="Use most recent error run")
    parser_view_run.add_argument("--last-ok", action="store_true", help="Use most recent successful run")
    parser_view_run.add_argument("--endpoint", help="Filter by entrypoint (e.g., api.chat_v2)")
    parser_view_run.set_defaults(func=cmd_view_run)

    # view failure
    parser_view_failure = view_subparsers.add_parser("failure", help="View failure context")
    parser_view_failure.add_argument("--run-id", help="Run ID (optional if using --last)")
    parser_view_failure.add_argument("--last", action="store_true", help="Use most recent error run")
    parser_view_failure.add_argument("--last-ok", action="store_true", help="Use most recent successful run")
    parser_view_failure.add_argument("--endpoint", help="Filter by entrypoint (e.g., api.chat_v2)")
    parser_view_failure.add_argument("--n", type=int, default=80, help="Number of events before exception (default: 80)")
    parser_view_failure.set_defaults(func=cmd_view_failure)

    # view tool-chain
    parser_view_tool = view_subparsers.add_parser("tool-chain", help="View tool chain")
    parser_view_tool.add_argument("--span-id", required=True, help="Span ID")
    parser_view_tool.set_defaults(func=cmd_view_tool_chain)

    # view llm
    parser_view_llm = view_subparsers.add_parser("llm", help="View LLM call")
    parser_view_llm.add_argument("--span-id", required=True, help="Span ID")
    parser_view_llm.set_defaults(func=cmd_view_llm)

    # blob command
    parser_blob = subparsers.add_parser("blob", help="View blob content")
    parser_blob.add_argument("--ref", required=True, help="Blob reference (e.g., blob:sha256:<hash>)")
    parser_blob.set_defaults(func=cmd_blob)

    # timeline command (enterprise-level debug view)
    parser_timeline = subparsers.add_parser("timeline", help="View timeline with human-readable timestamps")
    parser_timeline.add_argument("--run-id", help="Run ID (optional if using --last)")
    parser_timeline.add_argument("--last", action="store_true", 
                                 help="Use most recent run (default: True)")
    parser_timeline.add_argument("--last-error", action="store_true", help="Use most recent error run")
    parser_timeline.add_argument("--last-ok", action="store_true", help="Use most recent successful run")
    parser_timeline.add_argument("--endpoint", help="Filter by entrypoint (e.g., api.chat_v2)")
    parser_timeline.add_argument("--compact", action="store_true", 
                                 help="Compact view: only LLM calls with context (lossy projection, not suitable for root-cause analysis)")
    parser_timeline.add_argument(
        "--synthesize",
        action="store_true",
        default=False,
        help="Enable LLM-based synthesis of prompts/responses (configure via tracing/config.json and tracing/.env)",
    )
    parser_timeline.add_argument(
        "--quality",
        choices=["low", "high"],
        default="high",
        help="Quality level for synthesis: 'low' uses faster/cheaper model, 'high' uses more capable model (default: high). Models configured in tracing/config.json",
    )
    parser_timeline.set_defaults(func=cmd_timeline)

    args = parser.parse_args()

    # Check if command is missing (happens when no subcommand is provided)
    # This can happen during debugging when no arguments are passed
    if not args.command:
        # If we're in a debugger (detected by checking if sys.gettrace() is set),
        # default to timeline command for easier debugging
        if sys.gettrace() is not None:
            # Running in debugger - default to timeline
            # Need to manually set all attributes that argparse would have set for timeline command
            args.command = "timeline"
            args.run_id = None
            args.last = True
            args.last_error = False
            args.last_ok = False
            args.endpoint = None
            args.compact = False
            args.synthesize = True
            args.quality = "high"  # Default quality
            args.func = cmd_timeline
            # Also need to set attributes from parent parser
            if not hasattr(args, 'events_path'):
                args.events_path = "traces/events.jsonl"
            if not hasattr(args, 'blobs_path'):
                args.blobs_path = "traces/blobs"
        else:
            # Normal execution - show help and exit
            parser.print_help()
            sys.exit(1)

    # Check if view command needs a subcommand
    if args.command == "view" and not args.view_command:
        parser_view.print_help()
        sys.exit(1)

    # Execute command
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
