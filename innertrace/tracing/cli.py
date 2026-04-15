"""CLI for trace viewing and analysis."""

import argparse
import json
import os
import sys
from pathlib import Path

from .blob_store import BlobStore
from .synthesis import (
    get_synthesis_provider, 
    TruncationProvider,
    create_multi_provider_from_quality_config
)
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

    print(f"{'Run ID':<44} {'Entrypoint':<30} {'Status':<10} {'Start Time':<20}")
    print("-" * 110)

    for run in runs:
        run_id = run["run_id"]  # Full ID, no truncation
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
            entrypoint=args.endpoint,
            offset=0
        )
        
        if not run_id:
            print("No matching run found.", file=sys.stderr)
            sys.exit(1)
        
        print(f"Using run_id: {run_id}", file=sys.stderr)
    
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
            entrypoint=args.endpoint,
            offset=0
        )
        
        if not run_id:
            print("No matching run found.", file=sys.stderr)
            sys.exit(1)
        
        print(f"Using run_id: {run_id}", file=sys.stderr)
    
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
    
    # Apply defaults: --last is 0 by default if no run-id specified
    if not args.run_id and args.last is None and not args.last_error and not args.last_ok:
        args.last = 0
    
    # Resolve run_id from filters if needed
    run_id = args.run_id
    
    if not run_id:
        # Check if any filter is specified
        if args.last is None and not args.last_error and not args.last_ok:
            print("Error: Either --run-id or one of --last/--last-error/--last-ok must be specified.", file=sys.stderr)
            sys.exit(1)
        
        # Try to find run based on filters
        status = None
        offset = 0
        
        if args.last_error:
            status = "error"
        elif args.last_ok:
            status = "ok"
        elif args.last is not None:
            # args.last means any status (status=None) with specified offset
            offset = args.last
        
        run_id = find_last_run(
            events_path=args.events_path,
            status=status,
            entrypoint=args.endpoint,
            offset=offset
        )
        
        if not run_id:
            print("No matching run found.", file=sys.stderr)
            sys.exit(1)
        
        print(f"Using run_id: {run_id}", file=sys.stderr)
    
    # Configure synthesis provider from configuration and environment variables
    synthesis_provider = None
    if args.synthesize:
        # Load config.json
        config = load_config()
        
        # Determine quality level (default to high if not specified)
        quality = getattr(args, "quality", None)
        if not quality and config:
            quality = config.get("default_quality", "high")
        if not quality:
            quality = "high"
        
        # Check if external configuration is specified and use its quality
        if not quality and config and "external" in config:
            external_config = config.get("external", {})
            quality = external_config.get("quality")
        
        if quality not in ["low", "medium", "high", "highest"]:
            print(f"Warning: Invalid quality '{quality}', using 'high'", file=sys.stderr)
            quality = "high"
        
        # Nuova struttura config: quality_levels con provider multipli
        if not config or "quality_levels" not in config or quality not in config.get("quality_levels", {}):
            print(
                f"Error: No configuration found for quality '{quality}'.\n"
                f"Configure providers in tracing/config.json under 'quality_levels' -> '{quality}' -> 'providers'",
                file=sys.stderr,
            )
            sys.exit(1)
            
        quality_config = config["quality_levels"][quality]
        verbose = getattr(args, 'verbose', False)
        print(f"Using quality level '{quality}' with {len(quality_config.get('providers', []))} provider(s) configured", file=sys.stderr)
        # verbose e callback verranno impostati in timeline_view prima di chiamare synthesize_batch
        synthesis_provider = create_multi_provider_from_quality_config(quality_config, verbose=verbose)

    lines = timeline_view(
        run_id, args.events_path, compact=args.compact, synthesize=args.synthesize, 
        synthesis_provider=synthesis_provider, verbose=getattr(args, 'verbose', False)
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
    parser_timeline.add_argument("--last", type=int, nargs='?', const=0, default=None,
                                 help="Use n-th most recent run (0 = most recent, 1 = penultimate, etc.). Default: 0 if no run-id specified")
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
        choices=["low", "medium", "high", "highest"],
        default="high",
        help="Quality level for synthesis: 'low', 'medium', 'high', or 'highest'. Models and providers configured in tracing/config.json. If 'external' config is set, uses its quality automatically.",
    )
    parser_timeline.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show verbose output including synthesis provider fallback messages",
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
            args.last = 0  # Use most recent run (0 = most recent)
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
