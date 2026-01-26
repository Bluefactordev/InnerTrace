"""Projection views for trace analysis (LLM-safe, deterministic views)."""

import bisect
import json
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .synthesis import SynthesisProvider, TruncationProvider

# Configurazione sintesi
# MAX_SYNTHESIS_WORDS calcolato basandosi su max_tokens=500
# 500 token ≈ 375 parole (rapporto ~1.33 token/parola)
# Usiamo 400 parole per essere generosi e permettere sintesi complete
MAX_SYNTHESIS_WORDS = 400  # Allineato con max_tokens=500 (circa 375 parole, arrotondato a 400)

# Whitelist of parameters to include in run.start signature
# These are safe to display and useful for debugging
RUN_SIGNATURE_KEYS = {
    "args": [
        "model_id", "model",  # Model identifier
        "selected_tools", "tools",  # Tool selection
        "selected_agents", "agents",  # Agent selection
        "selected_validators", "validators",  # Validator selection
        "response_schema", "schema",  # Response schema
        "stream", "streaming",  # Streaming flag
        "temperature",  # Temperature parameter
        "max_tokens",  # Max tokens
        "conversation_id",  # Conversation ID (safe, no PII)
        "depth", "research_depth",  # Research/planning depth
        "code_execution_mode",  # Code execution mode (force/ask/never)
        "hitl_enabled",  # Human-in-the-loop enabled
        "hitl_auto_approve",  # HITL auto-approval
        "hitl_timeout",  # HITL timeout
        "use_rag",  # RAG enabled
        "planner_depth",  # Planner depth
        "apply_company_context",  # Company context flag
    ],
    "env": [
        "endpoint",  # API endpoint
        "client",  # Client identifier
    ]
}


# ═══════════════════════════════════════════════════════════════════════════════
# FILOSOFIA INNERTRACE v0.2
# ═══════════════════════════════════════════════════════════════════════════════
#
# InnerTrace NON decide, NON rassicura, NON giudica di default
# InnerTrace registra, struttura, rende leggibile
#
# COSA VA NEL CORE:
# ✓ Sintesi neutra delle risposte (sempre utile, sempre stabile)
# ✓ Strutturazione semantica del trace
# ✓ Informazioni che rendono leggibile cosa è successo
#
# COSA NON VA NEL CORE:
# ✗ Valutazioni di adeguatezza (advanced/stalled/regressed)
# ✗ Giudizi su topic alignment o intent progress
# ✗ Euristiche naive (keyword matching, conteggi grezzi)
# ✗ Segnali che inquinano il trace con falsi positivi
#
# Le valutazioni (se necessarie) vanno in:
# - Projection layer opzionale
# - Attivabile tramite flag espliciti
# - Chiaramente separate dalla verità primaria del trace
#
# ═══════════════════════════════════════════════════════════════════════════════


# Sintesi neutre generate in batch più avanti nel codice


def load_events(events_path: str = "traces/events.jsonl", run_id: Optional[str] = None) -> List[Dict]:
    """
    Load events from JSONL file.

    Args:
        events_path: Path to events file
        run_id: Optional run_id to filter by

    Returns:
        List of event dicts, sorted by timestamp
    """
    events = []
    path = Path(events_path)

    if not path.exists():
        # print(f"DEBUG: File not found: {path.absolute()}")
        return []

    total_lines = 0
    matching_events = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                event = json.loads(line)
                if run_id is None or event.get("run_id") == run_id:
                    events.append(event)
                    matching_events += 1
            except json.JSONDecodeError:
                continue

    if run_id:
        logging.debug(f"[load_events] Run {run_id}: {matching_events} eventi su {total_lines} righe totali nel file")

    # Sort by timestamp
    events.sort(key=lambda e: e.get("ts", 0))
    return events


def compact_run_view(run_id: str, events_path: str = "traces/events.jsonl") -> Dict:
    """
    Compact view of a run: router decisions, tool calls, sandbox execs, exceptions, llm metadata.

    Args:
        run_id: Run ID
        events_path: Path to events file

    Returns:
        Dict with run_id and list of compact items
    """
    events = load_events(events_path, run_id)

    items = []

    for event in events:
        event_type = event.get("type", "")
        ts = event.get("ts", 0)
        span_id = event.get("span_id")
        tags = event.get("tags", [])
        payload = event.get("payload", {})

        # Include relevant event types
        if event_type == "router.decision":
            items.append({
                "ts": ts,
                "span": span_id,
                "type": "router.decision",
                "chosen": payload.get("chosen"),
                "candidates": payload.get("candidates", []),
                "rule": payload.get("rule"),
                "tags": tags
            })

        elif event_type == "tool.call.start":
            items.append({
                "ts": ts,
                "span": span_id,
                "type": "tool.call.start",
                "tool": payload.get("tool"),
                "tags": tags
            })

        elif event_type == "tool.call.end":
            items.append({
                "ts": ts,
                "span": span_id,
                "type": "tool.call.end",
                "tool": payload.get("tool"),
                "status": payload.get("status"),
                "latency_ms": payload.get("latency_ms"),
                "tags": tags
            })

        elif event_type == "sandbox.exec.start":
            items.append({
                "ts": ts,
                "span": span_id,
                "type": "sandbox.exec.start",
                "sandbox": payload.get("sandbox", {}),
                "tags": tags
            })

        elif event_type == "sandbox.exec.end":
            items.append({
                "ts": ts,
                "span": span_id,
                "type": "sandbox.exec.end",
                "status": payload.get("status"),
                "tags": tags
            })

        elif event_type == "exception":
            items.append({
                "ts": ts,
                "span": span_id,
                "type": "exception",
                "exc_type": payload.get("exc_type"),
                "message": payload.get("message"),
                "where": payload.get("where", {}),
                "stack_ref": payload.get("stack_ref"),
                "tags": tags
            })

        elif event_type == "llm.call.start":
            items.append({
                "ts": ts,
                "span": span_id,
                "type": "llm.call.start",
                "model": payload.get("model"),
                "prompt_ref": payload.get("prompt_ref"),
                "tags": tags
            })

        elif event_type == "llm.call.end":
            items.append({
                "ts": ts,
                "span": span_id,
                "type": "llm.call.end",
                "response_ref": payload.get("response_ref"),
                "usage": payload.get("usage", {}),
                "finish_reason": payload.get("finish_reason"),
                "tags": tags
            })

    return {
        "run_id": run_id,
        "items": items
    }


def failure_context_view(run_id: str, n: int = 80, events_path: str = "traces/events.jsonl") -> Dict:
    """
    Context around first exception: last n relevant events + parent span tree.

    Args:
        run_id: Run ID
        n: Number of events before exception
        events_path: Path to events file

    Returns:
        Dict with exception info, preceding events, and span tree
    """
    events = load_events(events_path, run_id)

    # Find first exception
    exception_event = None
    exception_idx = -1
    for i, event in enumerate(events):
        if event.get("type") == "exception":
            exception_event = event
            exception_idx = i
            break

    if not exception_event:
        return {
            "run_id": run_id,
            "exception": None,
            "preceding_events": [],
            "span_tree": []
        }

    # Get last n relevant events before exception
    relevant_types = {
        "router.decision", "tool.call.start", "tool.call.end",
        "sandbox.exec.start", "sandbox.exec.end",
        "llm.call.start", "llm.call.end",
        "span.start", "span.end"
    }

    preceding = []
    for event in events[:exception_idx]:
        if event.get("type") in relevant_types:
            preceding.append({
                "ts": event.get("ts"),
                "type": event.get("type"),
                "span": event.get("span_id"),
                "actor": event.get("actor"),
                "payload": event.get("payload", {})
            })

    # Take last n
    preceding = preceding[-n:]

    # Build span tree from exception span to root
    exception_span_id = exception_event.get("span_id")
    span_tree = _build_span_tree(events, exception_span_id)

    return {
        "run_id": run_id,
        "exception": {
            "ts": exception_event.get("ts"),
            "span": exception_span_id,
            "exc_type": exception_event.get("payload", {}).get("exc_type"),
            "message": exception_event.get("payload", {}).get("message"),
            "where": exception_event.get("payload", {}).get("where"),
            "stack_ref": exception_event.get("payload", {}).get("stack_ref")
        },
        "preceding_events": preceding,
        "span_tree": span_tree
    }


def _build_span_tree(events: List[Dict], span_id: Optional[str]) -> List[Dict]:
    """Build span tree from span_id to root."""
    if not span_id:
        return []

    # Build span map
    spans = {}
    for event in events:
        if event.get("type") == "span.start":
            sid = event.get("span_id")
            spans[sid] = {
                "span_id": sid,
                "parent_span_id": event.get("parent_span_id"),
                "name": event.get("payload", {}).get("name"),
                "actor": event.get("actor"),
                "kind": event.get("payload", {}).get("kind")
            }

    # Walk up the tree
    tree = []
    current = span_id
    while current and current in spans:
        tree.append(spans[current])
        current = spans[current]["parent_span_id"]

    # Reverse to get root-to-leaf order
    return list(reversed(tree))


def tool_chain_view(span_id: str, events_path: str = "traces/events.jsonl") -> Dict:
    """
    All tool calls descending from a span.

    Args:
        span_id: Span ID
        events_path: Path to events file

    Returns:
        Dict with span_id and list of tool calls
    """
    events = load_events(events_path)

    # Find all descendant spans
    descendant_spans = _find_descendant_spans(events, span_id)
    descendant_spans.add(span_id)

    # Collect tool calls in those spans
    tool_calls = []
    for event in events:
        if event.get("span_id") not in descendant_spans:
            continue

        if event.get("type") == "tool.call.start":
            tool_calls.append({
                "ts": event.get("ts"),
                "span": event.get("span_id"),
                "tool": event.get("payload", {}).get("tool"),
                "args_ref": event.get("payload", {}).get("args_ref"),
                "args_preview": event.get("payload", {}).get("args_preview")
            })
        elif event.get("type") == "tool.call.end":
            tool_calls.append({
                "ts": event.get("ts"),
                "span": event.get("span_id"),
                "tool": event.get("payload", {}).get("tool"),
                "result_ref": event.get("payload", {}).get("result_ref"),
                "status": event.get("payload", {}).get("status"),
                "latency_ms": event.get("payload", {}).get("latency_ms")
            })

    return {
        "span_id": span_id,
        "tool_calls": tool_calls
    }


def _find_descendant_spans(events: List[Dict], span_id: str) -> set:
    """Find all descendant spans of a given span."""
    # Build parent map
    parent_map = {}
    for event in events:
        if event.get("type") == "span.start":
            sid = event.get("span_id")
            parent = event.get("parent_span_id")
            if parent:
                parent_map[sid] = parent

    # Find descendants
    descendants = set()
    to_check = [span_id]

    while to_check:
        current = to_check.pop()
        for sid, parent in parent_map.items():
            if parent == current and sid not in descendants:
                descendants.add(sid)
                to_check.append(sid)

    return descendants


def llm_call_view(span_id: str, events_path: str = "traces/events.jsonl") -> Dict:
    """
    Detailed view of an LLM call span.

    Args:
        span_id: Span ID
        events_path: Path to events file

    Returns:
        Dict with LLM call details (no full prompt/response, just refs and previews)
    """
    events = load_events(events_path)

    start_event = None
    end_event = None

    for event in events:
        if event.get("span_id") == span_id:
            if event.get("type") == "llm.call.start":
                start_event = event
            elif event.get("type") == "llm.call.end":
                end_event = event

    if not start_event:
        return {"span_id": span_id, "error": "LLM call not found"}

    result = {
        "span_id": span_id,
        "model": start_event.get("payload", {}).get("model"),
        "prompt_ref": start_event.get("payload", {}).get("prompt_ref"),
        "prompt_preview": start_event.get("payload", {}).get("prompt_preview"),
        "params": start_event.get("payload", {}).get("params", {}),
        "purpose": start_event.get("payload", {}).get("purpose")
    }

    if end_event:
        result["response_ref"] = end_event.get("payload", {}).get("response_ref")
        result["response_preview"] = end_event.get("payload", {}).get("response_preview")
        result["usage"] = end_event.get("payload", {}).get("usage", {})
        result["finish_reason"] = end_event.get("payload", {}).get("finish_reason")
        result["tool_calls"] = end_event.get("payload", {}).get("tool_calls", [])

    return result


def list_runs(events_path: str = "traces/events.jsonl", limit: int = 20) -> List[Dict]:
    """
    List recent runs.

    Args:
        events_path: Path to events file
        limit: Maximum number of runs to return

    Returns:
        List of run summaries
    """
    events = load_events(events_path)

    # Group by run_id
    runs = {}
    for event in events:
        run_id = event.get("run_id")
        if not run_id:
            continue

        if run_id not in runs:
            runs[run_id] = {
                "run_id": run_id,
                "start_ts": None,
                "end_ts": None,
                "status": None,
                "entrypoint": None
            }

        if event.get("type") == "run.start":
            runs[run_id]["start_ts"] = event.get("ts")
            runs[run_id]["entrypoint"] = event.get("payload", {}).get("entrypoint")
        elif event.get("type") == "run.end":
            runs[run_id]["end_ts"] = event.get("ts")
            runs[run_id]["status"] = event.get("payload", {}).get("status")

    # Sort by start time (most recent first) and limit
    run_list = list(runs.values())
    run_list.sort(key=lambda r: r.get("start_ts", 0), reverse=True)
    return run_list[:limit]


def extract_run_signature(payload: Dict) -> Dict[str, Any]:
    """
    Extract a safe, selective signature from run.start payload.
    
    Only includes whitelisted parameters that are:
    - Useful for debugging
    - Safe to display (no PII, no sensitive data)
    - Not too verbose
    
    Args:
        payload: Run start payload with 'args' and optionally 'env'
        
    Returns:
        Dict with extracted signature parameters
    """
    sig = {}
    
    for section, keys in RUN_SIGNATURE_KEYS.items():
        data = payload.get(section, {})
        if not isinstance(data, dict):
            continue
            
        for key in keys:
            if key in data:
                value = data[key]
                
                # Format lists/arrays compactly
                if isinstance(value, list):
                    if len(value) == 0:
                        continue
                    # Show count for long lists, full list for short ones
                    if len(value) > 5:
                        sig[key] = f"{len(value)} items"
                    else:
                        # Join with comma, truncate individual items if too long
                        items = [str(v)[:20] + "..." if len(str(v)) > 20 else str(v) for v in value]
                        sig[key] = ",".join(items)
                # Format booleans as yes/no
                elif isinstance(value, bool):
                    sig[key] = "yes" if value else "no"
                # Format None
                elif value is None:
                    continue
                # Truncate long strings
                elif isinstance(value, str) and len(value) > 50:
                    sig[key] = value[:47] + "..."
                else:
                    sig[key] = value
    
    return sig


def format_run_signature(sig: Dict[str, Any], max_length: int = 250) -> str:
    """
    Format signature dict into compact string for timeline.
    
    Args:
        sig: Signature dict from extract_run_signature
        max_length: Maximum length before truncation
        
    Returns:
        Formatted string like "model=gpt-4 tools=search_web stream=yes"
    """
    if not sig:
        return ""
    
    # Order keys for consistent output (most important first)
    priority_keys = [
        "model_id", "model",  # Model - most important
        "selected_tools", "tools",  # Tools - critical for debugging
        "selected_agents", "agents",  # Agents
        "depth", "research_depth",  # Depth - critical for planner mode
        "code_execution_mode",  # Code execution mode - critical
        "hitl_enabled",  # HITL - important flag
        "hitl_auto_approve",  # HITL settings
        "use_rag",  # RAG flag
        "selected_validators", "validators",  # Validators
        "stream", "streaming",  # Streaming
        "planner_depth",  # Planner depth
        "apply_company_context",  # Context flag
        "response_schema", "schema",  # Schema
        "endpoint", "client",  # Endpoint/client
        "conversation_id",  # Conversation ID (less important, can be truncated)
        "temperature", "max_tokens",  # Model params (less critical)
    ]
    
    ordered_items = []
    remaining = set(sig.keys())
    
    # Add priority keys first
    for key in priority_keys:
        if key in sig:
            ordered_items.append((key, sig[key]))
            remaining.discard(key)
    
    # Add remaining keys
    for key in sorted(remaining):
        ordered_items.append((key, sig[key]))
    
    # Format as key=value pairs, truncate if too long
    parts = []
    current_length = 0
    
    for k, v in ordered_items:
        part = f"{k}={v}"
        part_length = len(part) + 1  # +1 for space
        
        # If adding this part would exceed max_length, stop
        if current_length + part_length > max_length and parts:
            break
            
        parts.append(part)
        current_length += part_length
    
    result = " ".join(parts)
    
    # If we truncated, add indicator
    if len(ordered_items) > len(parts):
        result += " ..."
    
    return result


def find_last_run(
    events_path: str = "traces/events.jsonl",
    status: Optional[str] = None,
    entrypoint: Optional[str] = None,
    offset: int = 0
) -> Optional[str]:
    """
    Find the n-th most recent run ID matching optional filters.
    
    Includes both completed runs (with run.end) and running runs (only run.start).

    Args:
        events_path: Path to events file
        status: Optional status filter ("ok", "error", etc.). 
                If None, includes both completed and running runs.
                If specified, only matches completed runs with that status.
        entrypoint: Optional entrypoint filter (e.g., "api.chat_v2")
        offset: Offset from most recent (0 = most recent, 1 = penultimate, etc.)

    Returns:
        Run ID of the n-th most recent matching run, or None if not found
    """
    events = load_events(events_path)

    # Collect all runs with their metadata
    run_metadata = {}
    run_timestamps = {}  # Track most recent timestamp per run (start or end)

    for event in events:
        run_id = event.get("run_id")
        if not run_id:
            continue

        # Collect run metadata
        if run_id not in run_metadata:
            run_metadata[run_id] = {
                "entrypoint": None,
                "status": None,
                "start_ts": None,
                "end_ts": None,
                "is_completed": False
            }

        if event.get("type") == "run.start":
            run_metadata[run_id]["entrypoint"] = event.get("payload", {}).get("entrypoint")
            run_metadata[run_id]["start_ts"] = event.get("ts")
            run_timestamps[run_id] = event.get("ts")  # Use start_ts for running runs
        elif event.get("type") == "run.end":
            run_metadata[run_id]["status"] = event.get("payload", {}).get("status")
            run_metadata[run_id]["end_ts"] = event.get("ts")
            run_metadata[run_id]["is_completed"] = True
            run_timestamps[run_id] = event.get("ts")  # Use end_ts for completed runs

    # Filter by status and entrypoint
    filtered_runs = []
    for run_id, ts in run_timestamps.items():
        meta = run_metadata.get(run_id, {})
        
        # Apply filters
        if status:
            # If status filter specified, only match completed runs with that status
            if not meta.get("is_completed") or meta.get("status") != status:
                continue
        # If no status filter, include both completed and running runs
        
        if entrypoint and meta.get("entrypoint") != entrypoint:
            continue
        
        filtered_runs.append((run_id, ts))

    if not filtered_runs:
        return None

    # Sort by timestamp (most recent first) and return the run at offset
    filtered_runs.sort(key=lambda x: x[1], reverse=True)
    if offset < len(filtered_runs):
        return filtered_runs[offset][0]
    return None


# batch_synthesize_extracts() removed - replaced by synthesis.py providers


def _load_storytelling_prompts(conversation_id: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Load prompt/response extracts from storytelling logs.
    Returns map of node_name -> list of {prompt, response}
    """
    extracts = {}  # node_name -> list of calls
    
    # 1. Try to load from subdirectory with individual call files (most up-to-date)
    path = Path(f"logs/storytelling/{conversation_id}")
    if path.exists() and path.is_dir():
        call_files = sorted(path.glob("llm_call_*.json"))
        for f in call_files:
            try:
                with open(f, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
                    node = data.get("node", "unknown")
                    if node not in extracts:
                        extracts[node] = []
                    extracts[node].append({
                        "prompt": data.get("prompt", ""),
                        "response": data.get("response", "")
                    })
            except Exception:
                continue
    
    # 2. Fallback: Try to load from "full" storytelling file if folder is empty/missing
    if not extracts:
        # Match storytelling_conv-ID*.json
        root_path = Path("logs/storytelling")
        if root_path.exists():
            full_files = list(root_path.glob(f"storytelling_{conversation_id}*.json"))
            if full_files:
                try:
                    with open(full_files[0], 'r', encoding='utf-8') as jf:
                        data = json.load(jf)
                        # Extract from phases/llm_calls
                        for phase_data in data.get("phases", {}).values():
                            for call in phase_data.get("llm_calls", []):
                                node = call.get("node", "unknown")
                                if node not in extracts:
                                    extracts[node] = []
                                extracts[node].append({
                                    "prompt": call.get("prompt", ""),
                                    "response": call.get("response", "")
                                })
                except Exception:
                    pass

    return extracts


def timeline_view(
    run_id: str,
    events_path: str = "traces/events.jsonl",
    compact: bool = False,
    synthesize: bool = False,
    synthesis_provider: Optional[SynthesisProvider] = None,
    verbose: bool = False,
) -> List[str]:
    """
    Canonical timeline view with complete span hierarchy and delta times.
    
    This is the source of truth view - deterministic, complete, suitable for both
    human analysis and LLM-based debugging. Preserves full causal structure.

    Enterprise-level debug view showing:
    - ts_human (HH:MM:SS.sss)
    - Event type
    - dt_ms (delta from previous event)
    - Span hierarchy with indentation
    - Key payload details
    - Component context for LLM calls (when spans are available)

    Args:
        run_id: Run ID
        events_path: Path to events file
        compact: If True, return lossy projection showing only LLM calls with context.
                 If False (default), return complete canonical view.
        synthesize: If True, use synthesis provider to summarize LLM extracts.
        synthesis_provider: SynthesisProvider instance (optional, defaults to TruncationProvider)
    
    Returns:
        List of formatted timeline lines
    """

    # Use truncation provider as fallback if no provider specified
    if synthesize and synthesis_provider is None:
        synthesis_provider = TruncationProvider()

    logging.debug(f"[TIMELINE_VIEW] synthesis_provider: {synthesis_provider}, synthesize={synthesize}")
    logging.info(f"[TIMELINE_VIEW] synthesis_provider: {synthesis_provider}, synthesize={synthesize}")

    # Accumula messaggi di fallback se verbose=True
    fallback_messages = []
    if verbose and hasattr(synthesis_provider, 'verbose'):
        synthesis_provider.verbose = verbose
    if verbose and hasattr(synthesis_provider, 'fallback_callback'):
        synthesis_provider.fallback_callback = lambda msg: fallback_messages.append(msg)

    events = load_events(events_path, run_id=run_id)
    logging.debug(f"[TIMELINE_VIEW] Caricati {len(events)} eventi")

    if not events:
        return ["No events found for this run"]

    # Build complete span map with hierarchy
    span_info = {}  # span_id -> {name, actor, kind, parent_span_id, start_ts, depth}
    span_depths = {}
    llm_call_starts = {}  # span_id -> start_ts (for calculating duration)
    llm_call_index_map = {}  # span_id -> index within parent (for numbering)
    
    # Extract LLM prompts/responses from trace events for synthesis
    storytelling_extracts = {}  # component_name -> list of {prompt, response, span_id, parent_span_id}
    llm_call_data = {}  # span_id -> {prompt_ref, response_ref, component_name, parent_span_id}
    
    # First pass: build span hierarchy and track LLM call starts
    parent_llm_call_counts = {}  # parent_span_id -> list of llm_call span_ids in order

    logging.debug(f"[TIMELINE_VIEW] Primo passaggio: analisi {len(events)} eventi...")
    for event in events:
        span_id = event.get("span_id")
        parent_span_id = event.get("parent_span_id")
        event_type = event.get("type", "")
        payload = event.get("payload", {})

        if event_type == "span.start":
            if span_id not in span_info:
                span_info[span_id] = {
                    "name": payload.get("name", "unknown"),
                    "actor": event.get("actor", "unknown"),
                    "kind": payload.get("kind", "unknown"),
                    "parent_span_id": parent_span_id,
                    "start_ts": event.get("ts"),
                    "depth": 0
                }
            
            # Calculate depth
            if parent_span_id is None:
                span_depths[span_id] = 0
                span_info[span_id]["depth"] = 0
            else:
                parent_depth = span_depths.get(parent_span_id, 0)
                span_depths[span_id] = parent_depth + 1
                span_info[span_id]["depth"] = parent_depth + 1
        
        elif event_type == "llm.call.start":
            # Track LLM call start time for duration calculation
            llm_call_starts[span_id] = event.get("ts")
            
            # Track LLM calls per parent span for numbering
            if parent_span_id:
                if parent_span_id not in parent_llm_call_counts:
                    parent_llm_call_counts[parent_span_id] = []
                parent_llm_call_counts[parent_span_id].append(span_id)
            
            # Extract prompt_ref for synthesis (component name resolved in second pass)
            prompt_ref = payload.get("prompt_ref")
            if prompt_ref:
                # 🔧 FIX: Usa timestamp come chiave se span_id è None (per matching quando non c'è span hierarchy)
                key = span_id if span_id else f"ts_{event.get('ts')}"
                if key not in llm_call_data:
                    llm_call_data[key] = {}
                llm_call_data[key]["prompt_ref"] = prompt_ref
                llm_call_data[key]["parent_span_id"] = parent_span_id
                llm_call_data[key]["span_id"] = span_id  # Salva span_id per matching univoco
                llm_call_data[key]["ts"] = event.get("ts")  # Salva timestamp per matching quando span_id è None
        
        elif event_type == "llm.call.end":
            # Extract response_ref for synthesis
            response_ref = payload.get("response_ref")
            if response_ref:
                # 🔧 FIX: Usa timestamp come chiave se span_id è None (per matching quando non c'è span hierarchy)
                # Cerca prima se esiste già una entry per questo span_id o timestamp
                key = span_id if span_id else f"ts_{event.get('ts')}"
                # Se span_id è None, cerca la entry più vicina per timestamp
                if not span_id:
                    # Cerca la entry più vicina per timestamp (entro 1 secondo)
                    closest_key = None
                    closest_ts_diff = float('inf')
                    for k, data in llm_call_data.items():
                        if k.startswith("ts_") and "ts" in data:
                            ts_diff = abs(data["ts"] - event.get("ts", 0))
                            if ts_diff < closest_ts_diff and ts_diff < 1.0:  # Entro 1 secondo
                                closest_ts_diff = ts_diff
                                closest_key = k
                    if closest_key:
                        key = closest_key
                
                if key not in llm_call_data:
                    llm_call_data[key] = {}
                llm_call_data[key]["response_ref"] = response_ref
                if "parent_span_id" not in llm_call_data[key]:
                    llm_call_data[key]["parent_span_id"] = parent_span_id
                llm_call_data[key]["span_id"] = span_id  # Salva span_id per matching univoco
                if "ts" not in llm_call_data[key]:
                    llm_call_data[key]["ts"] = event.get("ts")  # Salva timestamp per matching quando span_id è None
    
    # Load extracts from blob store if synthesis is enabled (second pass: resolve component names)
    if synthesize:
        from .blob_store import BlobStore
        # Determine blob store path from events_path
        events_path_obj = Path(events_path)
        if events_path_obj.name == "events.jsonl":
            # If events.jsonl, use parent/blobs
            blob_store_path = str(events_path_obj.parent / "blobs")
        else:
            # Fallback: assume traces/blobs
            blob_store_path = "traces/blobs"
        blob_store = BlobStore(blob_store_path)
        
        for key, data in llm_call_data.items():
            # 🔧 FIX: Estrai span_id dalla data (può essere None)
            span_id = data.get("span_id")
            # Resolve component name from parent span
            parent_span_id = data.get("parent_span_id")
            component_name = "unknown"
            if parent_span_id and parent_span_id in span_info:
                component_name = span_info[parent_span_id].get("name", "unknown")
            else:
                # Try to get from span itself if it's an agent span
                if span_id and span_id in span_info:
                    span_kind = span_info[span_id].get("kind")
                    if span_kind == "agent":
                        component_name = span_info[span_id].get("name", "unknown")
            
            prompt_ref = data.get("prompt_ref")
            response_ref = data.get("response_ref")

            if component_name not in storytelling_extracts:
                storytelling_extracts[component_name] = []

            extract = {}
            try:
                if prompt_ref:
                    prompt_content = blob_store.get(prompt_ref)
                    extract["prompt"] = prompt_content
            except Exception as e:
                logging.debug(f"Could not load prompt from {prompt_ref}: {e}")

            try:
                if response_ref:
                    response_content = blob_store.get(response_ref)
                    extract["response"] = response_content
                else:
                    # Debug: response_ref è None
                    logging.warning(f"[EXTRACT] No response_ref for component={component_name}, span_id={span_id}")
                    print(f"Debug: No response_ref for {component_name} span={span_id}", file=sys.stderr)
            except Exception as e:
                logging.warning(f"[EXTRACT] Could not load response from {response_ref}: {e}")
                print(f"Debug: Failed to load response from {response_ref}: {e}", file=sys.stderr)
            
            if extract:
                # 🔧 FIX: Salva span_id, parent_span_id e timestamp per matching univoco nel timeline
                extract["span_id"] = span_id
                extract["parent_span_id"] = parent_span_id
                extract["ts"] = data.get("ts")  # Salva timestamp per matching quando span_id è None
                storytelling_extracts[component_name].append(extract)
                # Debug: log se prompt/response sono vuoti
                if extract.get("prompt") and len(extract["prompt"]) == 0:
                    logging.warning(f"[SYNTHESIS] Prompt vuoto per span_id={span_id}, component={component_name}")
                if extract.get("response") and len(extract["response"]) == 0:
                    logging.warning(f"[SYNTHESIS] Response vuota per span_id={span_id}, component={component_name}")
    
    # 🧠 Micromodel Synthesis Pass
    logging.debug(f"[TIMELINE_VIEW] synthesize={synthesize}, storytelling_extracts={len(storytelling_extracts) if storytelling_extracts else 0} componenti")
    logging.info(f"[TIMELINE_VIEW] synthesize={synthesize}, storytelling_extracts={len(storytelling_extracts) if storytelling_extracts else 0} componenti")

    # 🎯 v0.3: Sintesi separate di prompt e response + salvataggio file completi
    # Mappa: (component, idx) -> {"prompt_summary": str, "response_summary": str, "prompt_file": str, "response_file": str}
    prompt_response_summaries = {}

    if synthesize:
        if not storytelling_extracts:
            print(f"Warning: No extracts found to synthesize. synthesis_provider={synthesis_provider}", file=sys.stderr)
        elif synthesis_provider is None:
            print(f"Warning: synthesis_provider is None. Cannot synthesize extracts.", file=sys.stderr)
        else:
            print(f"Info: Found {len(storytelling_extracts)} components with extracts. Generating prompt and response summaries...", file=sys.stderr)
            logging.debug(f"[TIMELINE_VIEW] Generazione sintesi separate per {len(storytelling_extracts)} componenti")

            # Crea directory per salvare file completi (organizzati per run_id)
            events_dir = Path(events_path).parent if events_path else Path("traces")
            prompts_dir = events_dir / "prompts" / run_id
            prompts_dir.mkdir(parents=True, exist_ok=True)

            # Prepara batch separati per sintesi prompt e response
            prompts_to_synthesize = []
            responses_to_synthesize = []
            synthesis_keys = []  # (component, idx) per ogni coppia

            # Debug: conta distribuzione prompt/response
            debug_stats = {"total": 0, "has_prompt": 0, "has_response": 0, "has_both": 0}

            for component, calls in storytelling_extracts.items():
                for idx, call in enumerate(calls):
                    prompt = call.get("prompt", "")
                    response = call.get("response", "")

                    debug_stats["total"] += 1
                    if prompt:
                        debug_stats["has_prompt"] += 1
                    if response:
                        debug_stats["has_response"] += 1

                    if prompt and response:
                        debug_stats["has_both"] += 1
                        
                        # Salva file completi
                        # Nome file: {component}_{idx}_prompt.txt e {component}_{idx}_response.txt
                        safe_component = component.replace("/", "_").replace("\\", "_")
                        prompt_filename = f"{safe_component}_{idx}_prompt.txt"
                        response_filename = f"{safe_component}_{idx}_response.txt"
                        prompt_file_path = prompts_dir / prompt_filename
                        response_file_path = prompts_dir / response_filename
                        
                        # Salva file completi
                        try:
                            prompt_file_path.write_text(prompt, encoding='utf-8')
                            response_file_path.write_text(response, encoding='utf-8')
                        except Exception as e:
                            logging.warning(f"[TIMELINE_VIEW] Errore salvataggio file per {component} #{idx+1}: {e}")
                        
                        # Path relativo per visualizzazione nella timeline (rispetto alla directory di lavoro corrente)
                        # Se events_path è "traces/events.jsonl", prompts_dir sarà "traces/prompts/{run_id}"
                        # Il percorso relativo deve essere "traces/prompts/{run_id}/{filename}" per essere accessibile
                        events_path_str = str(events_path) if events_path else "traces/events.jsonl"
                        if "traces" in events_path_str:
                            prompt_file_rel = f"traces/prompts/{run_id}/{prompt_filename}"
                            response_file_rel = f"traces/prompts/{run_id}/{response_filename}"
                        else:
                            # Fallback: usa percorso relativo alla directory di events_path
                            prompts_dir_rel = prompts_dir.relative_to(Path.cwd()) if prompts_dir.is_relative_to(Path.cwd()) else prompts_dir
                            prompt_file_rel = f"{prompts_dir_rel}/{prompt_filename}"
                            response_file_rel = f"{prompts_dir_rel}/{response_filename}"
                        
                        # Prepara testi per sintesi (limita lunghezza per efficienza)
                        # 2000 caratteri ≈ 500 token di input, sufficiente per generare sintesi complete
                        prompt_text = prompt[:2000] if len(prompt) > 2000 else prompt
                        response_text = response[:2000] if len(response) > 2000 else response
                        
                        # Sintesi prompt: focus sull'intento
                        prompt_synthesis_text = f"Summarize the INTENT and PURPOSE of this prompt in maximum {MAX_SYNTHESIS_WORDS} words. Focus on what the user/system is asking for:\n\n{prompt_text}"
                        
                        # Sintesi response: focus su come soddisfa l'intento
                        response_synthesis_text = f"Summarize this response focusing on how it addresses and satisfies the intent. Maximum {MAX_SYNTHESIS_WORDS} words. Focus on the relevant parts that fulfill the request:\n\n{response_text}"
                        
                        prompts_to_synthesize.append(prompt_synthesis_text)
                        responses_to_synthesize.append(response_synthesis_text)
                        synthesis_keys.append((component, idx, prompt_file_rel, response_file_rel))

            # Debug output
            print(f"Debug: Extracts stats - total={debug_stats['total']}, has_prompt={debug_stats['has_prompt']}, has_response={debug_stats['has_response']}, has_both={debug_stats['has_both']}", file=sys.stderr)

            total_pairs = len(synthesis_keys)
            if total_pairs > 0:
                print(f"Info: Generating {total_pairs} prompt summaries and {total_pairs} response summaries...", file=sys.stderr)
                try:
                    # Batch synthesis async per prompt e response in parallelo
                    prompt_summaries = asyncio.run(synthesis_provider.synthesize_batch(prompts_to_synthesize, max_words=MAX_SYNTHESIS_WORDS))
                    response_summaries = asyncio.run(synthesis_provider.synthesize_batch(responses_to_synthesize, max_words=MAX_SYNTHESIS_WORDS))

                    # 🔧 DEBUG: Mappa risultati e logga fallimenti
                    successful = 0
                    failed = 0
                    empty = 0
                    for key_data, prompt_summary, response_summary in zip(synthesis_keys, prompt_summaries, response_summaries):
                        component, idx, prompt_file_rel, response_file_rel = key_data
                        key = (component, idx)
                        
                        # 🔧 [FIX_INNERTRACE] Log sintesi ricevute
                        prompt_summary_raw_len = len(prompt_summary) if prompt_summary else 0
                        response_summary_raw_len = len(response_summary) if response_summary else 0
                        logging.debug(f"[FIX_INNERTRACE] sintesi ricevuta - key={key}, prompt_raw_len={prompt_summary_raw_len}, response_raw_len={response_summary_raw_len}, prompt_is_none={prompt_summary is None}, response_is_none={response_summary is None}")
                        
                        prompt_summary_clean = prompt_summary.strip() if prompt_summary and len(prompt_summary.strip()) > 0 else None
                        response_summary_clean = response_summary.strip() if response_summary and len(response_summary.strip()) > 0 else None
                        
                        # 🔧 [FIX_INNERTRACE] Log sintesi dopo pulizia
                        prompt_summary_clean_len = len(prompt_summary_clean) if prompt_summary_clean else 0
                        response_summary_clean_len = len(response_summary_clean) if response_summary_clean else 0
                        logging.debug(f"[FIX_INNERTRACE] sintesi dopo clean - key={key}, prompt_clean_len={prompt_summary_clean_len}, response_clean_len={response_summary_clean_len}, prompt_clean_is_none={prompt_summary_clean is None}, response_clean_is_none={response_summary_clean is None}")
                        
                        if prompt_summary_clean and response_summary_clean:
                            prompt_response_summaries[key] = {
                                "prompt_summary": prompt_summary_clean,
                                "response_summary": response_summary_clean,
                                "prompt_file": prompt_file_rel,
                                "response_file": response_file_rel
                            }
                            successful += 1
                            # 🔧 [FIX_INNERTRACE] Log salvataggio riuscito
                            logging.debug(f"[FIX_INNERTRACE] sintesi salvata SUCCESS - key={key}, prompt_len={prompt_summary_clean_len}, response_len={response_summary_clean_len}")
                        else:
                            failed += 1
                            if not prompt_summary_clean:
                                print(f"Warning: Prompt synthesis failed for {component} #{idx+1} (summary is None or empty)", file=sys.stderr)
                                logging.warning(f"[PROMPT_SUMMARY] Failed for {component} #{idx+1}: summary is None or empty")
                                logging.debug(f"[FIX_INNERTRACE] prompt_summary FAILED - key={key}, prompt_summary_raw='{prompt_summary[:100] if prompt_summary else None}'")
                            if not response_summary_clean:
                                print(f"Warning: Response synthesis failed for {component} #{idx+1} (summary is None or empty)", file=sys.stderr)
                                logging.warning(f"[RESPONSE_SUMMARY] Failed for {component} #{idx+1}: summary is None or empty")
                                logging.debug(f"[FIX_INNERTRACE] response_summary FAILED - key={key}, response_summary_raw='{response_summary[:100] if response_summary else None}'")
                            
                            # Anche se una sintesi fallisce, salva comunque i file e le sintesi disponibili
                            prompt_summary_final = prompt_summary_clean or "[Synthesis failed]"
                            response_summary_final = response_summary_clean or "[Synthesis failed]"
                            prompt_response_summaries[key] = {
                                "prompt_summary": prompt_summary_final,
                                "response_summary": response_summary_final,
                                "prompt_file": prompt_file_rel,
                                "response_file": response_file_rel
                            }
                            # 🔧 [FIX_INNERTRACE] Log salvataggio con fallback
                            logging.debug(f"[FIX_INNERTRACE] sintesi salvata FALLBACK - key={key}, prompt_final='{prompt_summary_final[:50]}...', response_final='{response_summary_final[:50]}...', prompt_len={len(prompt_summary_final)}, response_len={len(response_summary_final)}")

                    print(f"Info: Synthesis results - successful={successful}, failed={failed}, empty={empty}, total={total_pairs}", file=sys.stderr)
                    if failed > 0 or empty > 0:
                        print(f"Warning: {failed + empty} out of {total_pairs} syntheses failed or returned empty.", file=sys.stderr)
                    
                    # 🔧 [FIX_INNERTRACE] Log finale statistiche sintesi salvate
                    total_saved = len(prompt_response_summaries)
                    logging.debug(f"[FIX_INNERTRACE] sintesi finali salvate - total_saved={total_saved}, expected={total_pairs}, successful={successful}, failed={failed}, empty={empty}")
                    if total_saved < total_pairs:
                        missing_keys = []
                        for key_data in synthesis_keys:
                            component, idx, _, _ = key_data
                            key = (component, idx)
                            if key not in prompt_response_summaries:
                                missing_keys.append(key)
                        logging.debug(f"[FIX_INNERTRACE] sintesi MANCANTI - missing_keys={missing_keys[:10]}")  # Primi 10 per non intasare
                except Exception as e:
                    print(f"Error: Summary generation failed: {type(e).__name__}: {str(e)}", file=sys.stderr)
                    logging.warning(f"[SUMMARY] Batch failed: {e}", exc_info=True)
            else:
                print(f"Warning: No prompt/response pairs found for synthesis.", file=sys.stderr)

    # 🎯 v0.2: Valutazioni rimosse dal core
    # InnerTrace registra, struttura, rende leggibile - NON giudica
    # Le valutazioni (se necessarie) vanno in projection layer opzionale

    # Build index map: for each LLM call, assign its number within parent
    for parent_span_id, llm_call_span_ids in parent_llm_call_counts.items():
        for idx, llm_span_id in enumerate(llm_call_span_ids):
            llm_call_index_map[llm_span_id] = idx + 1

    # Build map of open spans at each timestamp (for LLM calls without span_id)
    logging.debug(f"[TIMELINE_VIEW] Costruzione mappa span aperti...")
    open_spans_by_ts = {}  # ts -> list of open span_ids (ordered by most recent)
    sorted_ts = []  # Sorted list of timestamps for binary search
    active_spans = set()  # Currently open spans
    prev_active_spans = set()  # Track previous state to detect changes

    for event in events:
        event_ts = event.get("ts", 0)
        span_id = event.get("span_id")
        event_type = event.get("type", "")

        if event_type == "span.start":
            active_spans.add(span_id)
        elif event_type == "span.end":
            active_spans.discard(span_id)

        # Only store snapshot when active_spans actually changes (optimization)
        if active_spans != prev_active_spans:
            open_spans_by_ts[event_ts] = list(active_spans)
            sorted_ts.append(event_ts)  # Timestamps are already in order
            prev_active_spans = active_spans.copy()
    
    def find_parent_component(span_id: Optional[str], event_ts: Optional[float] = None) -> Optional[str]:
        """
        Find the parent component (agent) for a span by walking up the hierarchy.
        
        Deterministic resolution:
        1. If span_id exists, walk up via parent_span_id
        2. Stop at first span with kind="agent"
        3. Fallback to span.name if no agent found
        4. If no span_id, try timestamp-based lookup for open spans
        5. Never invent or infer - only use actual span data
        
        Returns:
            Component name (span.name) or None if not found
        """
        # If we have a span_id, walk up the hierarchy deterministically
        if span_id:
            current = span_id
            visited = set()
            
            while current and current not in visited:
                visited.add(current)
                info = span_info.get(current)
                if not info:
                    break
                
                # Stop at first agent span (deterministic rule)
                if info.get("kind") == "agent":
                    return info.get("name")
                
                # Walk up to parent via parent_span_id (only deterministic path)
                current = info.get("parent_span_id")
            
            # Fallback: return span.name if span exists (mechanical, no inference)
            if span_id in span_info:
                return span_info[span_id].get("name")
        
        # If no span_id, try timestamp-based lookup (only if spans exist)
        if event_ts is not None and open_spans_by_ts and sorted_ts:
            # Find the closest timestamp <= event_ts using binary search (O(log n))
            idx = bisect.bisect_right(sorted_ts, event_ts) - 1
            if idx >= 0:
                closest_ts = sorted_ts[idx]
                open_spans = open_spans_by_ts[closest_ts]
                
                # Find the most recent agent span among open spans (deterministic)
                for span_id_candidate in reversed(open_spans):
                    info = span_info.get(span_id_candidate)
                    if info and info.get("kind") == "agent":
                        return info.get("name")
        
        # No component found - return None (never invent)
        return None

    def get_llm_call_number(span_id: Optional[str]) -> str:
        """Get the call number for this LLM call within its parent span."""
        if not span_id:
            return ""
        
        index = llm_call_index_map.get(span_id)
        if index and index > 1:
            return f" #{index}"
        return ""

    # Check if we have any spans (for context resolution)
    has_spans = len(span_info) > 0

    # Format timeline
    logging.debug(f"[TIMELINE_VIEW] Generazione output timeline...")
    lines = []
    
    # Add synthesis fallback messages if verbose=True
    if verbose and fallback_messages:
        lines.append("# Synthesis Provider Fallback Log:")
        for msg in fallback_messages:
            lines.append(f"# {msg}")
        lines.append("")
    
    # Add warning if no spans but LLM calls exist
    if not has_spans:
        llm_calls = [e for e in events if e.get("type") in ("llm.call.start", "llm.call.end")]
        if llm_calls:
            lines.append("# ⚠️  NOTE: No span hierarchy found - component context unavailable")
            lines.append("#    To see component context (e.g., [meta_planner], [code_orchestrator]),")
            lines.append("#    components must create spans with kind='agent'")
            lines.append("")

    logging.debug(f"[TIMELINE_VIEW] Processando {len(events)} eventi per output...")
    event_count = 0
    for event in events:
        event_count += 1
        if event_count <= 5 or event_count % 100 == 0:
            logging.debug(f"[TIMELINE_VIEW] Processati {event_count}/{len(events)} eventi...")
        ts_human = event.get("ts_human", "??:??:??.???")
        event_type = event.get("type", "unknown")
        dt_ms = event.get("dt_ms")
        span_id = event.get("span_id")
        payload = event.get("payload", {})

        # Determine indentation based on span depth
        depth = span_depths.get(span_id, 0)
        
        # 🔧 FIX Indentation:
        # Span events stay at their depth.
        # Other events (tool calls, llm calls, router decisions) inside a span should be indented +1.
        if event_type not in ("span.start", "span.end", "run.start", "run.end"):
            indent = "  " * (depth + 1)
        else:
            indent = "  " * depth

        # Format delta time
        dt_str = f" (+{dt_ms}ms)" if dt_ms is not None and dt_ms > 0 else ""

        # Extract key details from payload
        details = ""
        if event_type == "run.start":
            entrypoint = payload.get('entrypoint', 'unknown')
            # Extract and format signature
            sig = extract_run_signature(payload)
            sig_str = format_run_signature(sig)
            
            if sig_str:
                details = f"({entrypoint}) {sig_str}"
            else:
                details = f"({entrypoint})"
        elif event_type == "run.end":
            status = payload.get('status', 'unknown')
            latency = payload.get('latency_ms', 0)
            details = f"({status}, {latency}ms total)"
        elif event_type == "span.start":
            kind = payload.get('kind', 'unknown')
            name = payload.get('name', 'unknown')
            details = f"[{kind}] {name}"
        elif event_type == "span.end":
            status = payload.get('status', 'ok')
            latency = payload.get('latency_ms', 0)
            details = f"({status}, {latency}ms)"
        elif event_type == "llm.call.start":
            model = payload.get('model', 'unknown')
            purpose = payload.get('purpose', '')
            event_ts = event.get("ts")
            
            # Find parent component context
            # Try span_id first, then parent_span_id, then timestamp-based lookup
            component = None
            if span_id:
                component = find_parent_component(span_id, event_ts)
            elif event.get("parent_span_id"):
                component = find_parent_component(event.get("parent_span_id"), event_ts)
            else:
                component = find_parent_component(None, event_ts)
            
            call_number = get_llm_call_number(span_id) if span_id else ""
            
            # If we found a component but no call_number, try to get it from parent
            if component and not call_number and event.get("parent_span_id"):
                # Count LLM calls in this component up to this point
                component_llm_count = 0
                for prev_event in events:
                    if prev_event.get("ts", 0) > event_ts:
                        break
                    if prev_event.get("type") == "llm.call.start":
                        prev_component = find_parent_component(
                            prev_event.get("span_id"),
                            prev_event.get("ts")
                        )
                        if prev_component == component:
                            component_llm_count += 1
                            if prev_event.get("ts") == event_ts:
                                break
                if component_llm_count > 1:
                    call_number = f" #{component_llm_count}"
            
            # Build context string
            context_str = ""
            if component:
                context_str = f" [{component}{call_number}]"
            
            
            details = f"({model}" + (f", {purpose}" if purpose else "") + f"){context_str}"
            
            # 🎭 Storytelling Extract: Show prompt snippet if available
            if storytelling_extracts:
                # 🔧 FIX: Cerca in tutti i componenti, non solo quello trovato
                # Il problema: component potrebbe essere None o "unknown" se non c'è span hierarchy
                # Soluzione: cerca per span_id in tutti i componenti
                matched_extract = None
                matched_component = None
                
                # Prima cerca per span_id in tutti i componenti
                if span_id:
                    for comp_name, node_calls in storytelling_extracts.items():
                        for extract in node_calls:
                            if extract.get("span_id") == span_id:
                                matched_extract = extract
                                matched_component = comp_name
                                break
                        if matched_extract:
                            break
                else:
                    # 🔧 FIX: Se span_id è None, cerca per timestamp (entro 1 secondo)
                    event_ts = event.get("ts")
                    if event_ts:
                        closest_extract = None
                        closest_ts_diff = float('inf')
                        for comp_name, node_calls in storytelling_extracts.items():
                            for extract in node_calls:
                                extract_ts = extract.get("ts")
                                if extract_ts:
                                    ts_diff = abs(extract_ts - event_ts)
                                    if ts_diff < closest_ts_diff and ts_diff < 1.0:  # Entro 1 secondo
                                        closest_ts_diff = ts_diff
                                        closest_extract = extract
                                        matched_component = comp_name
                        if closest_extract:
                            matched_extract = closest_extract
                
                # Fallback: se component è disponibile, cerca lì
                if not matched_extract and component:
                    node_calls = storytelling_extracts.get(component, [])
                    if span_id:
                        # Cerca per span_id
                        for extract in node_calls:
                            if extract.get("span_id") == span_id:
                                matched_extract = extract
                                matched_component = component
                                break
                    
                    # Fallback: usa idx se span_id non matcha (per retrocompatibilità)
                    if not matched_extract:
                        idx = int(call_number.strip().replace("#", "") or "1") - 1
                        if 0 <= idx < len(node_calls):
                            matched_extract = node_calls[idx]
                            matched_component = component
                
                # 🎯 v0.2: NON mostrare prompt raw - informazione disponibile in llm.call.end con sintesi neutra
                # llm.call.start serve solo a segnare l'inizio, i dettagli vanno in llm.call.end
                pass
            
        elif event_type == "llm.call.end":
            # Token and duration ONLY on llm.call.end (canonical location)
            usage = payload.get('usage', {})
            inp = usage.get('input_tokens', 0)
            out = usage.get('output_tokens', 0)
            total_tokens = inp + out if usage else 0
            event_ts = event.get("ts")
            
            # Calculate duration from matching llm.call.start
            duration_ms = None
            if span_id and span_id in llm_call_starts:
                # Direct match via span_id (deterministic)
                start_ts = llm_call_starts[span_id]
                end_ts = event.get("ts")
                if start_ts and end_ts:
                    duration_ms = int((end_ts - start_ts) * 1000)
            else:
                # Fallback: find matching start by looking backwards (deterministic matching)
                for prev_event in reversed(events):
                    if prev_event.get("ts", 0) > event_ts:
                        continue
                    if prev_event.get("type") == "llm.call.start":
                        # Match by model and purpose (deterministic attributes)
                        prev_model = prev_event.get("payload", {}).get("model")
                        prev_purpose = prev_event.get("payload", {}).get("purpose")
                        curr_model = payload.get("model")
                        curr_purpose = payload.get("purpose")
                        
                        if prev_model == curr_model and prev_purpose == curr_purpose:
                            start_ts = prev_event.get("ts")
                            if start_ts:
                                duration_ms = int((event_ts - start_ts) * 1000)
                            break
            
            # Find parent component for context (deterministic resolution)
            component = None
            if span_id:
                component = find_parent_component(span_id, event_ts)
            elif event.get("parent_span_id"):
                component = find_parent_component(event.get("parent_span_id"), event_ts)
            else:
                component = find_parent_component(None, event_ts)
            
            # Build details: tokens and duration (canonical format)
            if usage and total_tokens > 0:
                if duration_ms is not None:
                    duration_s = duration_ms / 1000.0
                    details = f"({total_tokens:,} tokens, {duration_s:.2f}s)"
                else:
                    details = f"({total_tokens:,} tokens)"
            elif duration_ms is not None:
                duration_s = duration_ms / 1000.0
                details = f"({duration_s:.2f}s)"
            else:
                details = ""
            
            # Add component context (only if found deterministically)
            if component:
                details += f" [{component}]"
                
                # 🎭 Storytelling Extract: Show response snippet if available
                if storytelling_extracts:
                    # 🔧 [FIX_INNERTRACE] Log inizio matching
                    logging.debug(f"[FIX_INNERTRACE] inizio matching extract - span_id={span_id}, component={component}, event_ts={event.get('ts')}, storytelling_extracts_components={list(storytelling_extracts.keys())}")
                    
                    # 🔧 FIX: Cerca in tutti i componenti, non solo quello trovato
                    matched_extract = None
                    matched_component = None
                    
                    # Prima cerca per span_id in tutti i componenti
                    if span_id:
                        # 🔧 [FIX_INNERTRACE] Log ricerca per span_id
                        logging.debug(f"[FIX_INNERTRACE] ricerca per span_id - span_id={span_id}")
                        for comp_name, node_calls in storytelling_extracts.items():
                            for extract_idx, extract in enumerate(node_calls):
                                extract_span_id = extract.get("span_id")
                                if extract_span_id == span_id:
                                    matched_extract = extract
                                    matched_component = comp_name
                                    # 🔧 [FIX_INNERTRACE] Log match trovato per span_id
                                    logging.debug(f"[FIX_INNERTRACE] match trovato per span_id - span_id={span_id}, component={comp_name}, extract_idx={extract_idx}, extract_has_prompt={bool(extract.get('prompt'))}, extract_has_response={bool(extract.get('response'))}")
                                    break
                            if matched_extract:
                                break
                        if not matched_extract:
                            # 🔧 [FIX_INNERTRACE] Log span_id non trovato
                            logging.debug(f"[FIX_INNERTRACE] span_id NON TROVATO - span_id={span_id}, cercato in {len(storytelling_extracts)} componenti")
                    else:
                        # 🔧 FIX: Se span_id è None, cerca per timestamp (entro 1 secondo)
                        event_ts = event.get("ts")
                        # 🔧 [FIX_INNERTRACE] Log ricerca per timestamp
                        logging.debug(f"[FIX_INNERTRACE] ricerca per timestamp - event_ts={event_ts}, span_id=None")
                        if event_ts:
                            closest_extract = None
                            closest_ts_diff = float('inf')
                            for comp_name, node_calls in storytelling_extracts.items():
                                for extract_idx, extract in enumerate(node_calls):
                                    extract_ts = extract.get("ts")
                                    if extract_ts:
                                        ts_diff = abs(extract_ts - event_ts)
                                        if ts_diff < closest_ts_diff and ts_diff < 1.0:  # Entro 1 secondo
                                            closest_ts_diff = ts_diff
                                            closest_extract = extract
                                            matched_component = comp_name
                            if closest_extract:
                                matched_extract = closest_extract
                                # 🔧 [FIX_INNERTRACE] Log match trovato per timestamp
                                logging.debug(f"[FIX_INNERTRACE] match trovato per timestamp - event_ts={event_ts}, closest_ts_diff={closest_ts_diff:.3f}s, component={matched_component}, extract_has_prompt={bool(closest_extract.get('prompt'))}, extract_has_response={bool(closest_extract.get('response'))}")
                            else:
                                # 🔧 [FIX_INNERTRACE] Log timestamp non trovato
                                logging.debug(f"[FIX_INNERTRACE] timestamp NON TROVATO - event_ts={event_ts}, cercato in {len(storytelling_extracts)} componenti")
                    
                    # Fallback: se component è disponibile, cerca lì
                    if not matched_extract and component:
                        # 🔧 [FIX_INNERTRACE] Log fallback per component
                        logging.debug(f"[FIX_INNERTRACE] fallback ricerca per component - component={component}, span_id={span_id}")
                        node_calls = storytelling_extracts.get(component, [])
                        if span_id:
                            # Cerca per span_id
                            for extract_idx, extract in enumerate(node_calls):
                                if extract.get("span_id") == span_id:
                                    matched_extract = extract
                                    matched_component = component
                                    # 🔧 [FIX_INNERTRACE] Log match trovato in fallback
                                    logging.debug(f"[FIX_INNERTRACE] match trovato in fallback span_id - component={component}, extract_idx={extract_idx}")
                                    break
                        
                        # Fallback: usa call_number se span_id non matcha (per retrocompatibilità)
                        if not matched_extract:
                            call_number = ""
                            if span_id:
                                call_number = get_llm_call_number(span_id)
                            idx = int(call_number.strip().replace("#", "") or "1") - 1
                            if 0 <= idx < len(node_calls):
                                matched_extract = node_calls[idx]
                                matched_component = component
                                # 🔧 [FIX_INNERTRACE] Log match trovato per call_number
                                logging.debug(f"[FIX_INNERTRACE] match trovato per call_number - component={component}, call_number={call_number}, idx={idx}, node_calls_len={len(node_calls)}")
                            else:
                                # 🔧 [FIX_INNERTRACE] Log call_number non valido
                                logging.debug(f"[FIX_INNERTRACE] call_number NON VALIDO - component={component}, call_number={call_number}, idx={idx}, node_calls_len={len(node_calls)}")
                    
                    if matched_extract and matched_component:
                        # 🎯 v0.3: Mostra sintesi separate di prompt e response + path ai file completi
                        node_calls = storytelling_extracts.get(matched_component, [])
                        extract_idx = node_calls.index(matched_extract) if matched_extract in node_calls else -1
                        
                        # 🔧 [FIX_INNERTRACE] Log matching
                        logging.debug(f"[FIX_INNERTRACE] llm.call.end matching - span_id={span_id}, component={component}, matched_component={matched_component}, extract_idx={extract_idx}, node_calls_len={len(node_calls)}")
                        
                        if extract_idx >= 0:
                            # Cerca sintesi separate e path ai file
                            lookup_key = (matched_component, extract_idx)
                            summaries_data = prompt_response_summaries.get(lookup_key)
                            
                            # 🔧 [FIX_INNERTRACE] Log lookup sintesi
                            logging.debug(f"[FIX_INNERTRACE] lookup sintesi - key={lookup_key}, found={summaries_data is not None}, total_keys_in_map={len(prompt_response_summaries)}")
                            
                            if summaries_data:
                                prompt_summary = summaries_data.get("prompt_summary", "")
                                response_summary = summaries_data.get("response_summary", "")
                                prompt_file = summaries_data.get("prompt_file", "")
                                response_file = summaries_data.get("response_file", "")
                                
                                # 🔧 [FIX_INNERTRACE] Log contenuto sintesi
                                prompt_summary_len = len(prompt_summary) if prompt_summary else 0
                                response_summary_len = len(response_summary) if response_summary else 0
                                prompt_summary_preview = prompt_summary[:50] if prompt_summary else "None"
                                response_summary_preview = response_summary[:50] if response_summary else "None"
                                logging.debug(f"[FIX_INNERTRACE] sintesi disponibili - prompt_len={prompt_summary_len}, response_len={response_summary_len}, prompt_preview='{prompt_summary_preview}...', response_preview='{response_summary_preview}...'")
                                
                                # Mostra sintesi prompt con path al file completo
                                if prompt_summary:
                                    # 🔧 FIX: Usa il formato corretto "PROMPT (synth):" invece di "PROMPT_INTENT:"
                                    details += f" | PROMPT (synth): {prompt_summary}"
                                    if prompt_file:
                                        details += f" [Full prompt: {prompt_file}]"
                                else:
                                    # 🔧 [FIX_INNERTRACE] Log se prompt_summary è vuoto
                                    logging.debug(f"[FIX_INNERTRACE] prompt_summary VUOTO - key={lookup_key}, summaries_data_keys={list(summaries_data.keys())}")
                                
                                # Mostra sintesi response con path al file completo
                                if response_summary:
                                    # 🔧 FIX: Usa il formato corretto "RESPONSE (synth):" invece di "RESPONSE_SUMMARY:"
                                    details += f" | RESPONSE (synth): {response_summary}"
                                    if response_file:
                                        details += f" [Full response: {response_file}]"
                                else:
                                    # 🔧 [FIX_INNERTRACE] Log se response_summary è vuoto
                                    logging.debug(f"[FIX_INNERTRACE] response_summary VUOTO - key={lookup_key}, summaries_data_keys={list(summaries_data.keys())}, response_summary_value='{response_summary}'")
                            else:
                                # 🔧 [FIX_INNERTRACE] Log se summaries_data non trovato
                                available_keys = list(prompt_response_summaries.keys())[:10]  # Primi 10 per non intasare
                                logging.debug(f"[FIX_INNERTRACE] summaries_data NON TROVATO - key={lookup_key}, available_keys_sample={available_keys}, matched_component={matched_component}, extract_idx={extract_idx}")
                                # 🔧 FALLBACK: Se sintesi non disponibili, mostra snippet originali
                                prompt_snippet = matched_extract.get("prompt", "")
                                resp_snippet = matched_extract.get("response", "")
                                if prompt_snippet:
                                    prompt_preview = prompt_snippet[:200].strip().replace("\n", " ")
                                    details += f" | PROMPT: {prompt_preview}..."
                                if resp_snippet:
                                    resp_preview = resp_snippet[:150].strip().replace("\n", " ")
                                    details += f" | RESPONSE: {resp_preview}..."
                        else:
                            # 🔧 [FIX_INNERTRACE] Log se extract_idx non valido
                            logging.debug(f"[FIX_INNERTRACE] extract_idx NON VALIDO - extract_idx={extract_idx}, matched_extract_in_node_calls={matched_extract in node_calls if matched_extract else False}, node_calls_len={len(node_calls)}")
                            # 🔧 FALLBACK: Se idx non trovato, mostra prompt e response originali
                            prompt_snippet = matched_extract.get("prompt", "")
                            resp_snippet = matched_extract.get("response", "")
                            if prompt_snippet:
                                prompt_preview = prompt_snippet[:200].strip().replace("\n", " ")
                                details += f" | PROMPT: {prompt_preview}..."
                            if resp_snippet:
                                resp_preview = resp_snippet[:150].strip().replace("\n", " ")
                                details += f" | RESPONSE: {resp_preview}..."
        elif event_type == "tool.call.start":
            tool = payload.get('tool', 'unknown')
            details = f"({tool})"
        elif event_type == "tool.call.end":
            tool = payload.get('tool', 'unknown')
            status = payload.get('status', 'ok')
            latency = payload.get('latency_ms', 0)
            details = f"({tool}, {status}, {latency}ms)"
        elif event_type == "exception":
            exc_type = payload.get('exc_type', 'Exception')
            message = payload.get('message', '')[:50]
            details = f"({exc_type}: {message})"
        elif event_type == "router.decision":
            chosen = payload.get('chosen', 'unknown')
            details = f"(→ {chosen})"

        # Format line
        line = f"[{ts_human}] {indent}{event_type:25s} {details}{dt_str}"
        lines.append(line)

    # Compact mode: lossy projection for quick overview
    # Shows only LLM calls with context (useful for cost/performance analysis)
    # NOT suitable for root-cause analysis (use full view for that)
    if compact:
        compact_lines = []
        for line in lines:
            # Include run boundaries and LLM calls only
            # Exclude function-level spans (use full view to see function traces)
            if ("run.start" in line or "run.end" in line or "llm.call" in line) and "[function]" not in line:
                compact_lines.append(line)
        return compact_lines

    # Default: return complete canonical view (source of truth)
    return lines
