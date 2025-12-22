"""
Story Projection - Domain model for storytelling semantic layer

v0.1: Neutral domain representation (NOT coupled to frontend format)
Reconstructs storytelling semantics from trace events without UI concerns.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LLMCallSemantics:
    """Domain model for an LLM call in storytelling context."""
    timestamp: str
    phase: str
    node: str
    prompt: str
    response: str
    model_id: str
    success: bool
    error: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class TaskQualitySemantics:
    """Domain model for task quality metrics."""
    timestamp: str
    task_id: str
    quality_score: float
    result_type: str
    key_insights: List[str]
    success_criteria_met: List[str]
    execution_difficulty: str
    retry_count: int


@dataclass
class PhaseQualitySemantics:
    """Domain model for phase quality metrics."""
    timestamp: str
    phase: str
    total_tasks: int
    success_rate: float
    avg_quality_score: float
    critical_failures: int
    bottlenecks: List[str]
    adaptations_count: int


@dataclass
class PhaseSemantics:
    """Domain model for a storytelling phase."""
    name: str
    llm_calls: List[LLMCallSemantics] = field(default_factory=list)
    generic_events: List[Dict[str, Any]] = field(default_factory=list)
    phase_quality: Optional[PhaseQualitySemantics] = None


@dataclass
class StoryProjection:
    """Neutral domain model for storytelling projection.

    This represents the semantic content WITHOUT UI formatting concerns.
    """
    conversation_id: str
    session_start: str
    session_end: str
    phases: Dict[str, PhaseSemantics]
    task_quality_metrics: List[TaskQualitySemantics]
    phase_quality_metrics: List[PhaseQualitySemantics]
    total_llm_calls: int
    total_events: int


class StoryProjector:
    """Projects trace events into neutral storytelling domain model.

    v0.1: Implements the PROJECTION LAYER (events.jsonl → domain semantics)
    NO UI concerns, NO frontend format dependencies.
    """

    def __init__(self, events_path: str = "traces/events.jsonl",
                 blobs_path: str = "traces/blobs"):
        """
        Initialize story projector.

        Args:
            events_path: Path to tracer events.jsonl
            blobs_path: Path to blob storage
        """
        self.events_path = Path(events_path)
        self.blobs_path = Path(blobs_path)

    def _load_events_jsonl(self) -> List[Dict[str, Any]]:
        """Load events from JSONL file."""
        import json
        events = []
        if not self.events_path.exists():
            return events

        try:
            with open(self.events_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Error loading events.jsonl: {e}")

        return events

    def _extract_conversation_id(self, event: Dict[str, Any]) -> Optional[str]:
        """Extract conversation_id from event."""
        # Check payload
        conv_id = event.get("payload", {}).get("conversation_id")
        if conv_id:
            return conv_id

        # Check run.start args
        if event.get("type") == "run.start":
            return event.get("payload", {}).get("args", {}).get("conversation_id")

        return None

    def _get_blob_content(self, blob_ref: Optional[str]) -> str:
        """Retrieve blob content from reference."""
        if not blob_ref or not blob_ref.startswith("blob:sha256:"):
            return ""

        try:
            hash_val = blob_ref.replace("blob:sha256:", "")
            # Try .txt first, then .json
            for ext in ["txt", "json"]:
                blob_file = self.blobs_path / "sha256" / f"{hash_val}.{ext}"
                if blob_file.exists():
                    with open(blob_file, 'r', encoding='utf-8') as f:
                        return f.read()
        except Exception as e:
            logger.warning(f"Could not load blob {blob_ref}: {e}")

        return ""

    def project(self, conversation_id: str) -> Optional[StoryProjection]:
        """Project trace events into storytelling domain model.

        Args:
            conversation_id: Conversation to reconstruct

        Returns:
            StoryProjection or None if no events found
        """
        # Load all events
        all_events = self._load_events_jsonl()
        if not all_events:
            return None

        # Filter by conversation_id
        conv_events = [
            e for e in all_events
            if self._extract_conversation_id(e) == conversation_id
        ]

        if not conv_events:
            return None

        # Collect story.* events
        story_events = [e for e in conv_events if e.get("type", "").startswith("story.")]

        if not story_events:
            return None

        # v0.1: Also collect llm.call events for story.link resolution
        llm_call_events = [e for e in conv_events if e.get("type") in ["llm.call.start", "llm.call.end"]]

        # Build span_id index for fast lookup (need both start and end)
        llm_events_by_span = {}
        for event in llm_call_events:
            span_id = event.get("span_id")
            if span_id:
                if span_id not in llm_events_by_span:
                    llm_events_by_span[span_id] = {}

                event_type = event.get("type")
                if event_type == "llm.call.start":
                    llm_events_by_span[span_id]["start"] = event
                elif event_type == "llm.call.end":
                    llm_events_by_span[span_id]["end"] = event

        # Initialize phase domain models
        phases = {
            "meta_planning": PhaseSemantics(name="meta_planning"),
            "project_planning": PhaseSemantics(name="project_planning"),
            "execution": PhaseSemantics(name="execution"),
            "synthesis": PhaseSemantics(name="synthesis"),
        }

        task_quality_metrics = []
        phase_quality_metrics = []

        # Process story events
        for event in story_events:
            event_type = event.get("type")
            payload = event.get("payload", {})
            phase = payload.get("phase", "unknown")

            # v0.1: story.link (linking to llm.call events)
            if event_type == "story.link" and payload.get("kind") == "llm":
                if phase in phases:
                    target_span_id = payload.get("target_span_id")
                    if target_span_id and target_span_id in llm_events_by_span:
                        llm_events = llm_events_by_span[target_span_id]
                        llm_start = llm_events.get("start", {})
                        llm_end = llm_events.get("end", {})

                        llm_start_payload = llm_start.get("payload", {})
                        llm_end_payload = llm_end.get("payload", {})

                        # Extract prompt from start, response from end
                        prompt = self._get_blob_content(llm_start_payload.get("prompt_ref")) or llm_start_payload.get("prompt", "")
                        response = self._get_blob_content(llm_end_payload.get("response_ref")) or llm_end_payload.get("response", "")

                        llm_call = LLMCallSemantics(
                            timestamp=event.get("ts_iso", ""),
                            phase=phase,
                            node=payload.get("node", ""),
                            prompt=prompt,
                            response=response,
                            model_id=payload.get("model_id") or llm_start_payload.get("model_id", ""),
                            success=payload.get("success", True),
                            error=payload.get("error"),
                            context=payload.get("context")
                        )
                        phases[phase].llm_calls.append(llm_call)
                    else:
                        logger.warning(f"story.link references unknown span_id: {target_span_id}")

            # story.quality.task (EXPERIMENTAL)
            elif event_type == "story.quality.task":
                task_quality = TaskQualitySemantics(
                    timestamp=event.get("ts_iso", ""),
                    task_id=payload.get("task_id", ""),
                    quality_score=payload.get("quality_score", 0.0),
                    result_type=payload.get("result_type", "unknown"),
                    key_insights=payload.get("key_insights", []),
                    success_criteria_met=payload.get("success_criteria_met", []),
                    execution_difficulty=payload.get("execution_difficulty", "low"),
                    retry_count=payload.get("retry_count", 0)
                )
                task_quality_metrics.append(task_quality)

            # story.quality.phase (EXPERIMENTAL)
            elif event_type == "story.quality.phase":
                phase_quality = PhaseQualitySemantics(
                    timestamp=event.get("ts_iso", ""),
                    phase=phase,
                    total_tasks=payload.get("total_tasks", 0),
                    success_rate=payload.get("success_rate", 0.0),
                    avg_quality_score=payload.get("avg_quality_score", 0.0),
                    critical_failures=payload.get("critical_failures", 0),
                    bottlenecks=payload.get("bottlenecks", []),
                    adaptations_count=payload.get("adaptations_count", 0)
                )
                phase_quality_metrics.append(phase_quality)

                if phase in phases:
                    phases[phase].phase_quality = phase_quality

            # Generic story events (story.phase, story.objective, story.task, etc.)
            else:
                if phase in phases:
                    phases[phase].generic_events.append({
                        "timestamp": event.get("ts_iso", ""),
                        "type": event_type,
                        "payload": payload
                    })

        # Calculate totals
        total_llm_calls = sum(len(p.llm_calls) for p in phases.values())

        return StoryProjection(
            conversation_id=conversation_id,
            session_start=story_events[0].get("ts_iso", "") if story_events else "",
            session_end=story_events[-1].get("ts_iso", "") if story_events else "",
            phases=phases,
            task_quality_metrics=task_quality_metrics,
            phase_quality_metrics=phase_quality_metrics,
            total_llm_calls=total_llm_calls,
            total_events=len(story_events)
        )
