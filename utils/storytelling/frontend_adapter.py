"""
Frontend Adapter - Maps domain model to legacy frontend JSON format

v0.1: Handles UI format concerns separately from domain logic.
Takes StoryProjection and produces EXACT format that frontend expects.
"""
from typing import Dict, Any
from dataclasses import asdict

from .story_projection import StoryProjection


class FrontendAdapter:
    """Adapts storytelling domain model to legacy frontend JSON format.

    v0.1: This layer handles ALL UI-specific concerns:
    - Exact key names ("phases", "quality_metrics", etc.)
    - Nested structure expected by frontend
    - Data type conversions for UI display
    """

    @staticmethod
    def to_frontend_json(projection: StoryProjection) -> Dict[str, Any]:
        """Convert StoryProjection to frontend-compatible JSON.

        Args:
            projection: Domain model from story_projection

        Returns:
            Dict with exact structure that frontend expects
        """
        # Build phases dict in frontend format
        phases_data = {}
        for phase_name, phase_model in projection.phases.items():
            # Convert LLMCallSemantics to dict
            llm_calls_list = []
            for llm_call in phase_model.llm_calls:
                llm_calls_list.append({
                    "timestamp": llm_call.timestamp,
                    "phase": llm_call.phase,
                    "node": llm_call.node,
                    "prompt": llm_call.prompt,
                    "response": llm_call.response,
                    "model_id": llm_call.model_id,
                    "success": llm_call.success,
                    "error": llm_call.error,
                    "context": llm_call.context
                })

            # Convert generic events
            events_list = []
            for generic_event in phase_model.generic_events:
                events_list.append({
                    "timestamp": generic_event["timestamp"],
                    "type": generic_event["type"].replace("story.", ""),
                    "data": generic_event["payload"]
                })

            # Convert phase quality if available
            phase_quality_dict = None
            if phase_model.phase_quality:
                phase_quality_dict = asdict(phase_model.phase_quality)

            phases_data[phase_name] = {
                "summary": "",  # Frontend expects this key (can be empty)
                "llm_calls": llm_calls_list,
                "events": events_list,
                "phase_quality": phase_quality_dict
            }

        # Build quality_metrics in frontend format
        task_metrics_list = [asdict(tm) for tm in projection.task_quality_metrics]
        phase_metrics_list = [asdict(pm) for pm in projection.phase_quality_metrics]

        # Calculate aggregate metrics (frontend expects these)
        avg_task_quality = (
            sum(tm.quality_score for tm in projection.task_quality_metrics) / len(projection.task_quality_metrics)
            if projection.task_quality_metrics else 0.0
        )

        # Calculate global difficulty score
        global_difficulty_score = 0.0
        if projection.task_quality_metrics:
            total_tasks = len(projection.task_quality_metrics)
            total_retries = sum(tm.retry_count for tm in projection.task_quality_metrics)
            low_quality_tasks = sum(1 for tm in projection.task_quality_metrics if tm.quality_score < 0.6)
            high_difficulty_tasks = sum(1 for tm in projection.task_quality_metrics if tm.execution_difficulty in ["high", "critical"])

            retry_factor = min(total_retries / total_tasks, 1.0) * 0.4
            quality_factor = (low_quality_tasks / total_tasks) * 0.3
            difficulty_factor = (high_difficulty_tasks / total_tasks) * 0.3
            global_difficulty_score = min(retry_factor + quality_factor + difficulty_factor, 1.0)

        quality_metrics = {
            "task_metrics": task_metrics_list,
            "phase_metrics": phase_metrics_list,
            "global_difficulty_score": global_difficulty_score,
            "total_tasks_completed": len(projection.task_quality_metrics),
            "avg_task_quality": avg_task_quality
        }

        # Return EXACT frontend format
        return {
            "conversation_id": projection.conversation_id,
            "session_start": projection.session_start,
            "session_end": projection.session_end,
            "total_llm_calls": projection.total_llm_calls,
            "total_events": projection.total_events,
            "phases": phases_data,
            "quality_metrics": quality_metrics
        }
