# Storytelling intelligente per il planner gerarchico
"""
Storytelling module for LLM orchestration semantic layer.

v0.1: Integrates with tracer for unified event storage.
"""

from .storytelling_manager import (
    StorytellingManager,
    init_storytelling,
    get_storytelling_manager,
    log_llm_call,
    log_storytelling_event,
    log_task_quality,
    log_phase_quality,
    calculate_global_difficulty_score,
)

from .storytelling_extractor import (
    StorytellingExtractor,
    get_storytelling_for_llm,
    get_available_conversations,
)

from .story_projection import (
    StoryProjector,
    StoryProjection,
    LLMCallSemantics,
    PhaseSemantics,
    TaskQualitySemantics,
    PhaseQualitySemantics,
)

from .frontend_adapter import FrontendAdapter

__all__ = [
    # Manager
    "StorytellingManager",
    "init_storytelling",
    "get_storytelling_manager",
    "log_llm_call",
    "log_storytelling_event",
    "log_task_quality",
    "log_phase_quality",
    "calculate_global_difficulty_score",
    # Extractor
    "StorytellingExtractor",
    "get_storytelling_for_llm",
    "get_available_conversations",
    # Projection (domain layer)
    "StoryProjector",
    "StoryProjection",
    "LLMCallSemantics",
    "PhaseSemantics",
    "TaskQualitySemantics",
    "PhaseQualitySemantics",
    # Adapter (UI layer)
    "FrontendAdapter",
]
