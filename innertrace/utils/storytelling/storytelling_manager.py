"""
Storytelling Manager - Sistema intelligente per tracciare e sintetizzare il progresso del planner

v0.1: Integrato con tracer per emettere eventi su events.jsonl oltre ai file JSON legacy
"""
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, asdict
from pathlib import Path

if TYPE_CHECKING:
    from tracing import Tracer

logger = logging.getLogger(__name__)

@dataclass
class LLMCall:
    """Rappresenta una singola chiamata LLM"""
    timestamp: str
    phase: str  # meta_planning, project_planning, execution, synthesis
    node: str   # meta_planner, project_planner, worker_executor, synthesis
    prompt: str
    response: str
    model_id: str
    success: bool
    error: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

@dataclass
class StorytellingEvent:
    """Rappresenta un evento del storytelling"""
    timestamp: str
    event_type: str
    phase: str
    data: Dict[str, Any]
    conversation_id: str

@dataclass
class TaskQualityMetrics:
    """Metriche qualitative compatte per un task"""
    timestamp: str
    task_id: str
    quality_score: float  # 0.0-1.0
    result_type: str  # "data_extraction|analysis|synthesis|search|other"
    key_insights: List[str]  # Max 3 insights principali
    success_criteria_met: List[str]  # Criteri di successo soddisfatti
    execution_difficulty: str  # "low|medium|high|critical"
    retry_count: int = 0

@dataclass
class PhaseQualityMetrics:
    """Metriche qualitative aggregate per fase"""
    timestamp: str
    phase: str
    total_tasks: int
    success_rate: float
    avg_quality_score: float
    critical_failures: int
    bottlenecks: List[str]
    adaptations_count: int

class StorytellingManager:
    """Manager per il storytelling intelligente

    v0.1: Supporta emissione eventi via tracer oltre a file JSON legacy.
    """

    def __init__(self, conversation_id: str, tracer: Optional['Tracer'] = None):
        """
        Initialize storytelling manager.

        Args:
            conversation_id: Unique conversation identifier
            tracer: Optional Tracer instance for event emission (v0.1+)
                   If provided, events will be emitted to events.jsonl
                   Legacy file writes are maintained for backward compatibility
        """
        self.conversation_id = conversation_id
        self.tracer = tracer
        self.llm_calls: List[LLMCall] = []
        self.events: List[StorytellingEvent] = []
        self.task_metrics: List[TaskQualityMetrics] = []
        self.phase_metrics: List[PhaseQualityMetrics] = []
        self.session_start = datetime.now().isoformat()

        # Crea directory per il session (legacy file storage)
        self.session_dir = Path(f"logs/storytelling/{conversation_id}")
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-load blob_store if tracer available
        self._blob_store = None
        if self.tracer:
            self._blob_store = self.tracer.blob_store
    
    def log_llm_call(self, phase: str, node: str, prompt: str, response: str,
                    model_id: str, success: bool = True, error: str = None,
                    context: Dict[str, Any] = None, span_id: Optional[str] = None):
        """Logga una chiamata LLM

        v0.1: Emette eventi via tracer (se disponibile) + file legacy

        Args:
            phase: Storytelling phase (meta_planning, project_planning, etc.)
            node: Node that made the call (meta_planner, worker_executor, etc.)
            prompt: LLM prompt
            response: LLM response
            model_id: Model identifier
            success: Whether the call succeeded
            error: Optional error message
            context: Optional context dictionary
            span_id: Optional span_id of the llm.call event (v0.1+)
                    If provided, will emit story.link instead of duplicating content
        """
        call = LLMCall(
            timestamp=datetime.now().isoformat(),
            phase=phase,
            node=node,
            prompt=prompt,
            response=response,
            model_id=model_id,
            success=success,
            error=error,
            context=context
        )

        self.llm_calls.append(call)

        # NEW v0.1: Emit event to tracer if available
        if self.tracer:
            try:
                # v0.1 CORRECT: Use story.link to reference llm.call span (no content duplication)
                if span_id:
                    # Filter context for serializability
                    filtered_context = None
                    if context:
                        filtered_context = {}
                        for k, v in context.items():
                            try:
                                json.dumps(v)
                                filtered_context[k] = v
                            except (TypeError, ValueError):
                                filtered_context[k] = f"<non-serializable: {type(v).__name__}>"

                    self.tracer.emit(
                        type="story.link",
                        actor="story.manager",
                        level="info" if success else "error",
                        tags=["storytelling", "llm_call", phase, node],
                        payload={
                            "conversation_id": self.conversation_id,
                            "kind": "llm",
                            "target_span_id": span_id,
                            "phase": phase,
                            "node": node,
                            "model_id": model_id,
                            "success": success,
                            "error": error,
                            "context": filtered_context
                        }
                    )
                else:
                    # FALLBACK (backward compat): If no span_id provided, log warning
                    # In production, all calls should provide span_id
                    logger.warning(
                        f"⚠️ STORYTELLING: log_llm_call called without span_id (phase={phase}, node={node}). "
                        "For v0.1+, please emit llm.call events via tracer and pass span_id to log_llm_call."
                    )
            except Exception as e:
                logger.warning(f"⚠️ STORYTELLING: Could not emit tracer event: {e}")

        # LEGACY: Write to file for backward compatibility
        try:
            call_dict = asdict(call)
            if call_dict.get('context'):
                filtered_context = {}
                for k, v in call_dict['context'].items():
                    try:
                        json.dumps(v)
                        filtered_context[k] = v
                    except (TypeError, ValueError):
                        filtered_context[k] = f"<non-serializable: {type(v).__name__}>"
                call_dict['context'] = filtered_context

            call_file = self.session_dir / f"llm_call_{len(self.llm_calls):03d}.json"
            with open(call_file, 'w', encoding='utf-8') as f:
                json.dump(call_dict, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"⚠️ STORYTELLING: Could not serialize LLM call to file: {e}")

        logger.info(f"🎭 STORYTELLING: Logged LLM call {phase}/{node} - Success: {success}")

    def log_task_quality(self, task_id: str, quality_score: float, result_type: str,
                        key_insights: List[str], success_criteria_met: List[str],
                        execution_difficulty: str = "low", retry_count: int = 0):
        """Logga metriche qualitative per un task completato

        v0.1: Emette eventi via tracer (se disponibile) + file legacy
        """
        # Limita insights a max 3 per mantenere compattezza
        limited_insights = key_insights[:3] if key_insights else []

        task_metric = TaskQualityMetrics(
            timestamp=datetime.now().isoformat(),
            task_id=task_id,
            quality_score=quality_score,
            result_type=result_type,
            key_insights=limited_insights,
            success_criteria_met=success_criteria_met[:5],  # Max 5 criteri
            execution_difficulty=execution_difficulty,
            retry_count=retry_count
        )

        self.task_metrics.append(task_metric)

        # NEW v0.1: Emit event to tracer if available
        if self.tracer:
            try:
                self.tracer.emit(
                    type="story.quality.task",
                    actor="story.manager",
                    level="info",
                    tags=["storytelling", "quality", result_type],
                    payload={
                        "conversation_id": self.conversation_id,
                        "task_id": task_id,
                        "quality_score": quality_score,
                        "result_type": result_type,
                        "key_insights": limited_insights,
                        "success_criteria_met": success_criteria_met[:5],
                        "execution_difficulty": execution_difficulty,
                        "retry_count": retry_count
                    }
                )
            except Exception as e:
                logger.warning(f"⚠️ STORYTELLING: Could not emit task quality event: {e}")

        # LEGACY: Write to file
        task_file = self.session_dir / f"task_quality_{task_id}.json"
        with open(task_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(task_metric), f, indent=2, ensure_ascii=False)

        logger.info(f"📊 STORYTELLING: Logged task quality {task_id} - Score: {quality_score:.2f}, Type: {result_type}")

    def log_phase_quality(self, phase: str, total_tasks: int, success_rate: float,
                         avg_quality_score: float, critical_failures: int = 0,
                         bottlenecks: List[str] = None, adaptations_count: int = 0):
        """Logga metriche qualitative aggregate per una fase

        v0.1: Emette eventi via tracer (se disponibile) + file legacy
        """
        phase_metric = PhaseQualityMetrics(
            timestamp=datetime.now().isoformat(),
            phase=phase,
            total_tasks=total_tasks,
            success_rate=success_rate,
            avg_quality_score=avg_quality_score,
            critical_failures=critical_failures,
            bottlenecks=bottlenecks[:3] if bottlenecks else [],  # Max 3 bottleneck
            adaptations_count=adaptations_count
        )

        self.phase_metrics.append(phase_metric)

        # NEW v0.1: Emit event to tracer if available
        if self.tracer:
            try:
                self.tracer.emit(
                    type="story.quality.phase",
                    actor="story.manager",
                    level="info",
                    tags=["storytelling", "quality", phase],
                    payload={
                        "conversation_id": self.conversation_id,
                        "phase": phase,
                        "total_tasks": total_tasks,
                        "success_rate": success_rate,
                        "avg_quality_score": avg_quality_score,
                        "critical_failures": critical_failures,
                        "bottlenecks": bottlenecks[:3] if bottlenecks else [],
                        "adaptations_count": adaptations_count
                    }
                )
            except Exception as e:
                logger.warning(f"⚠️ STORYTELLING: Could not emit phase quality event: {e}")

        # LEGACY: Write to file
        phase_file = self.session_dir / f"phase_quality_{phase}.json"
        with open(phase_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(phase_metric), f, indent=2, ensure_ascii=False)

        logger.info(f"📊 STORYTELLING: Logged phase quality {phase} - Success: {success_rate:.2%}, Avg Quality: {avg_quality_score:.2f}")

    def calculate_global_difficulty_score(self) -> float:
        """Calcola score di difficoltà globale dell'esecuzione (0.0-1.0)"""
        if not self.task_metrics:
            return 0.0

        # Fattori: retry count, critical failures, quality score bassi
        total_tasks = len(self.task_metrics)
        total_retries = sum(t.retry_count for t in self.task_metrics)
        low_quality_tasks = sum(1 for t in self.task_metrics if t.quality_score < 0.6)
        high_difficulty_tasks = sum(1 for t in self.task_metrics if t.execution_difficulty in ["high", "critical"])

        # Calcolo pesato
        retry_factor = min(total_retries / total_tasks, 1.0) * 0.4
        quality_factor = (low_quality_tasks / total_tasks) * 0.3
        difficulty_factor = (high_difficulty_tasks / total_tasks) * 0.3

        return min(retry_factor + quality_factor + difficulty_factor, 1.0)

    def log_event(self, event_type: str, phase: str, data: Dict[str, Any]):
        """Logga un evento del storytelling

        v0.1: Emette eventi via tracer (se disponibile) + file legacy
        """
        event = StorytellingEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            phase=phase,
            data=data,
            conversation_id=self.conversation_id
        )

        self.events.append(event)

        # NEW v0.1: Emit event to tracer if available
        if self.tracer:
            try:
                # Map event_type to story.* types
                story_type = f"story.{event_type}" if not event_type.startswith("story.") else event_type

                self.tracer.emit(
                    type=story_type,
                    actor="story.manager",
                    level="info",
                    tags=["storytelling", phase, event_type],
                    payload={
                        "conversation_id": self.conversation_id,
                        "phase": phase,
                        "event_type": event_type,
                        **data  # Merge data into payload
                    }
                )
            except Exception as e:
                logger.warning(f"⚠️ STORYTELLING: Could not emit storytelling event: {e}")

        # LEGACY: Write to file
        event_file = self.session_dir / f"event_{len(self.events):03d}.json"
        with open(event_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(event), f, indent=2, ensure_ascii=False)

        logger.info(f"🎭 STORYTELLING: Logged event {event_type} in {phase}")
    
    def generate_phase_summary(self, phase: str) -> str:
        """Genera un riassunto intelligente per una fase"""
        phase_calls = [call for call in self.llm_calls if call.phase == phase]
        phase_events = [event for event in self.events if event.phase == phase]
        
        if not phase_calls:
            return f"Fase {phase}: Nessuna attività registrata"
        
        summary_parts = [f"**Fase {phase.upper()}:**"]
        
        # Conta chiamate per nodo
        node_counts = {}
        for call in phase_calls:
            node_counts[call.node] = node_counts.get(call.node, 0) + 1
        
        for node, count in node_counts.items():
            success_count = sum(1 for call in phase_calls if call.node == node and call.success)
            summary_parts.append(f"- {node}: {success_count}/{count} chiamate riuscite")
        
        # Aggiungi eventi significativi
        for event in phase_events:
            if event.event_type in ["planning_complete", "execution_complete", "synthesis_complete"]:
                summary_parts.append(f"- ✅ {event.event_type}: {event.data}")
        
        return "\n".join(summary_parts)
    
    def generate_full_storytelling(self) -> Dict[str, Any]:
        """Genera il storytelling completo con metriche qualitative"""
        phases = ["meta_planning", "project_planning", "execution", "synthesis"]

        storytelling = {
            "conversation_id": self.conversation_id,
            "session_start": self.session_start,
            "session_end": datetime.now().isoformat(),
            "total_llm_calls": len(self.llm_calls),
            "total_events": len(self.events),
            "phases": {},
            # 🆕 Nuove metriche qualitative
            "quality_metrics": {
                "task_metrics": [asdict(tm) for tm in self.task_metrics],
                "phase_metrics": [asdict(pm) for pm in self.phase_metrics],
                "global_difficulty_score": self.calculate_global_difficulty_score(),
                "total_tasks_completed": len(self.task_metrics),
                "avg_task_quality": sum(tm.quality_score for tm in self.task_metrics) / len(self.task_metrics) if self.task_metrics else 0.0
            }
        }

        for phase in phases:
            storytelling["phases"][phase] = {
                "summary": self.generate_phase_summary(phase),
                "llm_calls": [asdict(call) for call in self.llm_calls if call.phase == phase],
                "events": [asdict(event) for event in self.events if event.phase == phase],
                # 🆕 Metriche qualitative per fase
                "phase_quality": next((asdict(pm) for pm in self.phase_metrics if pm.phase == phase), None)
            }

        # Salva storytelling completo
        storytelling_file = self.session_dir / "full_storytelling.json"
        with open(storytelling_file, 'w', encoding='utf-8') as f:
            json.dump(storytelling, f, indent=2, ensure_ascii=False)

        return storytelling
    
    def get_phase_prompts_responses(self, phase: str) -> List[Dict[str, str]]:
        """Ottiene tutti i prompt/response per una fase"""
        phase_calls = [call for call in self.llm_calls if call.phase == phase]
        
        return [
            {
                "timestamp": call.timestamp,
                "node": call.node,
                "prompt": call.prompt,
                "response": call.response,
                "model_id": call.model_id,
                "success": call.success,
                "error": call.error
            }
            for call in phase_calls
        ]

# Istanza globale per la sessione corrente
_current_storytelling_manager: Optional[StorytellingManager] = None

def init_storytelling(conversation_id: str, tracer: Optional['Tracer'] = None) -> StorytellingManager:
    """Inizializza il storytelling manager per una conversazione

    Args:
        conversation_id: Unique conversation identifier
        tracer: Optional Tracer instance for event emission (v0.1+)

    v0.1: Now supports optional tracer for event emission
    """
    global _current_storytelling_manager
    _current_storytelling_manager = StorytellingManager(conversation_id, tracer=tracer)
    return _current_storytelling_manager

def get_storytelling_manager() -> Optional[StorytellingManager]:
    """Ottiene il storytelling manager corrente"""
    return _current_storytelling_manager

def log_llm_call(phase: str, node: str, prompt: str, response: str,
                model_id: str, success: bool = True, error: str = None,
                context: Dict[str, Any] = None, span_id: Optional[str] = None):
    """Logga una chiamata LLM nel manager corrente

    Args:
        span_id: Optional span_id of the llm.call event (v0.1+)
                If provided, will emit story.link instead of duplicating content
    """
    if _current_storytelling_manager:
        _current_storytelling_manager.log_llm_call(
            phase, node, prompt, response, model_id, success, error, context, span_id
        )

def log_storytelling_event(event_type: str, phase: str, data: Dict[str, Any]):
    """Logga un evento nel manager corrente"""
    if _current_storytelling_manager:
        _current_storytelling_manager.log_event(event_type, phase, data)

# 🆕 Funzioni globali per metriche qualitative
def log_task_quality(task_id: str, quality_score: float, result_type: str,
                    key_insights: List[str], success_criteria_met: List[str],
                    execution_difficulty: str = "low", retry_count: int = 0):
    """Logga metriche qualitative per un task completato"""
    if _current_storytelling_manager:
        _current_storytelling_manager.log_task_quality(
            task_id, quality_score, result_type, key_insights,
            success_criteria_met, execution_difficulty, retry_count
        )

def log_phase_quality(phase: str, total_tasks: int, success_rate: float,
                     avg_quality_score: float, critical_failures: int = 0,
                     bottlenecks: List[str] = None, adaptations_count: int = 0):
    """Logga metriche qualitative aggregate per una fase"""
    if _current_storytelling_manager:
        _current_storytelling_manager.log_phase_quality(
            phase, total_tasks, success_rate, avg_quality_score,
            critical_failures, bottlenecks, adaptations_count
        )

def calculate_global_difficulty_score() -> float:
    """Calcola score di difficoltà globale dell'esecuzione"""
    if _current_storytelling_manager:
        return _current_storytelling_manager.calculate_global_difficulty_score()
    return 0.0
