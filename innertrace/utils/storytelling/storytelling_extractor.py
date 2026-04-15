"""
Storytelling Extractor - Sistema per estrarre e presentare lo storytelling intelligente

v0.1: Supporta lettura da events.jsonl (tracer) oltre ai file JSON legacy
v0.1: Uses story_projection (domain) + frontend_adapter (UI) separation
"""
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from .story_projection import StoryProjector
from .frontend_adapter import FrontendAdapter

logger = logging.getLogger(__name__)


def load_events_jsonl(events_path: Path) -> List[Dict[str, Any]]:
    """Load events from JSONL file."""
    events = []
    if not events_path.exists():
        return events

    try:
        with open(events_path, 'r', encoding='utf-8') as f:
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

@dataclass
class StorytellingSummary:
    """Riassunto dello storytelling per una conversazione"""
    conversation_id: str
    created_at: str
    total_events: int
    strategic_objectives: List[Dict[str, Any]]
    tactical_plans: List[Dict[str, Any]]
    completed_tasks: List[Dict[str, Any]]
    llm_calls: List[Dict[str, Any]]
    reflections: List[Dict[str, Any]]
    final_synthesis: Optional[Dict[str, Any]]
    duration_minutes: float
    statistics: Dict[str, int]
    # 🆕 Nuovi campi qualitativi
    task_quality_metrics: List[Dict[str, Any]]
    phase_quality_metrics: List[Dict[str, Any]]
    global_difficulty_score: float
    avg_task_quality: float
    execution_assessment: str

class StorytellingExtractor:
    """Estrattore di storytelling intelligente

    v0.1: Supporta lettura da events.jsonl (tracer) oltre ai file JSON legacy
    """

    def __init__(self, storytelling_dir: str = "logs/storytelling",
                 events_path: str = "traces/events.jsonl",
                 blobs_path: str = "traces/blobs"):
        """
        Initialize storytelling extractor.

        Args:
            storytelling_dir: Legacy storytelling directory
            events_path: Path to tracer events.jsonl (v0.1+)
            blobs_path: Path to blob storage (v0.1+)
        """
        self.storytelling_dir = Path(storytelling_dir)
        self.storytelling_dir.mkdir(parents=True, exist_ok=True)

        # v0.1: Support for events.jsonl reading
        self.events_path = Path(events_path)
        self.blobs_path = Path(blobs_path)

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

    def _reconstruct_from_events(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Reconstruct storytelling JSON from events.jsonl.

        v0.1: Two-layer architecture:
        1. StoryProjector: events.jsonl → neutral domain model (NO UI concerns)
        2. FrontendAdapter: domain model → exact frontend JSON format
        """
        # Layer 1: Project events into domain model
        projector = StoryProjector(
            events_path=str(self.events_path),
            blobs_path=str(self.blobs_path)
        )
        projection = projector.project(conversation_id)

        if not projection:
            return None

        # Layer 2: Adapt domain model to frontend format
        frontend_json = FrontendAdapter.to_frontend_json(projection)

        return frontend_json

    def _normalize_story_event(self, event: Dict[str, Any], default_phase: Optional[str] = None) -> Dict[str, Any]:
        """Normalize legacy and projected events into a common flat structure."""
        payload = event.get("data")
        if payload is None:
            payload = event.get("payload", {})
        if not isinstance(payload, dict):
            payload = {"value": payload}

        phase = event.get("phase") or payload.get("phase") or default_phase
        return {
            "timestamp": event.get("timestamp") or event.get("ts_iso") or "",
            "type": event.get("type") or event.get("event_type") or "unknown",
            "data": payload,
            "phase": phase,
        }

    def _flatten_story_events(self, data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Flatten either legacy top-level events or phase-based projection events."""
        if not data:
            return []

        flat_events: List[Dict[str, Any]] = []

        raw_events = data.get("events")
        if isinstance(raw_events, list) and raw_events:
            flat_events.extend(self._normalize_story_event(event) for event in raw_events if isinstance(event, dict))

        phases = data.get("phases", {})
        if isinstance(phases, dict):
            for phase_name, phase_data in phases.items():
                if not isinstance(phase_data, dict):
                    continue
                for event in phase_data.get("events", []) or []:
                    if isinstance(event, dict):
                        flat_events.append(self._normalize_story_event(event, default_phase=phase_name))
                for llm_call in phase_data.get("llm_calls", []) or []:
                    if isinstance(llm_call, dict):
                        flat_events.append({
                            "timestamp": llm_call.get("timestamp", ""),
                            "type": "llm_call",
                            "data": llm_call,
                            "phase": llm_call.get("phase") or phase_name,
                        })

        flat_events.sort(key=lambda item: item.get("timestamp", ""))
        return flat_events

    def _load_storytelling_from_event_files(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Load storytelling from legacy event_*.json / llm_call_*.json files in the conversation directory."""
        dir_path = self.storytelling_dir / conversation_id
        if not dir_path.exists() or not dir_path.is_dir():
            return None

        event_files = sorted(dir_path.glob("event_*.json"))
        llm_call_files = sorted(dir_path.glob("llm_call_*.json"))
        if not event_files and not llm_call_files:
            return None

        events: List[Dict[str, Any]] = []

        for file_path in event_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    events.append(self._normalize_story_event(payload))
            except Exception as e:
                logger.warning(f"Could not load storytelling event file {file_path}: {e}")

        for file_path in llm_call_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    events.append({
                        "timestamp": payload.get("timestamp", ""),
                        "type": "llm_call",
                        "data": payload,
                        "phase": payload.get("phase"),
                    })
            except Exception as e:
                logger.warning(f"Could not load storytelling llm file {file_path}: {e}")

        events.sort(key=lambda item: item.get("timestamp", ""))
        if not events:
            return None

        return {
            "conversation_id": conversation_id,
            "created_at": events[0].get("timestamp", ""),
            "events": events,
            "quality_metrics": {
                "task_metrics": [],
                "phase_metrics": [],
                "global_difficulty_score": 0.0,
                "avg_task_quality": 0.0,
            },
            "total_events": len(events),
        }

    def _merge_storytelling_sources(
        self,
        projected_storytelling: Optional[Dict[str, Any]],
        legacy_storytelling: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Merge projection data with richer legacy event files when both are available."""
        if not projected_storytelling and not legacy_storytelling:
            return None
        if not projected_storytelling:
            return legacy_storytelling
        if not legacy_storytelling:
            merged = dict(projected_storytelling)
            merged.setdefault("events", self._flatten_story_events(merged))
            if "total_events" not in merged:
                merged["total_events"] = len(merged.get("events", []))
            return merged

        merged = dict(projected_storytelling)
        legacy_events = self._flatten_story_events(legacy_storytelling)
        if legacy_events:
            merged["events"] = legacy_events
            merged["created_at"] = (
                merged.get("created_at")
                or legacy_storytelling.get("created_at")
                or legacy_events[0].get("timestamp", "")
            )
            merged["total_events"] = max(
                int(merged.get("total_events") or 0),
                len(legacy_events),
            )
        else:
            merged.setdefault("events", self._flatten_story_events(merged))

        return merged

    def get_available_conversations(self) -> List[str]:
        """Restituisce la lista delle conversazioni disponibili"""
        conversations = []
        for file_path in self.storytelling_dir.glob("storytelling_*.json"):
            # 🚨 CORREZIONE: Gestisci suffisso _unknown
            conv_id = file_path.stem.replace("storytelling_", "")
            if conv_id.endswith("_unknown"):
                conv_id = conv_id.replace("_unknown", "")
            conversations.append(conv_id)
        return sorted(conversations, reverse=True)  # Più recenti prima
    
    def load_storytelling(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Carica il file di storytelling per una conversazione

        v0.1: Prova events.jsonl PRIMA (projection), poi fallback a file legacy
        """
        projected_storytelling = None
        legacy_event_storytelling = None

        # 1. NEW v0.1: Try reconstructing from events.jsonl (preferred)
        try:
            projected_storytelling = self._reconstruct_from_events(conversation_id)
            if projected_storytelling:
                logger.info(f"✅ Loaded storytelling for {conversation_id} from events.jsonl")
        except Exception as e:
            logger.warning(f"Could not reconstruct from events.jsonl for {conversation_id}: {e}")

        # 2. LEGACY: Try granular event files in conversation directory
        try:
            legacy_event_storytelling = self._load_storytelling_from_event_files(conversation_id)
            if legacy_event_storytelling:
                logger.info(f"✅ Loaded storytelling for {conversation_id} from event files (legacy)")
        except Exception as e:
            logger.warning(f"Could not load storytelling event files for {conversation_id}: {e}")

        merged_storytelling = self._merge_storytelling_sources(projected_storytelling, legacy_event_storytelling)
        if merged_storytelling:
            return merged_storytelling

        # 3. LEGACY: Try directory with full_storytelling.json
        dir_path = self.storytelling_dir / conversation_id
        full_storytelling_path = dir_path / "full_storytelling.json"
        if full_storytelling_path.exists():
            try:
                with open(full_storytelling_path, 'r', encoding='utf-8') as f:
                    logger.info(f"✅ Loaded storytelling for {conversation_id} from full_storytelling.json (legacy)")
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading full_storytelling from directory {conversation_id}: {e}")

        # 4. LEGACY: Try direct file storytelling_*.json
        file_path = self.storytelling_dir / f"storytelling_{conversation_id}.json"
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    logger.info(f"✅ Loaded storytelling for {conversation_id} from direct file (legacy)")
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading storytelling {conversation_id}: {e}")
                return None

        # 5. LEGACY: Try with _unknown suffix (fallback)
        file_path_unknown = self.storytelling_dir / f"storytelling_{conversation_id}_unknown.json"
        if file_path_unknown.exists():
            try:
                with open(file_path_unknown, 'r', encoding='utf-8') as f:
                    logger.info(f"✅ Loaded storytelling for {conversation_id} from _unknown file (legacy)")
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading storytelling {conversation_id}_unknown: {e}")

        logger.warning(f"⚠️ Storytelling not found for {conversation_id} (tried events.jsonl, directory, direct file, _unknown)")
        return None
    
    def extract_summary(self, conversation_id: str) -> Optional[StorytellingSummary]:
        """Estrae un riassunto dello storytelling per una conversazione con metriche qualitative"""
        data = self.load_storytelling(conversation_id)
        if not data:
            return None

        events = self._flatten_story_events(data)

        # 🆕 Estrai metriche qualitative direttamente dai dati
        quality_metrics = data.get('quality_metrics', {})
        task_metrics = quality_metrics.get('task_metrics', [])
        phase_metrics = quality_metrics.get('phase_metrics', [])
        global_difficulty_score = quality_metrics.get('global_difficulty_score', 0.0)
        avg_task_quality = quality_metrics.get('avg_task_quality', 0.0)
        
        # Estrai obiettivi strategici
        strategic_objectives = []
        for event in events:
            if event.get('type') == 'meta_planning_complete':
                strategic_objectives = event.get('data', {}).get('strategic_objectives', [])
                break
        
        # Estrai piani tattici
        tactical_plans = []
        for event in events:
            if event.get('type') == 'project_planning_complete':
                tactical_plans.append({
                    'parent_task_id': event.get('data', {}).get('parent_task_id'),
                    'tactical_tasks': event.get('data', {}).get('tactical_tasks', [])
                })
        
        # Estrai task completati
        completed_tasks = []
        for event in events:
            if event.get('type') == 'task_execution_complete':
                completed_tasks.append({
                    'task_id': event.get('data', {}).get('task_id'),
                    'result': event.get('data', {}).get('result', ''),
                    'timestamp': event.get('timestamp')
                })
        
        # Estrai chiamate LLM
        llm_calls = []
        for event in events:
            if event.get('type') == 'llm_call':
                llm_calls.append({
                    'phase': event.get('data', {}).get('phase'),
                    'node': event.get('data', {}).get('node'),
                    'model_id': event.get('data', {}).get('model_id'),
                    'prompt': event.get('data', {}).get('prompt', ''),
                    'response': event.get('data', {}).get('response', ''),
                    'success': event.get('data', {}).get('success', True),
                    'timestamp': event.get('timestamp')
                })
        
        # Estrai riflessioni
        reflections = []
        for event in events:
            if event.get('type') == 'reflection_decision':
                reflections.append({
                    'decision': event.get('data', {}).get('decision'),
                    'reason': event.get('data', {}).get('reason'),
                    'timestamp': event.get('timestamp')
                })
        
        # Estrai sintesi finale
        final_synthesis = None
        for event in events:
            if event.get('type') == 'synthesis_complete':
                final_synthesis = event.get('data', {})
                break
        
        # Calcola durata
        duration_minutes = 0
        if events:
            try:
                start_time = datetime.fromisoformat(events[0]['timestamp'].replace('Z', '+00:00'))
                end_time = datetime.fromisoformat(events[-1]['timestamp'].replace('Z', '+00:00'))
                duration_minutes = (end_time - start_time).total_seconds() / 60
            except Exception as e:
                logger.warning(f"Errore nel calcolare la durata: {e}")
        
        # Statistiche
        statistics = {
            'total_events': len(events),
            'completed_tasks': len(completed_tasks),
            'llm_calls': len(llm_calls),
            'reflections': len(reflections),
            'strategic_objectives': len(strategic_objectives),
            'tactical_plans': len(tactical_plans),
            # 🆕 Statistiche qualitative
            'quality_tasks_analyzed': len(task_metrics),
            'phases_with_quality': len(phase_metrics)
        }

        # 🆕 Calcola assessment qualitativo dell'esecuzione
        execution_assessment = "Unknown"
        if avg_task_quality > 0:
            if avg_task_quality >= 0.8 and global_difficulty_score < 0.3:
                execution_assessment = "Smooth execution"
            elif avg_task_quality >= 0.6 and global_difficulty_score < 0.6:
                execution_assessment = "Standard execution"
            elif global_difficulty_score > 0.7:
                execution_assessment = "Challenging execution"
            else:
                execution_assessment = "Moderate execution"

        return StorytellingSummary(
            conversation_id=conversation_id,
            created_at=data.get('created_at') or data.get('session_start', '') or (events[0].get('timestamp', '') if events else ''),
            total_events=len(events),
            strategic_objectives=strategic_objectives,
            tactical_plans=tactical_plans,
            completed_tasks=completed_tasks,
            llm_calls=llm_calls,
            reflections=reflections,
            final_synthesis=final_synthesis,
            duration_minutes=duration_minutes,
            statistics=statistics,
            # 🆕 Nuovi campi qualitativi
            task_quality_metrics=task_metrics,
            phase_quality_metrics=phase_metrics,
            global_difficulty_score=global_difficulty_score,
            avg_task_quality=avg_task_quality,
            execution_assessment=execution_assessment
        )
    
    def format_storytelling_for_chat(self, conversation_id: str, max_length: int = 2000) -> str:
        """Formatta lo storytelling per la visualizzazione in chat"""
        summary = self.extract_summary(conversation_id)
        if not summary:
            return "❌ Storytelling non disponibile per questa conversazione."
        
        output = []
        output.append(f"🎭 **Storytelling Intelligente** - {conversation_id}")
        output.append(f"📊 **Statistiche**: {summary.statistics['completed_tasks']} task completati, {summary.statistics['llm_calls']} chiamate LLM, {summary.duration_minutes:.1f} min")

        # 🆕 Aggiungi metriche qualitative
        if summary.avg_task_quality > 0:
            quality_emoji = "🟢" if summary.avg_task_quality >= 0.8 else "🟡" if summary.avg_task_quality >= 0.6 else "🔴"
            difficulty_emoji = "💪" if summary.global_difficulty_score > 0.7 else "🏃" if summary.global_difficulty_score > 0.4 else "🚶"
            output.append(f"📈 **Qualità**: {quality_emoji} {summary.avg_task_quality:.1%} media, {difficulty_emoji} difficoltà {summary.global_difficulty_score:.1%}")
            output.append(f"🏆 **Assessment**: {summary.execution_assessment}")
        output.append("")

        # Obiettivi strategici
        if summary.strategic_objectives:
            output.append("📋 **Piano Strategico**:")
            for obj in summary.strategic_objectives[:3]:  # Limita a 3 per brevità
                output.append(f"  🎯 {obj.get('id', 'N/A')}: {obj.get('title', 'N/A')}")
            output.append("")
        
        # Task completati recenti
        if summary.completed_tasks:
            output.append("✅ **Task Completati** (ultimi 3):")
            for task in summary.completed_tasks[-3:]:
                result_preview = task.get('result', '')[:100] + "..." if len(task.get('result', '')) > 100 else task.get('result', '')
                output.append(f"  • {task.get('task_id', 'N/A')}: {result_preview}")
            output.append("")

        # 🆕 Task top performance (qualitativi)
        if summary.task_quality_metrics:
            # Ordina per quality score e prendi i top 3
            top_tasks = sorted(summary.task_quality_metrics, key=lambda x: x.get('quality_score', 0), reverse=True)[:3]
            if top_tasks:
                output.append("⭐ **Top Performance Tasks**:")
                for task in top_tasks:
                    score = task.get('quality_score', 0)
                    task_type = task.get('result_type', 'unknown')
                    insights = task.get('key_insights', [])
                    output.append(f"  🏆 {task.get('task_id', 'N/A')}: {score:.1%} ({task_type})")
                    if insights:
                        output.append(f"     💡 {insights[0]}")
                output.append("")

        # Riflessioni recenti
        if summary.reflections:
            output.append("🤔 **Decisioni Recenti**:")
            for reflection in summary.reflections[-2:]:  # Ultime 2
                output.append(f"  🔄 {reflection.get('decision', 'N/A')}: {reflection.get('reason', 'N/A')}")
            output.append("")

        # 🆕 Metriche fase
        if summary.phase_quality_metrics:
            output.append("📊 **Metriche per Fase**:")
            for phase in summary.phase_quality_metrics:
                phase_name = phase.get('phase', 'unknown')
                success_rate = phase.get('success_rate', 0)
                avg_quality = phase.get('avg_quality_score', 0)
                bottlenecks = phase.get('bottlenecks', [])
                output.append(f"  🔄 {phase_name.title()}: {success_rate:.1%} success, {avg_quality:.1%} qualità")
                if bottlenecks:
                    output.append(f"     ⚠️ {', '.join(bottlenecks)}")
            output.append("")

        # Sintesi finale se disponibile
        if summary.final_synthesis:
            output.append("📝 **Sintesi Finale**:")
            final_answer = summary.final_synthesis.get('final_answer', '')
            if final_answer:
                preview = final_answer[:200] + "..." if len(final_answer) > 200 else final_answer
                output.append(f"  {preview}")
        
        result = "\n".join(output)
        
        # Tronca se troppo lungo
        if len(result) > max_length:
            result = result[:max_length] + "\n\n... (troncato per brevità)"
        
        return result
    
    def format_storytelling_for_llm(self, conversation_id: str) -> str:
        """Formatta lo storytelling per l'uso da parte dei modelli LLM"""
        summary = self.extract_summary(conversation_id)
        if not summary:
            return "Storytelling non disponibile."
        
        output = []
        output.append(f"CONTESTO STORYTELLING - Conversazione: {conversation_id}")
        output.append(f"Durata: {summary.duration_minutes:.1f} minuti, Eventi: {summary.total_events}")
        output.append("")
        
        # Obiettivi strategici
        if summary.strategic_objectives:
            output.append("OBIETTIVI STRATEGICI:")
            for obj in summary.strategic_objectives:
                output.append(f"- {obj.get('id', 'N/A')}: {obj.get('title', 'N/A')} (Priorità: {obj.get('priority', 'N/A')})")
            output.append("")
        
        # Task completati
        if summary.completed_tasks:
            output.append("TASK COMPLETATI:")
            for task in summary.completed_tasks:
                output.append(f"- {task.get('task_id', 'N/A')}: {task.get('result', '')[:150]}...")
            output.append("")
        
        # Riflessioni
        if summary.reflections:
            output.append("DECISIONI E RIFLESSIONI:")
            for reflection in summary.reflections:
                output.append(f"- {reflection.get('decision', 'N/A')}: {reflection.get('reason', 'N/A')}")
            output.append("")
        
        return "\n".join(output)
    
    def get_realtime_updates(self, conversation_id: str, last_event_index: int = 0) -> Tuple[List[Dict[str, Any]], int]:
        """Restituisce gli aggiornamenti in tempo reale dello storytelling"""
        data = self.load_storytelling(conversation_id)
        if not data:
            return [], 0
        
        events = self._flatten_story_events(data)
        new_events = events[last_event_index:]
        
        return new_events, len(events)

# Funzioni di utilità per l'integrazione
def get_storytelling_for_chat(conversation_id: str) -> str:
    """Funzione di utilità per ottenere storytelling formattato per chat"""
    extractor = StorytellingExtractor()
    return extractor.format_storytelling_for_chat(conversation_id)

def get_storytelling_for_llm(conversation_id: str) -> str:
    """Funzione di utilità per ottenere storytelling formattato per LLM"""
    extractor = StorytellingExtractor()
    return extractor.format_storytelling_for_llm(conversation_id)

def get_available_conversations() -> List[str]:
    """Funzione di utilità per ottenere conversazioni disponibili"""
    extractor = StorytellingExtractor()
    return extractor.get_available_conversations()
