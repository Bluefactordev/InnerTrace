"""
Modelli Pydantic per il parsing strutturato con LangChain
"""
from pydantic import BaseModel, Field, root_validator
from typing import List, Optional, Dict, Any, Literal
from enum import Enum

class ExecutionFlow(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"

class TaskComplexity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class TaskDuration(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"

TaskOutputMode = Literal["text", "json_object", "json_list"]
TaskConsumerPolicy = Literal["direct_read", "inspect_then_process", "direct_process"]
TaskInputArtifactKind = Literal["none", "dataset_listing", "dataset_path_list", "document_content", "data_handle"]

class StrategicObjective(BaseModel):
    """Obiettivo strategico del meta-planner"""
    id: str = Field(description="ID univoco dell'obiettivo")
    title: str = Field(description="Titolo dell'obiettivo strategico")
    description: str = Field(description="Descrizione dettagliata dell'obiettivo")
    priority: int = Field(description="Priorità (1-10, 10=massima)", ge=1, le=10)  # 🔧 FIX: Aumentato da 5 a 10 per allineamento con prompt depth=10
    estimated_complexity: TaskComplexity = Field(description="Complessità stimata")

class MetaPlan(BaseModel):
    """Piano strategico del meta-planner"""
    strategic_objectives: List[StrategicObjective] = Field(description="Lista degli obiettivi strategici")
    execution_strategy: ExecutionFlow = Field(default=ExecutionFlow.SEQUENTIAL, description="Strategia di esecuzione")
    success_criteria: List[str] = Field(default_factory=list, description="Criteri di successo")

    @root_validator(pre=True)
    def fill_missing_fields(cls, values):
        """Popola campi mancanti che il LLM potrebbe aver omesso"""
        # Se il LLM ha restituito un dict con campi mancanti, li aggiungiamo
        if isinstance(values, dict):
            if 'strategic_objectives' not in values or not values['strategic_objectives']:
                # Fallback: crea un obiettivo generico se il LLM non ha specificato
                values['strategic_objectives'] = [{
                    'id': 'fallback_objective',
                    'title': 'Esecuzione diretta obiettivo',
                    'description': values.get('description', 'Esegui compito richiesto'),
                    'priority': 5,
                    'estimated_complexity': 'medium'
                }]
            if 'success_criteria' not in values or not values['success_criteria']:
                values['success_criteria'] = ["Completa l'obiettivo originale"]
        return values

class TacticalTask(BaseModel):
    """Task tattico del project-planner"""
    id: str = Field(description="ID univoco del task")
    title: str = Field(description="Titolo del task specifico")
    description: str = Field(description="Descrizione dettagliata dell'azione da compiere")
    required_tools: List[str] = Field(description="Strumenti richiesti per il task", default_factory=list)
    dependencies: List[str] = Field(description="ID dei task da cui dipende", default_factory=list)
    estimated_duration: TaskDuration = Field(description="Durata stimata del task")
    success_criteria: str = Field(description="Come verificare il successo del task")
    
    # 🔧 CARDINALITY-BASED ROUTING: Explicit Reasoning Schema (Opzione C)
    # CAMPI OBBLIGATORI SENZA DEFAULT - ValidationError se mancanti
    cardinality_analysis: str = Field(
        ...,  # NO DEFAULT - OBBLIGATORIO
        description="RAGIONAMENTO OBBLIGATORIO: Analizza il numero di input (file, URL, entry, record). Esempio: 'Il dataset contiene circa 100 file da analizzare'"
    )
    estimated_item_count: int = Field(
        ...,  # NO DEFAULT - OBBLIGATORIO
        description="Stima numerica ESATTA degli elementi. >3 = Batch obbligatorio"
    )
    
    # La strategia DEVE essere coerente con estimated_item_count
    execution_strategy: Literal["sequential_manual", "batch_programmatic"] = Field(
        ...,  # NO DEFAULT - OBBLIGATORIO
        description="OBBLIGATORIO: 'batch_programmatic' se >3 elementi, 'sequential_manual' SOLO se <=3"
    )
    input_artifact_kind: TaskInputArtifactKind = Field(
        ...,
        description="Tipo di artifact in ingresso: none, dataset_listing, dataset_path_list, document_content o data_handle"
    )
    consumer_policy: TaskConsumerPolicy = Field(
        ...,
        description="Policy di consumo artifact: direct_read, inspect_then_process o direct_process"
    )
    output_mode: TaskOutputMode = Field(
        ...,
        description="Forma dell'output del task: text, json_object o json_list"
    )
    persistence_required: bool = Field(
        ...,
        description="True se il risultato deve essere persistito come artifact per task downstream"
    )
    
    # Campi opzionali secondari
    requires_human_reasoning: Optional[bool] = Field(
        default=False,
        description="True se il task richiede ragionamento intermedio tra passi"
    )
    
    class Config:
        # 🔧 STRICT VALIDATION: Reject unknown fields and enforce required fields
        extra = "forbid"  # Reject any field not in schema
        validate_assignment = True  # Validate on assignment
    
class ProjectPlan(BaseModel):
    """Piano tattico del project-planner"""
    tactical_tasks: List[TacticalTask] = Field(description="Lista dei task tattici")
    execution_flow: ExecutionFlow = Field(default=ExecutionFlow.SEQUENTIAL, description="Flusso di esecuzione")
    quality_checks: List[str] = Field(description="Controlli di qualità", default_factory=list)

    @root_validator(pre=True)
    def fill_missing_fields(cls, values):
        """Popola campi mancanti che il LLM potrebbe aver omesso"""
        if isinstance(values, dict):
            if 'tactical_tasks' not in values or not values['tactical_tasks']:
                # Fallback: crea un task generico se il LLM non ha specificato
                values['tactical_tasks'] = [{
                    'id': 'fallback_task',
                    'title': 'Esecuzione diretta obiettivo',
                    'description': values.get('description', 'Esegui compito richiesto'),
                    'required_tools': [],
                    'dependencies': [],
                    'estimated_duration': 'medium',
                    'success_criteria': 'Completa l\'obiettivo originale',
                    'cardinality_analysis': 'Singolo task di fallback',
                    'estimated_item_count': 1,
                    'execution_strategy': 'sequential_manual',
                    'input_artifact_kind': 'none',
                    'consumer_policy': 'direct_read',
                    'output_mode': 'text',
                    'persistence_required': False,
                }]
        return values

class ExecutionResult(BaseModel):
    """Risultato dell'esecuzione di un task"""
    task_id: str = Field(description="ID del task eseguito")
    success: bool = Field(description="Se il task è stato completato con successo")
    result: str = Field(description="Risultato dell'esecuzione")
    error: Optional[str] = Field(description="Eventuale errore", default=None)
    execution_time: Optional[float] = Field(description="Tempo di esecuzione in secondi", default=None)
    tools_used: List[str] = Field(description="Strumenti utilizzati", default_factory=list)

class SynthesisResult(BaseModel):
    """Risultato della sintesi finale"""
    final_output: str = Field(description="Output finale del planner")
    summary: str = Field(description="Riassunto del processo")
    tasks_completed: int = Field(description="Numero di task completati")
    execution_time: float = Field(description="Tempo totale di esecuzione")
    quality_score: Optional[float] = Field(description="Punteggio di qualità (0-1)", default=None)

# === NUOVI MODELLI PER SUB-PLANNERS E QUALITY GATES ===

class ReflectionDecision(str, Enum):
    """Decisioni possibili del reflection node"""
    CONTINUE_PLANNING = "continue_planning"
    CONTINUE_EXECUTION = "continue_execution"
    DELEGATE_TO_SUBPLANNER = "delegate_to_subplanner"  # NUOVO: Delega a sub-planner
    SYNTHESIZE = "synthesize"
    TERMINATE = "terminate"

class QualityCheckType(str, Enum):
    """Tipi di quality gate"""
    SAMPLING = "sampling"  # Verifica a campione
    CROSS_VALIDATION = "cross_validation"  # Verifica incrociata
    PEER_REVIEW = "peer_review"  # Revisione tra pari
    CONFIDENCE_CHECK = "confidence_check"  # Controllo confidenza
    FINAL_VALIDATION = "final_validation"  # Validazione finale obbligatoria

class SubPlannerRequest(BaseModel):
    """Richiesta di creazione sub-planner per task complesso"""
    parent_task_id: str = Field(description="ID del task padre che richiede sub-planner")
    reason: str = Field(description="Perché serve un sub-planner")
    complexity_estimate: TaskComplexity = Field(description="Complessità stimata del sub-task")
    max_depth: int = Field(description="Profondità massima del sub-planner", ge=1, le=5)
    required_tools: List[str] = Field(description="Tool necessari al sub-planner", default_factory=list)

class QualityGateResult(BaseModel):
    """Risultato di un quality gate"""
    check_type: QualityCheckType = Field(description="Tipo di controllo eseguito")
    passed: bool = Field(description="Se il controllo è passato")
    score: Optional[float] = Field(description="Punteggio (0-1)", default=None)
    issues_found: List[str] = Field(description="Problemi identificati", default_factory=list)
    recommendations: List[str] = Field(description="Raccomandazioni per migliorare", default_factory=list)
    timestamp: str = Field(description="Timestamp del controllo")

class ReflectionOutput(BaseModel):
    """Output strutturato del reflection node"""
    decision: ReflectionDecision = Field(description="Decisione del reflection node")
    reason: str = Field(description="Motivazione della decisione")
    confidence: float = Field(description="Confidenza nella decisione (0-1)", ge=0.0, le=1.0)
    quality_assessment: Optional[QualityGateResult] = Field(description="Valutazione qualità attuale", default=None)
    subplanner_request: Optional[SubPlannerRequest] = Field(description="Richiesta sub-planner se necessario", default=None)
    next_actions: List[str] = Field(description="Azioni successive raccomandate", default_factory=list)
