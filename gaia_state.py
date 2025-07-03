"""State management and type definitions for GAIA agent"""

from typing import TypedDict, Literal, Annotated, Any, Optional, List, Dict
from langgraph.graph import add_messages
from datetime import datetime
import operator


class GaiaState(TypedDict):
    """Main state for GAIA agent execution"""
    # Core conversation state
    messages: Annotated[list, add_messages]
    
    # Question tracking
    original_question: str
    task_id: str
    question_type: Literal["number", "string", "list", "date", "unknown"]
    
    # Execution tracking
    step_count: int
    max_steps: int
    current_subtask: Optional[str]
    
    # Tool results accumulation
    web_search_results: Annotated[List[Dict], operator.add]
    file_contents: Dict[str, Any]
    image_analyses: Dict[str, str]
    code_outputs: List[str]
    
    # Reasoning chain
    reasoning_steps: Annotated[List[str], operator.add]
    intermediate_answers: List[str]
    confidence_scores: List[float]
    
    # Error tracking
    errors: List[Dict[str, Any]]
    retry_count: int
    
    # Final output
    raw_answer: Optional[str]
    formatted_answer: Optional[str]
    validation_status: Optional[Dict[str, bool]]
    
    # Metadata
    start_time: datetime
    end_time: Optional[datetime]
    total_tokens_used: int


class ToolCall(TypedDict):
    """Structure for tool calls"""
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[Any]
    error: Optional[str]
    timestamp: datetime


class ValidationResult(TypedDict):
    """Answer validation results"""
    is_valid: bool
    format_checks: Dict[str, bool]
    error_messages: List[str]
    suggested_correction: Optional[str]


def initialize_state(question: str, task_id: str) -> GaiaState:
    """Initialize a new GAIA state"""
    return GaiaState(
        messages=[],
        original_question=question,
        task_id=task_id,
        question_type="unknown",
        step_count=0,
        max_steps=5,  # GAIA Level 1 limit
        current_subtask=None,
        web_search_results=[],
        file_contents={},
        image_analyses={},
        code_outputs=[],
        reasoning_steps=[],
        intermediate_answers=[],
        confidence_scores=[],
        errors=[],
        retry_count=0,
        raw_answer=None,
        formatted_answer=None,
        validation_status=None,
        start_time=datetime.now(),
        end_time=None,
        total_tokens_used=0
    )


def update_step_count(state: GaiaState) -> Dict[str, Any]:
    """Increment step count and check limits"""
    new_count = state["step_count"] + 1
    
    if new_count > state["max_steps"]:
        return {
            "step_count": new_count,
            "errors": state["errors"] + [{
                "type": "step_limit_exceeded",
                "message": f"Exceeded maximum steps ({state['max_steps']})",
                "timestamp": datetime.now()
            }]
        }
    
    return {"step_count": new_count}


def add_reasoning_step(state: GaiaState, step: str, confidence: float = 0.5) -> Dict[str, Any]:
    """Add a reasoning step with confidence tracking"""
    return {
        "reasoning_steps": [step],
        "confidence_scores": state["confidence_scores"] + [confidence]
    }