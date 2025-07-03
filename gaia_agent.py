"""Main GAIA agent implementation using LangGraph"""

import os
from typing import Literal, Dict, Any, List, Optional
from datetime import datetime
import json

from langgraph.graph import StateGraph, END, START
try:
    from langgraph.types import Command
except ImportError:
    # For older versions
    Command = None

# Import checkpoint with fallback
try:
    from langgraph.checkpoint.memory import MemorySaver
    CheckpointSaver = MemorySaver
except ImportError:
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        CheckpointSaver = SqliteSaver
    except ImportError:
        # No checkpoint available
        CheckpointSaver = None

from langgraph.prebuilt import ToolNode, tools_condition

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
# RunnableRetry is not available in this version

# Import our modules
from gaia_state import GaiaState, initialize_state, update_step_count, add_reasoning_step
from gaia_tools import (
    web_search, browse_webpage, process_file, 
    analyze_image_with_vision, execute_python_code, find_pattern_in_text
)
from gaia_formatting import (
    format_gaia_answer, validate_answer_format, 
    fix_common_formatting_errors, detect_answer_type
)


class GaiaAgent:
    """Main GAIA benchmark agent"""
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 model_name: str = "gpt-4-1106-preview",
                 temperature: float = 0.1,
                 use_checkpointing: bool = True):
        """
        Initialize GAIA agent.
        
        Args:
            openai_api_key: OpenAI API key (uses env var if not provided)
            model_name: OpenAI model to use
            temperature: Model temperature for consistency
            use_checkpointing: Whether to enable state checkpointing
        """
        # Set API key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Initialize tools
        self.tools = [
            web_search,
            browse_webpage,
            process_file,
            analyze_image_with_vision,
            execute_python_code,
            find_pattern_in_text
        ]
        
        # Initialize primary model with tools
        self.model = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            top_p=0.95
        ).bind_tools(self.tools)
        
        # Initialize fallback model
        self.fallback_model = ChatOpenAI(
            model="gpt-4o",
            temperature=temperature
        ).bind_tools(self.tools)
        
        # Model with retry logic
        self.resilient_model = self.model.with_fallbacks([self.fallback_model])
        
        
        # Model without tools for synthesis
        self.synthesis_model = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            top_p=0.95
        )
        
        # Build the graph
        self.graph = self._build_graph()
        
        # Compile with checkpointing if enabled
        if use_checkpointing and CheckpointSaver:
            if hasattr(CheckpointSaver, 'from_conn_string'):
                # SqliteSaver style
                self.checkpointer = CheckpointSaver.from_conn_string("gaia_checkpoints.db")
            else:
                # MemorySaver style
                self.checkpointer = CheckpointSaver()
            self.app = self.graph.compile(checkpointer=self.checkpointer)
        else:
            if use_checkpointing and not CheckpointSaver:
                print("Warning: Checkpointing requested but no checkpointer available")
            self.app = self.graph.compile()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(GaiaState)
        
        # Add nodes
        workflow.add_node("analyze_question", self.analyze_question_node)
        workflow.add_node("reason", self.reasoning_node)
        workflow.add_node("use_tools", ToolNode(self.tools))
        workflow.add_node("synthesize", self.synthesis_node)
        workflow.add_node("format_answer", self.format_answer_node)
        workflow.add_node("validate", self.validation_node)
        
        # Define flow
        workflow.add_edge(START, "analyze_question")
        workflow.add_edge("analyze_question", "reason")
        
        # Conditional routing from reasoning
        # Note: route_reasoning never returns "end"
        workflow.add_conditional_edges(
            "reason",
            self.route_reasoning,
            {
                "tools": "use_tools",
                "synthesize": "synthesize",
                "continue": "reason"
            }
        )
        
        # Tool results go back to reasoning
        workflow.add_edge("use_tools", "reason")
        
        # Synthesis leads to formatting
        workflow.add_edge("synthesize", "format_answer")
        
        # Formatting leads to validation
        workflow.add_edge("format_answer", "validate")
        
        # Validation can retry or end
        workflow.add_conditional_edges(
            "validate",
            self.route_validation,
            {
                "retry": "format_answer",
                "end": END
            }
        )
        
        return workflow
    
    def analyze_question_node(self, state: GaiaState) -> Dict[str, Any]:
        """Analyze the question and plan approach"""
        system_prompt = """You are analyzing a GAIA benchmark question.
        
        Your task is to:
        1. Identify the type of answer expected (number, string, list, date)
        2. Break down what information is needed
        3. Plan the approach to find the answer
        4. Note any files or specific resources mentioned
        
        Be concise and focused on actionable analysis."""
        
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Question: {question}\n\nProvide your analysis:")
        ])
        
        result = self.synthesis_model.invoke(
            analysis_prompt.format_messages(question=state["original_question"])
        )
        
        # Extract question type from analysis
        analysis_text = result.content
        question_type = detect_answer_type(state["original_question"], "")
        
        return {
            "messages": [result],
            "question_type": question_type,
            "reasoning_steps": [f"Analysis: {analysis_text}"],
            "current_subtask": f"Find answer to: {state['original_question']}"
        }
    
    def reasoning_node(self, state: GaiaState) -> Dict[str, Any]:
        """Reasoning node that decides next action"""
        
        # Build context with proper tool result handling
        context_parts = []
        
        # Add task context
        context_parts.append(f"Question: {state['original_question']}")
        context_parts.append(f"Question Type: {state.get('question_type', 'Unknown')}")
        
        # Check if we just received tool results
        just_used_tools = False
        tool_result = None
        
        if state["messages"] and len(state["messages"]) >= 2:
            # Check if the last message is a tool result
            last_msg = state["messages"][-1]
            prev_msg = state["messages"][-2] if len(state["messages"]) > 1 else None
            
            # If previous message had tool calls and last message has content, it's a tool result
            if (prev_msg and hasattr(prev_msg, 'tool_calls') and prev_msg.tool_calls and
                hasattr(last_msg, 'content') and last_msg.content):
                just_used_tools = True
                tool_result = last_msg.content
                context_parts.append(f"\nTool Result: {tool_result}")
        
        # Add reasoning history
        if state.get("reasoning_steps"):
            context_parts.append(f"\nPrevious reasoning steps: {len(state['reasoning_steps'])}")
            # Show last 2 reasoning steps
            for step in state["reasoning_steps"][-2:]:
                context_parts.append(f"- {step[:100]}...")
        
        context = "\n".join(context_parts)
        
        # Modified prompt to handle tool results properly
        reasoning_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a reasoning agent. Based on the context, decide what to do next.

If you just received a tool result (shown as "Tool Result:"), you should:
1. Analyze the result
2. Decide if it answers the question or if more information is needed
3. If it answers the question, respond with "I have the answer: [answer]" to trigger synthesis
4. If more info is needed, explain what else is needed

Available actions:
- Use tools (search_web, execute_python_code, etc.) - but AVOID repeating the same tool call
- Synthesize answer if you have sufficient information
- Continue reasoning if more analysis is needed

Step {step_count}/{max_steps}"""),
            ("human", "{context}")
        ])
        
        result = self.model.invoke(
            reasoning_prompt.format_messages(
                context=context,
                step_count=state["step_count"] + 1,
                max_steps=state["max_steps"]
            )
        )
        
        # Update step count
        updates = update_step_count(state)
        updates["messages"] = [result]
        
        # Store tool results as intermediate answers
        if just_used_tools and tool_result:
            current_answers = state.get("intermediate_answers", [])
            updates["intermediate_answers"] = current_answers + [tool_result]
        
        # Store reasoning step
        updates["reasoning_steps"] = state.get("reasoning_steps", []) + [result.content]
        
        # Decide next action based on response
        next_action = self._decide_next_action(result, state)
        
        # Return updates only - routing is handled by route_reasoning
        return updates
    
    def synthesis_node(self, state: GaiaState) -> Dict[str, Any]:
        """Synthesize all information into a final answer"""
        
        # Gather all information
        all_info = self._gather_all_information(state)
        
        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are providing the final answer to a GAIA benchmark question.
            
            Rules:
            1. Give ONLY the answer - no explanation
            2. Be precise and exact
            3. Format according to the question requirements
            4. Do NOT include "FINAL ANSWER:" prefix
            
            Answer type: {answer_type}"""),
            ("human", """Question: {question}
            
            Information gathered:
            {information}
            
            Provide the exact answer:""")
        ])
        
        result = self.synthesis_model.invoke(
            synthesis_prompt.format_messages(
                question=state["original_question"],
                answer_type=state["question_type"],
                information=all_info
            )
        )
        
        raw_answer = result.content.strip()
        
        return {
            "messages": [result],
            "raw_answer": raw_answer,
            "reasoning_steps": [f"Synthesized answer: {raw_answer}"]
        }
    
    def format_answer_node(self, state: GaiaState) -> Dict[str, Any]:
        """Format the answer according to GAIA requirements"""
        
        if not state.get("raw_answer"):
            return {"formatted_answer": "", "errors": [{"type": "no_answer", "message": "No answer to format"}]}
        
        # Format the answer
        formatted = format_gaia_answer(
            state["raw_answer"],
            state["original_question"],
            state["question_type"]
        )
        
        return {
            "formatted_answer": formatted,
            "reasoning_steps": [f"Formatted answer: {formatted}"]
        }
    
    def validation_node(self, state: GaiaState) -> Dict[str, Any]:
        """Validate the formatted answer"""
        
        if not state.get("formatted_answer"):
            return {"validation_status": {"is_valid": False, "errors": ["No answer"]}}
        
        # Validate format
        validation = validate_answer_format(
            state["formatted_answer"],
            state["original_question"]
        )
        
        if validation["is_valid"]:
            # Valid answer - return updates for end
            return {
                "validation_status": validation,
                "end_time": datetime.now()
            }
        else:
            # Try to fix common errors
            fixed = fix_common_formatting_errors(
                state["formatted_answer"],
                state["original_question"],
                validation
            )
            
            if fixed != state["formatted_answer"] and state.get("retry_count", 0) < 3:
                # Can retry with fixed answer
                return {
                    "formatted_answer": fixed,
                    "retry_count": state.get("retry_count", 0) + 1,
                    "validation_status": validation
                }
            else:
                # No more retries - end with current validation
                return {
                    "validation_status": validation,
                    "end_time": datetime.now()
                }
    
    def route_reasoning(self, state: GaiaState) -> str:
        """Route from reasoning node based on state"""
        
        # If we just came from synthesis, we should end
        if state.get("raw_answer"):
            return "end"  # This should never happen as synthesis goes to format
        
        # Check if we've hit the step limit
        if state["step_count"] >= state["max_steps"]:
            return "synthesize"
        
        # Check if we have tool results that likely contain the answer
        if self._check_tool_results_for_answer(state):
            return "synthesize"
            
        # Get the last message
        last_message = state["messages"][-1] if state["messages"] else None
        
        if last_message:
            # Check for tool calls first
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
            
            # Otherwise use decision logic
            return self._decide_next_action(last_message, state)
        
        # Default to continue if no message
        return "continue"
    
    def route_validation(self, state: GaiaState) -> str:
        """Route from validation node"""
        
        validation = state.get("validation_status", {})
        
        if validation.get("is_valid", False):
            return "end"
        
        if state.get("retry_count", 0) < 3:
            return "retry"
        
        return "end"
    
    def _build_reasoning_context(self, state: GaiaState) -> str:
        """Build context string for reasoning"""
        context_parts = [
            f"Question: {state['original_question']}",
            f"Question Type: {state['question_type']}",
            f"Current Step: {state['step_count']}/{state['max_steps']}",
            f"Current Subtask: {state.get('current_subtask', 'None')}",
            ""
        ]
        
        # Add web search results
        if state.get("web_search_results"):
            context_parts.append("Web Search Results:")
            for result in state["web_search_results"][-3:]:  # Last 3 results
                context_parts.append(f"- {result}")
            context_parts.append("")
        
        # Add file contents summary
        if state.get("file_contents"):
            context_parts.append("Files Processed:")
            for filename, content in state["file_contents"].items():
                context_parts.append(f"- {filename}: {len(str(content))} chars")
            context_parts.append("")
        
        # Add reasoning steps
        if state.get("reasoning_steps"):
            context_parts.append("Reasoning So Far:")
            for step in state["reasoning_steps"][-5:]:  # Last 5 steps
                context_parts.append(f"- {step}")
            context_parts.append("")
        
        # Add intermediate answers
        if state.get("intermediate_answers"):
            context_parts.append("Potential Answers Found:")
            for answer in state["intermediate_answers"]:
                context_parts.append(f"- {answer}")
        
        return "\n".join(context_parts)
    
    def _decide_next_action(self, message: AIMessage, state: GaiaState) -> str:
        """Decide next action based on model response"""
        
        # Check for tool calls
        if hasattr(message, 'tool_calls') and message.tool_calls:
            return "tools"
        
        # Check for synthesis indicators
        content_lower = message.content.lower()
        synthesis_indicators = [
            "have enough information",
            "ready to answer",
            "found the answer",
            "can now provide",
            "the answer is"
        ]
        
        if any(indicator in content_lower for indicator in synthesis_indicators):
            return "synthesize"
        
        # Check if we're stuck
        if state["step_count"] >= state["max_steps"] - 1:
            return "synthesize"
        
        return "continue"
    
    def _has_sufficient_information(self, state: GaiaState) -> bool:
        """Check if we have enough information to answer"""
        
        # Simple heuristics
        has_web_results = bool(state.get("web_search_results"))
        has_file_data = bool(state.get("file_contents"))
        has_intermediate = bool(state.get("intermediate_answers"))
        
        # If we have multiple sources or clear answers, we might be ready
        sources_count = sum([has_web_results, has_file_data, has_intermediate])
        
        return sources_count >= 2 or (has_intermediate and state["step_count"] >= 3)
    
    
    def _check_tool_results_for_answer(self, state: GaiaState) -> bool:
        """Check if tool results contain the answer"""
        # Check if we have intermediate answers from tools
        if state.get("intermediate_answers"):
            return True
            
        # Check last few messages for tool results pattern
        if state["messages"] and len(state["messages"]) >= 2:
            # Look at last 2 messages
            for i in range(min(2, len(state["messages"]))):
                msg = state["messages"][-(i+1)]
                if hasattr(msg, 'content') and msg.content:
                    content = msg.content.lower()
                    # Check for answer indicators
                    if any(phrase in content for phrase in [
                        "i have the answer", "the answer is", "result:", "output:",
                        "sufficient information", "ready to synthesize"
                    ]):
                        return True
                    # If it's just a number or short result, likely an answer
                    if len(msg.content.strip()) < 20 and any(char.isdigit() for char in msg.content):
                        return True
        
        # Check code outputs
        if state.get("code_outputs") and state["code_outputs"]:
            return True
            
        return False
    
    def _gather_all_information(self, state: GaiaState) -> str:
        """Gather all information collected during reasoning"""
        
        info_parts = []
        
        # Web results
        if state.get("web_search_results"):
            info_parts.append("Web Search Results:")
            for result in state["web_search_results"]:
                info_parts.append(str(result))
            info_parts.append("")
        
        # File contents
        if state.get("file_contents"):
            info_parts.append("File Contents:")
            for filename, content in state["file_contents"].items():
                info_parts.append(f"{filename}:")
                info_parts.append(str(content)[:1000])  # Truncate
            info_parts.append("")
        
        # Code outputs
        if state.get("code_outputs"):
            info_parts.append("Calculations:")
            for output in state["code_outputs"]:
                info_parts.append(output)
            info_parts.append("")
        
        # Image analyses
        if state.get("image_analyses"):
            info_parts.append("Image Analysis:")
            for img, analysis in state["image_analyses"].items():
                info_parts.append(f"{img}: {analysis}")
        
        return "\n".join(info_parts)
    
    def run(self, question: str, task_id: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the agent on a GAIA question.
        
        Args:
            question: The GAIA question
            task_id: Task ID from GAIA
            file_path: Optional path to associated file
            
        Returns:
            Dictionary with answer and metadata
        """
        # Initialize state
        initial_state = initialize_state(question, task_id)
        
        # Add file reference if provided
        if file_path:
            initial_state["file_contents"] = {file_path: "pending"}
        
        # Run the graph
        config = RunnableConfig(
            configurable={"thread_id": task_id},
            recursion_limit=50  # Increased from default 25
        )
        
        try:
            final_state = self.app.invoke(initial_state, config)
            
            # Extract results
            result = {
                "task_id": task_id,
                "question": question,
                "answer": final_state.get("formatted_answer", ""),
                "final_answer": final_state.get("formatted_answer", ""),  # For compatibility
                "raw_answer": final_state.get("raw_answer", ""),
                "validation": final_state.get("validation_status", {}),
                "steps_used": final_state.get("step_count", 0),
                "reasoning_chain": final_state.get("reasoning_steps", []),
                "step_count": final_state.get("step_count", 0),  # For compatibility
                "reasoning_steps": final_state.get("reasoning_steps", []),  # For compatibility
                "validation_status": final_state.get("validation_status", {}),  # For compatibility
                "execution_time": (
                    final_state.get("end_time", datetime.now()) - 
                    final_state["start_time"]
                ).total_seconds() if final_state.get("end_time") else None
            }
            
            return result
            
        except Exception as e:
            return {
                "task_id": task_id,
                "question": question,
                "answer": "",
                "error": str(e),
                "validation": {"is_valid": False, "errors": [str(e)]}
            }


# Example usage
if __name__ == "__main__":
    # Initialize agent
    agent = GaiaAgent(
        model_name="gpt-4-1106-preview",
        temperature=0.1
    )
    
    # Test question
    test_question = "What was the actual enrollment count of the clinical trial on H. pylori in acne vulgaris patients from Jan-May 2018 as listed on the NIH website?"
    
    # Run agent
    result = agent.run(
        question=test_question,
        task_id="test_001"
    )
    
    print(json.dumps(result, indent=2))