import gradio as gr
import os
import json
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime
import logging

# Import our modules
from gaia_agent import GaiaAgent
from gaia_submission import GaiaRunner, GaiaSubmissionClient, analyze_results
from gaia_formatting import validate_answer_format

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
agent = None
runner = None
current_results = []


def initialize_agent(api_key: str, model_name: str = "gpt-4-1106-preview") -> str:
    """Initialize the GAIA agent with API key"""
    global agent, runner
    
    try:
        if not api_key:
            return "❌ Please provide an OpenAI API key"
        
        # Set environment variable
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Initialize agent
        agent = GaiaAgent(
            openai_api_key=api_key,
            model_name=model_name,
            temperature=0.1,
            use_checkpointing=True
        )
        
        # Initialize runner
        runner = GaiaRunner(agent)
        
        return f"✅ Agent initialized successfully with model: {model_name}"
        
    except Exception as e:
        return f"❌ Initialization failed: {str(e)}"


def test_single_question(question_text: str, task_id: str = "manual_test") -> Dict[str, Any]:
    """Test agent on a single manually entered question"""
    global agent
    
    if agent is None:
        return {
            "error": "Agent not initialized. Please provide API key first.",
            "answer": "",
            "reasoning": "",
            "validation": {}
        }
    
    try:
        # Run agent
        result = agent.run(
            question=question_text,
            task_id=task_id
        )
        
        return {
            "answer": result.get("answer", ""),
            "raw_answer": result.get("raw_answer", ""),
            "reasoning": "\n".join(result.get("reasoning_chain", [])),
            "validation": result.get("validation", {}),
            "steps": result.get("steps_used", 0),
            "time": result.get("execution_time", 0)
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "answer": "",
            "reasoning": "",
            "validation": {"is_valid": False, "errors": [str(e)]}
        }


def run_random_question() -> Dict[str, Any]:
    """Run agent on a random GAIA question"""
    global runner
    
    if runner is None:
        return {"error": "Runner not initialized. Please provide API key first."}
    
    try:
        # Get random question
        question_data = runner.api_client.get_random_question()
        
        # Run agent
        result = runner.run_single_question(question_data)
        
        return {
            "task_id": result["task_id"],
            "question": result["question"],
            "answer": result.get("answer", ""),
            "reasoning": "\n".join(result.get("reasoning_chain", [])),
            "validation": result.get("validation", {}),
            "steps": result.get("steps_used", 0),
            "time": result.get("processing_time", 0)
        }
        
    except Exception as e:
        return {"error": str(e)}


def run_evaluation(num_questions: int, save_progress: bool = True) -> pd.DataFrame:
    """Run evaluation on multiple questions"""
    global runner, current_results
    
    if runner is None:
        return pd.DataFrame({"Error": ["Runner not initialized"]})
    
    try:
        # Run evaluation
        results = runner.run_all_questions(
            limit=num_questions,
            save_progress=save_progress
        )
        
        current_results = results
        
        # Convert to DataFrame for display
        df_data = []
        for r in results:
            df_data.append({
                "Task ID": r["task_id"],
                "Question": r["question"][:50] + "..." if len(r["question"]) > 50 else r["question"],
                "Answer": r.get("answer", ""),
                "Valid": "✅" if r.get("validation", {}).get("is_valid", False) else "❌",
                "Steps": r.get("steps_used", 0),
                "Time (s)": f"{r.get('processing_time', 0):.1f}"
            })
        
        return pd.DataFrame(df_data)
        
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})


def get_evaluation_summary() -> str:
    """Get summary of current evaluation results"""
    global current_results
    
    if not current_results:
        return "No evaluation results available. Run evaluation first."
    
    analysis = analyze_results(current_results)
    
    summary = f"""
    📊 **Evaluation Summary**
    
    - Total Questions: {analysis['total_questions']}
    - Questions Answered: {analysis['answered']} ({analysis['answered']/analysis['total_questions']*100:.1f}%)
    - Valid Format: {analysis['valid_format']} ({analysis['valid_format']/analysis['total_questions']*100:.1f}%)
    - Errors: {analysis['errors']}
    
    **Performance Metrics:**
    - Average Time per Question: {analysis['avg_time']:.1f}s
    - Average Steps per Question: {analysis['avg_steps']:.1f}
    
    **Question Type Distribution:**
    {json.dumps(analysis['question_types'], indent=2)}
    """
    
    if analysis['common_validation_errors']:
        summary += f"\n\n**Common Validation Errors:**\n"
        for error in analysis['common_validation_errors'][:5]:
            summary += f"- {error}\n"
    
    return summary


def submit_to_leaderboard(username: str, space_url: str) -> str:
    """Submit results to GAIA leaderboard"""
    global runner, current_results
    
    if not username or not space_url:
        return "❌ Please provide both username and space URL"
    
    if not current_results:
        return "❌ No results to submit. Run evaluation first."
    
    try:
        # Ensure space URL is in correct format
        if not space_url.endswith("/tree/main"):
            space_url = space_url.rstrip("/") + "/tree/main"
        
        # Submit
        response = runner.submit_results(
            username=username,
            agent_code=space_url
        )
        
        score = response.get("score", "N/A")
        message = response.get("message", "Submission completed")
        
        return f"""
        ✅ **Submission Successful!**
        
        Score: {score}
        Message: {message}
        
        Check the leaderboard at: https://huggingface.co/spaces/gaia-benchmark/leaderboard
        """
        
    except Exception as e:
        return f"❌ Submission failed: {str(e)}"


# Create Gradio interface
def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="GAIA Benchmark Agent") as demo:
        gr.Markdown("""
        # 🤖 GAIA Benchmark Agent
        
        This is a LangGraph-based agent for the GAIA (General AI Assistants) benchmark.
        
        ## 🚀 Getting Started
        1. Enter your OpenAI API key below
        2. Test on individual questions or run full evaluation
        3. Submit your results to the leaderboard
        """)
        
        with gr.Tab("Setup"):
            with gr.Row():
                api_key_input = gr.Textbox(
                    label="OpenAI API Key",
                    placeholder="sk-...",
                    type="password"
                )
                model_dropdown = gr.Dropdown(
                    choices=["gpt-4-1106-preview", "gpt-4o", "gpt-4-turbo"],
                    value="gpt-4-1106-preview",
                    label="Model"
                )
            
            init_button = gr.Button("Initialize Agent", variant="primary")
            init_status = gr.Textbox(label="Status", interactive=False)
            
            init_button.click(
                fn=initialize_agent,
                inputs=[api_key_input, model_dropdown],
                outputs=init_status
            )
        
        with gr.Tab("Test Single Question"):
            gr.Markdown("### Test the agent on individual questions")
            
            with gr.Row():
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="Enter a GAIA-style question...",
                    lines=3
                )
            
            test_button = gr.Button("Run Agent", variant="primary")
            
            with gr.Row():
                answer_output = gr.Textbox(label="Answer", interactive=False)
                raw_answer_output = gr.Textbox(label="Raw Answer", interactive=False)
            
            with gr.Row():
                validation_output = gr.JSON(label="Validation")
                metrics_output = gr.JSON(label="Metrics")
            
            reasoning_output = gr.Textbox(
                label="Reasoning Chain",
                lines=10,
                interactive=False
            )
            
            def run_test(question):
                result = test_single_question(question)
                return (
                    result.get("answer", ""),
                    result.get("raw_answer", ""),
                    result.get("validation", {}),
                    {"steps": result.get("steps", 0), "time": result.get("time", 0)},
                    result.get("reasoning", "")
                )
            
            test_button.click(
                fn=run_test,
                inputs=question_input,
                outputs=[
                    answer_output,
                    raw_answer_output,
                    validation_output,
                    metrics_output,
                    reasoning_output
                ]
            )
        
        with gr.Tab("Random Question"):
            gr.Markdown("### Test on a random GAIA question from the benchmark")
            
            random_button = gr.Button("Get Random Question", variant="primary")
            
            random_task_id = gr.Textbox(label="Task ID", interactive=False)
            random_question = gr.Textbox(label="Question", lines=3, interactive=False)
            random_answer = gr.Textbox(label="Answer", interactive=False)
            random_validation = gr.JSON(label="Validation")
            random_reasoning = gr.Textbox(label="Reasoning", lines=10, interactive=False)
            
            def run_random():
                result = run_random_question()
                if "error" in result:
                    return result["error"], "", "", {}, ""
                return (
                    result.get("task_id", ""),
                    result.get("question", ""),
                    result.get("answer", ""),
                    result.get("validation", {}),
                    result.get("reasoning", "")
                )
            
            random_button.click(
                fn=run_random,
                outputs=[
                    random_task_id,
                    random_question,
                    random_answer,
                    random_validation,
                    random_reasoning
                ]
            )
        
        with gr.Tab("Full Evaluation"):
            gr.Markdown("### Run evaluation on multiple questions")
            
            with gr.Row():
                num_questions = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Number of Questions"
                )
                save_progress = gr.Checkbox(
                    value=True,
                    label="Save Progress"
                )
            
            eval_button = gr.Button("Run Evaluation", variant="primary")
            
            results_table = gr.DataFrame(
                label="Results",
                interactive=False
            )
            
            summary_output = gr.Markdown(label="Summary")
            
            eval_button.click(
                fn=run_evaluation,
                inputs=[num_questions, save_progress],
                outputs=results_table
            ).then(
                fn=get_evaluation_summary,
                outputs=summary_output
            )
        
        with gr.Tab("Submit to Leaderboard"):
            gr.Markdown("""
            ### Submit your results to the GAIA leaderboard
            
            Make sure you have:
            1. Run the full evaluation (all 20 questions)
            2. Your Hugging Face username
            3. The URL to this Space's code (should end with /tree/main)
            """)
            
            with gr.Row():
                username_input = gr.Textbox(
                    label="Hugging Face Username",
                    placeholder="your-username"
                )
                space_url_input = gr.Textbox(
                    label="Space Code URL",
                    placeholder="https://huggingface.co/spaces/your-username/your-space/tree/main"
                )
            
            submit_button = gr.Button("Submit Results", variant="primary")
            submission_result = gr.Markdown(label="Submission Result")
            
            submit_button.click(
                fn=submit_to_leaderboard,
                inputs=[username_input, space_url_input],
                outputs=submission_result
            )
        
        with gr.Tab("About"):
            gr.Markdown("""
            ## About This Agent
            
            This GAIA benchmark agent is built with:
            - **LangGraph** for multi-step reasoning workflows
            - **OpenAI GPT-4** for language understanding
            - **Custom tools** for web search, file processing, and more
            - **Strict formatting** to meet GAIA's exact match requirements
            
            ### Key Features:
            - ✅ Adaptive reasoning with step tracking
            - ✅ Multi-modal support (text, images, PDFs, spreadsheets)
            - ✅ Web search and browsing capabilities
            - ✅ Automatic answer formatting and validation
            - ✅ Error recovery and retry mechanisms
            
            ### GAIA Benchmark
            GAIA tests AI assistants on real-world tasks that require:
            - Information retrieval from the web
            - Multi-step reasoning
            - File analysis and data processing
            - Precise answer formatting
            
            Human performance: ~92%
            Current AI performance: ~15-75%
            
            ### Resources
            - [GAIA Paper](https://arxiv.org/abs/2311.12983)
            - [Official Leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard)
            - [Agents Course](https://huggingface.co/learn/agents-course)
            """)
    
    return demo


# Launch the app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()