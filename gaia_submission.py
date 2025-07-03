# gaia_submission.py
"""GAIA benchmark submission system with API integration"""

import os
import json
import requests
from typing import List, Dict, Any, Optional
import time
from datetime import datetime
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GaiaSubmissionClient:
    """Client for interacting with GAIA benchmark API"""
    
    def __init__(self, base_url: str = "https://agents-course-unit4-scoring.hf.space"):
        """
        Initialize GAIA API client.
        
        Args:
            base_url: Base URL for GAIA API
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def get_questions(self) -> List[Dict[str, Any]]:
        """
        Retrieve the full list of evaluation questions.
        
        Returns:
            List of question dictionaries
        """
        try:
            response = self.session.get(f"{self.base_url}/questions")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get questions: {e}")
            raise
    
    def get_random_question(self) -> Dict[str, Any]:
        """
        Fetch a single random question.
        
        Returns:
            Question dictionary
        """
        try:
            response = self.session.get(f"{self.base_url}/random-question")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get random question: {e}")
            raise
    
    def download_file(self, task_id: str, save_dir: str = "./files") -> Optional[str]:
        """
        Download file associated with a task.
        
        Args:
            task_id: Task ID
            save_dir: Directory to save files
            
        Returns:
            Path to downloaded file or None
        """
        try:
            response = self.session.get(f"{self.base_url}/files/{task_id}")
            
            if response.status_code == 404:
                logger.info(f"No file for task {task_id}")
                return None
            
            response.raise_for_status()
            
            # Create directory if needed
            os.makedirs(save_dir, exist_ok=True)
            
            # Get filename from headers or use default
            filename = response.headers.get('Content-Disposition', '')
            if 'filename=' in filename:
                filename = filename.split('filename=')[1].strip('"')
            else:
                # Guess extension from content type
                content_type = response.headers.get('Content-Type', '')
                ext_map = {
                    'application/pdf': 'pdf',
                    'image/png': 'png',
                    'image/jpeg': 'jpg',
                    'text/csv': 'csv',
                    'application/vnd.ms-excel': 'xls',
                    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx'
                }
                ext = ext_map.get(content_type, 'bin')
                filename = f"{task_id}.{ext}"
            
            filepath = os.path.join(save_dir, filename)
            
            # Save file
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded file for task {task_id}: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to download file for {task_id}: {e}")
            return None
    
    def submit_answers(self, 
                      username: str, 
                      agent_code: str, 
                      answers: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Submit answers for evaluation.
        
        Args:
            username: Hugging Face username
            agent_code: URL to agent code on HF Space
            answers: List of {"task_id": ..., "submitted_answer": ...}
            
        Returns:
            Submission result with score
        """
        payload = {
            "username": username,
            "agent_code": agent_code,
            "answers": answers
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/submit",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to submit answers: {e}")
            raise


class GaiaRunner:
    """Main runner for GAIA benchmark evaluation"""
    
    def __init__(self, agent, api_client: Optional[GaiaSubmissionClient] = None):
        """
        Initialize GAIA runner.
        
        Args:
            agent: The GAIA agent instance
            api_client: Optional API client (creates default if not provided)
        """
        self.agent = agent
        self.api_client = api_client or GaiaSubmissionClient()
        self.results = []
    
    def run_single_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run agent on a single question.
        
        Args:
            question_data: Question dictionary from API
            
        Returns:
            Result dictionary
        """
        task_id = question_data["task_id"]
        question = question_data["question"]
        
        logger.info(f"Processing task {task_id}")
        
        # Download associated file if exists
        file_path = None
        if question_data.get("has_file", False):
            file_path = self.api_client.download_file(task_id)
        
        # Run agent
        start_time = time.time()
        
        try:
            result = self.agent.run(
                question=question,
                task_id=task_id,
                file_path=file_path
            )
            
            # Add timing
            result["processing_time"] = time.time() - start_time
            
            # Ensure we have the required format
            if result.get("validation", {}).get("is_valid", False):
                logger.info(f"Task {task_id} completed successfully: {result['answer']}")
            else:
                logger.warning(f"Task {task_id} validation failed: {result.get('validation', {})}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}")
            return {
                "task_id": task_id,
                "question": question,
                "answer": "",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def run_all_questions(self, 
                         limit: Optional[int] = None,
                         save_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Run agent on all questions.
        
        Args:
            limit: Optional limit on number of questions
            save_progress: Whether to save progress incrementally
            
        Returns:
            List of results
        """
        # Get questions
        questions = self.api_client.get_questions()
        
        if limit:
            questions = questions[:limit]
        
        logger.info(f"Running on {len(questions)} questions")
        
        # Process each question
        self.results = []
        
        for question_data in tqdm(questions, desc="Processing questions"):
            result = self.run_single_question(question_data)
            self.results.append(result)
            
            # Save progress
            if save_progress:
                self._save_progress()
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
        
        logger.info(f"Completed all questions. Success rate: {self._calculate_success_rate():.1%}")
        
        return self.results
    
    def prepare_submission(self, 
                          username: str, 
                          agent_code: str) -> Dict[str, Any]:
        """
        Prepare submission payload from results.
        
        Args:
            username: Hugging Face username
            agent_code: URL to agent code
            
        Returns:
            Submission payload
        """
        # Extract answers
        answers = []
        
        for result in self.results:
            if result.get("answer"):  # Only include non-empty answers
                answers.append({
                    "task_id": result["task_id"],
                    "submitted_answer": result["answer"]
                })
        
        logger.info(f"Prepared {len(answers)} answers for submission")
        
        return {
            "username": username,
            "agent_code": agent_code,
            "answers": answers
        }
    
    def submit_results(self, username: str, agent_code: str) -> Dict[str, Any]:
        """
        Submit results to GAIA leaderboard.
        
        Args:
            username: Hugging Face username
            agent_code: URL to agent code
            
        Returns:
            Submission response
        """
        submission = self.prepare_submission(username, agent_code)
        
        logger.info(f"Submitting {len(submission['answers'])} answers...")
        
        response = self.api_client.submit_answers(
            username=submission["username"],
            agent_code=submission["agent_code"],
            answers=submission["answers"]
        )
        
        logger.info(f"Submission complete. Score: {response.get('score', 'N/A')}")
        
        return response
    
    def _calculate_success_rate(self) -> float:
        """Calculate success rate from results"""
        if not self.results:
            return 0.0
        
        valid_count = sum(
            1 for r in self.results 
            if r.get("validation", {}).get("is_valid", False)
        )
        
        return valid_count / len(self.results)
    
    def _save_progress(self):
        """Save current progress to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gaia_progress_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "results": self.results,
                "success_rate": self._calculate_success_rate()
            }, f, indent=2)
    
    def load_progress(self, filename: str):
        """Load progress from file"""
        with open(filename, 'r') as f:
            data = json.load(f)
            self.results = data["results"]
            logger.info(f"Loaded {len(self.results)} results from {filename}")


# Utility functions for testing and analysis
def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze results for insights.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Analysis dictionary
    """
    analysis = {
        "total_questions": len(results),
        "answered": sum(1 for r in results if r.get("answer")),
        "valid_format": sum(1 for r in results if r.get("validation", {}).get("is_valid", False)),
        "errors": sum(1 for r in results if r.get("error")),
        "avg_time": sum(r.get("processing_time", 0) for r in results) / len(results) if results else 0,
        "avg_steps": sum(r.get("steps_used", 0) for r in results) / len(results) if results else 0
    }
    
    # Question type breakdown
    type_counts = {}
    for r in results:
        q_type = detect_answer_type(r.get("question", ""), r.get("answer", ""))
        type_counts[q_type] = type_counts.get(q_type, 0) + 1
    
    analysis["question_types"] = type_counts
    
    # Common validation errors
    validation_errors = []
    for r in results:
        if not r.get("validation", {}).get("is_valid", False):
            errors = r.get("validation", {}).get("errors", [])
            validation_errors.extend(errors)
    
    analysis["common_validation_errors"] = validation_errors[:10]  # Top 10
    
    return analysis


# Example usage
if __name__ == "__main__":
    from gaia_agent import GaiaAgent
    from gaia_formatting import detect_answer_type
    
    # Initialize components
    agent = GaiaAgent(
        model_name="gpt-4-1106-preview",
        temperature=0.1
    )
    
    runner = GaiaRunner(agent)
    
    # Test on a single random question
    print("Testing on random question...")
    question = runner.api_client.get_random_question()
    result = runner.run_single_question(question)
    
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Valid: {result.get('validation', {}).get('is_valid', False)}")
    print(f"Time: {result.get('processing_time', 0):.1f}s")
    
    # For full evaluation (commented out to avoid accidental runs)
    # results = runner.run_all_questions(limit=5)  # Test on 5 questions
    # analysis = analyze_results(results)
    # print(json.dumps(analysis, indent=2))
    
    # For submission (requires valid username and code URL)
    # response = runner.submit_results(
    #     username="your-hf-username",
    #     agent_code="https://huggingface.co/spaces/your-username/your-space/tree/main"
    # )