"""Create Final GAIA Submission using the Working Agent"""

import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any
import time

from dotenv import load_dotenv
load_dotenv('.env')

from gaia_agent import GaiaAgent
from gaia_submission import GaiaSubmissionClient
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


def format_answer_for_gaia(raw_answer: str, question: str, reasoning_trace: str) -> str:
    """Format answer using the official GAIA system prompt"""
    
    # Official GAIA system prompt
    GAIA_SYSTEM_PROMPT = """You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."""
    
    # If the answer already looks properly formatted, use it
    if is_properly_formatted(raw_answer):
        return raw_answer
    
    # Otherwise, reformat using GAIA prompt
    try:
        formatter = ChatOpenAI(
            model="gpt-4-1106-preview",
            temperature=0.0,
            max_tokens=500
        )
        
        format_prompt = ChatPromptTemplate.from_messages([
            ("system", GAIA_SYSTEM_PROMPT),
            ("human", f"""Question: {question}

Based on this reasoning and information: {reasoning_trace}

I determined the answer is: {raw_answer}

Please provide the final answer in the correct GAIA format.""")
        ])
        
        result = formatter.invoke(format_prompt.format_messages())
        formatted_response = result.content.strip()
        
        # Extract the final answer
        if "FINAL ANSWER:" in formatted_response:
            answer_part = formatted_response.split("FINAL ANSWER:")[-1].strip()
            return answer_part
        else:
            return formatted_response
            
    except Exception as e:
        print(f"Warning: Could not reformat answer for {question[:50]}...: {e}")
        return raw_answer


def is_properly_formatted(answer: str) -> bool:
    """Check if answer follows GAIA format requirements"""
    answer = answer.strip()
    
    # Check for unwanted prefixes
    unwanted_prefixes = ["the answer is", "final answer:", "answer:", "result:", "the final answer is"]
    if any(answer.lower().startswith(prefix) for prefix in unwanted_prefixes):
        return False
    
    # Check for unwanted units (basic check)
    if any(unit in answer.lower() for unit in ["dollars", "percent", "percentage", " units"]):
        return False
    
    return True


def create_gaia_submission(num_questions: int = 50, output_file: str = None):
    """Create GAIA submission file using the working agent"""
    
    print("🚀 Creating GAIA Submission File")
    print("=" * 60)
    
    # Initialize components
    print("Initializing GAIA Agent (working version)...")
    agent = GaiaAgent(use_checkpointing=False)  # Use working version
    
    print("Initializing GAIA API client...")
    client = GaiaSubmissionClient()
    
    # Get questions
    print(f"\nFetching {num_questions} questions from GAIA API...")
    questions = client.get_questions()
    
    if not questions:
        print("❌ Failed to fetch questions")
        return None
    
    # Sample questions if requested less than total
    if num_questions < len(questions):
        import random
        random.seed(42)  # For reproducibility
        questions = random.sample(questions, num_questions)
    
    print(f"✅ Retrieved {len(questions)} questions")
    
    # Process each question
    submission_entries = []
    successful_count = 0
    
    print("\n" + "="*60)
    print("Processing Questions for Submission")
    print("="*60)
    
    for i, question_data in enumerate(questions, 1):
        task_id = question_data.get('task_id', 'unknown')
        question = question_data.get('question', '')
        level = question_data.get('level', 'Unknown')
        files = question_data.get('files', [])
        
        print(f"\n[{i}/{len(questions)}] Level {level} - Task {task_id}")
        print(f"Q: {question[:100]}{'...' if len(question) > 100 else ''}")
        
        try:
            # Run the working agent
            result = agent.run(question, task_id)
            
            if result:
                raw_answer = result.get('final_answer', 'No answer')
                reasoning_steps = result.get('reasoning_steps', [])
                
                # Create reasoning trace
                reasoning_trace = "\n".join(reasoning_steps) if reasoning_steps else "No detailed reasoning available"
                
                # Format answer for GAIA compliance
                formatted_answer = format_answer_for_gaia(raw_answer, question, reasoning_trace)
                
                # Create submission entry
                submission_entry = {
                    "task_id": task_id,
                    "model_answer": formatted_answer,
                    "reasoning_trace": reasoning_trace
                }
                
                submission_entries.append(submission_entry)
                
                print(f"✅ Answer: {formatted_answer}")
                
                if raw_answer != 'No answer':
                    successful_count += 1
                
            else:
                print("❌ Agent returned no result")
                
                # Still add entry for submission completeness
                submission_entry = {
                    "task_id": task_id,
                    "model_answer": "",
                    "reasoning_trace": "Agent failed to process this question"
                }
                submission_entries.append(submission_entry)
                
        except Exception as e:
            print(f"❌ Error processing question: {e}")
            
            # Add error entry
            submission_entry = {
                "task_id": task_id,
                "model_answer": "",
                "reasoning_trace": f"Error occurred: {str(e)}"
            }
            submission_entries.append(submission_entry)
    
    # Create submission file
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"gaia_final_submission_{timestamp}.jsonl"
    
    print(f"\n💾 Saving submission file: {output_file}")
    
    with open(output_file, 'w') as f:
        for entry in submission_entries:
            f.write(json.dumps(entry) + '\n')
    
    # Validation
    print("🔍 Validating submission file...")
    
    try:
        with open(output_file, 'r') as f:
            lines = f.readlines()
        
        validated_count = 0
        for i, line in enumerate(lines):
            line = line.strip()
            if line:
                entry = json.loads(line)
                required_fields = ["task_id", "model_answer"]
                
                if all(field in entry for field in required_fields):
                    validated_count += 1
                else:
                    print(f"❌ Line {i+1}: Missing required fields")
        
        print(f"✅ Validated {validated_count}/{len(lines)} entries")
        
    except Exception as e:
        print(f"❌ Validation error: {e}")
    
    # Final summary
    print("\n" + "="*60)
    print("📊 SUBMISSION SUMMARY")
    print("="*60)
    print(f"Total questions: {len(questions)}")
    print(f"Successful answers: {successful_count}")
    print(f"Success rate: {(successful_count/len(questions)*100):.1f}%")
    print(f"Submission file: {output_file}")
    print(f"Entries created: {len(submission_entries)}")
    
    print("\n🎯 READY FOR SUBMISSION!")
    print("Next steps:")
    print("1. Review the submission file")
    print("2. Upload to GAIA benchmark platform")
    print("3. Monitor evaluation results")
    
    return output_file


def test_submission_creation():
    """Test submission creation with a small number of questions"""
    
    print("🧪 Testing Submission Creation")
    print("=" * 40)
    
    # Test with 3 questions
    output_file = create_gaia_submission(num_questions=3, output_file="test_submission.jsonl")
    
    if output_file:
        print(f"\n✅ Test completed successfully!")
        print(f"Test file: {output_file}")
        
        # Show sample entries
        print("\n📋 Sample entries:")
        with open(output_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:3]):
                entry = json.loads(line.strip())
                print(f"{i+1}. {entry['task_id']}: {entry['model_answer']}")
    
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create GAIA Submission File")
    parser.add_argument('--num_questions', type=int, default=50,
                       help='Number of questions to process (default: 50)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output filename (default: auto-generated)')
    parser.add_argument('--test', action='store_true',
                       help='Run test mode with 3 questions')
    
    args = parser.parse_args()
    
    if args.test:
        test_submission_creation()
    else:
        create_gaia_submission(args.num_questions, args.output)
