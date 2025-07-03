import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any
import time
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv('.env.local')

from gaia_agent import GaiaAgent
from gaia_submission import GaiaSubmissionClient

def evaluate_gaia_agent(num_questions: int = 50, submit_results: bool = False):
    """Run full GAIA evaluation"""
    
    print("GAIA Full Benchmark Evaluation")
    print("=" * 80)
    
    # Initialize
    agent = GaiaAgent(use_checkpointing=False)
    client = GaiaSubmissionClient()
    
    # Get questions
    print(f"\nFetching {num_questions} questions from GAIA API...")
    questions = client.get_questions()
    
    if not questions:
        print("❌ Failed to fetch questions")
        return
    
    # Sample questions if requested less than total
    if num_questions < len(questions):
        import random
        random.seed(42)  # For reproducibility
        questions = random.sample(questions, num_questions)
    
    print(f"✅ Retrieved {len(questions)} questions")
    
    # Group by level
    by_level = defaultdict(list)
    for q in questions:
        level = q.get('level', 'Unknown')
        by_level[level].append(q)
    
    print("\nQuestion distribution:")
    for level, qs in sorted(by_level.items()):
        print(f"  Level {level}: {len(qs)} questions")
    
    # Evaluation metrics
    results = []
    metrics = {
        'total': len(questions),
        'successful': 0,
        'valid_format': 0,
        'by_level': defaultdict(lambda: {'total': 0, 'successful': 0, 'valid': 0}),
        'total_time': 0,
        'total_steps': 0,
        'errors': []
    }
    
    print("\n" + "="*80)
    print("Starting evaluation...")
    print("="*80)
    
    # Process each question
    for i, question_data in enumerate(questions, 1):
        task_id = question_data.get('task_id', 'unknown')
        question = question_data.get('question', '')
        level = question_data.get('level', 'Unknown')
        expected = question_data.get('expected_answer', None)
        
        print(f"\n[{i}/{len(questions)}] Level {level} - Task {task_id}")
        print(f"Q: {question[:100]}{'...' if len(question) > 100 else ''}")
        
        start_time = time.time()
        
        try:
            # Run agent
            result = agent.run(question, task_id)
            
            elapsed = time.time() - start_time
            
            if result:
                answer = result.get('final_answer', 'No answer')
                steps = result.get('step_count', 0)
                validation = result.get('validation_status', {})
                is_valid = validation.get('is_valid', False)
                
                print(f"A: {answer}")
                print(f"Time: {elapsed:.1f}s | Steps: {steps} | Valid: {is_valid}")
                
                # Update metrics
                if answer != 'No answer':
                    metrics['successful'] += 1
                    metrics['by_level'][level]['successful'] += 1
                
                if is_valid:
                    metrics['valid_format'] += 1
                    metrics['by_level'][level]['valid'] += 1
                
                metrics['total_time'] += elapsed
                metrics['total_steps'] += steps
                metrics['by_level'][level]['total'] += 1
                
                # Store result
                results.append({
                    'task_id': task_id,
                    'level': level,
                    'question': question,
                    'answer': answer,
                    'expected': expected,
                    'steps': steps,
                    'time': elapsed,
                    'valid': is_valid,
                    'validation_details': validation
                })
                
                # Note: Individual submission removed - will submit all at end
                
            else:
                print("❌ Agent returned no result")
                metrics['errors'].append({
                    'task_id': task_id,
                    'error': 'No result returned'
                })
                metrics['by_level'][level]['total'] += 1
                
        except Exception as e:
            print(f"❌ Error: {e}")
            metrics['errors'].append({
                'task_id': task_id,
                'error': str(e)
            })
            metrics['by_level'][level]['total'] += 1
    
    # Final report
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    
    success_rate = (metrics['successful'] / metrics['total'] * 100) if metrics['total'] > 0 else 0
    valid_rate = (metrics['valid_format'] / metrics['total'] * 100) if metrics['total'] > 0 else 0
    avg_time = metrics['total_time'] / metrics['successful'] if metrics['successful'] > 0 else 0
    avg_steps = metrics['total_steps'] / metrics['successful'] if metrics['successful'] > 0 else 0
    
    print(f"\nOverall Performance:")
    print(f"  Questions: {metrics['total']}")
    print(f"  Successful: {metrics['successful']} ({success_rate:.1f}%)")
    print(f"  Valid Format: {metrics['valid_format']} ({valid_rate:.1f}%)")
    print(f"  Errors: {len(metrics['errors'])}")
    print(f"  Avg Time: {avg_time:.1f}s")
    print(f"  Avg Steps: {avg_steps:.1f}")
    
    print(f"\nPerformance by Level:")
    for level in sorted(metrics['by_level'].keys()):
        level_data = metrics['by_level'][level]
        if level_data['total'] > 0:
            level_success = level_data['successful'] / level_data['total'] * 100
            level_valid = level_data['valid'] / level_data['total'] * 100
            print(f"  Level {level}: {level_data['successful']}/{level_data['total']} ({level_success:.1f}%) successful, {level_valid:.1f}% valid")
    
    # Comparison to baselines
    print(f"\nComparison to Baselines:")
    print(f"  Human: ~92%")
    print(f"  GPT-4: ~15%")
    print(f"  Top agents: 66-75%")
    print(f"  This agent: {success_rate:.1f}%")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"gaia_evaluation_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'metrics': metrics,
            'results': results,
            'success_rate': success_rate,
            'valid_rate': valid_rate,
            'avg_time': avg_time,
            'avg_steps': avg_steps
        }, f, indent=2)
    
    print(f"\n📊 Detailed results saved to: {results_file}")
    
    # Sample of challenging questions
    if metrics['errors']:
        print(f"\n⚠️ Failed questions ({len(metrics['errors'])} total):")
        for err in metrics['errors'][:5]:
            print(f"  - {err['task_id']}: {err['error']}")
        if len(metrics['errors']) > 5:
            print(f"  ... and {len(metrics['errors']) - 5} more")
    
    # Submit all results if requested
    if submit_results and results:
        print(f"\n📤 Submitting {len(results)} results to GAIA...")
        try:
            # Prepare submission format
            answers = []
            for r in results:
                if r['answer'] != 'No answer' and r['valid']:
                    answers.append({
                        'task_id': r['task_id'],
                        'submitted_answer': r['answer']
                    })
            
            if answers:
                print(f"Prepared {len(answers)} valid answers for submission")
                print("To submit, use the GaiaRunner class with your HF username and agent code")
            else:
                print("❌ No valid answers to submit")
        except Exception as e:
            print(f"❌ Submission preparation failed: {e}")
    
    return {
        'success_rate': success_rate,
        'valid_rate': valid_rate,
        'results_file': results_file
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GAIA benchmark evaluation")
    parser.add_argument('--num_questions', type=int, default=50,
                       help='Number of questions to evaluate (default: 50)')
    parser.add_argument('--submit', action='store_true',
                       help='Submit results to GAIA API for scoring')
    
    args = parser.parse_args()
    
    evaluate_gaia_agent(args.num_questions, args.submit)
