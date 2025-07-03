import os
from dotenv import load_dotenv
load_dotenv('.env.local')

from gaia_submission import GaiaSubmissionClient

def check_questions():
    client = GaiaSubmissionClient()
    questions = client.get_questions()
    
    if questions:
        print(f"Total questions available: {len(questions)}")
        
        # Count by level
        by_level = {}
        for q in questions:
            level = q.get('level', 'Unknown')
            by_level[level] = by_level.get(level, 0) + 1
        
        print("\nQuestions by level:")
        for level in sorted(by_level.keys()):
            print(f"  Level {level}: {by_level[level]}")
    else:
        print("Failed to fetch questions")

if __name__ == "__main__":
    check_questions()
