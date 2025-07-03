# gaia_formatting.py
"""Answer formatting and validation for GAIA benchmark compliance"""

import re
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import calendar


def detect_answer_type(question: str, raw_answer: str) -> str:
    """
    Detect the expected answer type from question and initial answer.
    
    Returns: "number", "string", "list", "date", or "unknown"
    """
    question_lower = question.lower()
    
    # Number indicators
    number_patterns = [
        r'\bhow many\b', r'\bcount\b', r'\bnumber of\b', r'\btotal\b',
        r'\bsum\b', r'\bpercentage\b', r'\bpercent\b', r'\b%\b',
        r'\bdollars?\b', r'\$', r'\bamount\b', r'\bprice\b',
        r'\byear\b(?! of)', r'\bage\b', r'\bscore\b', r'\bminutes?\b'
    ]
    
    # List indicators
    list_patterns = [
        r'\blist\b', r'\bname all\b', r'\bcomma[\s-]separated\b',
        r'\band\b.*\band\b.*\band\b', r'\bwhich\s+\w+s\b'
    ]
    
    # Date indicators
    date_patterns = [
        r'\bwhen\b', r'\bdate\b', r'\bmonth\b', r'\byear of\b',
        r'\bMM/DD/YY\b', r'\bDD/MM/YYYY\b'
    ]
    
    # Check patterns
    if any(re.search(pattern, question_lower) for pattern in number_patterns):
        # Verify answer looks like a number
        if re.match(r'^-?\d+\.?\d*$', raw_answer.strip().replace(',', '')):
            return "number"
    
    if any(re.search(pattern, question_lower) for pattern in list_patterns):
        # Check if answer contains commas
        if ',' in raw_answer:
            return "list"
    
    if any(re.search(pattern, question_lower) for pattern in date_patterns):
        return "date"
    
    # Default to string for single values, list for multiple
    if ',' in raw_answer and raw_answer.count(',') >= 1:
        return "list"
    
    return "string"


def format_number_answer(raw_answer: str, question: str) -> str:
    """
    Format numeric answers according to GAIA requirements.
    
    Rules:
    - No commas in numbers unless specifically requested
    - No units unless specifically requested
    - No decimal places unless in original answer
    """
    # Clean the answer
    answer = raw_answer.strip()
    
    # Remove common units unless specified in question
    units_to_remove = ['$', '€', '£', '%', 'percent', 'dollars', 'euros', 'pounds']
    question_lower = question.lower()
    
    for unit in units_to_remove:
        if unit not in question_lower and 'percent' not in question_lower:
            answer = answer.replace(unit, '').strip()
    
    # Remove commas
    answer = answer.replace(',', '')
    
    # Handle decimal places
    try:
        if '.' in answer:
            # Keep decimal places
            num = float(answer)
            # Remove trailing zeros
            answer = f"{num:g}"
        else:
            # Integer
            num = int(float(answer))
            answer = str(num)
    except ValueError:
        # If conversion fails, return cleaned string
        pass
    
    return answer


def format_string_answer(raw_answer: str, question: str) -> str:
    """
    Format string answers according to GAIA requirements.
    
    Rules:
    - No articles (a, an, the) unless part of proper noun
    - No abbreviations unless specified
    - Minimal words
    - Proper capitalization
    """
    answer = raw_answer.strip()
    
    # Remove leading/trailing quotes
    answer = answer.strip('"\'')
    
    # Remove articles at the beginning
    articles = ['the ', 'a ', 'an ']
    answer_lower = answer.lower()
    for article in articles:
        if answer_lower.startswith(article):
            # Check if it's part of a proper noun
            if not answer[len(article)].isupper():
                answer = answer[len(article):]
                break
    
    # Handle common abbreviations
    if 'abbreviation' not in question.lower():
        abbreviations = {
            'St.': 'Saint',
            'Mt.': 'Mount',
            'Dr.': 'Doctor',
            'Mr.': 'Mister',
            'Mrs.': 'Missus',
            'Ms.': 'Miss'
        }
        for abbr, full in abbreviations.items():
            answer = answer.replace(abbr, full)
    
    # Remove trailing punctuation
    answer = answer.rstrip('.,!?;:')
    
    return answer


def format_list_answer(raw_answer: str, question: str) -> str:
    """
    Format list answers according to GAIA requirements.
    
    Rules:
    - Comma-separated
    - Consistent formatting for each item
    - Alphabetical order if specified
    """
    # Split by various delimiters
    items = re.split(r'[,;]|\band\b', raw_answer)
    items = [item.strip() for item in items if item.strip()]
    
    # Format each item
    formatted_items = []
    for item in items:
        # Apply string formatting to each item
        formatted_item = format_string_answer(item, question)
        if formatted_item:
            formatted_items.append(formatted_item)
    
    # Check if alphabetical order is requested
    if 'alphabetical' in question.lower() or 'alphabetically' in question.lower():
        formatted_items.sort(key=lambda x: x.lower())
    
    # Join with comma and space
    return ', '.join(formatted_items)


def format_date_answer(raw_answer: str, question: str) -> str:
    """
    Format date answers according to specified format in question.
    
    Common formats:
    - MM/DD/YY
    - DD/MM/YYYY
    - Month DD, YYYY
    - YYYY-MM-DD
    """
    import dateutil.parser
    
    # Try to parse the date
    try:
        date_obj = dateutil.parser.parse(raw_answer)
    except:
        # If parsing fails, return as is
        return raw_answer.strip()
    
    # Look for format specification in question
    question_lower = question.lower()
    
    if 'MM/DD/YY' in question:
        return date_obj.strftime('%m/%d/%y')
    elif 'DD/MM/YYYY' in question:
        return date_obj.strftime('%d/%m/%Y')
    elif 'MM/DD/YYYY' in question:
        return date_obj.strftime('%m/%d/%Y')
    elif 'YYYY-MM-DD' in question:
        return date_obj.strftime('%Y-%m-%d')
    elif 'Month DD, YYYY' in question:
        return date_obj.strftime('%B %d, %Y')
    elif 'month day year' in question_lower:
        return date_obj.strftime('%B %d %Y')
    else:
        # Default format - check context
        if any(word in question_lower for word in ['american', 'us', 'usa']):
            return date_obj.strftime('%m/%d/%Y')
        else:
            return date_obj.strftime('%d/%m/%Y')


def format_gaia_answer(raw_answer: str, question: str, answer_type: Optional[str] = None) -> str:
    """
    Main formatting function that applies all GAIA formatting rules.
    
    Args:
        raw_answer: The unformatted answer
        question: The original question (for context)
        answer_type: Optional type hint ("number", "string", "list", "date")
        
    Returns:
        Formatted answer ready for submission
    """
    if not raw_answer:
        return ""
    
    # Remove any "FINAL ANSWER:" prefix (critical!)
    raw_answer = re.sub(r'^FINAL ANSWER:\s*', '', raw_answer, flags=re.IGNORECASE)
    raw_answer = raw_answer.strip()
    
    # Detect type if not provided
    if not answer_type:
        answer_type = detect_answer_type(question, raw_answer)
    
    # Apply type-specific formatting
    if answer_type == "number":
        formatted = format_number_answer(raw_answer, question)
    elif answer_type == "list":
        formatted = format_list_answer(raw_answer, question)
    elif answer_type == "date":
        formatted = format_date_answer(raw_answer, question)
    else:
        formatted = format_string_answer(raw_answer, question)
    
    return formatted


def validate_answer_format(answer: str, question: str) -> Dict[str, any]:
    """
    Validate that answer meets all GAIA format requirements.
    
    Returns:
        Dictionary with validation results and any error messages
    """
    validation = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "checks": {}
    }
    
    # Check 1: No "FINAL ANSWER" prefix
    if "FINAL ANSWER" in answer.upper():
        validation["is_valid"] = False
        validation["errors"].append("Answer contains 'FINAL ANSWER' prefix")
        validation["checks"]["no_prefix"] = False
    else:
        validation["checks"]["no_prefix"] = True
    
    # Check 2: No trailing punctuation (unless it's part of the answer)
    if answer and answer[-1] in '.!?;:' and 'punctuation' not in question.lower():
        validation["warnings"].append("Answer has trailing punctuation")
        validation["checks"]["no_trailing_punctuation"] = False
    else:
        validation["checks"]["no_trailing_punctuation"] = True
    
    # Check 3: Number format validation
    if detect_answer_type(question, answer) == "number":
        if ',' in answer and 'comma' not in question.lower():
            validation["is_valid"] = False
            validation["errors"].append("Number contains comma")
            validation["checks"]["number_format"] = False
        elif '$' in answer and 'dollar' not in question.lower() and '$' not in question:
            validation["warnings"].append("Number contains $ symbol")
            validation["checks"]["number_format"] = False
        else:
            validation["checks"]["number_format"] = True
    
    # Check 4: List format validation
    if detect_answer_type(question, answer) == "list":
        if ', ' not in answer and ',' in answer:
            validation["warnings"].append("List items not properly spaced")
            validation["checks"]["list_format"] = False
        else:
            validation["checks"]["list_format"] = True
    
    # Check 5: Empty answer
    if not answer or answer.isspace():
        validation["is_valid"] = False
        validation["errors"].append("Answer is empty")
        validation["checks"]["not_empty"] = False
    else:
        validation["checks"]["not_empty"] = True
    
    # Check 6: Excessive length
    if len(answer) > 100 and 'essay' not in question.lower():
        validation["warnings"].append(f"Answer seems long ({len(answer)} chars)")
        validation["checks"]["reasonable_length"] = False
    else:
        validation["checks"]["reasonable_length"] = True
    
    return validation


def fix_common_formatting_errors(answer: str, question: str, validation_result: Dict) -> str:
    """
    Attempt to fix common formatting errors based on validation results.
    
    Args:
        answer: The answer with potential formatting issues
        question: The original question
        validation_result: Results from validate_answer_format
        
    Returns:
        Fixed answer
    """
    fixed = answer
    
    # Fix "FINAL ANSWER" prefix
    if not validation_result["checks"].get("no_prefix", True):
        fixed = re.sub(r'^FINAL ANSWER:\s*', '', fixed, flags=re.IGNORECASE)
    
    # Fix number formatting
    if not validation_result["checks"].get("number_format", True):
        fixed = fixed.replace(',', '').replace('$', '').strip()
    
    # Fix trailing punctuation
    if not validation_result["checks"].get("no_trailing_punctuation", True):
        fixed = fixed.rstrip('.!?;:')
    
    # Fix list spacing
    if not validation_result["checks"].get("list_format", True):
        # Ensure proper spacing after commas
        fixed = re.sub(r',(?!\s)', ', ', fixed)
    
    return fixed.strip()


# Example usage and test function
def test_formatting():
    """Test the formatting functions with example cases"""
    test_cases = [
        {
            "question": "What is the population of Tokyo?",
            "raw_answer": "FINAL ANSWER: 13,960,000",
            "expected_type": "number",
            "expected": "13960000"
        },
        {
            "question": "List the primary colors in alphabetical order",
            "raw_answer": "red, blue, and yellow",
            "expected_type": "list",
            "expected": "blue, red, yellow"
        },
        {
            "question": "When was the Declaration of Independence signed? Format: MM/DD/YYYY",
            "raw_answer": "July 4, 1776",
            "expected_type": "date",
            "expected": "07/04/1776"
        },
        {
            "question": "What is the capital of France?",
            "raw_answer": "The city of Paris.",
            "expected_type": "string",
            "expected": "Paris"
        }
    ]
    
    for test in test_cases:
        formatted = format_gaia_answer(
            test["raw_answer"], 
            test["question"], 
            test["expected_type"]
        )
        validation = validate_answer_format(formatted, test["question"])
        
        print(f"Question: {test['question']}")
        print(f"Raw: {test['raw_answer']}")
        print(f"Formatted: {formatted}")
        print(f"Expected: {test['expected']}")
        print(f"Valid: {validation['is_valid']}")
        print(f"Match: {formatted == test['expected']}")
        print("-" * 50)


if __name__ == "__main__":
    test_formatting()