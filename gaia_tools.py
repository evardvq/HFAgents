
"""Tool implementations for GAIA agent"""

import os
import json
import base64
import requests
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
from bs4 import BeautifulSoup
import PyPDF2
from PIL import Image
import io

from langchain.tools import tool
from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI
import markdownify


@tool
def web_search(query: str, max_results: int = 3) -> str:
    """
    Search the web for current information.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        JSON string with search results
    """
    try:
        search = TavilySearchResults(max_results=max_results)
        results = search.run(query)
        
        # Parse and format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("snippet", ""),
                "date": result.get("date", "")
            })
        
        return json.dumps(formatted_results, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "query": query})


@tool
def browse_webpage(url: str) -> str:
    """
    Browse a specific webpage and extract its content as markdown.
    
    Args:
        url: The URL to browse
        
    Returns:
        Markdown formatted content of the webpage
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
        
        # Convert to markdown for better token efficiency
        markdown_content = markdownify.markdownify(
            str(soup),
            heading_style="ATX",
            bullets="-"
        )
        
        # Truncate if too long
        max_chars = 5000
        if len(markdown_content) > max_chars:
            markdown_content = markdown_content[:max_chars] + "\n... [Content truncated]"
        
        return f"# Content from {url}\n\n{markdown_content}"
        
    except Exception as e:
        return f"Error browsing {url}: {str(e)}"


@tool
def process_file(file_path: str, file_url: Optional[str] = None) -> str:
    """
    Process and extract information from various file types.
    
    Args:
        file_path: Path to the file or task_id for GAIA files
        file_url: Optional URL to download the file from
        
    Returns:
        Extracted content from the file
    """
    try:
        # If file_url is provided, download first
        if file_url:
            response = requests.get(file_url)
            response.raise_for_status()
            
            # Determine file extension from URL
            ext = file_url.split('.')[-1].lower()
            temp_path = f"/tmp/gaia_file.{ext}"
            
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            
            file_path = temp_path
        
        # Detect file type
        ext = file_path.split('.')[-1].lower()
        
        if ext == 'pdf':
            return _process_pdf(file_path)
        elif ext in ['csv', 'tsv']:
            return _process_csv(file_path)
        elif ext in ['xlsx', 'xls']:
            return _process_excel(file_path)
        elif ext in ['png', 'jpg', 'jpeg', 'gif', 'bmp']:
            return _process_image(file_path)
        elif ext in ['txt', 'md']:
            return _process_text(file_path)
        elif ext == 'json':
            return _process_json(file_path)
        else:
            return f"Unsupported file type: {ext}"
            
    except Exception as e:
        return f"Error processing file: {str(e)}"


def _process_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.extract_text()
            
            return text[:10000]  # Limit to 10k chars
    except Exception as e:
        return f"Error reading PDF: {str(e)}"


def _process_csv(file_path: str) -> str:
    """Process CSV file and return summary"""
    try:
        df = pd.read_csv(file_path)
        
        summary = f"CSV File Summary:\n"
        summary += f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n"
        summary += f"Columns: {', '.join(df.columns)}\n\n"
        
        # First few rows
        summary += "First 5 rows:\n"
        summary += df.head().to_string()
        
        # Basic statistics for numeric columns
        summary += "\n\nNumeric column statistics:\n"
        summary += df.describe().to_string()
        
        return summary
    except Exception as e:
        return f"Error reading CSV: {str(e)}"


def _process_excel(file_path: str) -> str:
    """Process Excel file and return summary"""
    try:
        excel_file = pd.ExcelFile(file_path)
        summary = f"Excel File Summary:\n"
        summary += f"Sheets: {', '.join(excel_file.sheet_names)}\n\n"
        
        # Process first sheet or all if small
        for sheet_name in excel_file.sheet_names[:3]:  # Max 3 sheets
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            summary += f"\n--- Sheet: {sheet_name} ---\n"
            summary += f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n"
            summary += f"Columns: {', '.join(df.columns)}\n"
            summary += f"First 5 rows:\n{df.head().to_string()}\n"
        
        return summary
    except Exception as e:
        return f"Error reading Excel: {str(e)}"


def _process_text(file_path: str) -> str:
    """Process text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content[:10000]  # Limit to 10k chars
    except Exception as e:
        return f"Error reading text file: {str(e)}"


def _process_json(file_path: str) -> str:
    """Process JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return json.dumps(data, indent=2)[:10000]
    except Exception as e:
        return f"Error reading JSON: {str(e)}"


def _process_image(file_path: str) -> str:
    """Process image file - returns basic info"""
    try:
        img = Image.open(file_path)
        info = f"Image Information:\n"
        info += f"Format: {img.format}\n"
        info += f"Size: {img.size}\n"
        info += f"Mode: {img.mode}\n"
        
        # Note: For actual image analysis, use analyze_image_with_vision
        info += "\nNote: Use analyze_image_with_vision for content analysis"
        
        return info
    except Exception as e:
        return f"Error reading image: {str(e)}"


@tool
def analyze_image_with_vision(image_path: str, question: str) -> str:
    """
    Analyze image content using GPT-4 Vision to answer specific questions.
    
    Args:
        image_path: Path to the image file
        question: Specific question about the image
        
    Returns:
        Analysis result from GPT-4 Vision
    """
    try:
        # Read and encode image
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Determine image type
        image_type = image_path.split('.')[-1].lower()
        if image_type == 'jpg':
            image_type = 'jpeg'
        
        # Create vision model
        vision_model = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=500)
        
        # Prepare message
        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Please analyze this image to answer the following question: {question}"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_type};base64,{image_base64}"
                    }
                }
            ]
        }
        
        response = vision_model.invoke([message])
        return response.content
        
    except Exception as e:
        return f"Error analyzing image: {str(e)}"


@tool
def execute_python_code(code: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Execute Python code safely for calculations and data processing.
    
    Args:
        code: Python code to execute
        context: Optional context variables
        
    Returns:
        Output from code execution
    """
    try:
        # Create isolated namespace
        namespace = {
            '__builtins__': {
                'len': len,
                'range': range,
                'sum': sum,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'sorted': sorted,
                'int': int,
                'float': float,
                'str': str,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
            }
        }
        
        # Add pandas and numpy for calculations
        import numpy as np
        namespace['np'] = np
        namespace['pd'] = pd
        
        # Add context if provided
        if context:
            namespace.update(context)
        
        # Capture output
        from io import StringIO
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        
        try:
            exec(code, namespace)
            output = mystdout.getvalue()
            
            # Also capture last expression result
            if not output and 'result' in namespace:
                output = str(namespace['result'])
                
            return output if output else "Code executed successfully (no output)"
            
        finally:
            sys.stdout = old_stdout
            
    except Exception as e:
        return f"Error executing code: {str(e)}"


@tool
def find_pattern_in_text(text: str, pattern: str, context_chars: int = 100) -> str:
    """
    Find specific patterns or information in text.
    
    Args:
        text: Text to search in
        pattern: Pattern to find (can be substring or description)
        context_chars: Number of characters to include around match
        
    Returns:
        Matching excerpts with context
    """
    try:
        import re
        
        # Simple substring search
        matches = []
        text_lower = text.lower()
        pattern_lower = pattern.lower()
        
        # Find all occurrences
        start = 0
        while True:
            pos = text_lower.find(pattern_lower, start)
            if pos == -1:
                break
                
            # Extract context
            context_start = max(0, pos - context_chars)
            context_end = min(len(text), pos + len(pattern) + context_chars)
            
            excerpt = text[context_start:context_end]
            if context_start > 0:
                excerpt = "..." + excerpt
            if context_end < len(text):
                excerpt = excerpt + "..."
                
            matches.append({
                "position": pos,
                "excerpt": excerpt
            })
            
            start = pos + 1
            
            if len(matches) >= 5:  # Limit results
                break
        
        if matches:
            result = f"Found {len(matches)} matches for '{pattern}':\n\n"
            for i, match in enumerate(matches, 1):
                result += f"Match {i} (position {match['position']}):\n{match['excerpt']}\n\n"
            return result
        else:
            return f"No matches found for '{pattern}'"
            
    except Exception as e:
        return f"Error searching text: {str(e)}"