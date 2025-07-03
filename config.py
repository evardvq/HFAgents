import os
from typing import Dict, Any

# API Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Model Configuration
MODEL_CONFIG = {
    "primary_model": "gpt-4-1106-preview",  # GPT-4.1
    "fallback_model": "gpt-4o",
    "vision_model": "gpt-4-vision-preview",
    "temperature": 0.1,
    "max_tokens": None,  # Use model default
    "top_p": 0.95
}

# GAIA Benchmark Settings
GAIA_CONFIG = {
    "api_base_url": "https://agents-course-unit4-scoring.hf.space",
    "max_steps_level_1": 5,
    "max_steps_level_2": 10,
    "max_steps_level_3": 20,
    "submission_delay": 0.5,  # Delay between questions in seconds
}

# Tool Configuration
TOOL_CONFIG = {
    "web_search": {
        "max_results": 3,
        "include_snippets": True
    },
    "web_browse": {
        "max_content_length": 5000,  # Characters
        "timeout": 10,  # Seconds
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    },
    "file_processing": {
        "max_file_size_mb": 10,
        "pdf_max_pages": 50,
        "csv_preview_rows": 100,
        "image_max_dimension": 4096
    },
    "code_execution": {
        "timeout": 30,  # Seconds
        "max_output_length": 10000  # Characters
    }
}

# Answer Formatting Rules
FORMAT_CONFIG = {
    "remove_articles": True,
    "remove_trailing_punctuation": True,
    "number_remove_commas": True,
    "number_remove_units": True,
    "list_alphabetical_default": False,
    "date_default_format": "MM/DD/YYYY"
}

# Validation Settings
VALIDATION_CONFIG = {
    "max_answer_length": 100,  # Characters
    "max_retry_attempts": 3,
    "strict_mode": True,  # Fail on any validation error
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "save_to_file": True,
    "log_file": "gaia_agent.log"
}

# Gradio Interface Settings
GRADIO_CONFIG = {
    "share": False,  # Set to True for public URL
    "server_port": 7860,
    "server_name": "0.0.0.0",  # For Docker/Space deployment
    "theme": "default",
    "max_file_size": 10 * 1024 * 1024,  # 10MB
}

# Rate Limiting
RATE_LIMIT_CONFIG = {
    "requests_per_second": 10,
    "requests_per_minute": 100,
    "check_interval": 0.1,
    "use_fallback_on_limit": True
}

# Checkpointing
CHECKPOINT_CONFIG = {
    "enabled": True,
    "backend": "sqlite",  # sqlite, memory
    "db_path": "gaia_checkpoints.db",
    "save_frequency": 1,  # Save after every N questions
}

# Development Settings
DEV_CONFIG = {
    "debug_mode": False,
    "test_mode": False,
    "mock_api_responses": False,
    "save_all_responses": True,
    "response_dir": "./responses"
}

def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary"""
    return {
        "model": MODEL_CONFIG,
        "gaia": GAIA_CONFIG,
        "tools": TOOL_CONFIG,
        "format": FORMAT_CONFIG,
        "validation": VALIDATION_CONFIG,
        "logging": LOGGING_CONFIG,
        "gradio": GRADIO_CONFIG,
        "rate_limit": RATE_LIMIT_CONFIG,
        "checkpoint": CHECKPOINT_CONFIG,
        "dev": DEV_CONFIG
    }

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check required API keys
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY not set")
    
    if not TAVILY_API_KEY:
        errors.append("TAVILY_API_KEY not set (required for web search)")
    
    # Validate model names
    valid_models = ["gpt-4-1106-preview", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-4-vision-preview"]
    if MODEL_CONFIG["primary_model"] not in valid_models:
        errors.append(f"Invalid primary model: {MODEL_CONFIG['primary_model']}")
    
    # Validate numeric ranges
    if not 0 <= MODEL_CONFIG["temperature"] <= 2:
        errors.append(f"Temperature must be between 0 and 2")