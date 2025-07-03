# GAIA Benchmark Agent with LangGraph

A high-performance agent for the GAIA (General AI Assistants) benchmark, built with LangGraph and OpenAI GPT-4. This implementation achieves competitive performance through sophisticated multi-step reasoning, tool integration, and precise answer formatting.

## 🏆 Features

- **Multi-step reasoning** with explicit step counting (GAIA Level 1 limit: 5 steps)
- **Comprehensive tool suite**:
  - Web search and browsing with markdown conversion
  - Multi-format file processing (PDF, CSV, Excel, images, JSON)
  - Image analysis with GPT-4 Vision
  - Python code execution for calculations
  - Pattern finding in text
- **Exact answer formatting** compliant with GAIA requirements
- **Robust error handling** with retry mechanisms
- **Gradio interface** for easy testing and submission
- **Full API integration** with GAIA benchmark endpoints

## 📋 Requirements

- Python 3.8+
- OpenAI API key (GPT-4 access required)
- Tavily API key (for web search)
- Hugging Face account (for submission)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository (or copy files to your HF Space)
git clone <your-repo-url>
cd gaia-agent

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY="your-openai-key"
export TAVILY_API_KEY="your-tavily-key"
```

### 2. Run Locally

```python
from gaia_agent import GaiaAgent

# Initialize agent
agent = GaiaAgent(
    model_name="gpt-4-1106-preview",
    temperature=0.1
)

# Test on a question
result = agent.run(
    question="What is the capital of France?",
    task_id="test_001"
)

print(f"Answer: {result['answer']}")
print(f"Valid: {result['validation']['is_valid']}")
```

### 3. Launch Gradio Interface

```bash
python app.py
```

Navigate to `http://localhost:7860` to access the interface.

## 🏗️ Architecture

### Core Components

1. **`gaia_state.py`** - State management and types
   - Comprehensive state tracking
   - Step counting and limits
   - Error accumulation

2. **`gaia_tools.py`** - Tool implementations
   - Web search and browsing
   - File processing (multiple formats)
   - Image analysis
   - Code execution

3. **`gaia_formatting.py`** - Answer formatting
   - Type detection (number, string, list, date)
   - Format compliance validation
   - Common error fixes

4. **`gaia_agent.py`** - Main agent logic
   - LangGraph workflow
   - Reasoning nodes
   - Tool coordination

5. **`gaia_submission.py`** - API integration
   - Question retrieval
   - File downloads
   - Result submission

6. **`app.py`** - Gradio interface
   - Interactive testing
   - Batch evaluation
   - Leaderboard submission

### LangGraph Workflow

```
START → analyze_question → reason → {tools/synthesize/continue} → format_answer → validate → END
                             ↑                    ↓
                             ←────────────────────
```

## 🎯 Usage Guide

### Testing Single Questions

1. Initialize the agent in the Gradio interface
2. Go to "Test Single Question" tab
3. Enter a GAIA-style question
4. Review answer, validation, and reasoning chain

### Running Full Evaluation

1. Go to "Full Evaluation" tab
2. Select number of questions (1-20)
3. Click "Run Evaluation"
4. Monitor progress and results

### Submitting to Leaderboard

1. Complete full evaluation (all 20 questions)
2. Go to "Submit to Leaderboard" tab
3. Enter your HF username
4. Enter your Space URL (ending with `/tree/main`)
5. Click "Submit Results"

## 🔧 Configuration

### Model Selection

The agent supports multiple OpenAI models:
- `gpt-4-1106-preview` (recommended)
- `gpt-4o` (cost-effective fallback)
- `gpt-4-turbo` (alternative)

### Tool Configuration

Customize tools in `gaia_tools.py`:
- Adjust search result limits
- Modify file size limits
- Configure timeout values

### Answer Formatting

Fine-tune formatting rules in `gaia_formatting.py`:
- Number formatting (comma handling)
- String processing (article removal)
- List formatting (alphabetical ordering)
- Date format detection

## 📊 Performance Tips

1. **Optimize Token Usage**
   - Use markdown conversion for web pages
   - Truncate long file contents
   - Limit search results

2. **Improve Accuracy**
   - Lower temperature (0.1 recommended)
   - Use explicit step counting
   - Implement validation retry

3. **Handle Rate Limits**
   - Add delays between questions
   - Implement exponential backoff
   - Use fallback models

## 🐛 Troubleshooting

### Common Issues

1. **"No module named 'langgraph'"**
   ```bash
   pip install langgraph>=0.2.0
   ```

2. **"Invalid API key"**
   - Verify OpenAI API key has GPT-4 access
   - Check environment variables

3. **"Answer validation failed"**
   - Review formatting requirements
   - Check for "FINAL ANSWER:" prefix (remove it!)
   - Verify answer type detection

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 GAIA Benchmark Details

### Question Types
- **Level 1**: 0-5 steps, basic tool usage
- **Level 2**: 5-10 steps, multiple tools
- **Level 3**: Complex multi-tool coordination

### Answer Requirements
- Exact match evaluation
- No "FINAL ANSWER:" prefix in submissions
- Specific formatting per type:
  - Numbers: No commas unless specified
  - Strings: No articles, minimal words
  - Lists: Comma-separated, alphabetical if requested
  - Dates: Follow specified format

### Performance Metrics
- Human baseline: ~92%
- GPT-4 baseline: ~15%
- Top agents: 66-75%

## 🤝 Contributing

1. Test improvements locally
2. Validate on subset of questions
3. Document changes clearly
4. Submit with performance metrics

## 📄 License

This project is open source and follows the same license as the GAIA benchmark.

## 🔗 Resources

- [GAIA Paper](https://arxiv.org/abs/2311.12983)
- [Official Leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard)
- [Hugging Face Agents Course](https://huggingface.co/learn/agents-course)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

## 💡 Tips for Success

1. **Start Simple**: Test on individual questions before full evaluation
2. **Monitor Steps**: Stay within the 5-step limit for Level 1
3. **Validate Formatting**: Always check answer format before submission
4. **Save Progress**: Enable checkpointing for long evaluations
5. **Iterate**: Analyze failures and improve tool usage

Good luck with the GAIA benchmark! 🚀