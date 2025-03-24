import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set. Please add it to your .env file.")

# LangSmith Configuration
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "software-development-agent")

# Model Configuration
MODEL_NAME = "claude-3-opus-20240229"  # Claude 3.7 Sonnet

# Agent Configuration
MAX_ITERATIONS = 10
TEMPERATURE = 0.2
MAX_TOKENS = 4096

# Development Tools Configuration
SUPPORTED_LANGUAGES = [
    "python",
    "javascript",
    "typescript",
    "html",
    "css",
    "java",
    "c",
    "cpp",
    "csharp",
    "go",
    "rust",
    "php",
    "ruby",
    "swift",
    "kotlin"
]

# Default system prompt for the agent
DEFAULT_SYSTEM_PROMPT = """You are a software development agent powered by Claude 3.7 Sonnet.
Your purpose is to assist in developing custom software based on user requirements.
You can understand code, generate code, debug issues, and provide explanations.
Always strive to write clean, efficient, and well-documented code.
When generating code, include appropriate error handling and follow best practices for the language.

When creating multiple files, please follow these guidelines:
1. Provide clear filenames for each code block (e.g., "Save this to app.py" or "Filename: utils.js")
2. When specifying file paths, use the format "directory/filename.ext" (e.g., "models/user.py")
3. If you're creating a project structure, clearly indicate the directory hierarchy
4. Make sure to include all necessary files for the project to work correctly
""" 