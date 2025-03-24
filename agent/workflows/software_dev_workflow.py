from typing import Dict, List, Any, Optional, TypedDict, Annotated, Literal, Union
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
import sys
import os
import json
import re

from utils import ensure_directory

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import ANTHROPIC_API_KEY, MODEL_NAME, DEFAULT_SYSTEM_PROMPT, TEMPERATURE, MAX_TOKENS
from agent.tools import get_tools

# Define the system prompt
SYSTEM_PROMPT = """
You are a software development agent powered by Claude 3.7 Sonnet.
Your purpose is to assist in developing custom software based on user requirements.

You have access to the following tools:
{tool_descriptions}

Follow these steps to complete software development tasks:
1. Understand the user's requirements
2. Plan the development approach
3. Execute the plan using the available tools
4. Provide the final solution or explanation

When generating code, please follow these guidelines:
1. Provide clear filenames for each code block (e.g., "Save this to app.py" or "Filename: utils.js")
2. Make sure the code is complete, functional, and follows best practices
3. Include appropriate error handling and documentation
4. When specifying file paths, use the format "directory/filename.ext" (e.g., "models/user.py")

When you have completed the task, provide a detailed explanation of the solution.
"""

def run_software_dev_workflow(user_input: str, output_folder: Optional[str] = None) -> str:
    """
    Run the software development workflow.
    
    Args:
        user_input: The user input
        output_folder: Folder where generated code will be saved
        
    Returns:
        The final answer or last response
    """
    # Initialize the LLM
    llm = ChatAnthropic(
        model=MODEL_NAME,
        anthropic_api_key=ANTHROPIC_API_KEY,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )
    
    # Get the tools
    tools = get_tools()
    
    # Create the prompt
    tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
    system_prompt = SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)
    
    # Create the messages
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]
    
    # Get the response from the LLM
    response = llm.invoke(messages)
    
    # If output folder is specified, save code blocks to files
    if output_folder:
        from utils import extract_code_from_markdown, write_file, ensure_directory
        
        # Ensure the output folder exists
        ensure_directory(output_folder)
        
        # Extract code blocks
        content = response.content
        code_blocks = extract_code_from_markdown(content)
        
        # Try to extract project structure and create directories
        create_project_structure(content, output_folder)
        
        for i, block in enumerate(code_blocks):
            language = block.get('language', '').strip()
            code = block.get('code', '').strip()
            
            if not code:
                continue
                
            # Try to extract filename from the content
            filename = extract_filename_from_content(content, i, code_blocks)
            
            # If no filename found, use a default one
            if not filename:
                extension = get_extension_for_language(language)
                filename = f"generated_code_{i+1}{extension}"
            
            # Clean the code (remove any trailing special characters)
            code = re.sub(r'%+$', '', code)  # Remove trailing % characters
            code = code.rstrip()  # Remove trailing whitespace
            
            # Handle file paths with directories
            if '/' in filename:
                dir_path = os.path.dirname(filename)
                full_dir_path = os.path.join(output_folder, dir_path)
                ensure_directory(full_dir_path)
            
            # Check if the filename is a directory
            file_path = os.path.join(output_folder, filename)
            if os.path.isdir(file_path):
                # If it's a directory, add a default filename
                extension = get_extension_for_language(language)
                file_path = os.path.join(file_path, f"index{extension}")
            
            # Save the code to a file
            write_file(file_path, code)
    
    return response.content

def create_project_structure(content: str, output_folder: str) -> None:
    """
    Extract project structure from the content and create directories.
    
    Args:
        content: The content to extract project structure from
        output_folder: Folder where the project structure will be created
    """
    # Look for project structure patterns
    structure_pattern = r'```(?:bash|shell|text)?\s*project\/.*?```'
    structure_match = re.search(structure_pattern, content, re.DOTALL)
    
    if structure_match:
        structure_text = structure_match.group(0)
        
        # Extract directory paths
        dir_pattern = r'(?:├|└)── ([a-zA-Z0-9_\-\.\/]+)\/'
        dir_matches = re.finditer(dir_pattern, structure_text)
        
        for match in dir_matches:
            dir_path = match.group(1)
            full_path = os.path.join(output_folder, dir_path)
            ensure_directory(full_path)
            
            # Create __init__.py files for Python packages
            if dir_path.endswith('/'):
                init_file = os.path.join(full_path, '__init__.py')
                if not os.path.exists(init_file):
                    from utils import write_file
                    write_file(init_file, '')

def extract_filename_from_content(content: str, block_index: int, code_blocks: List[Dict[str, str]]) -> Optional[str]:
    """
    Try to extract a filename from the content for a specific code block.
    
    Args:
        content: The content to extract the filename from
        block_index: The index of the code block
        code_blocks: The extracted code blocks
        
    Returns:
        The extracted filename, or None if no filename was found
    """
    # Look for patterns like "Save this to filename.py" or "Create a file named filename.py"
    patterns = [
        r'save (?:this|the code) (?:to|as) [\'"]?([a-zA-Z0-9_\-\.\/]+)[\'"]?',
        r'create a file (?:named|called) [\'"]?([a-zA-Z0-9_\-\.\/]+)[\'"]?',
        r'filename:? [\'"]?([a-zA-Z0-9_\-\.\/]+)[\'"]?',
        r'file:? [\'"]?([a-zA-Z0-9_\-\.\/]+)[\'"]?',
        r'save (?:this|the code) in [\'"]?([a-zA-Z0-9_\-\.\/]+)[\'"]?',
        r'name the file [\'"]?([a-zA-Z0-9_\-\.\/]+)[\'"]?',
        r'`([a-zA-Z0-9_\-\.\/]+)`'
    ]
    
    # First, try to find a filename specifically for this code block
    if 0 <= block_index < len(code_blocks):
        code_block = code_blocks[block_index]
        language = code_block.get('language', '').strip().lower()
        
        # Look for common filenames based on language
        if language == 'python':
            if 'hello world' in content.lower():
                return 'hello_world.py'
            if 'main' in code_block.get('code', '').lower():
                return 'main.py'
        elif language == 'javascript':
            if 'hello world' in content.lower():
                return 'hello_world.js'
            if 'main' in code_block.get('code', '').lower():
                return 'main.js'
        elif language == 'html':
            return 'index.html'
        elif language == 'css':
            return 'styles.css'
    
    # Try to find a filename in the content
    for pattern in patterns:
        matches = re.finditer(pattern, content, re.IGNORECASE)
        
        # Count to find the correct block
        current_block = 0
        for match in matches:
            if current_block == block_index:
                return match.group(1)
            current_block += 1
    
    return None

def get_extension_for_language(language: str) -> str:
    """
    Get the file extension for a language.
    
    Args:
        language: The programming language
        
    Returns:
        The file extension
    """
    language_extensions = {
        'python': '.py',
        'javascript': '.js',
        'typescript': '.ts',
        'html': '.html',
        'css': '.css',
        'java': '.java',
        'c': '.c',
        'cpp': '.cpp',
        'csharp': '.cs',
        'go': '.go',
        'rust': '.rs',
        'php': '.php',
        'ruby': '.rb',
        'swift': '.swift',
        'kotlin': '.kt'
    }
    
    return language_extensions.get(language.lower(), '.txt') 