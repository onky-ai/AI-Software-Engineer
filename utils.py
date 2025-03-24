import os
import json
import re
from typing import Dict, List, Any, Optional

def ensure_directory(directory_path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory to create
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def save_json(data: Any, file_path: str) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to the file
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(file_path: str) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        The loaded data
    """
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, 'r') as f:
        return json.load(f)

def read_file(file_path: str) -> Optional[str]:
    """
    Read the contents of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        The contents of the file, or None if the file doesn't exist
    """
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, 'r') as f:
        return f.read()

def write_file(file_path: str, content: str) -> None:
    """
    Write content to a file.
    
    Args:
        file_path: Path to the file
        content: Content to write
    """
    # Ensure the directory exists
    directory = os.path.dirname(file_path)
    if directory:
        ensure_directory(directory)
    
    # Clean the content before writing
    content = clean_code(content)
    
    with open(file_path, 'w') as f:
        f.write(content)

def clean_code(code: str) -> str:
    """
    Clean code by removing trailing special characters and whitespace.
    
    Args:
        code: The code to clean
        
    Returns:
        The cleaned code
    """
    # Remove trailing % characters
    code = re.sub(r'%+$', '', code)
    
    # Remove trailing whitespace from each line
    lines = code.splitlines()
    cleaned_lines = [line.rstrip() for line in lines]
    
    # Join the lines back together
    cleaned_code = '\n'.join(cleaned_lines)
    
    # Remove trailing whitespace from the entire string
    cleaned_code = cleaned_code.rstrip()
    
    return cleaned_code

def format_code_for_llm(code: str, language: str) -> str:
    """
    Format code for sending to the LLM.
    
    Args:
        code: The code to format
        language: The programming language
        
    Returns:
        Formatted code with markdown code blocks
    """
    return f"```{language}\n{code}\n```"

def extract_code_from_markdown(markdown: str) -> List[Dict[str, str]]:
    """
    Extract code blocks from markdown.
    
    Args:
        markdown: The markdown text
        
    Returns:
        List of dictionaries with 'language' and 'code' keys
    """
    code_blocks = []
    lines = markdown.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('```'):
            language = line[3:].strip()
            code_lines = []
            i += 1
            
            while i < len(lines) and not lines[i].startswith('```'):
                code_lines.append(lines[i])
                i += 1
                
            if i < len(lines):  # Skip the closing ```
                i += 1
                
            code_blocks.append({
                'language': language,
                'code': '\n'.join(code_lines)
            })
        else:
            i += 1
            
    return code_blocks 