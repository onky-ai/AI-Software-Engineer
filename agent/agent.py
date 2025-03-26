import json
from typing import Dict, List, Any, Optional
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

import sys
import os
import re

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ANTHROPIC_API_KEY, MODEL_NAME, DEFAULT_SYSTEM_PROMPT, TEMPERATURE, MAX_TOKENS
from utils import write_file, ensure_directory, extract_code_from_markdown

class SoftwareDevelopmentAgent:
    """
    A software development agent powered by Claude 3.7 Sonnet.
    """
    
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        output_folder: Optional[str] = None
    ):
        """
        Initialize the software development agent.
        
        Args:
            system_prompt: Custom system prompt to use
            temperature: Temperature for the model
            max_tokens: Maximum tokens for the model response
            output_folder: Folder where generated code will be saved
        """
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.output_folder = output_folder
        
        # Create output folder if specified
        if self.output_folder:
            ensure_directory(self.output_folder)
        
        # Initialize the LLM
        self.llm = ChatAnthropic(
            model=MODEL_NAME,
            anthropic_api_key=ANTHROPIC_API_KEY,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Conversation history
        self.history = []
    
    def add_to_history(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role: The role of the message sender (system, human, or assistant)
            content: The content of the message
        """
        self.history.append({"role": role, "content": content})
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history.
        
        Returns:
            The conversation history
        """
        return self.history
    
    def clear_history(self) -> None:
        """
        Clear the conversation history.
        """
        self.history = []

    # This method is used to query the agent with a user input.
    # It stores the user input in the history.
    # It is used to generate code for the code generation node in the workflow.
    def query(self, user_input: str, output_schema: Optional[BaseModel] = None) -> Any:
        """
        Query the agent with a user input.
        
        Args:
            user_input: The user input
            output_schema: Optional schema for structured output
            
        Returns:
            The agent's response, either as a string or structured according to output_schema
        """
        # Add the user input to the history
        self.add_to_history("human", user_input)
        
        # Create messages for the LLM
        messages = [SystemMessage(content=self.system_prompt)]
        
        # Add the conversation history
        for message in self.history:
            if message["role"] == "human":
                messages.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                messages.append(AIMessage(content=message["content"]))
        
        # Get the response from the LLM
        if output_schema:
            # Get structured response if schema is provided
            llm_with_structured_output = self.llm.with_structured_output(output_schema)
            response = llm_with_structured_output.invoke(messages)
            # Add the raw response to the history (convert structured output to string)
            self.add_to_history("assistant", json.dumps(response.model_dump(), indent=2))
        else:
            # Get regular response
            response = self.llm.invoke(messages)
            # Add the response to the history
            self.add_to_history("assistant", response.content)
            
        return response if output_schema else response.content
  
    # # This method is used to query the agent with a user input and get a structured output.
    # def query_with_structured_output(self, user_input: str, output_schema: BaseModel) -> Dict:
    #     """
    #     Query the agent with a user input and get a structured output.
        
    #     Args:
    #         user_input: The user input
    #         output_schema: The schema for the structured output
            
    #     Returns:
    #         The agent's response as a structured output according to the schema
    #     """
    #     # Add the user input to the history
    #     self.add_to_history("human", user_input)
        
    #     # Create messages for the LLM
    #     messages = [SystemMessage(content=self.system_prompt)]
        
    #     # Add the conversation history
    #     for message in self.history:
    #         if message["role"] == "human":
    #             messages.append(HumanMessage(content=message["content"]))
    #         elif message["role"] == "assistant":
    #             messages.append(AIMessage(content=message["content"]))
        
    #     # Get the structured response from the LLM
    #     llm_with_structured_output = self.llm.with_structured_output(output_schema)
    #     structured_response = llm_with_structured_output.invoke(messages)
        
    #     # Add the raw response to the history (convert structured output to string)
    #     # Convert BaseModel to dict first, then to JSON string
    #     self.add_to_history("assistant", json.dumps(structured_response.model_dump(), indent=2))
        
    #     return structured_response

    def generate_code(self, prompt: str, language: str, filename: Optional[str] = None) -> str:
        """
        Generate code based on a prompt.
        
        Args:
            prompt: The prompt for code generation
            language: The programming language
            filename: Optional filename to save the generated code
            
        Returns:
            The generated code
        """
        code_prompt = f"""
        Generate {language} code based on the following requirements:
        
        {prompt}
        
        Please provide only the code without explanations. Make sure the code is complete, functional, and follows best practices.
        Format the code with proper markdown code blocks using ```{language} as the opening marker.
        """
        
        # do not add user prompt and code prompt to the history
        response = self._query_code_generation(code_prompt)
        
        # Extract code from the response
        code_blocks = extract_code_from_markdown(response)
        
        if code_blocks:
            # Use the first code block that matches the requested language
            for block in code_blocks:
                if block['language'].lower() == language.lower():
                    code = block['code']
                    
                    # Save the code to a file if output folder is specified and filename is provided
                    if self.output_folder and filename:
                        file_path = os.path.join(self.output_folder, filename)
                        write_file(file_path, code)
                    
                    return code
            
            # If no matching language block found, use the first block
            code = code_blocks[0]['code']
            
            # Save the code to a file if output folder is specified and filename is provided
            if self.output_folder and filename:
                file_path = os.path.join(self.output_folder, filename)
                write_file(file_path, code)
            
            return code
        else:
            return response
    
    def explain_code(self, code: str, language: str) -> str:
        """
        Explain the provided code.
        
        Args:
            code: The code to explain
            language: The programming language
            
        Returns:
            The explanation
        """
        explain_prompt = f"""
        Please explain the following {language} code in detail:
        
        ```{language}
        {code}
        ```
        
        Include information about:
        1. What the code does
        2. How it works
        3. Any important patterns or techniques used
        4. Potential improvements or issues
        """
        
        return self.query(explain_prompt)
    
    def debug_code(self, code: str, error_message: str, language: str) -> str:
        """
        Debug the provided code.
        
        Args:
            code: The code to debug
            error_message: The error message
            language: The programming language
            
        Returns:
            The debugged code or explanation
        """
        debug_prompt = f"""
        Please debug the following {language} code that is producing this error:
        
        Error:
        {error_message}
        
        Code:
        ```{language}
        {code}
        ```
        
        Identify the issue and provide a fixed version of the code.
        """
        
        return self.query(debug_prompt) 

    # This method is used to generate code for the code generation node in the workflow.
    # It does not store the user input in the history.
    def _query_code_generation(self, user_input: str) -> str:
        """
        Query the agent with a user input.
        
        Args:
            user_input: The user input
            
        Returns:
            The agent's response
        """
        
        # Create messages for the LLM
        messages = [SystemMessage(content=self.system_prompt)]
        
        # Add the conversation history
        for message in self.history:
            if message["role"] == "human":
                messages.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                messages.append(AIMessage(content=message["content"]))
        
        # Get the response from the LLM
        messages.append(HumanMessage(content=user_input))
        response = self.llm.invoke(messages)
        
        return response.content
      
    def _extract_filename_from_content(self, content: str, block_index: int) -> Optional[str]:
        """
        Try to extract a filename from the content for a specific code block.
        
        Args:
            content: The content to extract the filename from
            block_index: The index of the code block
            
        Returns:
            The extracted filename, or None if no filename was found
        """
        # Look for patterns like "Save this to filename.py" or "Create a file named filename.py"
        patterns = [
            r'save (?:this|the code) (?:to|as) [\'"]?([a-zA-Z0-9_\-\.\/]+)[\'"]?',
            r'create a file (?:named|called) [\'"]?([a-zA-Z0-9_\-\.\/]+)[\'"]?',
            r'filename:?\s*[\'"]?([a-zA-Z0-9_\-\.\/]+)[\'"]?',
            r'file:?\s*[\'"]?([a-zA-Z0-9_\-\.\/]+)[\'"]?',
            r'save (?:this|the code) in [\'"]?([a-zA-Z0-9_\-\.\/]+)[\'"]?',
            r'name the file [\'"]?([a-zA-Z0-9_\-\.\/]+)[\'"]?',
            r'`([a-zA-Z0-9_\-\.\/]+)`'
        ]
            
        # Try to find a filename in the content
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            
            # Count to find the correct block
            current_block = 0
            for match in matches:
                if current_block == block_index:
                    filename = match.group(1)
                    # Clean the filename by removing any prefix like "Filename:" or "File:"
                    filename = re.sub(r'^(?:filename|file):\s*', '', filename, flags=re.IGNORECASE)
                    return filename.strip()
                current_block += 1
        
        return None
    
    def _get_extension_for_language(self, language: str) -> str:
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
    
    def _create_project_structure(self, content: str) -> None:
        """
        Extract project structure from the content and create directories.
        
        Args:
            content: The content to extract project structure from
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
                full_path = os.path.join(self.output_folder, dir_path)
                ensure_directory(full_path)
                
                # Create __init__.py files for Python packages
                if dir_path.endswith('/'):
                    init_file = os.path.join(full_path, '__init__.py')
                    if not os.path.exists(init_file):
                        write_file(init_file, '')
    
    def _save_code_blocks_to_files(self, content: str) -> None:
        """
        Extract code blocks from the content and save them to files.
        
        Args:
            content: The content to extract code blocks from
        """
        code_blocks = extract_code_from_markdown(content)
        
        for i, block in enumerate(code_blocks):
            language = block.get('language', '').strip()
            code = block.get('code', '').strip()
            
            if not code:
                continue
                
            # Try to extract filename from the content
            filename = self._extract_filename_from_content(content, i)
            
            # If no filename found, use a default one
            if not filename:
                extension = self._get_extension_for_language(language)
                filename = f"generated_code_{i+1}{extension}"
            
            # Clean the code (remove any trailing special characters)
            code = re.sub(r'%+$', '', code)  # Remove trailing % characters
            code = code.rstrip()  # Remove trailing whitespace
            
            # Handle file paths with directories
            if '/' in filename:
                dir_path = os.path.dirname(filename)
                full_dir_path = os.path.join(self.output_folder, dir_path)
                ensure_directory(full_dir_path)
            
            # Check if the filename is a directory
            file_path = os.path.join(self.output_folder, filename)
            if os.path.isdir(file_path):
                # If it's a directory, add a default filename
                extension = self._get_extension_for_language(language)
                file_path = os.path.join(file_path, f"index{extension}")
            
            # Save the code to a file
            write_file(file_path, code)