from typing import Dict, List, Any, Optional, Callable, ClassVar, Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import os
import subprocess
import sys

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import read_file, write_file

class FileReadInput(BaseModel):
    """Input for the file read tool."""
    file_path: str = Field(..., description="Path to the file to read")

class FileWriteInput(BaseModel):
    """Input for the file write tool."""
    file_path: str = Field(..., description="Path to the file to write")
    content: str = Field(..., description="Content to write to the file")

class CommandRunInput(BaseModel):
    """Input for the command run tool."""
    command: str = Field(..., description="Command to run")

class FileReadTool(BaseTool):
    """Tool for reading files."""
    name: ClassVar[str] = "file_read"
    description: ClassVar[str] = "Read the contents of a file"
    args_schema: ClassVar[Type[BaseModel]] = FileReadInput
    
    def _run(self, file_path: str) -> str:
        """
        Read the contents of a file.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            The contents of the file
        """
        content = read_file(file_path)
        if content is None:
            return f"Error: File '{file_path}' not found"
        return content
    
    async def _arun(self, file_path: str) -> str:
        """Async implementation of the tool."""
        return self._run(file_path)

class FileWriteTool(BaseTool):
    """Tool for writing files."""
    name: ClassVar[str] = "file_write"
    description: ClassVar[str] = "Write content to a file"
    args_schema: ClassVar[Type[BaseModel]] = FileWriteInput
    
    def _run(self, file_path: str, content: str) -> str:
        """
        Write content to a file.
        
        Args:
            file_path: Path to the file to write
            content: Content to write to the file
            
        Returns:
            A success message
        """
        try:
            write_file(file_path, content)
            return f"Successfully wrote to file '{file_path}'"
        except Exception as e:
            return f"Error writing to file '{file_path}': {str(e)}"
    
    async def _arun(self, file_path: str, content: str) -> str:
        """Async implementation of the tool."""
        return self._run(file_path, content)

class CommandRunTool(BaseTool):
    """Tool for running shell commands."""
    name: ClassVar[str] = "command_run"
    description: ClassVar[str] = "Run a shell command"
    args_schema: ClassVar[Type[BaseModel]] = CommandRunInput
    
    def _run(self, command: str) -> str:
        """
        Run a shell command.
        
        Args:
            command: Command to run
            
        Returns:
            The command output
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"Error running command: {e.stderr}"
    
    async def _arun(self, command: str) -> str:
        """Async implementation of the tool."""
        return self._run(command)

class ListDirectoryTool(BaseTool):
    """Tool for listing directory contents."""
    name: ClassVar[str] = "list_directory"
    description: ClassVar[str] = "List the contents of a directory"
    
    def _run(self, directory_path: str = ".") -> str:
        """
        List the contents of a directory.
        
        Args:
            directory_path: Path to the directory to list
            
        Returns:
            A string representation of the directory contents
        """
        try:
            items = os.listdir(directory_path)
            result = []
            
            for item in items:
                full_path = os.path.join(directory_path, item)
                if os.path.isdir(full_path):
                    result.append(f"{item}/")
                else:
                    result.append(item)
            
            return "\n".join(result)
        except Exception as e:
            return f"Error listing directory '{directory_path}': {str(e)}"
    
    async def _arun(self, directory_path: str = ".") -> str:
        """Async implementation of the tool."""
        return self._run(directory_path)

# Create a list of all available tools
def get_tools() -> List[BaseTool]:
    """
    Get a list of all available tools.
    
    Returns:
        A list of tools
    """
    return [
        FileReadTool(),
        FileWriteTool(),
        CommandRunTool(),
        ListDirectoryTool()
    ] 