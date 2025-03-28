from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

class RequirementsOutput(BaseModel):
    requirements: List[str] = Field(description="List of clear requirements extracted from the task")
    file_dependencies: List[str] = Field(description="Dependencies between requirements", default_factory=list)
        
class ProjectStructureOutput(BaseModel):
    """Output model for the project structure step of the workflow."""
    files: List[str] = Field(description="List of files to be created", default_factory=list)
    description: str = Field(description="Description of the files to be created")

class DesignOutput(BaseModel):
    architecture: str = Field(description="Overview of the system architecture")
    components: Any = Field(description="Main components of the system")
    data_models: Any = Field(description="Data models used in the system")
    api_endpoints: Any = Field(default=[], description="API endpoints if applicable")
    dependencies: Any = Field(description="Dependencies and libraries needed")

class DocumentationOutput(BaseModel):
    overview: str = Field(description="Project overview and purpose")
    installation: str = Field(description="Installation instructions")
    usage: str = Field(description="Usage instructions and examples")
    api_docs: Dict[str, Any] = Field(description="API documentation by component", default_factory=dict)
    examples: List[str] = Field(description="Usage examples", default_factory=list)
    file_descriptions: Dict[str, Any] = Field(description="Description of each file", default_factory=dict)

class FileGenerationOutput(BaseModel):
    content: str = Field(description="The generated file content")
    quality_score: Dict[str, float] = Field(description="Quality scores for the generated content", default_factory=dict)
    missing_elements: Dict[str, List[str]] = Field(description="Missing elements in the code", default_factory=dict)
    suggestions: Dict[str, List[str]] = Field(description="Suggestions for improvement", default_factory=dict)
