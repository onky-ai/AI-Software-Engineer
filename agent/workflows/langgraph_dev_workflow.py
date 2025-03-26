from typing import Dict, List, Any, Optional, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
import os
import sys
from langsmith import traceable
from pydantic import BaseModel, Field

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent.agent import SoftwareDevelopmentAgent
from agent.docker_utils import run_in_docker
from config import DEFAULT_SYSTEM_PROMPT, LANGCHAIN_PROJECT, LANGCHAIN_ENDPOINT
from agent.langsmith_utils import is_tracing_enabled, trace_tool_usage, create_trace_id

# Define the state type for our workflow
class WorkflowState(TypedDict):
    task: str
    current_step: str
    requirements: List[str]
    design: Dict[str, Any]
    project_structure: Dict[str, Any]
    code_files: Dict[str, str]
    file_dependencies: Dict[str, List[str]]
    documentation: str
    messages: List[Dict[str, str]]
    output_folder: Optional[str]
    trace_id: Optional[str]

# Define the requirements output model
class RequirementsOutput(BaseModel):
    requirements: List[str] = Field(description="List of clear requirements extracted from the task")
    file_dependencies: List[str] = Field(description="Dependencies between requirements", default_factory=list)
        
class ProjectStructureOutput(BaseModel):
    """Output model for the project structure step of the workflow."""
    files: List[str] = Field(description="List of files to be created", default_factory=list)
    description: str = Field(description="Description of the files to be created")

class DesignOutput(BaseModel):
    architecture: str = Field(description="Overview of the system architecture")
    components: List[str] = Field(description="Main components of the system")
    data_models: List[str] = Field(description="Data models used in the system")
    api_endpoints: Optional[List[str]] = Field(default=[], description="API endpoints if applicable")
    dependencies: List[str] = Field(description="Dependencies and libraries needed")
        

# Define the workflow steps
def analyze_requirements(state: WorkflowState) -> WorkflowState:
    """Analyze the task and extract requirements"""
    agent = SoftwareDevelopmentAgent()
    
    prompt = f"""
    Analyze the following software development task and extract clear requirements:
 
    {state["task"]}
    
    Please provide a structured list of requirements that need to be implemented and dependencies between them. 
    Keep all requirements super minimal - focus only on the absolute essential requirements needed to accomplish the task.
    """
    
    response = agent.query(prompt, RequirementsOutput)
    state["requirements"] = response.requirements
    state["file_dependencies"] = response.file_dependencies
    state["messages"].append({"role": "system", "content": f"Requirements analyzed: {len(state['requirements'])} requirements identified"})
    state["current_step"] = "requirements_analyzed"
    
    print(f"Analyze Requirements: {state['requirements']}")

    return state

def create_design(state: WorkflowState) -> WorkflowState:
    """Create a high-level design based on requirements"""
    agent = SoftwareDevelopmentAgent()
    
    requirements_text = "\n".join([f"- {req}" for req in state["requirements"]])
    dependencies_text = "\n".join([f"- {dep}" for dep in state["file_dependencies"]])

    prompt = f"""
    Based on these requirements:

    {requirements_text}

    and dependencies between them:

    {dependencies_text}

    Create a high-level software design that focuses on simplicity
    1. Architecture overview
    2. Main components
    3. Data models
    4. API endpoints (if applicable)
    5. Dependencies and libraries needed
    
    Provide the design in a structured format.
    """
    
    response = agent.query(prompt, DesignOutput)

    state["design"] = {
            "description": response.architecture,
            "components": response.components,
            "data_models": response.data_models,
            "dependencies": response.dependencies,
            "api_endpoints": response.api_endpoints
        }
    state["messages"].append({"roke" : "system", "content" : "Design created"})
    state["current_step"] = "desing_created"

    return state

def propose_project_structure(state: WorkflowState) -> WorkflowState:
    """Propose a project structure based on the design"""
    agent = SoftwareDevelopmentAgent()
    
    prompt = f"""
    Based on this design:
    
    1. Design description:
    
    {state["design"]["description"]}
   
    2. Design components:
    
    {state["design"]["components"]}
    
    3. Design data models:
    
    {state["design"]["data_models"]}
    
    4. Design API endpoints:
    
    {state["design"]["api_endpoints"]}
    
    5. Design dependencies:
    
    {state["design"]["dependencies"]}

    Create a minimal project structure with only the essential directories and files needed.
    Each file must be a separate artifact as a code generation task will be generated for each file separately.
    Focus on simplicity and avoid creating any unnecessary files or directories.

    Return only the files with full path, one per line, without any additional text.
    """
      
    response = agent.query(prompt, ProjectStructureOutput)
     
    state["project_structure"] = { "description": response.description, "files": response.files }
    state["messages"].append({"role": "system", "content": "Project structure proposed"})
    state["current_step"] = "project_structure_proposed"
  
    return state

def generate_files(state: WorkflowState) -> WorkflowState:
    """Generate each file as a separate artifact"""
    agent = SoftwareDevelopmentAgent(output_folder=state["output_folder"])
    state["code_files"] = {}
    state["file_dependencies"] = {}
    
    files_to_create = state["project_structure"]["files"]
    
    # Generate code for each file
    for file_name in files_to_create:
        # Get the language from the file extension
        language = file_name.split(".")[-1] if "." in file_name else "txt"
        
        prompt = f"""
            Create the code for the file `{file_name}` based on:
            
            Requirements:
            {chr(10).join([f"- {req}" for req in state["requirements"]])}
            
            Design:
            {state["design"]["description"]}
            
            Project Structure:
            {state["project_structure"]["files"]}
            
            Provide only the code, properly formatted and complete.
            Make sure to include all necessary imports and dependencies.
            """
        
        # Generate the code
        content = agent.generate_code(prompt, language, file_name)
        
        # Save the file
        if state["output_folder"]:
            # Create any necessary directories
            file_path = os.path.join(state["output_folder"], file_name)
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            with open(file_path, "w") as f:
                f.write(content)
            
            # Trace tool usage for file creation
            if is_tracing_enabled() and state.get("trace_id"):
                trace_tool_usage(
                    tool_name="file_write",
                    input_data={
                        "file_name": file_name,
                        "content": content
                    },
                    output_data=f"File {file_name} created successfully",
                    metadata={
                        "step": "generate_files",
                        "file_name": file_name,
                        "trace_id": state["trace_id"]
                    }
                )
        
        state["code_files"][file_name] = content
    
    state["current_step"] = "files_generated"
    return state

# @traceable(run_type="chain", name="verify_completeness")
# def verify_completeness(state: WorkflowState) -> WorkflowState:
#     """Verify each file's completeness"""
#     agent = SoftwareDevelopmentAgent()
    
#     for file_name, code in state["code_files"].items():
#         language = file_name.split(".")[-1]
        
#         prompt = f"""
#         Verify the completeness of this {language} file:
        
#         File: {file_name}
#         Code:
#         ```{language}
#         {code}
#         ```
        
#         Analyze the code for completeness, including imports, dependencies,
#         function implementations, error handling, and documentation.
#         """
        
#         try:
#             completeness_output = agent.query(prompt)
            
#             if not completeness_output.complete:
#                 # Generate updated code if issues found
#                 missing_elements = completeness_output.missing_elements.get(file_name, [])
#                 suggestions = completeness_output.suggestions.get(file_name, [])
                
#                 update_prompt = f"""
#                 Update the code for {file_name} to address these issues:
#                 {completeness_output}
                
#                 Original code:
#                 ```{language}
#                 {code}
#                 ```
#                 """
                
#                 file_output = agent.query_with_structured_output(update_prompt, FileGenerationOutput)
#                 state["code_files"][file_name] = file_output.content
#                 state["messages"].append({
#                     "role": "system", 
#                     "content": f"Updated file {file_name} with completeness score: {completeness_output.quality_score.get(file_name, 0)}"
#                 })
#         except Exception as e:
#             print(f"Error in completeness verification for {file_name}: {e}")
#             # Fallback to simple verification
#             response = agent.query(prompt)
#             if "missing" in response.lower() or "needed" in response.lower():
#                 updated_code = agent.generate_code(prompt, language, file_name)
#                 state["code_files"][file_name] = updated_code
    
#     state["current_step"] = "completeness_verified"
#     return state

# @traceable(run_type="chain", name="create_documentation")
# def create_documentation(state: WorkflowState) -> WorkflowState:
#     """Create documentation for the project"""
#     agent = SoftwareDevelopmentAgent()
    
#     files_list = "\n".join([f"- {file_name}" for file_name in state["code_files"].keys()])
#     dependencies = json.dumps(state["file_dependencies"], indent=2)
    
#     prompt = f"""
#     Create comprehensive documentation for this software project:
    
#     Requirements:
#     {chr(10).join([f"- {req}" for req in state["requirements"]])}
    
#     Design:
#     {state["design"]["description"]}
    
#     Project Structure:
#     {state["project_structure"]["description"]}
    
#     Files:
#     {files_list}
    
#     Dependencies:
#     {dependencies}
#     """
    
#     try:
#         doc_output = agent.query(prompt)
        
#         # Create README.md content
#         documentation = f"""# {doc_output.overview}

# ## Installation
# {doc_output.installation}

# ## Usage
# {doc_output.usage}

# ## API Documentation
# {chr(10).join([f"### {component}{chr(10)}{docs}" for component, docs in doc_output.api_docs.items()])}

# ## Examples
# {chr(10).join([f"- {example}" for example in doc_output.examples])}

# ## File Structure
# {chr(10).join([f"- {file}: {desc}" for file, desc in doc_output.file_descriptions.items()])}
# """
        
#         state["documentation"] = documentation
        
#         # Save documentation if output folder is specified
#         if state["output_folder"]:
#             doc_path = os.path.join(state["output_folder"], "README.md")
#             with open(doc_path, "w") as f:
#                 f.write(documentation)
            
#             if is_tracing_enabled() and state.get("trace_id"):
#                 trace_tool_usage(
#                     tool_name="file_write",
#                     input_data={
#                         "file_name": "README.md",
#                         "content": documentation
#                     },
#                     output_data=f"File README.md created successfully",
#                     metadata={
#                         "step": "create_documentation",
#                         "file_name": "README.md",
#                         "trace_id": state["trace_id"]
#                     }
#                 )
        
#         state["messages"].append({"role": "system", "content": "Documentation created"})
#         state["current_step"] = "documentation_created"
        
#     except Exception as e:
#         print(f"Error in documentation creation: {e}")
#         # Fallback to simple documentation
#         response = agent.query(prompt)
#         state["documentation"] = response
        
#         if state["output_folder"]:
#             doc_path = os.path.join(state["output_folder"], "README.md")
#             with open(doc_path, "w") as f:
#                 f.write(response)
    
#     return state

@traceable(run_type="chain", name="router")
def router(state: WorkflowState) -> Literal["analyze_requirements", "create_design", "propose_project_structure", "generate_files", "verify_dependencies", "verify_completeness", "create_documentation", "END"]:
    """Route to the next step based on the current state"""
    current_step = state.get("current_step", "")
    
    if current_step == "":
        return "analyze_requirements"
    elif current_step == "requirements_analyzed":
        return "create_design"
    elif current_step == "design_created":
        return "propose_project_structure"
    elif current_step == "project_structure_proposed":
        return "generate_files"
    elif current_step == "files_generated":
        return "verify_dependencies"
    elif current_step == "dependencies_verified":
        return "verify_completeness"
    elif current_step == "completeness_verified":
        return "create_documentation"
    elif current_step == "documentation_created":
        return "END"
    else:
        return "analyze_requirements"

# Create the workflow graph with only 2 nodes: create_design and generate_files
@traceable(run_type="chain", name="create_workflow_graph")
def create_workflow_graph() -> StateGraph:
    """Create the workflow graph"""
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("analyze_requirements", analyze_requirements)
    workflow.add_node("create_design", create_design)
    workflow.add_node("propose_project_structure", propose_project_structure)
    workflow.add_node("generate_files", generate_files)

    # Add edge from START to first node
    workflow.add_edge(START, "analyze_requirements")
    workflow.add_edge("analyze_requirements", "create_design")
    workflow.add_edge("create_design", "propose_project_structure")
    workflow.add_edge("propose_project_structure", "generate_files")
    workflow.add_edge("generate_files", END)


    return workflow

# Python runner will run the tests in the container
def python_runner_tester(state: WorkflowState, host_project_dir: Optional[str] = None) -> Dict:
    """
    Tester node implementation using Docker execution.
    """
    if not host_project_dir:
        print("Error: Output folder not specified.")
        return {"test_results": {"setup_error": "Output folder not specified"}, "error_logs": [], "build_status": "Testing Failed"}
    
    print("--- Entering Tester Node ---")
    test_results = {}
    error_logs = state.get("error_logs", [])
    dependencies_updated = False # Flag to track if we installed deps

    # --- Step 1: Ensure project directory exists (File Manager node should create it) ---
    if not os.path.exists(host_project_dir):
         print(f"Error: Project directory {host_project_dir} not found.")
         error_logs.append(f"Tester Error: Project directory {host_project_dir} not found.")
         return {"test_results": {"setup_error": "Project directory missing"}, "error_logs": error_logs, "build_status": "Testing Failed"}

    # --- Step 2: Install Dependencies (if requirements.txt exists) ---
    requirements_path = os.path.join(host_project_dir, "requirements.txt")
    if os.path.exists(requirements_path):
        print("Found requirements.txt, installing dependencies in Docker...")
        # Adjust command if pip is not directly runnable or needs activation
        dep_command = ["pip", "install", "-r", "requirements.txt"]
        exit_code, stdout, stderr = run_in_docker(dep_command, host_project_dir)

        if exit_code != 0:
            print(f"ERROR: Failed to install dependencies. Exit code: {exit_code}")
            print(f"Stderr:\n{stderr}")
            error_logs.append(f"Dependency Installation Failed (Exit Code {exit_code}):\n{stderr}")
            # Consider stopping here or trying tests anyway
            return {"test_results": {"dependency_error": "Failed"}, "error_logs": error_logs, "build_status": "Testing Failed"}
        else:
            print("Dependencies installed successfully.")
            print(f"Stdout:\n{stdout}")
            dependencies_updated = True
            # Add stdout to logs if verbose logging is desired
            # error_logs.append(f"Dependency Installation Log:\n{stdout}")
    else:
        print("No requirements.txt found, skipping dependency installation.")


    # --- Step 3: Run Tests (Example using pytest) ---
    # Adapt this command based on the testing framework expected/generated
    test_command = ["pytest"] # Or ["python", "run_tests.py"], etc.
    print(f"Running tests using command: {' '.join(test_command)}")
    exit_code, stdout, stderr = run_in_docker(test_command, host_project_dir)

    print(f"Test execution finished. Exit Code: {exit_code}")
    print(f"Stdout:\n{stdout}")
    if stderr:
        print(f"Stderr:\n{stderr}")

    # --- Step 4: Parse Results ---
    # This is highly dependent on the test runner's output format
    # Simple example: Check exit code and look for failure keywords in stdout/stderr
    if exit_code == 0:
        # Crude check, proper parsing (e.g., JUnit XML) is better
        if "failed" in stdout.lower() or "error" in stdout.lower():
             test_results["summary"] = "pass_with_failures" # Some tests might have passed
             error_logs.append(f"Tests ran, but reported failures/errors (Exit Code 0):\nStdout:\n{stdout}\nStderr:\n{stderr}")
        else:
             test_results["summary"] = "pass"
    # Specific exit codes for pytest indicate failures: https://docs.pytest.org/en/7.1.x/reference/exit-codes.html
    elif exit_code == 1: # Pytest: tests collected and run but some failed
         test_results["summary"] = "fail"
         error_logs.append(f"Pytest reported test failures (Exit Code 1):\nStdout:\n{stdout}\nStderr:\n{stderr}")
    elif exit_code == 5: # Pytest: No tests collected
         test_results["summary"] = "no_tests_found"
         error_logs.append(f"Pytest reported no tests collected (Exit Code 5):\nStdout:\n{stdout}\nStderr:\n{stderr}")
    else: # Other non-zero exit codes often mean execution error
        test_results["summary"] = "execution_error"
        error_logs.append(f"Test execution failed (Exit Code {exit_code}):\nStdout:\n{stdout}\nStderr:\n{stderr}")

    # TODO: Add more sophisticated parsing of stdout/stderr or generated report files
    # (e.g., read a JUnit XML file created within the container in the mounted volume)

    # Determine overall build status based on tests
    build_status = "Testing Failed"
    if test_results.get("summary") == "pass":
        build_status = "Tests Passed"


    print(f"--- Exiting Tester Node (Status: {build_status}) ---")
    return {"test_results": test_results, "error_logs": error_logs, "build_status": build_status}

@traceable(run_type="chain", name="generate_workflow_graph")
def generate_workflow_graph(task: str, output_folder: Optional[str] = None, save_visualization: bool = True) -> Dict[str, Any]:
    """
    Generate and compile the workflow graph without running it.
    
    Args:
        task: The software development task description
        output_folder: Optional folder to save generated files
        save_visualization: Whether to save a visualization of the graph
        
    Returns:
        A dictionary containing the compiled graph and initial state
    """
    # Create output folder if specified
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    
    # Generate a trace ID if tracing is enabled
    trace_id = create_trace_id() if is_tracing_enabled() else None
    
    # Initialize the state
    initial_state = WorkflowState(
        task=task,
        current_step="",
        requirements=[],
        design={},
        project_structure={},
        code_files={},
        file_dependencies={},
        documentation="",
        messages=[],
        output_folder=output_folder,
        trace_id=trace_id
    )
    
    try:
        # Create and compile the workflow graph
        workflow = create_workflow_graph().compile()
        
        # Save workflow graph visualization if requested
        if save_visualization and output_folder:
            try:
                # Get the graph visualization as PNG
                image_data = workflow.get_graph(xray=True).draw_mermaid_png()
                
                # Save the image to the output folder
                graph_path = os.path.join(output_folder, "workflow_graph.png")
                with open(graph_path, "wb") as f:
                    f.write(image_data)
                
                print(f"Workflow graph visualization saved to: {graph_path}")
                
                # Trace tool usage for file creation
                if is_tracing_enabled() and trace_id:
                    trace_tool_usage(
                        tool_name="save_graph_visualization",
                        input_data={
                            "file_path": graph_path
                        },
                        output_data=f"Graph visualization saved to {graph_path}",
                        metadata={
                            "trace_id": trace_id
                        }
                    )
            except Exception as e:
                print(f"Error saving workflow graph visualization: {e}")
        
        return {
            "graph": workflow,
            "initial_state": initial_state,
            "trace_id": trace_id
        }
    
    except Exception as e:
        error_message = f"Error generating workflow graph: {str(e)}"
        print(error_message)
        return {
            "error": error_message,
            "trace_id": trace_id
        }

@traceable(run_type="chain", name="run_software_dev_workflow", project=LANGCHAIN_PROJECT)
def run_software_dev_workflow(task: str, output_folder: Optional[str] = None, compile_only: bool = False) -> str:
    """
    Run the software development workflow.
    
    Args:
        task: The software development task description
        output_folder: Optional folder to save generated files
        compile_only: If True, only compile the graph without running it
        
    Returns:
        A summary of the workflow execution or a message if compile_only is True
    """
    # Create output folder if specified
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    
    # Generate a trace ID if tracing is enabled
    trace_id = create_trace_id() if is_tracing_enabled() else None
    
    # If compile_only is True, only generate and compile the graph without running it
    if compile_only:
        result = generate_workflow_graph(task, output_folder)
        
        if "error" in result:
            return f"Error generating workflow graph: {result['error']}"
        
        return "Workflow graph compiled successfully. Graph visualization saved to the output folder."
    
    # Initialize the state
    initial_state = WorkflowState(
        task=task,
        current_step="",
        requirements=[],
        design={},
        project_structure={},
        code_files={},
        file_dependencies={},
        documentation="",
        messages=[],
        output_folder=output_folder,
        trace_id=trace_id
    )
    
    try:
        # Create and compile the workflow
        workflow = create_workflow_graph().compile()
        
        # Run the workflow with proper tracing
        if is_tracing_enabled() and trace_id:
            # Set up the run metadata
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
            
            # Run the workflow
            result = workflow.invoke(initial_state)
            
            # Log the result to LangSmith using trace_llm_call
            trace_tool_usage(
                tool_name="workflow_execution",
                input_data={"task": task, "output_folder": output_folder},
                output_data={"files_created": list(result["code_files"].keys())},
                metadata={"trace_id": trace_id}
            )
        else:
            # Run without special tracing
            result = workflow.invoke(initial_state)
        
        # Generate a summary of what was done
        summary = "# Software Development Workflow Summary\n\n"
        
        if result["code_files"]:
            summary += "## Files Created\n"
            for file_name in result["code_files"].keys():
                summary += f"- {file_name}\n"
            summary += "\n"

        # Save the summary to the output folder
        if output_folder:
            with open(os.path.join(output_folder, "workflow_summary.md"), "w") as f:
                f.write(summary)
                
            # Add LangSmith trace information if tracing was enabled
            if is_tracing_enabled() and trace_id:
                langsmith_info = "\n\n## LangSmith Tracing\n"
                langsmith_info += f"This workflow execution was traced in LangSmith with trace ID: {trace_id}\n"
                langsmith_info += f"View the trace at: {LANGCHAIN_ENDPOINT}/o/default/projects/{LANGCHAIN_PROJECT}/traces/{trace_id}\n"
                
                with open(os.path.join(output_folder, "langsmith_trace.md"), "w") as f:
                    f.write(langsmith_info)
                
                summary += langsmith_info
        
        return summary
    
    except Exception as e:
        error_message = f"Error in workflow execution: {str(e)}"
        print(error_message)
        
        # Fall back to the simpler approach if the workflow fails
        agent = SoftwareDevelopmentAgent(output_folder=output_folder)
        response = agent.query(task)
        return response

def test_workflow():
    """Test function to verify the workflow and catch any serialization issues."""
    try:
        # Test simple task
        task = "Create a simple calculator with add and subtract functions"
        result = run_software_dev_workflow(task, output_folder="test_output_workflow")
        print("Test completed successfully")
        print(result)
        return True
    except Exception as e:
        print(f"Test failed with error: {e}")
        return False

if __name__ == "__main__":
    test_workflow()
