import re
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Literal, Union
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
import json
import os
import sys
from langsmith import traceable
from langsmith.run_trees import RunTree

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent.agent import SoftwareDevelopmentAgent
from config import DEFAULT_SYSTEM_PROMPT, LANGCHAIN_PROJECT, LANGCHAIN_ENDPOINT
from agent.langsmith_utils import is_tracing_enabled, trace_llm_call, trace_tool_usage, create_trace_id

# Define the state type for our workflow
class WorkflowState(TypedDict):
    task: str
    current_step: str
    requirements: List[str]
    design: Dict[str, Any]
    implementation_plan: List[str]
    code_files: Dict[str, str]
    documentation: str
    messages: List[Dict[str, str]]
    output_folder: Optional[str]
    trace_id: Optional[str]

# Define the workflow steps
@traceable(run_type="chain", name="analyze_requirements")
def analyze_requirements(state: WorkflowState) -> WorkflowState:
    """Analyze the task and extract requirements"""
    agent = SoftwareDevelopmentAgent()
    
    prompt = f"""
    Analyze the following software development task and extract clear requirements:
    
    {state["task"]}
    
    Please provide a structured list of requirements that need to be implemented.
    Return only the requirements, no other text. 
    List each requirement on a separate line.
    """
    
    response = agent.query(prompt)
    
    # Trace the LLM call if tracing is enabled
    if is_tracing_enabled() and state.get("trace_id"):
        trace_llm_call(
            model=agent.llm.model,
            prompt=prompt,
            response=response,
            metadata={
                "step": "analyze_requirements",
                "trace_id": state["trace_id"]
            }
        )
    
    state["requirements"] = [req.strip() for req in response.split("\n") if req.strip()]
    state["messages"].append({"role": "system", "content": f"Requirements analyzed: {len(state['requirements'])} requirements identified"})
    state["current_step"] = "requirements_analyzed"
    
    print(f"Analyze Requirements: {state['requirements']}")
    return state

@traceable(run_type="chain", name="create_design")
def create_design(state: WorkflowState) -> WorkflowState:
    """Create a high-level design based on requirements"""
    agent = SoftwareDevelopmentAgent()
    
    requirements_text = "\n".join([f"- {req}" for req in state["requirements"]])
    prompt = f"""
    Based on these requirements:
    
    {requirements_text}
    
    Create a high-level software design including:
    1. Architecture overview
    2. Main components
    3. Data models
    4. API endpoints (if applicable)
    5. Dependencies and libraries needed
    
    Provide the design in a structured format.
    """
    
    response = agent.query(prompt)
    
    # Trace the LLM call if tracing is enabled
    if is_tracing_enabled() and state.get("trace_id"):
        trace_llm_call(
            model=agent.llm.model,
            prompt=prompt,
            response=response,
            metadata={
                "step": "create_design",
                "trace_id": state["trace_id"]
            }
        )
    
    # Parse the response into a structured design
    state["design"] = {
        "description": response,
        "components": [],
        "data_models": [],
        "dependencies": []
    }
    
    state["messages"].append({"role": "system", "content": "Design created"})
    state["current_step"] = "design_created"
    
    return state

@traceable(run_type="chain", name="create_implementation_plan")
def create_implementation_plan(state: WorkflowState) -> WorkflowState:
    """Create an implementation plan based on the design"""
    agent = SoftwareDevelopmentAgent()
    
    prompt = f"""
    Based on this design:
    
    {state["design"]["description"]}
    
    Create a step-by-step implementation plan. List the files that need to be created
    and the order in which they should be implemented.
    """
    
    response = agent.query(prompt)
    
    # Trace the LLM call if tracing is enabled
    if is_tracing_enabled() and state.get("trace_id"):
        trace_llm_call(
            model=agent.llm.model,
            prompt=prompt,
            response=response,
            metadata={
                "step": "create_implementation_plan",
                "trace_id": state["trace_id"]
            }
        )
    
    state["implementation_plan"] = [step.strip() for step in response.split("\n") if step.strip()]
    state["messages"].append({"role": "system", "content": f"Implementation plan created with {len(state['implementation_plan'])} steps"})
    state["current_step"] = "implementation_plan_created"
    
    return state

@traceable(run_type="chain", name="implement_code")
def implement_code(state: WorkflowState) -> WorkflowState:
    """Implement the code based on the implementation plan"""
    agent = SoftwareDevelopmentAgent(output_folder=state["output_folder"])
    state["code_files"] = {}
    
    # Extract file names from implementation plan
    file_pattern = r'`([^`]+\.[a-zA-Z0-9]+)`'
    all_files = []
    
    for step in state["implementation_plan"]:
        matches = re.findall(file_pattern, step)
        all_files.extend(matches)
    
    # If no files were found in the format, ask the agent to identify files
    if not all_files:
        prompt = f"""
        Based on the implementation plan and design, list all the files that need to be created
        for this project. Format each filename on a new line.
        
        Design: {state["design"]["description"]}
        """
        
        response = agent.query(prompt)
        
        # Trace the LLM call if tracing is enabled
        if is_tracing_enabled() and state.get("trace_id"):
            trace_llm_call(
                model=agent.llm.model,
                prompt=prompt,
                response=response,
                metadata={
                    "step": "implement_code_file_list",
                    "trace_id": state["trace_id"]
                }
            )
        
        all_files = [line.strip() for line in response.split("\n") if line.strip() and "." in line]
    
    # Remove duplicates while preserving order
    unique_files = []
    for file in all_files:
        if file not in unique_files:
            unique_files.append(file)
    
    # Generate code for each file
    for file_name in unique_files:
        prompt = f"""
        Create the code for the file `{file_name}` based on:
        
        Requirements:
        {chr(10).join([f"- {req}" for req in state["requirements"]])}
        
        Design:
        {state["design"]["description"]}
        
        Provide only the code, properly formatted and complete.
        Make sure to include all necessary imports and dependencies.
        """
        
        # Get the language from the file extension
        language = file_name.split(".")[-1]
        
        # Generate the code
        code = agent.generate_code(prompt, language, file_name)
        
        # Trace the LLM call if tracing is enabled
        if is_tracing_enabled() and state.get("trace_id"):
            trace_llm_call(
                model=agent.llm.model,
                prompt=prompt,
                response=code,
                metadata={
                    "step": f"implement_code_{file_name}",
                    "file_name": file_name,
                    "language": language,
                    "trace_id": state["trace_id"]
                }
            )
        
        state["code_files"][file_name] = code
        
        # The agent will automatically save the file to the output folder
        state["messages"].append({"role": "system", "content": f"Created file: {file_name}"})
        
        # Trace tool usage for file creation
        if is_tracing_enabled() and state.get("trace_id"):
            trace_tool_usage(
                tool_name="file_write",
                input_data={
                    "file_name": file_name,
                    "content": code
                },
                output_data=f"File {file_name} created successfully",
                metadata={
                    "step": "implement_code",
                    "file_name": file_name,
                    "trace_id": state["trace_id"]
                }
            )
    
    # Create any necessary directory structure
    if state["output_folder"]:
        # Extract directory structure from design or implementation plan
        dir_structure_prompt = f"""
        Based on the implementation plan and design, list all the directories that need to be created
        for this project. Format each directory path on a new line.
        
        Design: {state["design"]["description"]}
        Implementation Plan: {chr(10).join(state["implementation_plan"])}
        """
        
        dir_response = agent.query(dir_structure_prompt)
        
        # Trace the LLM call if tracing is enabled
        if is_tracing_enabled() and state.get("trace_id"):
            trace_llm_call(
                model=agent.llm.model,
                prompt=dir_structure_prompt,
                response=dir_response,
                metadata={
                    "step": "directory_structure",
                    "trace_id": state["trace_id"]
                }
            )
        
        directories = [line.strip() for line in dir_response.split("\n") if line.strip() and "/" in line and not "." in line.split("/")[-1]]
        
        # Create directories
        from utils import ensure_directory
        for directory in directories:
            dir_path = os.path.join(state["output_folder"], directory)
            ensure_directory(dir_path)
            state["messages"].append({"role": "system", "content": f"Created directory: {directory}"})
            
            # Trace tool usage for directory creation
            if is_tracing_enabled() and state.get("trace_id"):
                trace_tool_usage(
                    tool_name="create_directory",
                    input_data={
                        "directory": directory
                    },
                    output_data=f"Directory {directory} created successfully",
                    metadata={
                        "step": "implement_code",
                        "directory": directory,
                        "trace_id": state["trace_id"]
                    }
                )
    
    state["current_step"] = "code_implemented"
    return state

@traceable(run_type="chain", name="create_documentation")
def create_documentation(state: WorkflowState) -> WorkflowState:
    """Create documentation for the project"""
    agent = SoftwareDevelopmentAgent()
    
    files_list = "\n".join([f"- {file_name}" for file_name in state["code_files"].keys()])
    
    prompt = f"""
    Create comprehensive documentation for this software project:
    
    Requirements:
    {chr(10).join([f"- {req}" for req in state["requirements"]])}
    
    Design:
    {state["design"]["description"]}
    
    Files implemented:
    {files_list}
    
    Include:
    1. Project overview
    2. Installation instructions
    3. Usage examples
    4. API documentation (if applicable)
    5. Configuration options
    """
    
    response = agent.query(prompt)
    
    # Trace the LLM call if tracing is enabled
    if is_tracing_enabled() and state.get("trace_id"):
        trace_llm_call(
            model=agent.llm.model,
            prompt=prompt,
            response=response,
            metadata={
                "step": "create_documentation",
                "trace_id": state["trace_id"]
            }
        )
    
    state["documentation"] = response
    
    # Save documentation if output folder is specified
    if state["output_folder"]:
        doc_path = os.path.join(state["output_folder"], "README.md")
        with open(doc_path, "w") as f:
            f.write(state["documentation"])
        
        # Trace tool usage for file creation
        if is_tracing_enabled() and state.get("trace_id"):
            trace_tool_usage(
                tool_name="file_write",
                input_data={
                    "file_name": "README.md",
                    "content": state["documentation"]
                },
                output_data=f"File README.md created successfully",
                metadata={
                    "step": "create_documentation",
                    "file_name": "README.md",
                    "trace_id": state["trace_id"]
                }
            )
    
    state["messages"].append({"role": "system", "content": "Documentation created"})
    state["current_step"] = "documentation_created"
    
    return state

@traceable(run_type="chain", name="router")
def router(state: WorkflowState) -> Literal["analyze_requirements", "create_design", "create_implementation_plan", "implement_code", "create_documentation", "END"]:
    """Route to the next step based on the current state"""
    current_step = state.get("current_step", "")
    
    if current_step == "":
        return "analyze_requirements"
    elif current_step == "requirements_analyzed":
        return "create_design"
    elif current_step == "design_created":
        return "create_implementation_plan"
    elif current_step == "implementation_plan_created":
        return "implement_code"
    elif current_step == "code_implemented":
        return "create_documentation"
    elif current_step == "documentation_created":
        return "END"
    else:
        return "analyze_requirements"

# Create the workflow graph
@traceable(run_type="chain", name="create_workflow_graph")
def create_workflow_graph() -> StateGraph:
    """Create the workflow graph"""
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("analyze_requirements", analyze_requirements)
    workflow.add_node("create_design", create_design)
    workflow.add_node("create_implementation_plan", create_implementation_plan)
    workflow.add_node("implement_code", implement_code)
    workflow.add_node("create_documentation", create_documentation)
    
    # Add edge from START to first node
    workflow.add_edge(START, "analyze_requirements")


    # Add conditional edges between nodes
    workflow.add_conditional_edges(
        "analyze_requirements",
        router,
        {
            "create_design": "create_design",
            "create_implementation_plan": "create_implementation_plan",
            "implement_code": "implement_code",
            "create_documentation": "create_documentation",
            "END": END
        }
    )
    
    workflow.add_conditional_edges(
        "create_design",
        router,
        {
            "create_implementation_plan": "create_implementation_plan",
            "implement_code": "implement_code",
            "create_documentation": "create_documentation",
            "END": END
        }
    )
    
    workflow.add_conditional_edges(
        "create_implementation_plan",
        router,
        {
            "implement_code": "implement_code",
            "create_documentation": "create_documentation",
            "END": END
        }
    )
    
    workflow.add_conditional_edges(
        "implement_code",
        router,
        {
            "create_documentation": "create_documentation",
            "END": END
        }
    )
    
    workflow.add_conditional_edges(
        "create_documentation",
        router,
        {
            "END": END
        }
    )
    
    return workflow

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
        implementation_plan=[],
        code_files={},
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
        implementation_plan=[],
        code_files={},
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
        
        if result["requirements"]:
            summary += "## Requirements\n"
            for req in result["requirements"]:
                summary += f"- {req}\n"
            summary += "\n"
        
        if result["design"]:
            summary += "## Design\n"
            summary += result["design"].get("description", "No design description available.")
            summary += "\n\n"
        
        if result["code_files"]:
            summary += "## Files Created\n"
            for file_name in result["code_files"].keys():
                summary += f"- {file_name}\n"
            summary += "\n"
        
        if result["documentation"]:
            summary += "## Documentation\n"
            summary += result["documentation"]
        
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
