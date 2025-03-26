import os
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check if the API key is set
if not os.getenv("ANTHROPIC_API_KEY"):
    print("Error: ANTHROPIC_API_KEY environment variable is not set.")
    print("Please create a .env file with your API key or set it in your environment.")
    print("Example .env file:")
    print("ANTHROPIC_API_KEY=your_api_key_here")
    exit(1)

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Software Development Agent")
    parser.add_argument("--mode", choices=["interactive", "workflow", "compile"], default="interactive",
                        help="Mode to run the agent in (interactive, workflow, or compile)")
    parser.add_argument("--query", type=str, help="Query to send to the agent in workflow or compile mode")
    parser.add_argument("--folder", type=str, default="generated_code",
                        help="Folder where generated code will be saved")
    args = parser.parse_args()
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(args.folder):
        os.makedirs(args.folder)
    
    if args.mode == "interactive":
        run_interactive_mode(args.folder)
    elif args.mode == "workflow":
        if not args.query:
            print("Error: --query is required in workflow mode")
            exit(1)
        run_workflow_mode(args.query, args.folder)
    elif args.mode == "compile":
        if not args.query:
            print("Error: --query is required in compile mode")
            exit(1)
        run_compile_mode(args.query, args.folder)

def run_interactive_mode(output_folder):
    """
    Run the agent in interactive mode.
    
    Args:
        output_folder: Folder where generated code will be saved
    """
    from agent import SoftwareDevelopmentAgent
    
    print("Software Development Agent (Interactive Mode)")
    print(f"Generated code will be saved to: {os.path.abspath(output_folder)}")
    print("Type 'exit' to quit")
    print()
    
    agent = SoftwareDevelopmentAgent(output_folder=output_folder)
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ["exit", "quit", "q"]:
            break
            
        if not user_input.strip():
            print("Please enter a valid query.")
            continue
        
        response = agent.query(user_input)
        
        if output_folder:
            agent._save_code_blocks_to_files(response)
            
            # Also try to extract project structure and create directories
            agent._create_project_structure(response)
            
        print("\nAgent:", response)
        print()

def run_compile_mode(query, output_folder):
    """
    Run the agent in compile-only mode (generate graph without running workflow).
    
    Args:
        query: The query to send to the agent
        output_folder: Folder where graph visualization will be saved
    """
    from agent.workflows import run_software_dev_workflow
    
    print("Software Development Agent (Compile Mode)")
    print(f"Query: {query}")
    print(f"Graph visualization will be saved to: {os.path.abspath(output_folder)}")
    print()
    
    response = run_software_dev_workflow(query, output_folder, compile_only=True)
    print("\nResult:", response)

def run_workflow_mode(query, output_folder):
    """
    Run the agent in workflow mode.
    
    Args:
        query: The query to send to the agent
        output_folder: Folder where generated code will be saved
    """
    from agent.workflows import run_software_dev_workflow
    
    print("Software Development Agent (Workflow Mode)")
    print(f"Query: {query}")
    print(f"Generated code will be saved to: {os.path.abspath(output_folder)}")
    print()
    
    response = run_software_dev_workflow(query, output_folder, compile_only=False)
    print("\nAgent:", response)

if __name__ == "__main__":
    main() 