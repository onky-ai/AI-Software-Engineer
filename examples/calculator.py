import os
# Add the parent directory to the path so we can import from the root
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.workflows import run_software_dev_workflow

def main():
    # Example task
    task = """Create a Python calculator application with addition, subtraction, 
    multiplication, and division functions. Implement a proper folder structure 
    with separate modules for operations, user interface, and tests. Include 
    appropriate error handling for division by zero and invalid inputs."""
    # Run the workflow with output folder
    response = run_software_dev_workflow(task, output_folder="generated_calculator")
    
    print(response)

if __name__ == "__main__":
    main() 