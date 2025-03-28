import os
# Add the parent directory to the path so we can import from the root
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.workflows import run_software_dev_workflow

def main():
    # Example task
    task = """
    Create a simple Flask application that:
    1. Has a route for /hello-world that returns "Hello, World!"
    2. Connects to a PostgreSQL database
    3. Has a route for /users that returns all users from the database
    4. Uses environment variables for sensitive information
    """
    
    # Run the workflow with output folder
    response = run_software_dev_workflow(task, output_folder="generated_flask_app")
    
    # Print the response
    print(response)

if __name__ == "__main__":
    main() 