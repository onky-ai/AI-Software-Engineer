import os
import sys

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import SoftwareDevelopmentAgent

def main():
    """Run a simple example of the software development agent."""
    # Create the agent
    agent = SoftwareDevelopmentAgent()
    
    # Example 1: Generate a simple Python function
    print("Example 1: Generate a simple Python function")
    prompt = "Write a Python function that calculates the Fibonacci sequence up to n terms."
    response = agent.generate_code(prompt, "python")
    print("\nGenerated Code:")
    print(response)
    print("\n" + "-" * 80 + "\n")
    
    # Example 2: Explain some code
    print("Example 2: Explain some code")
    code = """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
    """
    response = agent.explain_code(code, "python")
    print("\nExplanation:")
    print(response)
    print("\n" + "-" * 80 + "\n")
    
    # Example 3: Debug some code
    print("Example 3: Debug some code")
    buggy_code = """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)
    
result = calculate_average([])
print(result)
    """
    error_message = "ZeroDivisionError: division by zero"
    response = agent.debug_code(buggy_code, error_message, "python")
    print("\nDebugging Result:")
    print(response)

if __name__ == "__main__":
    main() 