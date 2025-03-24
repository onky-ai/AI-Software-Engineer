"""
LangSmith utilities for tracing, monitoring, and evaluation.
"""

import os
import sys
import uuid
from typing import Any, Dict, Optional
from langsmith import Client, traceable
from langsmith.run_trees import RunTree

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    LANGCHAIN_TRACING_V2,
    LANGCHAIN_ENDPOINT,
    LANGCHAIN_API_KEY,
    LANGCHAIN_PROJECT
)

def get_langsmith_client() -> Optional[Client]:
    """
    Get a LangSmith client if tracing is enabled.
    
    Returns:
        A LangSmith client or None if tracing is disabled
    """
    if not LANGCHAIN_TRACING_V2 or not LANGCHAIN_API_KEY:
        return None
    
    return Client(
        api_key=LANGCHAIN_API_KEY,
        api_url=LANGCHAIN_ENDPOINT
    )

@traceable(run_type="chain")
def trace_llm_call(model: str, prompt: str, response: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Trace an LLM call to LangSmith.
    
    Args:
        model: The model name
        prompt: The prompt sent to the model
        response: The response from the model
        metadata: Additional metadata to include in the trace
        
    Returns:
        A dictionary with the traced data
    """
    return {
        "model": model,
        "prompt": prompt,
        "response": response,
        "metadata": metadata or {}
    }

@traceable(run_type="tool")
def trace_tool_usage(tool_name: str, input_data: Any, output_data: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Trace a tool usage to LangSmith.
    
    Args:
        tool_name: The name of the tool
        input_data: The input data sent to the tool
        output_data: The output data from the tool
        metadata: Additional metadata to include in the trace
        
    Returns:
        A dictionary with the traced data
    """
    return {
        "tool_name": tool_name,
        "input": input_data,
        "output": output_data,
        "metadata": metadata or {}
    }

def create_trace_id() -> str:
    """
    Create a unique trace ID.
    
    Returns:
        A unique trace ID
    """
    return str(uuid.uuid4())

def is_tracing_enabled() -> bool:
    """
    Check if tracing is enabled.
    
    Returns:
        True if tracing is enabled, False otherwise
    """
    return LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY 