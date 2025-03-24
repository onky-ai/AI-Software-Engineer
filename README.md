# Software Development Agent

A software development agent built with LangChain and LangGraph, powered by Claude 3.7 Sonnet.

## Overview

This agent is designed to assist in developing custom software by leveraging the capabilities of Claude 3.7 Sonnet through LangChain and LangGraph frameworks.

## Features

- Connects to Claude 3.7 Sonnet API
- Uses LangChain for structured interactions with the LLM
- Implements LangGraph for complex agent workflows
- Integrates with LangSmith for tracing, monitoring, and debugging
- Assists in software development tasks

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   
   # Optional: LangSmith Configuration
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   LANGCHAIN_API_KEY=your_langsmith_api_key_here
   LANGCHAIN_PROJECT=software-development-agent
   ```
4. Run the agent:
   ```
   python main.py
   ```

## Usage

The agent supports multiple modes of operation:

### Interactive Mode

Run the agent in interactive mode to engage in a conversation:

```
python main.py --mode=interactive --folder=output_folder
```

### Workflow Mode

Run the agent in workflow mode to execute the full development workflow:

```
python main.py --mode=workflow --query="Create a simple calculator in Python" --folder=output_folder
```

### Compile Mode

Run the agent in compile-only mode to generate and visualize the workflow graph without executing it:

```
python main.py --mode=compile --query="Create a simple calculator in Python" --folder=output_folder
```

## LangSmith Integration

This agent integrates with LangSmith for tracing, monitoring, and debugging. To enable LangSmith integration:

1. Sign up for LangSmith at [smith.langchain.com](https://smith.langchain.com/)
2. Get your API key from the LangSmith dashboard
3. Set the following environment variables in your `.env` file:
   ```
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   LANGCHAIN_API_KEY=your_langsmith_api_key_here
   LANGCHAIN_PROJECT=software-development-agent
   ```

With LangSmith integration enabled, you can:
- View detailed traces of workflow execution
- Debug issues with specific steps or components
- Analyze LLM inputs and outputs
- Monitor performance and usage

## Project Structure

- `main.py`: Entry point for the application
- `agent/`: Core agent implementation
  - `agent.py`: Main agent definition
  - `tools.py`: Custom tools for the agent
  - `workflows/`: LangGraph workflows
    - `software_dev_workflow.py`: Simple workflow implementation
    - `langgraph_dev_workflow.py`: Advanced workflow with LangGraph
  - `langsmith_utils.py`: Utilities for LangSmith integration
- `config.py`: Configuration settings
- `utils.py`: Utility functions 