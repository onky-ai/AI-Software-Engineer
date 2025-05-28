# AI Software Engineer

A software development agent built with LangChain and LangGraph, powered by Claude 3.7 Sonnet.

## Overview

This agent is designed to assist in developing custom software by leveraging the capabilities of Claude 3.7 Sonnet through LangChain and LangGraph frameworks. It provides a structured approach to software development, from requirements analysis to code generation and verification.

## Features

### Open Source Features

- Connects to Claude 3.7 Sonnet API
- Uses LangChain for structured interactions with the LLM
- Implements LangGraph for complex agent workflows
- Integrates with LangSmith for tracing, monitoring, and debugging
- Assists in software development tasks
- Supports multiple development modes (interactive, workflow, compile)
- Provides comprehensive code verification and completeness checks
- Docker integration for secure code execution
- Comprehensive testing framework

### Enterprise Features

For enterprise customers, additional features are available under a separate license:

- Advanced security and compliance features
- Multi-tenant architecture support
- Enterprise-grade authentication and authorization
- Advanced monitoring and analytics
- Priority support and SLA guarantees
- Custom integrations and workflows
- On-premises deployment options

For enterprise licensing inquiries, contact: enterprise@onky.ai

## Setup

1. Clone this repository:
   ```bash
   git clone git@github.com:onky-ai/AI-Software-Engineer.git
   cd AI-Software-Engineer
   ```

2. Install dependencies:
   ```bash
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
   ```bash
   python main.py
   ```

## Usage

The agent supports multiple modes of operation:

### Interactive Mode

Run the agent in interactive mode to engage in a conversation:

```bash
python main.py --mode=interactive --folder=output_folder
```

### Workflow Mode

Run the agent in workflow mode to execute the full development workflow:

```bash
python main.py --mode=workflow --query="Create a simple calculator in Python" --folder=output_folder
```

### Compile Mode

Run the agent in compile-only mode to generate and visualize the workflow graph without executing it:

```bash
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

```
agent/
├── __init__.py
├── agent.py              # Main agent implementation
├── models/              # Pydantic models for structured outputs
│   └── default.py
├── utils/              # Utility functions
│   ├── __init__.py
│   ├── docker_utils.py
│   └── langsmith_utils.py
└── workflows/          # LangGraph workflows
    ├── __init__.py
    ├── default.py
    └── langgraph_dev_workflow.py
├── main.py            # Entry point for the application
├── config.py          # Configuration settings
└── requirements.txt   # Project dependencies
```

## Development Workflow

The agent follows a structured development workflow:

1. **Requirements Analysis**: Analyzes the task and extracts clear requirements
2. **Design Creation**: Creates a detailed design based on requirements
3. **Project Structure**: Proposes a suitable project structure
4. **Code Generation**: Generates code files based on the design
5. **Completeness Verification**: Verifies and improves code completeness
6. **Documentation**: Creates comprehensive documentation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project uses a dual licensing model:

### Open Source License

The code in this repository, except for the contents of the "enterprise" directory, is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). This applies to the open core components of the software.

### Enterprise License

All code in the "enterprise" directory and enterprise-specific features are licensed under a separate Enterprise License. See `enterprise/LICENSE.md` for details.

For enterprise licensing inquiries, contact: enterprise@onky.ai

See the LICENSE file for the complete AGPL-3.0 license text. 