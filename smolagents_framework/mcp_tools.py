"""
MCP Tools Integration with SmolaGents
======================================

This script demonstrates how to integrate tools from a Model Context Protocol (MCP) server
with a SmolaGents CodeAgent. It uses a local Ollama model to power the agent and loads
tools from the PubMed MCP server to answer medical questions.

Requirements:
    - Ollama running locally on http://127.0.0.1:11434
    - Python 3.13 installed
    - uv package manager installed (for uvx command)
    - Required packages: smolagents, langchain-community, mcp

Note:
    This script is meant to be run as a standalone Python script, not in Jupyter notebooks,
    due to Windows subprocess compatibility issues.
"""

import os
from smolagents import ToolCollection, CodeAgent, LiteLLMModel
from mcp import StdioServerParameters


def initialize_model():
    """
    Initialize the LiteLLMModel that will power the agent.
    Uses Qwen 2.5 Coder model running locally via Ollama.
    
    Returns:
        LiteLLMModel: Configured model instance
    """
    model = LiteLLMModel(
        model_id="ollama_chat/qwen2.5-coder:7b",
        api_base="http://127.0.0.1:11434",
        max_tokens=2048,
        num_ctx=8192
    )
    return model


def setup_mcp_server():
    """
    Configure the MCP server parameters for PubMed integration.
    
    Returns:
        StdioServerParameters: Server configuration for the MCP connection
    """
    server_parameters = StdioServerParameters(
        command="uvx",  # Use uv package manager to run the MCP server
        args=["--quiet", "pubmedmcp@0.1.3"],  # PubMed MCP server version
        env={"UV_PYTHON": "3.13", **os.environ},  # Specify Python version and inherit environment
    )
    return server_parameters


def run_agent_with_mcp_tools(model, server_parameters):
    """
    Create and run a CodeAgent with MCP tools from the PubMed server.
    
    Args:
        model: The LiteLLMModel instance to use
        server_parameters: MCP server configuration
    """
    # Load tools from the MCP server and create an agent
    with ToolCollection.from_mcp(server_parameters, trust_remote_code=True) as tool_collection:
        # Initialize the CodeAgent with MCP tools and base tools
        agent = CodeAgent(
            tools=[*tool_collection.tools],  # Unpack all MCP tools
            model=model,
            add_base_tools=True  # Include default smolagents tools
        )
        
        # Run the agent with a specific query
        result = agent.run("Please find a remedy for hangover.")
        
        return result


def main():
    """
    Main entry point for the script.
    Initializes the model, sets up MCP server, and runs the agent.
    """
    print("Initializing LiteLLM model...")
    model = initialize_model()
    
    print("Setting up MCP server parameters...")
    server_parameters = setup_mcp_server()
    
    print("Connecting to MCP server and running agent...")
    print("-" * 60)
    
    result = run_agent_with_mcp_tools(model, server_parameters)
    
    print("-" * 60)
    print("Agent completed successfully!")


if __name__ == "__main__":
    main()