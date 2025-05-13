import os
import json
from typing import Dict, Any, Optional, List, Union
import logging
from functools import wraps
import inspect
import time

from openai import OpenAI
from agents import Agent, Runner, WebSearchTool, ModelSettings

from logger_utils import trace_api_calls, DepthContext

class APIWrapper:
    """Wrapper around OpenAI API that logs all requests and responses"""

    def __init__(self, logger, api_key=None):
        """
        Initialize the API wrapper

        Args:
            logger: The logger to use for API call tracing
            api_key: OpenAI API key (uses env var if not provided)
        """
        self.logger = logger
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)

        # Apply the trace_api_calls decorator to all relevant methods
        self.trace_decorator = trace_api_calls(logger)

    @property
    def openai_client(self):
        """Get the underlying OpenAI client"""
        return self.client

    async def create_agent(self,
                    name: str,
                    instructions: str,
                    model: str,
                    tools: List[Any] = None,
                    model_settings: ModelSettings = None,
                    handoffs: List[Agent] = None) -> Agent:
        """
        Create an agent with detailed logging

        Args:
            name: Agent name
            instructions: Agent instructions
            model: Model name
            tools: Agent tools (optional)
            model_settings: Model settings (optional)
            handoffs: Agent handoffs (optional)

        Returns:
            Agent instance
        """
        self.logger.info(f"Creating agent '{name}' with model: {model}")
        if tools:
            tool_names = [str(t.__name__) if hasattr(t, '__name__') else str(t) for t in tools]
            self.logger.debug(f"Agent tools: {', '.join(tool_names)}")

        with DepthContext(self.logger, f"Agent Initialization: {name}"):
            agent = Agent(
                name=name,
                instructions=instructions,
                model=model,
                tools=tools or [],
                model_settings=model_settings or ModelSettings(temperature=0),
                handoffs=handoffs or []
            )

        return agent

    async def run_agent(self, agent, input_text):
        """
        Run an agent with detailed logging

        Args:
            agent: The agent to run
            input_text: Input text for the agent

        Returns:
            Agent run result
        """
        agent_name = getattr(agent, 'name', 'unnamed_agent')
        self.logger.info(f"Running agent '{agent_name}'")
        self.logger.debug(f"Agent input: {input_text[:100]}..." if len(input_text) > 100 else input_text)

        with DepthContext(self.logger, f"Agent Run: {agent_name}"):
            start_time = time.time()
            result = await Runner.run(agent, input=input_text)
            duration = time.time() - start_time

            # Log token usage if available
            if hasattr(result, 'usage') and result.usage:
                self.logger.debug(f"Token usage: {result.usage}")

            # Log completion tokens
            self.logger.debug(f"Agent completed in {duration:.2f}s")

            # Log output (truncated if very long)
            output = result.final_output
            truncated_output = output[:500] + "..." if len(output) > 500 else output
            self.logger.debug(f"Agent output: {truncated_output}")

        return result

    # Apply the trace decorator to OpenAI client methods we want to track
    @trace_api_calls
    async def chat_completion(self, model, messages, **kwargs):
        """Create a chat completion with tracing"""
        return await self.client.chat.completions.create(model=model, messages=messages, **kwargs)

    @trace_api_calls
    async def create_response(self, model, input_text, tools=None, **kwargs):
        """Create a response with tracing"""
        return await self.client.responses.create(model=model, input=input_text, tools=tools, **kwargs)

    @trace_api_calls
    async def web_search(self, query):
        """Perform a web search with tracing"""
        tool = WebSearchTool()
        return await tool(query)

# Create a singleton instance to be imported and used throughout the app
_api_wrapper_instance = None

def initialize_api_wrapper(logger):
    """Initialize the API wrapper singleton"""
    global _api_wrapper_instance
    _api_wrapper_instance = APIWrapper(logger)
    return _api_wrapper_instance

def get_api_wrapper():
    """Get the API wrapper singleton instance"""
    if _api_wrapper_instance is None:
        raise RuntimeError("API wrapper not initialized. Call initialize_api_wrapper first.")
    return _api_wrapper_instance
