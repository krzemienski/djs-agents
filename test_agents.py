#!/usr/bin/env python
"""Simple test file to verify that we can use the Agents library"""

import os
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, ModelSettings

load_dotenv()  # Load environment variables from .env file

async def main():
    print("Testing OpenAI Agents library...")

    # Create a simple agent
    agent = Agent(
        name="test-agent",
        instructions="You are a test agent. Just say hello.",
        model="gpt-4o-mini",
        model_settings=ModelSettings(temperature=0)
    )

    # Run the agent
    result = await Runner.run(agent, input="Say hello")
    print(f"Agent response: {result.final_output}")
    print("Test completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
