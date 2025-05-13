#!/usr/bin/env python
"""
visualize_agents.py - Visualization tool for Deep Job Search agent architecture
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from agents import Agent, ModelSettings, WebSearchTool, function_tool
from agents.extensions.visualization import draw_graph

# Import our main functions
from deep_job_search import planner_prompt, searcher_prompt, processor_prompt, verifier_prompt

# Load environment variables
load_dotenv()

# Function to create the visualization
def create_visualization(output_path="agent_graph"):
    """Generate a visualization of the agent architecture"""
    print("Creating agent visualization...")

    # Define our function tools similar to the main application
    @function_tool
    def extract_job_listings(search_results: str) -> list:
        """Extract job listings from search results text"""
        return []

    @function_tool
    def verify_job_url(job_url: str) -> bool:
        """Verify if a job URL is valid and contains an apply button or form"""
        return True

    # Create agents similar to the main application
    planner = Agent(
        name="Planner",
        instructions=planner_prompt(),
        model="gpt-4.1",
        model_settings=ModelSettings(temperature=0)
    )

    searcher = Agent(
        name="Searcher",
        instructions=searcher_prompt(),
        tools=[WebSearchTool(), extract_job_listings],
        model="gpt-4o-mini",
        model_settings=ModelSettings(temperature=0)
    )

    processor = Agent(
        name="Processor",
        instructions=processor_prompt(),
        model="gpt-4o-mini",
        model_settings=ModelSettings(temperature=0)
    )

    verifier = Agent(
        name="Verifier",
        instructions=verifier_prompt(),
        tools=[],  # No web search for basic verifier
        model="o3",
        model_settings=ModelSettings(temperature=0)
    )

    # Set up the handoffs to show proper agent relationships
    planner.handoffs = []
    searcher.handoffs = [processor]
    processor.handoffs = [verifier]
    verifier.handoffs = []

    # Create the main orchestrator agent which delegates to all other agents
    orchestrator = Agent(
        name="DeepJobSearch",
        instructions="Orchestrate job search using specialized agents",
        handoffs=[planner, searcher, processor, verifier],
        tools=[],
        model="gpt-4o",
        model_settings=ModelSettings(temperature=0)
    )

    # Generate the visualization
    graph = draw_graph(orchestrator, filename=output_path)
    print(f"Visualization saved to {output_path}.png")

    # Return the generated graph for potential further use
    return graph

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Generate a visualization of the Deep Job Search agent architecture")
    parser.add_argument("--output", default="agent_graph", help="Output path for the visualization (without extension)")
    args = parser.parse_args()

    # Ensure output directory exists
    output_path = Path(args.output)
    if '/' in args.output or '\\' in args.output:
        # If path contains directories, create them
        output_path.parent.mkdir(exist_ok=True, parents=True)
    else:
        # Otherwise, put it in the results directory
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        output_path = results_dir / args.output

    # Generate visualization
    graph = create_visualization(str(output_path))

    # Optionally display the visualization if running in an environment with a display
    try:
        graph.view()
        print("Graph displayed. Close the window to continue.")
    except Exception as e:
        print(f"Couldn't display graph directly: {e}")
        print(f"Graph is still saved at {output_path}.png")
