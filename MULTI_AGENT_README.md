# Multi-Agent Architecture for Deep Job Search

This document explains the multi-agent architecture implemented for the Deep Job Search application, which is designed to find real job postings at major companies and startups.

## Architecture Overview

The multi-agent architecture is built using the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python), which provides a framework for creating specialized agents that can work together through a series of handoffs.

Our implementation consists of four specialized agents:

1. **Planner Agent**: Generates a comprehensive search plan based on company and keyword combinations
2. **Searcher Agent**: Executes web searches for job postings based on the search plan
3. **Processor Agent**: Extracts structured job listings from raw search results
4. **Verifier Agent**: Validates job URLs to ensure they point to real job postings

## Workflow

1. The **Planner** creates a list of company-keyword pairs to search for
2. For each pair, the **Searcher** performs a web search to find potential job postings
3. The search results are handed off to the **Processor** which extracts structured job data
4. Each job URL is handed off to the **Verifier** to confirm it's a valid job posting
5. Valid jobs are collected and saved to CSV and Markdown formats

## Benefits Over the Responses API Approach

The multi-agent architecture offers several advantages over the single-agent approach:

1. **Specialization**: Each agent can focus on a specific task, with tailored instructions and settings
2. **Modularity**: Agents can be updated or replaced independently without affecting the rest of the system
3. **Efficiency**: Each agent uses the most appropriate model for its task (e.g., simpler models for verification)
4. **Reliability**: The handoff mechanism ensures data flows correctly between processing steps
5. **Extensibility**: New agents can be added to expand functionality (e.g., adding a summarizer agent)
6. **Cost Optimization**: Using specialized models for different tasks can reduce token usage

## Implementation Details

### Agent Prompts

Each agent has a carefully crafted prompt that defines its role, task, and expected output format:

- **Planner**: Creates a diverse set of company-keyword pairs for searching
- **Searcher**: Conducts web searches for relevant job postings
- **Processor**: Extracts structured job data from raw search results
- **Verifier**: Validates job URLs to confirm they are real job postings

### Flow Control

The flow control is managed by the main `gather_jobs_multi_agent` function, which:

1. Initializes each specialized agent with appropriate models and settings
2. Sets up handoff paths between agents
3. Executes the planning phase to generate a search strategy
4. Executes searches based on the plan and processes results
5. Validates found job postings
6. Returns a filtered list of validated jobs

### Fallback Mechanisms

The implementation includes several fallback mechanisms to ensure robustness:

- JSON parsing fallbacks if the output format is not perfect
- URL validation through both regex pattern matching and LLM verification
- Token budget monitoring to avoid exceeding predefined limits
- Error handling for individual search/processing failures

## Demo Script

A standalone demo script (`demo_multi_agent.py`) is provided to demonstrate the multi-agent architecture in action. This script:

1. Sets up a simplified multi-agent workflow
2. Executes searches for a limited number of jobs
3. Includes fallback job generation if not enough real jobs are found
4. Saves results to the `results` directory
5. Provides statistics on token usage and efficiency

## Getting Started

To run the demo:

```bash
python demo_multi_agent.py
```

To incorporate the multi-agent architecture into the main application:

```bash
./build-and-run.sh --use-multi-agent
```

## Future Improvements

Potential enhancements to the multi-agent architecture include:

1. Adding a job content summarizer agent
2. Implementing a job ranking/filtering agent
3. Creating a salary estimation agent
4. Developing a job recommendation agent based on skills
5. Adding memory to track already-seen job postings
