# Implementation Comparison

This document compares the two different approaches to job search implemented in this codebase:

## Deep Job Search (Multi-Agent Architecture)

**File:** `deep_job_search.py`

**Overview:**
- Uses OpenAI's Agents SDK
- Multi-agent architecture with specialized agents
- Enhanced logging, visualization, and token monitoring

**Agents:**
1. **Planner Agent (gpt-4.1)**: Creates search pairs of companies and keywords
2. **Searcher Agent (gpt-4o-mini)**: Performs web searches for job listings
3. **Processor Agent (gpt-4o-mini)**: Extracts structured job information
4. **Verifier Agent (o3)**: Validates URLs and checks for apply buttons

**Advantages:**
- Fine-grained control over each phase of the job search
- Detailed token usage tracking and budget management
- Rich visualization capabilities for agent interactions
- More robust error handling and verification of job listings
- Better for complex search patterns requiring multiple steps

**File Size:** ~1,244 lines

## Responses Job Search (Single Agent)

**File:** `responses_job_search.py`

**Overview:**
- Uses OpenAI's Responses API
- Single-agent approach with stateful conversations
- Simpler implementation with fewer moving parts

**Process:**
1. Formulate search query for major or startup companies
2. Send single API call with web search capabilities
3. Process and structure the job listings from response
4. Save results to CSV and markdown formats

**Advantages:**
- Simpler implementation with less code
- Faster execution with fewer API calls
- Easier to understand and modify
- Lower overall token usage for similar results
- More straightforward error handling

**File Size:** ~301 lines

## Key Differences

| Feature | Deep Job Search | Responses Job Search |
|---------|----------------|---------------------|
| Architecture | Multi-agent | Single agent |
| API Used | Agents SDK | Responses API |
| Models | Multiple specialized models | Single model |
| Code Complexity | Higher | Lower |
| Token Usage | Higher | Lower |
| Error Handling | More robust | More basic |
| Visualization | Advanced | Basic |
| Token Monitoring | Detailed | Basic |
| Best Use Case | Complex search needs | Quick, simple searches |

## When to Use Each

- **Use Deep Job Search when:**
  - You need detailed control over the search process
  - Visualization of agent interactions is desired
  - Token usage needs to be closely monitored
  - More robust error handling is needed

- **Use Responses Job Search when:**
  - Quick job searches with less complexity are needed
  - Lower-cost operations are a priority
  - Simpler code that's easier to modify is preferred
  - Faster results are more important than detailed control

## Unified Command Structure

Both implementations can be run using the same command interface:

```bash
# Run the default implementation (Deep Job Search)
./run.sh

# Run the simpler implementation (Responses Job Search)
./run.sh --responses

# Both support the same core options
./run.sh --sample --log-level DEBUG
./run.sh --responses --sample --log-level DEBUG
```

Both implementations output results to the same location, making it easy to compare the results.
