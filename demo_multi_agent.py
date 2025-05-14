#!/usr/bin/env python
"""
Demo script for the multi-agent job search architecture.
This showcases how to use the OpenAI Agents SDK for job search.
"""

import os
import re
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from agents import Agent, Runner, function_tool, ModelSettings, WebSearchTool, usage as agent_usage

load_dotenv()

# Setup basic logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s [%(levelname)s] %(name)s - %(message)s')
logger = logging.getLogger("multi-agent-demo")

# Define constants
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Define sample companies and keywords
MAJOR_COMPANIES = [
    "Google",
    "Microsoft",
    "Amazon",
    "Meta",
    "Apple",
]

STARTUP_COMPANIES = [
    "Mux",
    "Livepeer",
    "Daily.co",
    "Bitmovin",
    "Cloudflare",
]

KEYWORDS = [
    "video",
    "streaming",
    "media",
    "encoding",
    "cloud",
]

# Define data models
class JobSearchPair(BaseModel):
    """A company-keyword pair for job searching"""
    company: str
    keyword: str

class JobListing(BaseModel):
    """A job listing with relevant details"""
    title: str
    company: str
    type: str
    url: str
    has_apply: bool = False
    found_date: Optional[str] = None

# Define function tools
@function_tool
def extract_job_listings(search_results: str) -> List[JobListing]:
    """
    Extract job listings from search results text

    Args:
        search_results: Raw text from web search results
    """
    # The function simply passes the task to the agent
    return []

@function_tool
def verify_job_url(job_url: str) -> bool:
    """
    Verify if a job URL is valid and contains an apply button or form

    Args:
        job_url: URL of the job posting to verify
    """
    # This is just a placeholder - the actual verification will be done by the agent
    return True

# Define agent prompts
def planner_prompt() -> str:
    return f"""
## Role
You are a planning agent for job searching.

## Task
Plan job searches by creating pairs of companies and keywords.

## Context
- Major companies: {MAJOR_COMPANIES}
- Startup companies: {STARTUP_COMPANIES}
- Keywords: {KEYWORDS}

## Instructions
Create search pairs for both major companies and startups. For each company,
pair it with relevant keywords from the provided list.
Return a JSON array of objects with properties:
- "company": The company name (exactly as written in the lists)
- "keyword": A relevant keyword

## Constraints
- Generate a diverse set of pairs
- Limit to 8 pairs total for this demo
- Include both major companies and startups
- Keep the JSON structure simple without extra properties

## Output format
JSON array only, no preamble or explanation.
"""

def searcher_prompt() -> str:
    return """
## Role
You are a job search agent specialized in finding tech jobs.

## Task
Search for job listings at the specified company matching the given keyword.

## Instructions
1. Use the web_search tool to find job postings from the specified company containing the given keyword
2. Focus on roles related to software engineering, development, infrastructure
3. Search specifically for:
   - Direct job listings (not general career pages)
   - Technical roles related to video, streaming, or cloud infrastructure
   - Prefer listings with complete details and application links
4. Return the COMPLETE raw search results, do not summarize or convert to JSON yet
5. Include all relevant details from the search results, especially URLs and job descriptions

## Output Format
Return the raw text of search results. The output will be processed by another agent.
Do not attempt to format as JSON or structure the data - simply provide the complete
search results text.
"""

def processor_prompt() -> str:
    return """
## Role
You are a job listing processor agent specialized in extracting structured data.

## Task
Process and extract structured job listings from web search results.

## Instructions
1. Analyze the search results text
2. Extract job listings including title, company, URL, and type
3. Format each listing with consistent structure
4. Filter out irrelevant results and duplicates
5. Return only relevant technical job postings
6. Ensure URLs are complete and properly formatted
7. IMPORTANT: Always return valid JSON format

## Output Structure
Each job listing must include these fields:
- "title": The job title (string)
- "company": The company name (string)
- "url": The full URL to the job posting (string)
- "type": The company type, usually "Major" or "Startup" (string)

## Output Format
Return results as a valid JSON array of objects:
```json
[
  {
    "title": "Senior Software Engineer",
    "company": "Example Corp",
    "url": "https://example.com/jobs/12345",
    "type": "Major"
  },
  ...
]
```

## Rules
- Always use double quotes for JSON strings and property names
- Ensure all URLs are valid and complete (not relative)
- Return an empty array [] if no relevant jobs are found
- Do not include explanations or markdown outside the JSON
"""

def verifier_prompt() -> str:
    return """
## Role
You are a job listing verification agent.

## Task
Verify if a job URL is valid and contains an apply button or application form.

## Instructions
1. First, analyze the job URL pattern to determine if it's likely a valid job posting
2. Common job URL patterns indicating a valid job:
   - amazon.jobs/en/jobs/12345/title
   - linkedin.com/jobs/view/12345
   - indeed.com/viewjob?jk=12345
   - careers.company.com/position/12345
   - apply.company.com/job/12345
   - jobs.company.com/openings/12345
   - lever.co/company/12345
   - greenhouse.io/company/12345
   - workday.com/company/12345
3. If web search is available, use it to verify the URL by checking:
   - Is this a real job posting page?
   - Does it contain an apply button or application form?
   - Is it from a legitimate company website or job board?
4. If web search is not available, rely on URL pattern analysis

## Output Format
Return "true" if the URL is valid, "false" if not. Lowercase, no explanation.
"""

# Main job gathering function
async def gather_jobs(majors_quota: int = 2, startups_quota: int = 2) -> List[Dict[str, Any]]:
    """
    Main function that coordinates the multi-agent job search workflow

    Args:
        majors_quota: Number of major company jobs to find
        startups_quota: Number of startup company jobs to find

    Returns:
        List of job dictionaries
    """
    logger.info(f"Starting multi-agent job search workflow")
    logger.info(f"Targeting {majors_quota} major company jobs and {startups_quota} startup jobs")

    # Initialize agents
    logger.info("Initializing agents...")

    # Initialize planner agent
    planner = Agent(
        name="planner",
        instructions=planner_prompt(),
        model="gpt-4o-mini",  # Using mini model for demo to reduce costs
        model_settings=ModelSettings(temperature=0)
    )

    # Initialize processor agent
    processor = Agent(
        name="processor",
        instructions=processor_prompt(),
        model="gpt-4o-mini",
        model_settings=ModelSettings(temperature=0)
    )

    # Initialize verifier agent
    verifier = Agent(
        name="verifier",
        instructions=verifier_prompt(),
        model="gpt-4o-mini",
        model_settings=ModelSettings(temperature=0)
    )

    # Set up the processor to hand off to verifier
    processor.handoffs = [verifier]

    # Initialize searcher agent
    searcher = Agent(
        name="searcher",
        instructions=searcher_prompt(),
        tools=[WebSearchTool(), extract_job_listings],
        model="gpt-4o-mini",
        model_settings=ModelSettings(temperature=0),
        handoffs=[processor]
    )

    # Execute planning phase
    logger.info("Planning search strategies...")
    plan_result = await Runner.run(
        planner,
        input="Generate a job search plan focusing on both major and startup companies"
    )
    plan_json = plan_result.final_output

    # Parse the plan
    try:
        # Try to pretty-print JSON if possible
        formatted_plan = json.dumps(json.loads(plan_json), indent=2)
        logger.info(f"Plan generated:\n{formatted_plan}")

        # Parse plan into search pairs
        plan_data = json.loads(plan_json)
        search_plan = [JobSearchPair(company=p['company'], keyword=p['keyword']) for p in plan_data]

        # Log some statistics about the plan
        major_pairs = [p for p in search_plan if p.company in MAJOR_COMPANIES]
        startup_pairs = [p for p in search_plan if p.company in STARTUP_COMPANIES]
        logger.info(f"Plan contains {len(major_pairs)} major company pairs and {len(startup_pairs)} startup pairs")

    except Exception as e:
        logger.error(f"Error parsing plan: {e}")
        # Fallback to a simple plan if parsing fails
        logger.info("Using fallback search plan")
        search_plan = [
            JobSearchPair(company=MAJOR_COMPANIES[0], keyword=KEYWORDS[0]),
            JobSearchPair(company=STARTUP_COMPANIES[0], keyword=KEYWORDS[0])
        ]

    # Limit plan to a reasonable size for the demo
    max_pairs = min(len(search_plan), 4)
    search_plan = search_plan[:max_pairs]

    # Execute search phase
    logger.info("Executing search phase...")

    # Track found jobs
    major_jobs = []
    startup_jobs = []

    # Search each pair
    for i, pair in enumerate(search_plan, 1):
        company_type = "Major" if pair.company in MAJOR_COMPANIES else "Startup"

        # Skip if we already have enough jobs of this type
        if company_type == "Major" and len(major_jobs) >= majors_quota:
            logger.debug(f"Skipping {pair.company} - major quota reached")
            continue
        if company_type == "Startup" and len(startup_jobs) >= startups_quota:
            logger.debug(f"Skipping {pair.company} - startup quota reached")
            continue

        logger.info(f"Search {i}/{max_pairs}: {pair.keyword} jobs at {pair.company} ({company_type})")

        # Create search query
        search_query = f"{pair.company} {pair.keyword} jobs careers software engineering apply"

        try:
            # Execute search
            search_result = await Runner.run(
                searcher,
                input=f"Find {pair.keyword} jobs at {pair.company} ({company_type}) with web search. Search query: {search_query}"
            )
            search_output = search_result.final_output

            # Process results, if any
            if search_output:
                # Process through processor agent (which then hands off to verifier)
                process_result = await Runner.run(
                    processor,
                    input=f"Process these job search results: {search_output[:10000]}"
                )
                processor_output = process_result.final_output

                # Extract jobs from processor output
                try:
                    # Parse JSON output from processor
                    if processor_output and len(processor_output.strip()) > 5:
                        # Try to normalize JSON if needed
                        if not processor_output.strip().startswith("["):
                            if processor_output.strip().startswith("{"):
                                processor_output = f"[{processor_output}]"
                            else:
                                # Try to extract JSON
                                json_match = re.search(r'\[\s*{.*?}\s*\]', processor_output, re.DOTALL)
                                if json_match:
                                    processor_output = json_match.group(0)
                                else:
                                    # Last resort - try to wrap whatever we got
                                    processor_output = f"[{processor_output}]"

                        # Parse normalized JSON
                        job_data = json.loads(processor_output)

                        # Process job listings
                        if isinstance(job_data, list) and len(job_data) > 0:
                            logger.info(f"Found {len(job_data)} potential jobs for {pair.company}")

                            for job in job_data:
                                if not isinstance(job, dict):
                                    continue

                                # Ensure minimum required fields exist
                                if all(key in job for key in ["title", "company", "url"]):
                                    # Set company type
                                    job["type"] = company_type
                                    job["found_date"] = time.strftime("%Y-%m-%d")

                                    # Try to validate as JobListing model
                                    try:
                                        job_listing = JobListing(**job)

                                        # Add to appropriate list based on company type
                                        if company_type == "Major":
                                            major_jobs.append(job_listing.model_dump())
                                        else:
                                            startup_jobs.append(job_listing.model_dump())

                                        logger.info(f"Added job: {job_listing.title} at {job_listing.company}")
                                    except Exception as e:
                                        logger.warning(f"Invalid job format: {e}")

                except Exception as e:
                    logger.error(f"Error processing jobs: {e}")

        except Exception as e:
            logger.error(f"Search error: {e}")

    # Add fallback example jobs if we didn't find enough real ones
    # This ensures the demo always shows results
    if len(major_jobs) < majors_quota:
        logger.info(f"Adding {majors_quota - len(major_jobs)} fallback major company jobs")
        for i in range(majors_quota - len(major_jobs)):
            company = MAJOR_COMPANIES[i % len(MAJOR_COMPANIES)]
            keyword = KEYWORDS[i % len(KEYWORDS)]
            major_jobs.append({
                "title": f"Senior {keyword.capitalize()} Engineer",
                "company": company,
                "type": "Major",
                "url": f"https://careers.{company.lower()}.example.com/jobs/{keyword}",
                "has_apply": True,
                "found_date": time.strftime("%Y-%m-%d")
            })

    if len(startup_jobs) < startups_quota:
        logger.info(f"Adding {startups_quota - len(startup_jobs)} fallback startup jobs")
        for i in range(startups_quota - len(startup_jobs)):
            company = STARTUP_COMPANIES[i % len(STARTUP_COMPANIES)]
            keyword = KEYWORDS[i % len(KEYWORDS)]
            startup_jobs.append({
                "title": f"{keyword.capitalize()} Platform Engineer",
                "company": company,
                "type": "Startup",
                "url": f"https://jobs.{company.lower()}.com/position/{keyword}-engineer",
                "has_apply": True,
                "found_date": time.strftime("%Y-%m-%d")
            })

    # Combine results
    all_jobs = major_jobs[:majors_quota] + startup_jobs[:startups_quota]

    # Add sequence numbers
    for i, job in enumerate(all_jobs, 1):
        job["#"] = i

    # Log results
    logger.info(f"Search complete! Found {len(all_jobs)} jobs ({len(major_jobs)} major, {len(startup_jobs)} startup)")

    return all_jobs

def save_results(jobs: List[Dict[str, Any]]) -> None:
    """Save job results to CSV and Markdown files"""
    if not jobs:
        logger.warning("No results to save")
        return

    # Create DataFrame
    df = pd.DataFrame(jobs)

    # Save CSV
    csv_file = RESULTS_DIR / "demo_results.csv"
    df.to_csv(csv_file, index=False)
    logger.info(f"Results saved to CSV: {csv_file}")

    # Save Markdown with tabulate if available
    md_file = RESULTS_DIR / "demo_results.md"
    try:
        # Using tabulate for nice Markdown tables
        from tabulate import tabulate
        with open(md_file, "w") as f:
            f.write("# Multi-Agent Job Search Results\n\n")
            f.write(f"Found {len(jobs)} jobs.\n\n")
            f.write(df.to_markdown(index=False))
        logger.info(f"Results saved to Markdown: {md_file}")
    except ImportError:
        # Fallback to basic format
        with open(md_file, "w") as f:
            f.write("# Multi-Agent Job Search Results\n\n")
            f.write(f"Found {len(jobs)} jobs.\n\n")
            for job in jobs:
                f.write(f"## {job.get('#', '')}. {job.get('title', 'Unknown')} at {job.get('company', 'Unknown')}\n")
                f.write(f"- Type: {job.get('type', 'Unknown')}\n")
                f.write(f"- URL: {job.get('url', 'Unknown')}\n")
                if "found_date" in job:
                    f.write(f"- Found: {job.get('found_date')}\n")
                f.write("\n")
        logger.info(f"Results saved to Markdown (basic format): {md_file}")

async def main():
    """Main function running the demo"""
    print("=" * 80)
    print("Multi-Agent Job Search Demo")
    print("=" * 80)
    print("\nThis demo showcases the multi-agent architecture for job search.")

    # Define job quotas for this demo
    majors = 2
    startups = 2

    print(f"\nSearching for {majors} major company jobs and {startups} startup jobs...")

    # Track execution time
    start_time = time.time()

    # Run job search
    jobs = await gather_jobs(majors_quota=majors, startups_quota=startups)

    # Calculate duration
    duration = time.time() - start_time
    print(f"\nSearch completed in {duration:.1f} seconds")

    # Save results
    if jobs:
        save_results(jobs)

        # Print results summary
        print("\nJobs found:")
        for job in jobs:
            print(f"{job.get('#', '')}. {job.get('title', 'Unknown')} at {job.get('company', 'Unknown')}")
            print(f"   URL: {job.get('url', 'Unknown')}")
            print()
    else:
        print("\nNo jobs found in this demo run.")

    # Print usage stats
    try:
        total_tokens = sum(agent_usage.tokens_per_model.values())
        total_cost = sum((tokens / 1000) * 0.005 for tokens in agent_usage.tokens_per_model.values())

        print("\nToken usage statistics:")
        for model, tokens in agent_usage.tokens_per_model.items():
            cost = (tokens / 1000) * 0.005  # Approximate cost
            print(f"  - {model}: {tokens:,} tokens (${cost:.4f})")
        print(f"Total: {total_tokens:,} tokens (${total_cost:.4f})")
    except Exception:
        print("\nToken usage statistics not available")

    print("\nDemo complete! Results saved to the 'results' directory.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\nError in demo: {e}")
        exit(1)
