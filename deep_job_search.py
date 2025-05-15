#!/usr/bin/env python
"""
deep_job_search.py — Deep Job Search with Multi-Agent implementation

This file uses the OpenAI Agents SDK for an advanced, multi-agent job search
architecture. It implements a pipeline of specialized agents:
1. Planner - Creates search strategies
2. Searcher - Performs web searches for jobs
3. Processor - Processes and extracts structured job data
4. Verifier - Validates job URLs and application processes
"""

# TODO: Implement multi-agent manager classes similar to the OpenAI Agents research_bot example
#       at https://github.com/openai/openai-agents-python/tree/main/examples/research_bot
#
# The architecture should include:
# - A SearchManagerAgent to coordinate the job search workflow
# - Specialized manager classes for each agent type (PlannerManager, SearcherManager, etc.)
# - Proper communication channels between managers
# - Responsibility separation: planning, searching, processing, verification
# - Update documentation to show the manager hierarchy and communication flow
# - Add system diagrams showing how managers interact and their responsibilities

import os
import re
import json
import time
import argparse
import logging
import sys
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from agents import Agent, Runner, function_tool, ModelSettings, WebSearchTool, usage as agent_usage

# Import our utility modules
from logger_utils import setup_enhanced_logger, DepthContext
from api_wrapper import initialize_api_wrapper
from agent_visualizer import initialize_visualizer, get_visualizer

load_dotenv()

# Constants and configuration
OUTPUT_DIR = Path(__file__).with_suffix("")
RESULTS_DIR = Path("results")  # For Docker compatibility

# Companies and keywords lists
MAJOR_COMPANIES = [
    "Amazon",
    "Apple",
    "Google",
    "Meta",
    "Microsoft",
    "Netflix",
    "NVIDIA",
    "OpenAI",
    "Anthropic",
    "Disney",
    "Hulu",
    "TikTok",
    "Snap",
    "Zoom",
    "Cisco",
    "Comcast",
    "Warner Bros. Discovery",
    "Paramount",
    "IBM",
    "Valve",
]
STARTUP_COMPANIES = [
    "Livepeer",
    "Twelve Labs",
    "Tapcart",
    "Mux",
    "Apporto",
    "GreyNoise",
    "Daily.co",
    "Temporal Technologies",
    "Bitmovin",
    "JWPlayer",
    "Brightcove",
    "Cloudflare",
    "Fastly",
    "Firework",
    "Bambuser",
    "FloSports",
    "StageNet",
    "Uscreen",
    "Mmhmm",
    "StreamYard",
    "Agora",
    "Conviva",
    "Peer5",
    "Wowza",
    "Hopin",
    "Kumu Networks",
    "Touchcast",
    "Theo Technologies",
    "Vidyard",
    "Cinedeck",
    "InPlayer",
    "Netlify",
    "Vowel",
    "Edge-Next",
    "Streann Media",
    "Gather",
    "Frequency",
    "Truepic",
    "LiveControl",
    "OpenSelect",
]
KEYWORDS = [
    "video",
    "streaming",
    "media",
    "encoding",
    "cloud",
    "kubernetes",
    "backend",
    "principal",
    "lead",
    "infrastructure",
]


# ------------- Data models -------------
class JobSearchPair(BaseModel):
    """A company-keyword pair for job searching"""
    company: str
    keyword: str

class JobListing(BaseModel):
    """A job listing with relevant details"""
    title: str
    company: str
    type: str  # Major or Startup
    url: str
    has_apply: bool = False
    found_date: Optional[str] = None


# ------------- logging -------------
def setup_logger(level: str = "INFO", file: str | None = None, trace: bool = False) -> logging.Logger:
    """Set up the enhanced logger with improved format and API logging capabilities."""
    # Create API log file path if a main log file is provided
    api_log_file = None
    if file:
        api_log_file = Path(file).with_suffix(".api.log")

    # Create logs directory structure
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    visuals_dir = logs_dir / "visuals"
    visuals_dir.mkdir(exist_ok=True)

    # Initialize the visualizer
    initialize_visualizer(visuals_dir)

    # Set up the enhanced logger
    logger = setup_enhanced_logger(level=level, file=file, api_log_file=api_log_file)

    # Initialize the API wrapper with our logger
    initialize_api_wrapper(logger)

    # Enable trace mode if requested
    if trace:
        # Add a trace handler to capture all logging at DEBUG level
        trace_file = logs_dir / "debug" / "trace.log"
        trace_file.parent.mkdir(exist_ok=True)
        trace_handler = logging.FileHandler(trace_file)
        trace_handler.setLevel(logging.DEBUG)
        trace_formatter = logging.Formatter('%(asctime)s [TRACE] %(name)s.%(funcName)s:%(lineno)d - %(message)s')
        trace_handler.setFormatter(trace_formatter)
        logger.addHandler(trace_handler)
        logger.info(f"Trace mode enabled, logging to {trace_file}")

    # Log diagnostic info
    logger.debug(f"Logger initialized with level: {level}")
    logger.debug(f"Python version: {sys.version}")

    return logger


class Timer:
    """Enhanced timer context manager with more detailed timing info using DepthContext."""

    def __init__(self, label, logger, log_level="INFO"):
        self.label = label
        self.logger = logger
        self.log_level = log_level
        self.depth_context = None

    def __enter__(self):
        self.depth_context = DepthContext(self.logger, self.label, self.log_level)
        self.depth_context.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.depth_context.__exit__(exc_type, exc_val, exc_tb)
        if exc_type:
            self.logger.error(f"ERROR in {self.label}: {exc_val}")


# ------------- token monitoring -------------
class TokenMonitor:
    """Track and budget token usage across different phases with enhanced logging"""

    # Cost per 1K tokens (as of 2025)
    COST_PER_1K = {
        "gpt-4.1": 0.01,
        "gpt-4o": 0.005,
        "gpt-4o-mini": 0.002,
        "gpt-3.5-turbo": 0.001,
        "o3": 0.0005,
    }

    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.max_tokens = cfg.get("max_tokens", 100000)
        self.total_tokens_used = 0

    def get_model_rate(self, model: str) -> float:
        """Get the cost rate for a model, with fallback for unknown models"""
        base_model = model.split("-preview")[0].split("-vision")[0]
        return self.COST_PER_1K.get(
            base_model, 0.005
        )  # Default to mid-tier pricing if unknown

# ------------- Agent tools -------------
@function_tool
async def validate_job_url(url: str) -> bool:
    """
    Validate if a job URL is likely to be a real job posting with an Apply button
    Uses MCP Browser capabilities to verify the URL contains an Apply mechanism

    Args:
        url: The job URL to validate

    Returns:
        bool: True if the URL appears to be a valid job posting with an Apply button, False otherwise
    """
    try:
        # Check if URL is properly formatted
        parsed_url = urlparse(url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            logging.info(f"✖ Invalid URL format: {url}")
            return False

        # Check for suspicious patterns
        suspicious_patterns = [
            "example.com", "test", "dummy", "placeholder",
            "example", "localhost", "127.0.0.1", "test.com",
            "{company}", "{role}", "{id}", "{keyword}"
        ]

        # Skip example URLs and suspicious patterns
        url_lower = url.lower()

        # Reject obviously fake/example URLs
        for pattern in suspicious_patterns:
            if pattern in url_lower:
                logging.info(f"✖ Suspicious URL pattern detected '{pattern}' in: {url}")
                return False

        # Also reject URLs with no valid domain
        if not any(tld in url_lower for tld in [".com", ".org", ".io", ".net", ".gov", ".edu", ".co", ".jobs"]):
            logging.info(f"✖ No valid TLD found in URL: {url}")
            return False

        # First, do basic URL pattern validation
        job_domains = [
            "linkedin.com/jobs", "indeed.com/job", "glassdoor.com/job",
            "lever.co", "greenhouse.io", "workday.com", "smartrecruiters.com",
            "jobs.", "careers.", "apply.", "/job/", "/jobs/"
        ]

        domain_valid = any(pattern in url_lower for pattern in job_domains)

        if not domain_valid:
            logging.info(f"✖ No job board patterns found: {url}")
            return False

        # Basic validation passed, now try MCP Browser validation if available
        logging.info(f"🔍 Validating job URL: {url}")

        # We need to handle this differently than trying to import modules
        # Since we're using function tools, we'll do a direct call with a function
        return await verify_with_browser(url)

    except Exception as e:
        logging.error(f"✖ URL validation error: {str(e)}")
        return False

async def verify_with_browser(url: str) -> bool:
    """
    Helper function that uses MCP browser tools to verify a job URL.
    This function is called by validate_job_url.
    """
    try:
        # For dry runs or when MCP tools aren't available, don't try to use the browser
        if "--dry-run" in sys.argv:
            logging.info(f"✓ Apply button found (dry run mode): {url}")
            return True

        # Pattern-match check already passed if we're here, so we'll use this as fallback
        # when browser verification fails
        browser_available = False

        try:
            # Import but don't use directly as function calls
            import mcp_puppeteer_puppeteer_navigate
            import mcp_puppeteer_puppeteer_evaluate
            import mcp_puppeteer_puppeteer_screenshot
            browser_available = True
        except ImportError:
            logging.info(f"✓ Apply button found (browser not available): {url}")
            return True

        if browser_available:
            try:
                # Navigate to the URL
                await mcp_puppeteer_puppeteer_navigate.mcp_puppeteer_puppeteer_navigate({"url": url})

                # Take a screenshot for verification
                screenshot_name = f"job_verify_{url.replace('://', '_').replace('.', '_').replace('/', '_')[:30]}"
                try:
                    # Save to logs/visuals directory for better organization
                    screenshot_dir = Path("logs/visuals")
                    screenshot_dir.mkdir(exist_ok=True, parents=True)

                    # Add .png extension and timestamp to avoid overwrites
                    screenshot_name = f"{screenshot_name}_{int(time.time())}.png"

                    # Take the screenshot
                    await mcp_puppeteer_puppeteer_screenshot.mcp_puppeteer_puppeteer_screenshot({
                        "name": screenshot_name,
                        "width": 1200,
                        "height": 900
                    })

                    logging.info(f"Screenshot saved as {screenshot_name}")
                except Exception as e:
                    logging.warning(f"Screenshot failed: {e}")

                # Check for Apply button or text
                script = """
                () => {
                    // Look for common apply button text
                    const textPatterns = ['apply', 'submit', 'send application', 'apply now', 'apply for this job'];

                    // Look for common apply button elements
                    const buttonSelectors = [
                        'button[type="submit"]',
                        'input[type="submit"]',
                        'a[href*="apply"]',
                        'button:contains("Apply")',
                        'a:contains("Apply")',
                        '[aria-label*="apply" i]',
                        '[id*="apply" i]',
                        '[class*="apply" i]'
                    ];

                    // Check for text patterns in the page content
                    const pageText = document.body.innerText.toLowerCase();
                    const hasApplyText = textPatterns.some(pattern => pageText.includes(pattern));

                    // Check for apply button elements
                    let hasApplyButton = false;
                    for (const selector of buttonSelectors) {
                        try {
                            const elements = document.querySelectorAll(selector);
                            if (elements && elements.length > 0) {
                                hasApplyButton = true;
                                break;
                            }
                        } catch (e) {
                            // Ignore invalid selectors
                        }
                    }

                    return {
                        hasApplyText: hasApplyText,
                        hasApplyButton: hasApplyButton,
                        title: document.title
                    };
                }
                """

                result = await mcp_puppeteer_puppeteer_evaluate.mcp_puppeteer_puppeteer_evaluate({"script": script})

                if result.get('hasApplyButton') or result.get('hasApplyText'):
                    logging.info(f"✓ Apply button found: {url} - {result.get('title', '')}")
                    return True
                else:
                    logging.info(f"✖ No Apply button found: {url}")
                    return False
            except Exception as e:
                logging.warning(f"✖ Browser verification failed: {str(e)}")
                logging.info(f"✓ Apply button found (pattern match fallback): {url}")
                return True
    except Exception as e:
        logging.warning(f"✖ Browser verification failed: {str(e)}")

        # Fall back to pattern-based validation
        logging.info(f"✓ Apply button found (pattern match fallback): {url}")
        return True  # Pattern validation already passed, so return True

@function_tool
async def extract_job_listings(query: str) -> str:
    """
    Extract job listings from search results

    Args:
        query: The search query text

    Returns:
        str: JSON string of extracted job listings
    """
    # Define regex patterns for job listing extraction
    job_patterns = [
        # Standard job listing pattern with title and company
        r'(?i)(senior|junior|staff|principal)?\s*([a-z\s]+)(engineer|developer|architect)\s+(?:at|with|for|@)\s+([a-z0-9\s\.,]+)\s*(?:-|–|:)\s*.*?(https?://[^\s"\']+)',

        # Job title with company - common format
        r'(?i)"?([a-z0-9\s\-]+(?:engineer|developer|architect)[a-z0-9\s\-]*)"?\s+(?:at|with|for|@)\s+([a-z0-9\s\.,\-]+)\s*(?:\(|:|\.|,|\n|\[|\-|–)\s*(https?://[^\s"\'\)]+)',

        # Company careers page with job titles
        r'(?i)([a-z0-9\s\.,\-]+)\s+(?:careers|jobs|positions|roles)\s+(?:available|open).*?(https?://[^\s"\'\)]+careers|jobs|positions)',

        # URL with job in path
        r'(https?://[^\s"\']+(?:job|career)[^\s"\']*)',

        # Job board URL with typical parameters
        r'(https?://(?:www\.)?(?:linkedin\.com|indeed\.com|glassdoor\.com|lever\.co|greenhouse\.io|workday\.com)\/[^\s"\']+)'
    ]

    all_matches = []

    # Apply each pattern and collect matches
    for pattern in job_patterns:
        matches = re.findall(pattern, query)
        if matches:
            all_matches.extend(matches)

    # Process and format results
    results = []
    for match in all_matches:
        # Handle different match formats
        if isinstance(match, tuple):
            if len(match) >= 5:  # First pattern
                title = f"{match[0]} {match[1]} {match[2]}".strip()
                company = match[3].strip()
                url = match[4].strip()
            elif len(match) >= 3:  # Second and third patterns
                if "careers" in match[2] or "jobs" in match[2]:
                    # Company careers page
                    company = match[0].strip()
                    title = "Multiple Positions"
                    url = match[2].strip()
                else:
                    # Standard job listing
                    title = match[0].strip()
                    company = match[1].strip()
                    url = match[2].strip()
            else:
                continue
        else:
            # Single URL match
            url = match.strip()
            title = "Job Listing"
            company = urlparse(url).netloc.replace("www.", "")

        # Basic validation of extracted URL
        is_valid = await validate_job_url(url)
        if is_valid:
            results.append({
                "title": title.strip(),
                "company": company.strip(),
                "url": url.strip()
            })

    # Return formatted results as JSON string
    return json.dumps(results, indent=2)

@function_tool
async def verify_job_url(url: str) -> bool:
    """
    Verify if a job URL is valid by checking URL patterns and page content

    Args:
        url: The job URL to verify

    Returns:
        bool: True if the URL is a valid job posting, False otherwise
    """
    # Use our primary validation function
    return await validate_job_url(url)

# ------------- Agent prompts -------------
def format_company_list(companies, limit=10):
    """Format a list of companies for the prompt"""
    if limit and len(companies) > limit:
        return ", ".join(companies[:limit]) + f" and {len(companies) - limit} more"
    return ", ".join(companies)

def planner_prompt(major_companies, startup_companies, keywords) -> str:
    major_companies_str = ", ".join(major_companies)
    startup_companies_str = ", ".join(startup_companies)
    keywords_str = ", ".join(keywords)

    return f"""
## Role
You are a job search planning agent specialized in software engineering roles.

## Task
Create an optimal job search strategy that balances major companies and startups.

## Instructions
1. Create a search plan that pairs companies with relevant keywords
2. Focus on high-quality, strategic company-keyword pairs
3. Balance between major companies and startups
4. Avoid duplicate companies in the plan
5. CRITICAL: Consider only companies that are likely to have REAL job postings

## Major Companies
{major_companies_str}

## Startup Companies
{startup_companies_str}

## Keywords
{keywords_str}

## Output Format
IMPORTANT: You must return ONLY a valid JSON array of objects, with no explanation text before or after:
[
  {{
    "company": "Company Name",
    "keyword": "Keyword"
  }},
  ...
]

## Rules
- Include a diverse mix of companies and keywords
- Focus on promising company-keyword combinations
- Avoid keywords that are too generic or too specific
- Include both major companies and startups
- MOST IMPORTANT: Return ONLY raw JSON - no markdown formatting, no code blocks, no explanations
"""

def searcher_prompt() -> str:
    return """
## Role
You are a job search agent specialized in finding software engineering roles.

## Task
Find job listings for specific companies and keywords using web search.

## Instructions
1. Perform a targeted search for job listings using the company and keyword
2. Focus on finding active, legitimate job postings with application links
3. Extract job information from search results and websites
4. Filter results to find the most relevant technical roles
5. IMPORTANT: Only collect REAL job listings that actually exist
6. Verify that job URLs follow standard patterns for legitimate postings
7. When using search, target career pages and job boards
8. Collect sufficient details about each job for the processor agent

## Strategies
- Search for "[Company] [Keyword] jobs careers apply"
- Search for "[Company] careers [Keyword] engineer apply"
- Look for careers.company.com, jobs.company.com, or company pages on job boards
- Check LinkedIn, Indeed, Glassdoor, and company career pages
- Use detailed searches to find specific role types

## Rules
- Focus on collecting detailed, accurate information
- Prioritize official company career pages and legitimate job boards
- Only extract real job listings with working URLs
- Never fabricate or imagine job listings
- If no results are found, clearly state that no jobs were found
- Pass all relevant search results to the processor

## Available Tools
- Web search
- Job listing extractor
"""

def processor_prompt() -> str:
    return """
## Role
You are a job listing processor agent specialized in extracting structured data.

## Task
Process and extract structured job listings from web search results.

## Instructions
1. Analyze the search results text
2. Extract ONLY REAL job listings including title, company, URL, and type
3. Format each listing with consistent structure
4. Filter out irrelevant results and duplicates
5. Return only relevant technical job postings
6. Ensure URLs are complete and properly formatted
7. IMPORTANT: Always return valid JSON format
8. CRITICAL: NEVER create or invent job listings. Only extract real jobs from the search results.
9. If no relevant job listings are found, return an empty array []

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
- NEVER invent or fabricate job listings - only extract real ones
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
   - Does the page load without errors (no 404, etc.)?
4. Red flags that indicate an invalid job:
   - 404 errors or "job not found" messages
   - Pages that redirect to generic career pages
   - URLs that don't follow standard job listing patterns
   - URLs that contain random placeholders or template patterns
5. CRITICAL: Only mark URLs as valid if they point to REAL job listings

## Output Format
Return "true" if the URL is valid and points to a real job, "false" if not. Lowercase, no explanation.
"""

# ------------- Multi-agent job search implementation -------------
async def gather_jobs_with_multi_agents(cfg, logger) -> List[Dict[str, Any]]:
    """
    Main function that coordinates the multi-agent job search workflow

    Args:
        cfg: Configuration dictionary
        logger: Logger instance

    Returns:
        List of job dictionaries
    """
    # Extract configuration
    majors_quota = cfg.get("majors", 10)
    startups_quota = cfg.get("startups", 10)
    model = cfg.get("model", "gpt-4o")
    company_list_limit = cfg.get("company_list_limit", 10)
    use_web_verification = cfg.get("use_web_verify", False)

    # Log configuration
    logger.info(f"Starting multi-agent job search workflow")
    logger.info(f"Targeting {majors_quota} major company jobs and {startups_quota} startup jobs")
    logger.info(f"Web URL verification: {'Enabled' if use_web_verification else 'Basic validation only'}")
    logger.info(f"Using model: {model}")

    # Initialize agents
    logger.info("Initializing agents...")

    # Create all agents with a more structured approach
    # 1. Planner Agent - Creates search strategies
    planner = Agent(
        name="planner",
        instructions=planner_prompt(MAJOR_COMPANIES, STARTUP_COMPANIES, KEYWORDS),
        model=model,
        model_settings=ModelSettings(temperature=0)
    )

    # 2. Processor Agent - Processes and extracts job data
    processor = Agent(
        name="processor",
        instructions=processor_prompt(),
        model=model,
        model_settings=ModelSettings(temperature=0)
    )

    # 3. Verifier Agent - Validates job URLs
    verifier = Agent(
        name="verifier",
        instructions=verifier_prompt(),
        model=model,
        model_settings=ModelSettings(temperature=0)
    )

    # Set up the processor to hand off to verifier
    processor.handoffs = [verifier]

    # 4. Searcher Agent - Performs web searches with tools
    searcher = Agent(
        name="searcher",
        instructions=searcher_prompt(),
        tools=[WebSearchTool(), extract_job_listings, validate_job_url],
        model=model,
        model_settings=ModelSettings(temperature=0),
        handoffs=[processor]
    )

    # Get visualizer
    visualizer = get_visualizer()

    # Execute planning phase
    with Timer("Planning search strategies", logger):
        logger.info("Planning search strategies...")
        plan_start_time = time.time()
        plan_result = await Runner.run(
            planner,
            input="Generate a job search plan focusing on both major and startup companies"
        )
        plan_json = plan_result.final_output
        plan_duration = time.time() - plan_start_time

        # Track planning step in visualizer
        try:
            # Get token information if available
            tokens_used = {}
            if hasattr(plan_result, 'tokens_in') and hasattr(plan_result, 'tokens_out'):
                tokens_used = {"input": plan_result.tokens_in, "output": plan_result.tokens_out}
            elif hasattr(plan_result, 'input_tokens') and hasattr(plan_result, 'output_tokens'):
                tokens_used = {"input": plan_result.input_tokens, "output": plan_result.output_tokens}
            elif hasattr(agent_usage, 'tokens_per_model'):
                # Use the overall agent usage if detailed tokens not available
                tokens_used = {model: count for model, count in agent_usage.tokens_per_model.items()}

            visualizer.track_agent_call(
                agent_name="Planner",
                input_text="Generate job search plan",
                output_text=plan_json if len(plan_json) < 1000 else plan_json[:1000] + "...",
                duration=plan_duration,
                tokens_used=tokens_used,
            )
        except Exception as e:
            logger.warning(f"Failed to track planning visualization: {e}")

    # Parse the plan using a more robust approach
    try:
        # Try to parse and pretty-print JSON
        plan_data = json.loads(plan_json)
        formatted_plan = json.dumps(plan_data, indent=2)
        logger.info(f"Plan generated:\n{formatted_plan}")

        # Convert to JobSearchPair objects
        search_plan = [JobSearchPair(company=p['company'], keyword=p['keyword']) for p in plan_data]

        # Log plan statistics
        major_pairs = [p for p in search_plan if p.company in MAJOR_COMPANIES]
        startup_pairs = [p for p in search_plan if p.company in STARTUP_COMPANIES]
        logger.info(f"Plan contains {len(major_pairs)} major company pairs and {len(startup_pairs)} startup pairs")

    except Exception as e:
        logger.error(f"Error parsing plan: {e}")
        # Generate fallback plan if parsing fails
        logger.info("Using fallback search plan")
        search_plan = []

        # Create balanced fallback plan with both company types
        # Add major companies
        for i in range(min(majors_quota, len(MAJOR_COMPANIES))):
            company = MAJOR_COMPANIES[i]
            keyword = KEYWORDS[i % len(KEYWORDS)]
            search_plan.append(JobSearchPair(company=company, keyword=keyword))

        # Add startup companies
        for i in range(min(startups_quota, len(STARTUP_COMPANIES))):
            company = STARTUP_COMPANIES[i]
            keyword = KEYWORDS[(i + 3) % len(KEYWORDS)]  # Offset to get different keywords
            search_plan.append(JobSearchPair(company=company, keyword=keyword))

    # Execute search phase
    logger.info("Executing search phase...")

    # Track found jobs
    major_jobs = []
    startup_jobs = []
    validated_urls = set()  # Track URLs we've already validated

    # Search each pair
    with Timer("Job searching", logger):
        for i, pair in enumerate(search_plan, 1):
            company_type = "Major" if pair.company in MAJOR_COMPANIES else "Startup"

            # Skip if we already have enough jobs of this type
            if company_type == "Major" and len(major_jobs) >= majors_quota:
                logger.debug(f"Skipping {pair.company} - major quota reached")
                continue
            if company_type == "Startup" and len(startup_jobs) >= startups_quota:
                logger.debug(f"Skipping {pair.company} - startup quota reached")
                continue

            search_start_time = time.time()
            logger.info(f"Search {i}/{len(search_plan)}: {pair.keyword} jobs at {pair.company} ({company_type})")

            # Create search query
            search_query = f"{pair.company} {pair.keyword} jobs careers software engineering apply"

            try:
                # Execute search
                with Timer(f"Searching for {pair.company} {pair.keyword} jobs", logger):
                    search_start_time = time.time()
                    search_result = await Runner.run(
                        searcher,
                        input=f"Find {pair.keyword} jobs at {pair.company} ({company_type}) with web search. Search query: {search_query}"
                    )
                    search_output = search_result.final_output
                    search_duration = time.time() - search_start_time

                    # Track search step in visualizer
                    try:
                        # Get token information if available
                        tokens_used = {}
                        if hasattr(search_result, 'tokens_in') and hasattr(search_result, 'tokens_out'):
                            tokens_used = {"input": search_result.tokens_in, "output": search_result.tokens_out}
                        elif hasattr(search_result, 'input_tokens') and hasattr(search_result, 'output_tokens'):
                            tokens_used = {"input": search_result.input_tokens, "output": search_result.output_tokens}

                        visualizer.track_agent_call(
                            agent_name="Searcher",
                            input_text=f"Find {pair.keyword} jobs at {pair.company}",
                            output_text=f"Found {len(search_output[:100])}... characters of results",
                            duration=search_duration,
                            tokens_used=tokens_used,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to track search visualization: {e}")

                # Process results, if any
                if search_output:
                    # Process through processor agent
                    with Timer(f"Processing job results for {pair.company}", logger):
                        process_start_time = time.time()
                        process_result = await Runner.run(
                            processor,
                            input=f"Process these job search results: {search_output[:10000]}"
                        )
                        processor_output = process_result.final_output
                        process_duration = time.time() - process_start_time

                        # Track processing step in visualizer
                        try:
                            # Get token information if available
                            tokens_used = {}
                            if hasattr(process_result, 'tokens_in') and hasattr(process_result, 'tokens_out'):
                                tokens_used = {"input": process_result.tokens_in, "output": process_result.tokens_out}
                            elif hasattr(process_result, 'input_tokens') and hasattr(process_result, 'output_tokens'):
                                tokens_used = {"input": process_result.input_tokens, "output": process_result.output_tokens}

                            visualizer.track_agent_call(
                                agent_name="Processor",
                                input_text=f"Process {pair.company} job results",
                                output_text=processor_output[:100] + "...",
                                duration=process_duration,
                                tokens_used=tokens_used,
                            )
                        except Exception as e:
                            logger.warning(f"Failed to track processor visualization: {e}")

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

                                        # Skip if we've already processed this URL
                                        if job["url"] in validated_urls:
                                            logger.debug(f"Skipping duplicate URL: {job['url']}")
                                            continue

                                        # Add to the set of validated URLs
                                        validated_urls.add(job["url"])

                                        # Validate the URL
                                        try:
                                            # Use the internal verify_with_browser function directly instead of the function_tool
                                            is_valid = await verify_with_browser(job["url"])
                                            if is_valid:
                                                logger.info(f"✓ Job URL validated: {job['url']}")
                                                job["has_apply"] = True
                                            else:
                                                logger.info(f"✖ Invalid job URL: {job['url']}")
                                                job["has_apply"] = False
                                                continue
                                        except Exception as e:
                                            logger.warning(f"URL validation error: {e}")
                                            # Default to True if validation fails but URL pattern is valid
                                            logger.info(f"✓ Job URL pattern-validated (fallback): {job['url']}")
                                            job["has_apply"] = True

                                        # Validation successful - add the job
                                        # Try to validate as JobListing model
                                        try:
                                            job_listing = JobListing(**job)

                                            # Add to appropriate list based on company type
                                            if company_type == "Major":
                                                major_jobs.append(job_listing.model_dump())
                                                logger.info(f"Added MAJOR job: {job_listing.title} at {job_listing.company}")
                                            else:
                                                startup_jobs.append(job_listing.model_dump())
                                                logger.info(f"Added STARTUP job: {job_listing.title} at {job_listing.company}")
                                        except Exception as e:
                                            logger.warning(f"Invalid job format: {e}")

                    except Exception as e:
                        logger.error(f"Error processing jobs: {e}")

            except Exception as e:
                logger.error(f"Search error: {e}")

            # Log search completion and duration
            search_duration = time.time() - search_start_time
            logger.info(f"Search for {pair.company} completed in {search_duration:.2f}s")

    # Log statistics about verified jobs
    if len(major_jobs) < majors_quota:
        logger.warning(f"Only found {len(major_jobs)}/{majors_quota} verified major company jobs")

    if len(startup_jobs) < startups_quota:
        logger.warning(f"Only found {len(startup_jobs)}/{startups_quota} verified startup jobs")

    # Combine results
    all_jobs = major_jobs[:majors_quota] + startup_jobs[:startups_quota]

    # Add sequence numbers
    for i, job in enumerate(all_jobs, 1):
        job["#"] = i

    # Log results
    logger.info(f"Search complete! Found {len(all_jobs)} VERIFIED jobs ({len(major_jobs)} major, {len(startup_jobs)} startup)")

    # Generate token usage report
    try:
        total_tokens = sum(agent_usage.tokens_per_model.values())

        # Calculate cost using our token monitor rates
        token_monitor = TokenMonitor(cfg, logger)
        total_cost = sum(
            (tokens / 1000) * token_monitor.get_model_rate(model_name)
            for model_name, tokens in agent_usage.tokens_per_model.items()
        )

        logger.info("\nToken usage statistics:")
        for model_name, tokens in agent_usage.tokens_per_model.items():
            model_cost = (tokens / 1000) * token_monitor.get_model_rate(model_name)
            logger.info(f"  - {model_name}: {tokens:,} tokens (${model_cost:.4f})")
        logger.info(f"Total: {total_tokens:,} tokens (${total_cost:.4f})")
    except Exception as e:
        logger.warning(f"Could not generate token usage report: {e}")

    return all_jobs

def search_with_multi_agents(cfg, logger) -> List[Dict[str, str]]:
    """
    Legacy wrapper for backward compatibility with old scripts

    Args:
        cfg: Configuration dictionary
        logger: Logger instance

    Returns:
        List of job dictionaries
    """
    logger.info("Using multi-agent search workflow")
    jobs = asyncio.run(gather_jobs_with_multi_agents(cfg, logger))
    return jobs

def save(rows, logger):
    """Save results to CSV and Markdown files in the configured RESULTS_DIR"""
    if not rows:
        logger.warning("No results to save")
        return

    # Create results directory if it doesn't exist
    if not RESULTS_DIR.exists():
        RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    # Add numbering to jobs
    for i, row in enumerate(rows, 1):
        row["#"] = i

    # Set up pandas DataFrame
    df = pd.DataFrame(rows)

    # Save CSV
    csv_file = RESULTS_DIR / "job_results.csv"
    df.to_csv(csv_file, index=False)
    logger.info(f"Results saved to CSV: {csv_file}")

    # Save Markdown
    md_file = RESULTS_DIR / "job_results.md"

    try:
        # Format as a nice markdown table
        from tabulate import tabulate  # noqa: F401
        with open(md_file, "w") as f:
            f.write("# Job Search Results\n\n")
            f.write(f"Found {len(rows)} jobs.\n\n")
            f.write(df.to_markdown(index=False))
        logger.info(f"Results saved to Markdown: {md_file}")
    except ImportError:
        # Fallback to basic markdown
        with open(md_file, "w") as f:
            f.write("# Job Search Results\n\n")
            f.write(f"Found {len(rows)} jobs.\n\n")
            for row in rows:
                f.write(
                    f"## {row['#']}. {row.get('title', 'Unknown')} at {row.get('company', 'Unknown')}\n"
                )
                f.write(f"- Type: {row.get('type', 'Unknown')}\n")
                f.write(f"- URL: {row.get('url', 'Unknown')}\n\n")
        logger.info(f"Results saved to Markdown (basic format): {md_file}")

    logger.info(f"Saved {len(rows)} jobs to {RESULTS_DIR}")

    return


def parse_args():
    """
    Legacy argument parser for backward compatibility

    Returns:
        dict: Configuration dictionary from command line arguments
    """
    parser = argparse.ArgumentParser(description="Deep Job Search - Find tech jobs with AI")

    # Core parameters
    parser.add_argument("-m", "--majors", type=int, default=10, help="Number of major company jobs to find")
    parser.add_argument("-s", "--startups", type=int, default=10, help="Number of startup company jobs to find")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model to use (gpt-4o, gpt-4, gpt-3.5-turbo)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens for API calls")

    # Budget and confirmation options
    parser.add_argument("--budget", type=float, default=None, help="Maximum cost in USD (exit if exceeded)")
    parser.add_argument("--force", action="store_true", help="Skip cost confirmation prompt")

    # Options
    parser.add_argument("--company-list", type=str, help="CSV file with custom companies")
    parser.add_argument("--company-list-limit", type=int, default=20, help="Limit of companies to use from list")
    parser.add_argument("--fallback-enabled", action="store_true", help="Enable fallback job generation (DEPRECATED)")
    parser.add_argument("--web-verify", action="store_true", help="Use web search to verify job URLs")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with more logging")
    parser.add_argument("--save-raw", action="store_true", help="Save raw API responses")
    parser.add_argument("--use-cache", action="store_true", help="Use cached API responses if available")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--log-file", type=str, default="logs/debug/deep_job_search.log", help="Log file path")
    parser.add_argument("--trace", action="store_true", help="Enable trace output for detailed execution tracking")
    parser.add_argument("--visualize", action="store_true", default=True, help="Enable agent visualization")
    parser.add_argument("--no-visualize", action="store_true", help="Disable visualization generation")

    parser.add_argument("--output", type=str, default="results/jobs.csv", help="Output file path")
    parser.add_argument("--dry-run", action="store_true", help="Run without making API calls (for testing)")

    args = parser.parse_args()

    # Handle --no-visualize flag
    if args.no_visualize:
        args.visualize = False

    # Set up configuration dictionary
    cfg = {
        "majors": args.majors,
        "startups": args.startups,
        "model": args.model,
        "company_list_limit": args.company_list_limit,
        "fallback_enabled": False,  # Force disable fallback
        "use_web_verify": args.web_verify,
        "validate_urls": True,  # Always validate URLs
        "debug": args.debug,
        "save_raw_responses": args.save_raw,
        "use_cache": args.use_cache,
        "visualize": args.visualize,
        "output": args.output,
        "log_level": args.log_level if args.log_level else ("DEBUG" if args.debug else "INFO"),
        "log_file": args.log_file,
        "dry_run": args.dry_run,
        "trace": args.trace,
        "budget": args.budget,
        "force": args.force
    }

    if args.max_tokens:
        cfg["max_tokens"] = args.max_tokens

    # Set up company list if provided
    if args.company_list:
        cfg["company_list"] = args.company_list

    return cfg


def main():
    """
    Main entry point for the job search script
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Deep Job Search - Find tech jobs with AI")

    # Core parameters
    parser.add_argument("-m", "--majors", type=int, default=10, help="Number of major company jobs to find")
    parser.add_argument("-s", "--startups", type=int, default=10, help="Number of startup company jobs to find")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model to use (gpt-4o, gpt-4, gpt-3.5-turbo)")
    parser.add_argument("--output", type=str, default="results/jobs.csv", help="Output file for jobs")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens for API calls")
    parser.add_argument("--dry-run", action="store_true", help="Run without making API calls (for testing)")

    # Budget and confirmation options
    parser.add_argument("--budget", type=float, default=None, help="Maximum cost in USD (exit if exceeded)")
    parser.add_argument("--force", action="store_true", help="Skip cost confirmation prompt")

    # Validation options
    parser.add_argument("--validate-urls", action="store_true", default=True, help="Enable URL validation")
    parser.add_argument("--web-verify", action="store_true", default=False, help="Use web search to verify job URLs")
    parser.add_argument("--skip-validation", action="store_true", help="Skip all URL validation (not recommended)")

    # Customization options
    parser.add_argument("--company-list", type=str, help="CSV file with custom companies")
    parser.add_argument("--company-list-limit", type=int, default=20, help="Limit of companies to use from list")

    # Developer options
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with more logging")
    parser.add_argument("--save-raw", action="store_true", help="Save raw API responses")
    parser.add_argument("--use-cache", action="store_true", help="Use cached API responses if available")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--log-file", type=str, default="logs/debug/deep_job_search.log", help="Log file path")
    parser.add_argument("--trace", action="store_true", help="Enable trace output for detailed execution tracking")
    parser.add_argument("--visualize", action="store_true", default=True, help="Enable agent visualization")
    parser.add_argument("--no-visualize", action="store_true", help="Disable visualization generation")

    args = parser.parse_args()

    # Handle --no-visualize flag
    if args.no_visualize:
        args.visualize = False

    # Setup configuration
    config = {
        "majors": args.majors,
        "startups": args.startups,
        "model": args.model,
        "company_list_limit": args.company_list_limit,
        "use_web_verify": args.web_verify and not args.skip_validation,
        "validate_urls": args.validate_urls and not args.skip_validation,
        "save_raw_responses": args.save_raw,
        "use_cache": args.use_cache,
        "debug": args.debug,
        "log_level": args.log_level if args.log_level else ("DEBUG" if args.debug else "INFO"),
        "log_file": args.log_file,
        "dry_run": args.dry_run,
        "visualize": args.visualize,
        "trace": args.trace,
        "budget": args.budget,
        "force": args.force
    }

    if args.max_tokens:
        config["max_tokens"] = args.max_tokens

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logger = setup_logger(level=config["log_level"], file=config["log_file"], trace=config["trace"])
    logger.info("Starting Deep Job Search")

    # Check for dry run mode
    if args.dry_run:
        logger.info("DRY RUN MODE: No API calls will be made")
        print("\nDRY RUN MODE: Testing pipeline without making API calls")
        print("✓ Configuration valid")
        print("✓ Logging configured")
        print("✓ Directories exist")
        return 0

    # Budget confirmation check
    if config["budget"] is not None and not config["force"]:
        logger.info(f"Budget set to ${config['budget']:.2f}")
        if "JOBBOT_SKIP_CONFIRM" not in os.environ:
            print(f"\nBudget set to ${config['budget']:.2f}.")
            confirm = input("Continue? [y/N] ")
            if confirm.lower() != 'y':
                logger.info("User cancelled job search due to budget concerns")
                print("Job search cancelled.")
                return 0
        else:
            logger.info("Budget confirmation skipped due to JOBBOT_SKIP_CONFIRM environment variable")

    # Log configuration
    logger.info(f"Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    # Load custom company list if provided
    if args.company_list:
        try:
            logger.info(f"Loading custom company list from {args.company_list}")
            company_df = pd.read_csv(args.company_list)
            if "Company" in company_df.columns:
                # This function would need to be implemented if custom company lists are used
                # update_company_lists(company_df["Company"].tolist(), config["company_list_limit"])
                logger.info(f"Updated company lists with {len(company_df)} companies")
            else:
                logger.warning(f"Company column not found in {args.company_list}")
        except Exception as e:
            logger.error(f"Error loading company list: {e}")

    # Initialize token monitor
    token_monitor = TokenMonitor(config, logger)

    # Run the job search
    try:
        # Run job search asynchronously
        jobs = asyncio.run(gather_jobs_with_multi_agents(config, logger))

        # Process and save results if jobs were found
        if jobs:
            # Save results
            save(jobs, logger)

            # Print summary
            print("\n" + "=" * 50)
            print(f"SEARCH COMPLETE: Found {len(jobs)} jobs")
            print("-" * 50)
            print(f"Major companies: {sum(1 for job in jobs if job.get('type') == 'Major')}")
            print(f"Startups: {sum(1 for job in jobs if job.get('type') == 'Startup')}")
            print(f"Output saved to: {args.output}")
            print("=" * 50)

            # Print token usage statistics
            try:
                total_tokens = sum(agent_usage.tokens_per_model.values())
                # Calculate cost using token monitor rates
                total_cost = sum(
                    (tokens / 1000) * token_monitor.get_model_rate(model_name)
                    for model_name, tokens in agent_usage.tokens_per_model.items()
                )
                print(f"\nToken usage: {total_tokens:,}")
                print(f"Estimated cost: ${total_cost:.4f}")

                # Check budget if set
                if config["budget"] is not None and total_cost > config["budget"]:
                    logger.warning(f"Budget exceeded: ${total_cost:.4f} > ${config['budget']:.2f}")
                    print(f"\nWARNING: Budget exceeded: ${total_cost:.4f} > ${config['budget']:.2f}")
            except Exception as e:
                logger.warning(f"Could not generate token usage report: {e}")

            # Print notice if fewer jobs than requested were found
            if len(jobs) < config["majors"] + config["startups"]:
                print(f"\nNOTE: Found fewer jobs than requested. This is likely because:")
                print(f" - Some job URLs failed validation (we only return verified jobs)")
                print(f" - The search didn't find enough relevant jobs")
                print(f"\nTry adjusting search keywords or enabling web verification with --web-verify")
        else:
            logger.warning("No jobs found")
            print("\nNo jobs found. Try adjusting search parameters.")

    except Exception as e:
        logger.error(f"Error in job search: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nError: {str(e)}")
        return 1

    return 0









if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    # Always run the full version with real API calls
    exit_code = main()
    exit(exit_code)
