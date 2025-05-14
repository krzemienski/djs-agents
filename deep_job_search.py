#!/usr/bin/env python
"""
deep_job_search.py — Deep Job Search with Responses API implementation

This file uses the Responses API for a simple, efficient job search.
"""

import os, re, json, time, argparse, logging, sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

# Import our utility modules
from logger_utils import setup_enhanced_logger, DepthContext
from api_wrapper import initialize_api_wrapper, get_api_wrapper
from agent_visualizer import initialize_visualizer, get_visualizer

load_dotenv()

# Constants and configuration
OUTPUT_DIR = Path(__file__).with_suffix("")
RESULTS_DIR = Path("results")  # For Docker compatibility

# Companies and keywords lists
MAJOR_COMPANIES = [
    'Amazon','Apple','Google','Meta','Microsoft','Netflix','NVIDIA',
    'OpenAI','Anthropic','Disney','Hulu','TikTok','Snap','Zoom',
    'Cisco','Comcast','Warner Bros. Discovery','Paramount','IBM','Valve'
]
STARTUP_COMPANIES = [
    'Livepeer','Twelve Labs','Tapcart','Mux','Apporto','GreyNoise',
    'Daily.co','Temporal Technologies','Bitmovin','JWPlayer','Brightcove',
    'Cloudflare','Fastly','Firework','Bambuser','FloSports','StageNet',
    'Uscreen','Mmhmm','StreamYard','Agora','Conviva','Peer5','Wowza',
    'Hopin','Kumu Networks','Touchcast','Theo Technologies','Vidyard',
    'Cinedeck','InPlayer','Netlify','Vowel','Edge-Next','Streann Media',
    'Gather','Frequency','Truepic','LiveControl','OpenSelect'
]
KEYWORDS = ['video','streaming','media','encoding','cloud','kubernetes','backend','principal','lead','infrastructure']

# ------------- Data models -------------
class JobListing(BaseModel):
    """A job listing with relevant details"""
    title: str
    company: str
    type: str  # Major or Startup
    url: str
    has_apply: bool = False

# ------------- logging -------------
def setup_logger(level:str='INFO', file:str|None=None)->logging.Logger:
    """Set up the enhanced logger with improved format and API logging capabilities."""
    # Create API log file path if a main log file is provided
    api_log_file = None
    if file:
        api_log_file = Path(file).with_suffix('.api.log')

    # Create logs directory structure
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)

    visuals_dir = logs_dir / 'visuals'
    visuals_dir.mkdir(exist_ok=True)

    # Initialize the visualizer
    initialize_visualizer(visuals_dir)

    # Set up the enhanced logger
    logger = setup_enhanced_logger(
        level=level,
        file=file,
        api_log_file=api_log_file
    )

    # Initialize the API wrapper with our logger
    initialize_api_wrapper(logger)

    # Log diagnostic info
    logger.debug(f"Logger initialized with level: {level}")
    logger.debug(f"Python version: {sys.version}")

    return logger

class Timer:
    """Enhanced timer context manager with more detailed timing info using DepthContext."""
    def __init__(self, label, logger, log_level='INFO'):
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
        'gpt-4.1': 0.01,
        'gpt-4o': 0.005,
        'gpt-4o-mini': 0.002,
        'gpt-3.5-turbo': 0.001,
        'o3': 0.0005,
    }

    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.max_tokens = cfg.get('max_tokens', 100000)
        self.total_tokens_used = 0

    def get_model_rate(self, model: str) -> float:
        """Get the cost rate for a model, with fallback for unknown models"""
        base_model = model.split('-preview')[0].split('-vision')[0]
        return self.COST_PER_1K.get(base_model, 0.005)  # Default to mid-tier pricing if unknown

# ------------- Responses API implementation -------------
def format_company_list(companies, limit=10):
    """Format a list of companies for the prompt"""
    if limit and len(companies) > limit:
        return ", ".join(companies[:limit]) + f" and {len(companies) - limit} more"
    return ", ".join(companies)

def parse_responses_output(response, logger):
    """
    Parse the output from the Responses API to extract job data

    Args:
        response: Response object from the OpenAI Responses API
        logger: Logger instance

    Returns:
        List of job dictionaries or empty list on error
    """
    try:
        with Timer("Parsing job data", logger):
            # Extract text from the response
            # Try different methods to get the result text based on what's available
            result_text = ""

            # Focus on the output field which seems to be most reliable
            if hasattr(response, 'output') and response.output:
                # The output field typically contains the message from the assistant
                if isinstance(response.output, list):
                    # Add bounds check to avoid IndexError
                    output_idx = 1 if len(response.output) > 1 else 0
                    if output_idx < len(response.output):
                        output_message = response.output[output_idx]
                        if hasattr(output_message, 'content') and isinstance(output_message.content, list):
                            logger.debug("Using output_message.content")
                            for content_item in output_message.content:
                                if hasattr(content_item, 'text'):
                                    logger.debug("Found text in content item")
                                    result_text = content_item.text
                                    break

                        # If we still don't have text, try other methods
                        if not result_text and hasattr(output_message, 'text'):
                            logger.debug("Using output_message.text")
                            result_text = output_message.text

                        # If nothing worked, convert to string
                        if not result_text:
                            logger.debug("Using str(output_message)")
                            result_text = str(output_message)

            # Still no result? Fall back to text attribute
            if not result_text and hasattr(response, 'text') and response.text:
                logger.debug("Using response.text attribute")
                if hasattr(response.text, 'value'):
                    result_text = response.text.value
                else:
                    result_text = str(response.text)

            # Last resort, try output_text
            if not result_text and hasattr(response, 'output_text') and response.output_text:
                logger.debug("Using response.output_text")
                result_text = str(response.output_text)

            # If we still don't have text, use the entire response
            if not result_text:
                logger.debug("Using fallback approach with whole response")
                result_text = str(response)

            logger.info(f"Search returned {len(result_text)} characters of results")

            # Now look for job listings in the response text
            jobs = []

            logger.debug(f"Result text type: {type(result_text)}")

            # Check if the response text itself might already be JSON or contains job data
            if isinstance(result_text, dict) or isinstance(result_text, list):
                try:
                    # If it's already a list or dict, try to use it directly
                    if isinstance(result_text, list):
                        jobs = result_text
                    else:
                        # If it's a dict with job data, wrap it in a list
                        if 'title' in result_text and 'company' in result_text and 'url' in result_text:
                            jobs = [result_text]

                    if jobs:
                        logger.info(f"Used direct data structure: {len(jobs)} jobs found")
                        return jobs
                except Exception as e:
                    logger.debug(f"Error using direct structure: {e}")

            # Convert to string if not already
            if not isinstance(result_text, str):
                result_text = str(result_text)

            # Look for a JSON array in the text using a safer approach
            # Find the first '[' and last ']' characters to extract potential JSON array
            try:
                start_idx = result_text.find('[')
                if start_idx != -1:
                    # Find the matching closing bracket
                    bracket_count = 0
                    for i in range(start_idx, len(result_text)):
                        if result_text[i] == '[':
                            bracket_count += 1
                        elif result_text[i] == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                # We found the matching closing bracket
                                end_idx = i + 1
                                potential_json = result_text[start_idx:end_idx]
                                try:
                                    jobs = json.loads(potential_json)
                                    logger.info(f"Successfully parsed JSON data: {len(jobs)} jobs found")
                                    return jobs
                                except json.JSONDecodeError as e:
                                    logger.debug(f"JSON decode error: {e}")
                                break
            except Exception as e:
                logger.debug(f"Error during JSON bracket extraction: {e}")

            # If no JSON found, try to parse markdown table
            table_match = re.search(r'\|\s*Title\s*\|\s*Company\s*\|\s*URL\s*\|', result_text)
            if table_match:
                # Parse markdown table
                jobs = []
                lines = result_text.split('\n')
                start_idx = None

                for i, line in enumerate(lines):
                    if '|' in line and ('Title' in line or 'title' in line) and ('Company' in line or 'company' in line):
                        start_idx = i + 2  # Skip header and separator
                        break

                if start_idx:
                    for i in range(start_idx, len(lines)):
                        line = lines[i].strip()
                        if not line or '|' not in line:
                            continue

                        parts = [p.strip() for p in line.split('|')]
                        if len(parts) >= 4:
                            # Handle row with Title, Company, URL columns
                            title = parts[1]
                            company = parts[2]
                            url = parts[3]

                            if 'http' in url:
                                job_type = "Major" if company in MAJOR_COMPANIES else "Startup"
                                jobs.append({
                                    "title": title,
                                    "company": company,
                                    "url": url,
                                    "type": job_type
                                })
                logger.info(f"Successfully parsed markdown table: {len(jobs)} jobs found")
                return jobs

            # If parsing table fails, use regex to find potential job listings
            logger.info("Falling back to regex pattern matching for job data")
            jobs = []

            # Regex patterns for different job title formats
            title_at_company_pattern = r'([A-Za-z\s\-–&]+) at ([A-Za-z\s\-–&]+)'
            company_hiring_pattern = r'([A-Za-z\s\-–&]+) is hiring:?\s*([A-Za-z\s\-–&]+)'
            url_pattern = r'(https?://[^\s]+)'

            # Words to ignore as titles (false positives like action verbs)
            ignore_words = ['apply', 'learn', 'more', 'view', 'click', 'check', 'see', 'visit', 'join']

            # Find "<title> at <company>" matches
            title_matches = re.findall(title_at_company_pattern, result_text)
            url_matches = re.findall(url_pattern, result_text)

            for i, (title, company) in enumerate(title_matches):
                # Skip if the title is a common action verb (false positive)
                if title.strip().lower() in ignore_words:
                    continue

                if i < len(url_matches):
                    url = url_matches[i]
                    job_type = "Major" if company.strip() in MAJOR_COMPANIES else "Startup"
                    jobs.append({
                        "title": title.strip(),
                        "company": company.strip(),
                        "url": url,
                        "type": job_type
                    })

            # Also find "<company> is hiring: <title>" matches
            hiring_matches = re.findall(company_hiring_pattern, result_text)
            start_idx = len(jobs)

            for i, (company, title) in enumerate(hiring_matches):
                # Skip if the title is a common action verb (false positive)
                if title.strip().lower() in ignore_words:
                    continue

                if start_idx + i < len(url_matches):
                    url = url_matches[start_idx + i]
                    job_type = "Major" if company.strip() in MAJOR_COMPANIES else "Startup"
                    jobs.append({
                        "title": title.strip(),
                        "company": company.strip(),
                        "url": url,
                        "type": job_type
                    })

            logger.info(f"Found {len(jobs)} jobs using regex pattern matching")
            return jobs

    except Exception as e:
        logger.error(f"Error parsing job data: {e}")
        # Return empty list on error
        return []

def search_jobs_with_responses(query, model, logger):
    """
    Search for jobs using the Responses API

    Args:
        query: Search query or instructions
        model: Model to use for the search
        logger: Logger instance

    Returns:
        List of job dictionaries
    """
    logger.info(f"Searching for jobs with query: {query[:100]}...")

    # Create OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Get visualizer
    visualizer = get_visualizer()

    # Start timing
    start_time = time.time()

    with Timer("Job search", logger):
        try:
            # Create a response with web search enabled
            response = client.responses.create(
                model=model,
                input=query,
                tools=[{"type": "web_search"}]
            )

            # Record API call in visualizer
            duration = time.time() - start_time
            visualizer.track_api_call(
                function_name="responses.create",
                api_type="responses",
                success=True,
                duration=duration
            )

            # Log response structure for debugging
            logger.debug(f"Response type: {type(response)}")
            logger.debug(f"Response attributes: {dir(response)}")

            # Debug the output field specifically
            if hasattr(response, 'output'):
                logger.debug(f"Output type: {type(response.output)}")
                if isinstance(response.output, list):
                    logger.debug(f"Output length: {len(response.output)}")
                    for i, item in enumerate(response.output):
                        logger.debug(f"Output[{i}] type: {type(item)}")
                        logger.debug(f"Output[{i}] attributes: {dir(item)}")
        except Exception as e:
            # Record failed API call in visualizer
            duration = time.time() - start_time
            visualizer.track_api_call(
                function_name="responses.create",
                api_type="responses",
                success=False,
                duration=duration
            )
            logger.error(f"Error creating Responses API request: {e}")
            return []

        # Access response properties for debugging
    logger.debug(f"Response has text attribute: {hasattr(response, 'text')}")
    logger.debug(f"Response has output_text attribute: {hasattr(response, 'output_text')}")

    # Parse the response
    return parse_responses_output(response, logger)

def search_companies_with_responses(company_type, companies, count, model, logger, limit=None):
    """
    Unified search function for both major and startup companies

    Args:
        company_type: "Major" or "Startup"
        companies: List of companies to search
        count: Number of jobs to search for
        model: Model to use for search
        logger: Logger instance
        limit: Limit for company list formatting (optional)

    Returns:
        List of job dictionaries
    """
    companies_text = format_company_list(companies, limit)
    keywords = ", ".join(KEYWORDS[:5])

    logger.info(f"Searching for {count} jobs at {company_type.lower()} companies using Responses API")

    query = f"""
    I need you to find {count} software engineering job listings at {company_type.lower()} companies in the video/streaming industry.

    Focus on these companies: {companies_text}
    Look for roles with keywords like: {keywords}

    For each job listing, I need:
    1. The exact job title
    2. The company name
    3. The direct URL to the job posting (not a careers page)

    Return the results as a structured JSON array with fields: title, company, url, type
    Set the "type" field to "{company_type}" for all these companies.

    Only include real job postings with direct application links.
    """

    return search_jobs_with_responses(query, model, logger)

def search_major_companies_with_responses(count, model, logger, limit=None):
    """Search for jobs at major companies using Responses API"""
    return search_companies_with_responses("Major", MAJOR_COMPANIES, count, model, logger, limit)

def search_startup_companies_with_responses(count, model, logger, limit=None):
    """Search for jobs at startup companies using Responses API"""
    return search_companies_with_responses("Startup", STARTUP_COMPANIES, count, model, logger, limit)

def search_with_responses(cfg, logger) -> List[Dict[str, str]]:
    """
    Gather job listings using the OpenAI Responses API approach.
    This is a simpler and more efficient alternative to the multi-agent approach.

    Args:
        cfg: Configuration dictionary
        logger: Logger instance

    Returns:
        List of job listing dictionaries
    """
    # Get visualizer
    visualizer = get_visualizer()

    # Reset visualizer for new run
    visualizer.reset()

    # Start timing
    api_start_time = time.time()

    # Log function start with configuration
    logger.info(f"Job search started with Responses API using configuration: {json.dumps(cfg, indent=2)}")

    # Estimate token usage and cost
    tokens_estimate = (cfg['majors'] + cfg['startups']) * 2000  # Rough estimate

    # Calculate cost based on model
    model = cfg.get('model', 'gpt-4o')
    rate = TokenMonitor.COST_PER_1K.get(model, 0.005)  # Default to gpt-4o rate
    cost_estimate = (tokens_estimate / 1000) * rate

    logger.info(f"Estimated resource usage:")
    logger.info(f"  - Tokens: ~{tokens_estimate:,} tokens")
    logger.info(f"  - Cost: ~${cost_estimate:.4f}")

    if cost_estimate > 0.50:  # Arbitrary threshold for a "high" cost
        logger.warning(f"⚠️ COST WARNING: Estimated cost (${cost_estimate:.4f}) exceeds $0.50")
        logger.warning(f"Consider reducing job counts or using less expensive models")

        # Give user a chance to abort if cost is high
        if not cfg.get('force', False) and not os.environ.get('JOBBOT_SKIP_CONFIRM', ''):
            logger.info("Continue? (y/n, or set JOBBOT_SKIP_CONFIRM=1 to skip this prompt)")
            try:
                response = input().strip().lower()
                if response != 'y':
                    logger.info("Aborting job search")
                    return []
            except (KeyboardInterrupt, EOFError):
                logger.info("\nAborting job search")
                return []

    # Check for estimate-only mode
    if os.environ.get('JOBBOT_ESTIMATE_ONLY', ''):
        logger.info(f"Estimate-only mode enabled. Exiting now.")
        return []

    # Search for jobs
    major_jobs = []
    startup_jobs = []

    # Get company list limit
    company_list_limit = cfg.get('company_list_limit', 10)

    with Timer("Overall job search", logger):
        if cfg['majors'] > 0:
            major_jobs = search_major_companies_with_responses(
                count=cfg['majors'],
                model=model,
                logger=logger,
                limit=company_list_limit
            )
            logger.info(f"Found {len(major_jobs)} major company jobs")

            # Ensure has_apply field exists for compatibility
            for job in major_jobs:
                job['has_apply'] = True

        if cfg['startups'] > 0:
            startup_jobs = search_startup_companies_with_responses(
                count=cfg['startups'],
                model=model,
                logger=logger,
                limit=company_list_limit
            )
            logger.info(f"Found {len(startup_jobs)} startup jobs")

            # Ensure has_apply field exists for compatibility
            for job in startup_jobs:
                    job['has_apply'] = True

    # Combine results
    all_jobs = major_jobs[:cfg['majors']] + startup_jobs[:cfg['startups']]
    logger.info(f"Total jobs found: {len(all_jobs)}")

    # Generate visualizations
    visualizer = get_visualizer()
    try:
        # Track overall API usage for visualization
        api_duration = time.time() - api_start_time
        token_usage = {cfg.get('model', 'gpt-4o'): tokens_estimate}

        # Track search as a component step
        visualizer.track_agent_call(
            agent_name="JobSearch",
            input_text=f"Search for {cfg['majors']} major and {cfg['startups']} startup jobs",
            output_text=f"Found {len(all_jobs)} jobs",
            duration=api_duration,
            tokens_used=token_usage
        )

        # Generate visualization files in logs/visuals
        timeline_file = visualizer.generate_timeline_diagram(
            title=f"Job Search Execution Timeline ({cfg.get('model', 'gpt-4o')})"
        )
        report_file = visualizer.generate_report()

        logger.info(f"Generated visualization timeline: {timeline_file}")
        logger.info(f"Generated execution report: {report_file}")
    except Exception as e:
        logger.warning(f"Failed to generate visualizations: {e}")

    return all_jobs

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
        row['#'] = i

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
        from tabulate import tabulate
        with open(md_file, 'w') as f:
            f.write(f"# Job Search Results\n\n")
            f.write(f"Found {len(rows)} jobs.\n\n")
            f.write(df.to_markdown(index=False))
        logger.info(f"Results saved to Markdown: {md_file}")
    except ImportError:
        # Fallback to basic markdown
        with open(md_file, 'w') as f:
            f.write(f"# Job Search Results\n\n")
            f.write(f"Found {len(rows)} jobs.\n\n")
            for row in rows:
                f.write(f"## {row['#']}. {row.get('title', 'Unknown')} at {row.get('company', 'Unknown')}\n")
                f.write(f"- Type: {row.get('type', 'Unknown')}\n")
                f.write(f"- URL: {row.get('url', 'Unknown')}\n\n")
        logger.info(f"Results saved to Markdown (basic format): {md_file}")

    logger.info(f"Saved {len(rows)} jobs to {RESULTS_DIR}")

    return

def parse():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Deep Job Search: OpenAI powered job search tool for finding software engineering jobs in video/streaming companies")
    parser.add_argument("--majors", type=int, default=10, help="Number of major company jobs to find")
    parser.add_argument("--startups", type=int, default=10, help="Number of startup company jobs to find")
    parser.add_argument("--sample", action="store_true", help="Run with minimal settings (2 major, 2 startup jobs)")
    parser.add_argument("--max-tokens", type=int, default=100000, help="Max tokens to use")
    parser.add_argument("--budget", type=float, help="Maximum cost in USD (exit if exceeded)")
    parser.add_argument("--force", action="store_true", help="Skip cost confirmation prompt")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    parser.add_argument("--log-file", help="Log to this file in addition to console")
    parser.add_argument("--trace", action="store_true", help="Enable trace output")
    parser.add_argument("--use-web-verify", action="store_true", help="Use web search for URL verification")
    parser.add_argument("--model", default="gpt-4o",
                        help="Model to use for Responses API implementation (default: gpt-4o)")
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="Generate visualization diagrams (default: True)")
    parser.add_argument("--no-visualize", action="store_false", dest="visualize",
                        help="Disable visualization generation")
    parser.add_argument("--company-list-limit", type=int, default=10,
                        help="Maximum number of companies to list in prompts (default: 10)")

    # No legacy arguments needed

    args = parser.parse_args()

    # Handle sample mode
    if args.sample:
        args.majors = 2
        args.startups = 2
        args.log_level = "DEBUG"

    # Convert to config dict
    cfg = {
        'majors': args.majors,
        'startups': args.startups,
        'max_tokens': args.max_tokens,
        'budget': args.budget,
        'force': args.force,
        'use_web_verify': args.use_web_verify,
        'log_level': args.log_level,
        'log_file': args.log_file,
        'trace': args.trace,
        'model': args.model,
        'company_list_limit': args.company_list_limit,
    }

    return cfg

def main():
    """Main function to run the job search"""
    # Parse arguments
    cfg = parse()

    # Configure results directory for Docker compatibility
    global RESULTS_DIR
    # Check if we're running in a Docker container
    if os.environ.get("RUNNING_IN_CONTAINER") == "1":
        # In Docker, results should go to /app/results which is mounted to the host
        RESULTS_DIR = Path("/app/results")
    else:
        # Otherwise use local results directory
        RESULTS_DIR = Path("results")

    # Create required directories
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    # Initialize logger
    logger = setup_logger(level=cfg['log_level'], file=cfg['log_file'])

    # Check if visualization is enabled
    if cfg.get('visualize', True):
        # Initialize visualizer if not already initialized
        visuals_dir = Path('logs/visuals')
        initialize_visualizer(visuals_dir)

    # Log startup info
    logger.info(f"deep_job_search starting with Python {sys.version}")

    # Check for estimate-only mode
    if os.environ.get('JOBBOT_ESTIMATE_ONLY', ''):
        logger.info(f"Estimate-only mode enabled. Exiting now.")
        return 0

    # Log execution start
    logger.info("Job search execution started")

    # Run the job search
    rows = search_with_responses(cfg, logger)

    # Save results
    if rows:
        save(rows, logger)

    return 0

if __name__=='__main__':
    exit_code = main()
    exit(exit_code)
