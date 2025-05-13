#!/usr/bin/env python
"""
responses_job_search.py — Alternative implementation using OpenAI Responses API

This demonstrates a simpler implementation using the stateful Responses API
as an alternative to the multi-agent approach.
"""

import os, re, json, time, argparse, logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Import our logging utilities if available
try:
    from logger_utils import setup_enhanced_logger, DepthContext
    from api_wrapper import initialize_api_wrapper, get_api_wrapper
    HAS_LOGGING_UTILS = True
except ImportError:
    HAS_LOGGING_UTILS = False

# Load environment variables
load_dotenv()

# Constants
RESULTS_DIR = Path("results")

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

# Setup logging
def setup_logger(level:str='INFO', file:str|None=None) -> logging.Logger:
    """Set up logger with improved format and API logging capabilities."""
    if HAS_LOGGING_UTILS:
        # Use enhanced logging if available
        api_log_file = None
        if file:
            api_log_file = Path(file).with_suffix('.api.log')

        # Create logs directory
        logs_dir = Path('logs')
        logs_dir.mkdir(exist_ok=True)

        # Set up the enhanced logger
        logger = setup_enhanced_logger(
            level=level,
            file=file,
            api_log_file=api_log_file
        )

        # Initialize the API wrapper
        initialize_api_wrapper(logger)

        return logger
    else:
        # Basic logging setup as fallback
        logger = logging.getLogger('responses_job_search')
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR
        }
        logger.setLevel(level_map.get(level, logging.INFO))

        # Console handler
        console = logging.StreamHandler()
        console.setLevel(level_map.get(level, logging.INFO))
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logger.addHandler(console)

        # File handler if specified
        if file:
            file_handler = logging.FileHandler(file)
            file_handler.setLevel(level_map.get(level, logging.INFO))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

# Initialize OpenAI client and logger
logger = setup_logger(level='INFO', file='logs/responses_job_search.log')
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class Timer:
    """Simple context manager for timing operations."""
    def __init__(self, label, logger):
        self.label = label
        self.logger = logger
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting {self.label}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.logger.info(f"{self.label} completed in {duration:.2f}s")
        if exc_type:
            self.logger.error(f"ERROR in {self.label}: {exc_val}")

def format_company_list(companies, limit=10):
    """Format a list of companies for the prompt"""
    if limit and len(companies) > limit:
        return ", ".join(companies[:limit]) + f" and {len(companies) - limit} more"
    return ", ".join(companies)

def search_jobs(query, model="gpt-4o", max_tokens=4000):
    """
    Search for jobs using the Responses API

    Args:
        query: Search query or instructions
        model: Model to use for the search
        max_tokens: Maximum tokens to generate

    Returns:
        List of job dictionaries
    """
    logger.info(f"Searching for jobs with query: {query[:100]}...")

    with Timer("Job search", logger):
        # Create a response with web search enabled
        response = client.responses.create(
            model=model,
            input=query,
            tools=[{"type": "web_search"}],
            max_tokens=max_tokens
        )

    # Extract the response text
    result_text = response.output[0].content[0].text
    logger.info(f"Search returned {len(result_text)} characters of results")

    # Try to extract structured job data
    try:
        with Timer("Parsing job data", logger):
            # Look for JSON in the response
            json_match = re.search(r'\[\s*\{.*\}\s*\]', result_text, re.DOTALL)
            if json_match:
                jobs_json = json_match.group(0)
                jobs = json.loads(jobs_json)
                logger.info(f"Successfully parsed JSON data: {len(jobs)} jobs found")
                return jobs

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
            title_pattern = r'([A-Za-z\s\-–&]+) at ([A-Za-z\s\-–&]+)'
            url_pattern = r'(https?://[^\s]+)'

            title_matches = re.findall(title_pattern, result_text)
            url_matches = re.findall(url_pattern, result_text)

            for i, (title, company) in enumerate(title_matches):
                if i < len(url_matches):
                    url = url_matches[i]
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
        # Return raw text for manual inspection
        return [{"title": "Parsing Error", "company": "Error", "url": "Error", "type": "Error", "raw_text": result_text}]

def save_results(jobs, output_dir=None):
    """Save job results to CSV and Markdown"""
    if output_dir is None:
        output_dir = RESULTS_DIR

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)

    # Add numbering to jobs
    for i, job in enumerate(jobs, 1):
        job['#'] = i

    # Create DataFrame
    df = pd.DataFrame(jobs)

    # Save to CSV
    csv_path = output_dir / 'responses_job_results.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to CSV: {csv_path}")

    # Save to Markdown
    md_path = output_dir / 'responses_job_results.md'
    try:
        # Try using pandas to_markdown if tabulate is available
        import tabulate
        with open(md_path, 'w') as f:
            f.write("# Job Search Results\n\n")
            f.write(df.to_markdown(index=False))
        logger.info(f"Results saved to Markdown: {md_path}")
    except ImportError:
        # Fallback to simple markdown
        with open(md_path, 'w') as f:
            f.write("# Job Search Results\n\n")
            f.write(f"Found {len(jobs)} jobs.\n\n")
            for job in jobs:
                f.write(f"## {job['#']}. {job.get('title', 'Unknown')} at {job.get('company', 'Unknown')}\n")
                f.write(f"- Type: {job.get('type', 'Unknown')}\n")
                f.write(f"- URL: {job.get('url', 'Unknown')}\n\n")
        logger.info(f"Results saved to Markdown: {md_path}")

def search_major_companies(count=5, model="gpt-4o", max_tokens=4000):
    """Search for jobs at major companies"""
    companies = format_company_list(MAJOR_COMPANIES)
    keywords = ", ".join(KEYWORDS[:5])

    logger.info(f"Searching for {count} jobs at major companies")

    query = f"""
    I need you to find {count} software engineering job listings at major companies in the video/streaming industry.

    Focus on these companies: {companies}
    Look for roles with keywords like: {keywords}

    For each job listing, I need:
    1. The exact job title
    2. The company name
    3. The direct URL to the job posting (not a careers page)

    Return the results as a structured JSON array with fields: title, company, url, type
    Set the "type" field to "Major" for all these companies.

    Only include real job postings with direct application links.
    """

    return search_jobs(query, model=model, max_tokens=max_tokens)

def search_startup_companies(count=5, model="gpt-4o", max_tokens=4000):
    """Search for jobs at startup companies"""
    companies = format_company_list(STARTUP_COMPANIES)
    keywords = ", ".join(KEYWORDS[:5])

    logger.info(f"Searching for {count} jobs at startup companies")

    query = f"""
    I need you to find {count} software engineering job listings at startups in the video/streaming industry.

    Focus on these startups: {companies}
    Look for roles with keywords like: {keywords}

    For each job listing, I need:
    1. The exact job title
    2. The company name
    3. The direct URL to the job posting (not a careers page)

    Return the results as a structured JSON array with fields: title, company, url, type
    Set the "type" field to "Startup" for all these companies.

    Only include real job postings with direct application links.
    """

    return search_jobs(query, model=model, max_tokens=max_tokens)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Job search using OpenAI Responses API")
    parser.add_argument("--majors", type=int, default=5, help="Number of major company jobs to find (default: 5)")
    parser.add_argument("--startups", type=int, default=5, help="Number of startup jobs to find (default: 5)")
    parser.add_argument("--model", default="gpt-4o", help="Model to use (default: gpt-4o)")
    parser.add_argument("--max-tokens", type=int, default=4000, help="Maximum tokens to generate (default: 4000)")
    parser.add_argument("--budget", type=float, help="Maximum cost in USD (estimates only, not enforced)")
    parser.add_argument("--sample", action="store_true", help="Run with minimal settings (2 major, 2 startup jobs)")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    parser.add_argument("--log-file", help="Log to this file in addition to console")
    args = parser.parse_args()

    # Update logger based on arguments
    global logger
    logger = setup_logger(level=args.log_level, file=args.log_file)

    # Configure results directory for Docker compatibility
    global RESULTS_DIR
    # Check if we're running in a Docker container
    if os.environ.get("RUNNING_IN_CONTAINER") == "1":
        # In Docker, results should go to /app/results which is mounted to the host
        RESULTS_DIR = Path("/app/results")
    else:
        # Otherwise use local results directory
        RESULTS_DIR = Path("results")

    # Handle sample mode
    if args.sample:
        args.majors = 2
        args.startups = 2

    # Print cost estimate
    tokens_estimate = (args.majors + args.startups) * 2000  # Rough estimate

    # Calculate cost based on model
    cost_per_1k = {
        "gpt-4": 0.01,
        "gpt-4o": 0.005,
        "gpt-4o-mini": 0.002,
        "gpt-3.5-turbo": 0.001
    }
    rate = cost_per_1k.get(args.model, 0.005)  # Default to gpt-4o rate
    cost_estimate = (tokens_estimate / 1000) * rate

    logger.info(f"Estimated resource usage:")
    logger.info(f"  - Tokens: ~{tokens_estimate:,} tokens")
    logger.info(f"  - Cost: ~${cost_estimate:.4f}")

    # Check budget constraint if specified
    if args.budget and cost_estimate > args.budget:
        logger.warning(f"⚠️ ERROR: Estimated cost ${cost_estimate:.4f} exceeds budget ${args.budget:.2f}")
        logger.warning("Exiting without running search")
        return 1

    logger.info(f"Searching for {args.majors} major company jobs and {args.startups} startup jobs using {args.model}")

    # Search for jobs
    major_jobs = []
    startup_jobs = []

    with Timer("Overall job search", logger):
        if args.majors > 0:
            major_jobs = search_major_companies(count=args.majors, model=args.model, max_tokens=args.max_tokens)
            logger.info(f"Found {len(major_jobs)} major company jobs")

        if args.startups > 0:
            startup_jobs = search_startup_companies(count=args.startups, model=args.model, max_tokens=args.max_tokens)
            logger.info(f"Found {len(startup_jobs)} startup jobs")

    # Combine results
    all_jobs = major_jobs + startup_jobs
    logger.info(f"Total jobs found: {len(all_jobs)}")

    # Save results
    save_results(all_jobs)
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
