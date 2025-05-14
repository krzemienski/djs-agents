#!/usr/bin/env python
"""
deep_job_search.py — Deep Job Search with Responses API implementation

This file uses the Responses API for a simple, efficient job search.
"""

import os
import re
import json
import time
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Union

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

# Import our utility modules
from logger_utils import setup_enhanced_logger, DepthContext
from api_wrapper import initialize_api_wrapper
from agent_visualizer import initialize_visualizer, get_visualizer

load_dotenv()

# Constants and configuration
OUTPUT_DIR = Path(__file__).with_suffix("")
RESULTS_DIR = Path("results")  # For Docker compatibility
DEBUG_DIR = Path("logs/debug")  # Directory for saving raw responses

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
class JobListing(BaseModel):
    """A job listing with relevant details"""

    title: str
    company: str
    type: str  # Major or Startup
    url: str
    has_apply: bool = False


# ------------- logging -------------
def save_raw_response(response: Any, response_type: str, logger: logging.Logger) -> str:
    """
    Save the raw response data to a file for debugging purposes.

    Args:
        response: The response object from the API
        response_type: Type of response (major/startup)
        logger: Logger instance

    Returns:
        Path to the saved file
    """
    timestamp = int(time.time())

    # Ensure debug directory exists
    DEBUG_DIR.mkdir(exist_ok=True, parents=True)

    # Create a unique filename
    filename = DEBUG_DIR / f"raw_response_{response_type}_{timestamp}.txt"

    try:
        # Try to get the content in various ways
        content = ""

        # Save full response attributes
        attrs = dir(response)
        content += f"Response attributes: {attrs}\n\n"

        # Try to get text attributes
        if hasattr(response, "text"):
            content += f"Response.text: {str(response.text)}\n\n"

        if hasattr(response, "output_text"):
            content += f"Response.output_text: {str(response.output_text)}\n\n"

        # Try to get output field
        if hasattr(response, "output") and response.output:
            content += f"Response.output type: {type(response.output)}\n"
            if isinstance(response.output, list):
                for i, item in enumerate(response.output):
                    content += f"Output[{i}] type: {type(item)}\n"
                    if hasattr(item, "content") and isinstance(item.content, list):
                        for j, content_item in enumerate(item.content):
                            content += f"Output[{i}].content[{j}] type: {type(content_item)}\n"
                            if hasattr(content_item, "text"):
                                content += f"Output[{i}].content[{j}].text: {content_item.text[:1000]}...\n"
                    elif hasattr(item, "content"):
                        content += f"Output[{i}].content: {str(item.content)[:1000]}...\n"

        # Write to file
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)

        logger.debug(f"Saved raw response to {filename}")
        return str(filename)
    except Exception as e:
        logger.error(f"Error saving raw response: {e}")
        return ""

def setup_logger(level: str = "INFO", file: str | None = None) -> logging.Logger:
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

    # Create debug directory for raw responses
    debug_dir = logs_dir / "debug"
    debug_dir.mkdir(exist_ok=True)

    # Initialize the visualizer
    initialize_visualizer(visuals_dir)

    # Set up the enhanced logger
    logger = setup_enhanced_logger(level=level, file=file, api_log_file=api_log_file)

    # Initialize the API wrapper with our logger
    initialize_api_wrapper(logger)

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


# ------------- Responses API implementation -------------
def format_company_list(companies, limit=10):
    """Format a list of companies for the prompt"""
    if limit and len(companies) > limit:
        return ", ".join(companies[:limit]) + f" and {len(companies) - limit} more"
    return ", ".join(companies)


def parse_responses_output(response, logger, response_type="unknown"):
    """
    Parse the output from the Responses API to extract job data

    Args:
        response: Response object from the OpenAI Responses API
        logger: Logger instance
        response_type: Type of response (major/startup)

    Returns:
        List of job dictionaries or empty list on error
    """
    try:
        with Timer("Parsing job data", logger):
            # Save raw response for debugging
            save_raw_response(response, response_type, logger)

            # Extract text from the response
            # Try different methods to get the result text based on what's available
            result_text = ""

            # Focus on the output field which seems to be most reliable
            if hasattr(response, "output") and response.output:
                # The output field typically contains the message from the assistant
                if isinstance(response.output, list):
                    # Add bounds check to avoid IndexError
                    output_idx = 1 if len(response.output) > 1 else 0
                    if output_idx < len(response.output):
                        output_message = response.output[output_idx]
                        if hasattr(output_message, "content") and isinstance(
                            output_message.content, list
                        ):
                            logger.debug("Using output_message.content")
                            for content_item in output_message.content:
                                if hasattr(content_item, "text"):
                                    logger.debug("Found text in content item")
                                    result_text = content_item.text
                                    break

                        # If we still don't have text, try other methods
                        if not result_text and hasattr(output_message, "text"):
                            logger.debug("Using output_message.text")
                            result_text = output_message.text

                        # If nothing worked, try direct content attribute
                        if not result_text and hasattr(output_message, "content"):
                            logger.debug("Using output_message.content directly")
                            if isinstance(output_message.content, str):
                                result_text = output_message.content
                            else:
                                result_text = str(output_message.content)

                        # Last resort for output_message
                        if not result_text:
                            logger.debug("Using str(output_message)")
                            result_text = str(output_message)

            # Still no result? Fall back to text attribute
            if not result_text and hasattr(response, "text") and response.text:
                logger.debug("Using response.text attribute")
                if hasattr(response.text, "value"):
                    result_text = response.text.value
                else:
                    result_text = str(response.text)

            # Last resort, try output_text
            if (
                not result_text
                and hasattr(response, "output_text")
                and response.output_text
            ):
                logger.debug("Using response.output_text")
                result_text = str(response.output_text)

            # If we still don't have text, use the entire response
            if not result_text:
                logger.debug("Using fallback approach with whole response")
                result_text = str(response)

            logger.info(f"Search returned {len(result_text)} characters of results")

            # Save extracted result text to file for debugging
            debug_file = DEBUG_DIR / f"extracted_text_{response_type}_{int(time.time())}.txt"
            try:
                DEBUG_DIR.mkdir(exist_ok=True, parents=True)
                with open(debug_file, "w", encoding="utf-8") as f:
                    f.write(result_text)
                logger.debug(f"Saved extracted text to {debug_file}")
            except Exception as e:
                logger.error(f"Error saving extracted text: {e}")

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
                        if (
                            "title" in result_text
                            and "company" in result_text
                            and "url" in result_text
                        ):
                            jobs = [result_text]

                    if jobs:
                        logger.info(
                            f"Used direct data structure: {len(jobs)} jobs found"
                        )
                        return jobs
                except Exception as e:
                    logger.debug(f"Error using direct structure: {e}")

            # Convert to string if not already
            if not isinstance(result_text, str):
                result_text = str(result_text)

            # Try all JSON parsing methods

            # Method 1: Look for complete JSON blocks using regex
            json_pattern = r'(\[\s*\{.*?\}\s*\])'
            json_matches = re.findall(json_pattern, result_text, re.DOTALL)

            for json_str in json_matches:
                try:
                    potential_jobs = json.loads(json_str)
                    if isinstance(potential_jobs, list) and len(potential_jobs) > 0:
                        # Verify it contains job data
                        if all(isinstance(job, dict) for job in potential_jobs):
                            jobs.extend(potential_jobs)
                            logger.info(f"Successfully parsed JSON array: {len(potential_jobs)} jobs found")
                except Exception as e:
                    logger.debug(f"Error parsing JSON match: {e}")

            if jobs:
                return jobs

            # Method 2: Look for a JSON array in the text using bracket matching
            try:
                start_idx = result_text.find("[")
                if start_idx != -1:
                    # Find the matching closing bracket
                    bracket_count = 0
                    for i in range(start_idx, len(result_text)):
                        if result_text[i] == "[":
                            bracket_count += 1
                        elif result_text[i] == "]":
                            bracket_count -= 1
                            if bracket_count == 0:
                                # We found the matching closing bracket
                                end_idx = i + 1
                                potential_json = result_text[start_idx:end_idx]
                                try:
                                    jobs = json.loads(potential_json)
                                    logger.info(
                                        f"Successfully parsed JSON data: {len(jobs)} jobs found"
                                    )
                                    return jobs
                                except json.JSONDecodeError as e:
                                    logger.debug(f"JSON decode error: {e}")
                                break
            except Exception as e:
                logger.debug(f"Error during JSON bracket extraction: {e}")

            # If no JSON found, try to parse markdown table
            table_match = re.search(
                r"\|\s*Title\s*\|\s*Company\s*\|\s*URL\s*\|", result_text, re.IGNORECASE
            )
            if table_match:
                # Parse markdown table
                jobs = []
                lines = result_text.split("\n")
                start_idx = None

                for i, line in enumerate(lines):
                    if (
                        "|" in line
                        and ("Title" in line or "title" in line)
                        and ("Company" in line or "company" in line)
                    ):
                        start_idx = i + 2  # Skip header and separator
                        break

                if start_idx:
                    for i in range(start_idx, len(lines)):
                        line = lines[i].strip()
                        if not line or "|" not in line:
                            continue

                        parts = [p.strip() for p in line.split("|")]
                        if len(parts) >= 4:
                            # Handle row with Title, Company, URL columns
                            title = parts[1]
                            company = parts[2]
                            url = parts[3]

                            if "http" in url:
                                job_type = response_type.capitalize()
                                jobs.append(
                                    {
                                        "title": title,
                                        "company": company,
                                        "url": url,
                                        "type": job_type,
                                    }
                                )
                logger.info(
                    f"Successfully parsed markdown table: {len(jobs)} jobs found"
                )
                if jobs:
                    return jobs

            # Try to find a list format with numbering
            numbered_jobs = re.findall(r'\d+\.\s+([^:]+):\s+([^(]+)\s*\(([^)]+)\)', result_text)
            if numbered_jobs:
                for title, company, url in numbered_jobs:
                    if 'http' in url:
                        job_type = response_type.capitalize()
                        jobs.append({
                            "title": title.strip(),
                            "company": company.strip(),
                            "url": url.strip(),
                            "type": job_type
                        })
                logger.info(f"Found {len(jobs)} jobs using numbered format parsing")
                if jobs:
                    return jobs

            # If parsing table fails, use regex to find potential job listings
            logger.info("Falling back to regex pattern matching for job data")

            # Enhanced regex patterns
            job_patterns = [
                # Pattern 1: <title> at <company> with nearby URL
                (r'(?P<title>[A-Za-z0-9\s\-–&,\.]+)\s+at\s+(?P<company>[A-Za-z0-9\s\-–&,\.]+)', 50),

                # Pattern 2: <company> is hiring <title>
                (r'(?P<company>[A-Za-z0-9\s\-–&,\.]+)\s+is\s+hiring:?\s*(?P<title>[A-Za-z0-9\s\-–&,\.]+)', 50),

                # Pattern 3: Job: <title>, Company: <company>
                (r'(?:Job|Title|Position):\s*(?P<title>[^,\n]+)(?:[,\n]|\s+and|\s+at)\s*(?:Company|Employer):\s*(?P<company>[^,\n]+)', 50),

                # Pattern 4: <title> - <company>
                (r'(?P<title>[A-Za-z0-9\s\-–&,\.]+)\s+-\s+(?P<company>[A-Za-z0-9\s\-–&,\.]+)', 30),

                # Pattern 5: <company> - <title>
                (r'(?P<company>[A-Za-z0-9\s\-–&,\.]+)\s+-\s+(?P<title>[A-Za-z0-9\s\-–&,\.]+)', 30)
            ]

            url_pattern = r'(https?://[^\s"\',]+)'
            url_matches = re.findall(url_pattern, result_text)

            # Words to ignore as titles (false positives like action verbs)
            ignore_words = [
                "apply", "learn", "more", "view", "click", "check", "see", "visit", "join",
                "next", "back", "skip", "search", "find", "contact", "home", "about", "jobs",
                "careers", "terms", "privacy", "help", "support", "login", "sign"
            ]

            jobs = []

            # Process each pattern
            for pattern_tuple, proximity in job_patterns:
                matches = re.finditer(pattern_tuple, result_text)

                for match in matches:
                    match_dict = match.groupdict()
                    title = match_dict.get('title', '').strip()
                    company = match_dict.get('company', '').strip()

                    # Skip if the title is a common action verb (false positive)
                    if title.lower() in ignore_words:
                        continue

                    # Find URL near this match
                    match_pos = match.start()

                    # Try to find URL within proximity characters
                    nearby_text = result_text[max(0, match_pos - proximity):min(len(result_text), match_pos + len(match.group(0)) + proximity)]
                    nearby_urls = re.findall(url_pattern, nearby_text)

                    if nearby_urls:
                        url = nearby_urls[0]  # Take the first URL near this job listing
                        job_type = response_type.capitalize()

                        # Check if this job is already in our list (avoid duplicates)
                        is_duplicate = False
                        for existing_job in jobs:
                            if (existing_job['title'].lower() == title.lower() and
                                existing_job['company'].lower() == company.lower()):
                                is_duplicate = True
                                break

                        if not is_duplicate:
                            jobs.append({
                                "title": title,
                                "company": company,
                                "url": url,
                                "type": job_type,
                                "has_apply": True
                            })

            # Last resort: Look for any URL with "job" in it
            if not jobs:
                job_urls = [url for url in url_matches if 'job' in url.lower() or 'career' in url.lower()]

                for i, url in enumerate(job_urls):
                    job_type = response_type.capitalize()
                    jobs.append({
                        "title": f"software engineering job listings",
                        "company": f"{response_type} companies in the video",
                        "url": url,
                        "type": job_type,
                        "has_apply": True
                    })

            logger.info(f"Found {len(jobs)} jobs using regex pattern matching")
            return jobs

    except Exception as e:
        logger.error(f"Error parsing job data: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Return empty list on error
        return []


def search_jobs_with_responses(query, model, logger, timeout=120):
    """
    Search for jobs using the Responses API

    Args:
        query: Search query or instructions
        model: Model to use for the search
        logger: Logger instance
        timeout: Maximum time in seconds to wait for response (default: 120)

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
                tools=[{"type": "web_search"}],
                max_output_tokens=4000,  # Ensure we get a complete response
                timeout=timeout
            )

            # Record API call in visualizer
            duration = time.time() - start_time
            visualizer.track_api_call(
                function_name="responses.create",
                api_type="responses",
                success=True,
                duration=duration,
            )

            # Log response structure for debugging
            logger.debug(f"Response type: {type(response)}")
            logger.debug(f"Response attributes: {dir(response)}")

            # Debug the output field specifically
            if hasattr(response, "output"):
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
                duration=duration,
            )
            logger.error(f"Error creating Responses API request: {e}")
            return []

        # Access response properties for debugging
    logger.debug(f"Response has text attribute: {hasattr(response, 'text')}")
    logger.debug(
        f"Response has output_text attribute: {hasattr(response, 'output_text')}"
    )

    # Parse the response
    # Determine response type from query
    response_type = "major" if "major companies" in query.lower() else "startup"
    return parse_responses_output(response, logger, response_type)


def search_companies_with_responses(
    company_type, companies, count, model, logger, limit=None, timeout=120
):
    """
    Unified search function for both major and startup companies

    Args:
        company_type: "Major" or "Startup"
        companies: List of companies to search
        count: Number of jobs to search for
        model: Model to use for search
        logger: Logger instance
        limit: Limit for company list formatting (optional)
        timeout: Maximum time in seconds to wait for response (default: 120)

    Returns:
        List of job dictionaries
    """
    companies_text = format_company_list(companies, limit)
    keywords = ", ".join(KEYWORDS[:5])

    logger.info(
        f"Searching for {count} jobs at {company_type.lower()} companies using Responses API"
    )

    query = f"""
    I need you to find {count} software engineering job listings at {company_type.lower()} companies in the video/streaming industry.

    Focus on these companies: {companies_text}
    Look for roles with keywords like: {keywords}

    For each job listing, I need:
    1. The exact job title
    2. The company name
    3. The direct URL to the job posting (not a careers page)

    IMPORTANT: Return the results in the following JSON format:
    ```json
    [
      {{
        "title": "Software Engineer",
        "company": "CompanyName",
        "url": "https://example.com/job",
        "type": "{company_type}"
      }},
      ...more jobs...
    ]
    ```

    If you cannot find enough jobs in JSON format, provide what you can find in a markdown table with columns for Title, Company, and URL.

    Only include real job postings with direct application links. Use "type": "{company_type}" for all these companies.
    """

    return search_jobs_with_responses(query, model, logger, timeout=timeout)


def search_major_companies_with_responses(count, model, logger, limit=None, timeout=120):
    """Search for jobs at major companies using Responses API"""
    return search_companies_with_responses(
        "Major", MAJOR_COMPANIES, count, model, logger, limit, timeout
    )


def search_startup_companies_with_responses(count, model, logger, limit=None, timeout=120):
    """Search for jobs at startup companies using Responses API"""
    return search_companies_with_responses(
        "Startup", STARTUP_COMPANIES, count, model, logger, limit, timeout
    )


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
    logger.info(
        f"Job search started with Responses API using configuration: {json.dumps(cfg, indent=2)}"
    )

    # Estimate token usage and cost
    tokens_estimate = (cfg["majors"] + cfg["startups"]) * 2000  # Rough estimate

    # Calculate cost based on model
    model = cfg.get("model", "gpt-4o")
    rate = TokenMonitor.COST_PER_1K.get(model, 0.005)  # Default to gpt-4o rate
    cost_estimate = (tokens_estimate / 1000) * rate

    logger.info("Estimated resource usage:")
    logger.info(f"  - Tokens: ~{tokens_estimate:,} tokens")
    logger.info(f"  - Cost: ~${cost_estimate:.4f}")

    if cost_estimate > 0.50:  # Arbitrary threshold for a "high" cost
        logger.warning(
            f"⚠️ COST WARNING: Estimated cost (${cost_estimate:.4f}) exceeds $0.50"
        )
        logger.warning("Consider reducing job counts or using less expensive models")

        # Give user a chance to abort if cost is high
        if not cfg.get("force", False) and not os.environ.get(
            "JOBBOT_SKIP_CONFIRM", ""
        ):
            logger.info(
                "Continue? (y/n, or set JOBBOT_SKIP_CONFIRM=1 to skip this prompt)"
            )
            try:
                response = input().strip().lower()
                if response != "y":
                    logger.info("Aborting job search")
                    return []
            except (KeyboardInterrupt, EOFError):
                logger.info("\nAborting job search")
                return []

    # Check for estimate-only mode
    if os.environ.get("JOBBOT_ESTIMATE_ONLY", ""):
        logger.info("Estimate-only mode enabled. Exiting now.")
        return []

    # Search for jobs
    major_jobs = []
    startup_jobs = []

    # Get company list limit
    company_list_limit = cfg.get("company_list_limit", 10)

    with Timer("Overall job search", logger):
        if cfg["majors"] > 0:
            major_jobs = search_major_companies_with_responses(
                count=cfg["majors"],
                model=model,
                logger=logger,
                limit=company_list_limit,
                timeout=cfg.get("timeout", 120),
            )
            logger.info(f"Found {len(major_jobs)} major company jobs")

            # Ensure has_apply field exists for compatibility
            for job in major_jobs:
                job["has_apply"] = True

        if cfg["startups"] > 0:
            startup_jobs = search_startup_companies_with_responses(
                count=cfg["startups"],
                model=model,
                logger=logger,
                limit=company_list_limit,
                timeout=cfg.get("timeout", 120),
            )
            logger.info(f"Found {len(startup_jobs)} startup jobs")

            # Ensure has_apply field exists for compatibility
            for job in startup_jobs:
                job["has_apply"] = True

    # Combine results
    all_jobs = major_jobs[: cfg["majors"]] + startup_jobs[: cfg["startups"]]
    logger.info(f"Total jobs found: {len(all_jobs)}")

    # Generate visualizations
    visualizer = get_visualizer()
    try:
        # Track overall API usage for visualization
        api_duration = time.time() - api_start_time
        token_usage = {cfg.get("model", "gpt-4o"): tokens_estimate}

        # Track search as a component step
        visualizer.track_agent_call(
            agent_name="JobSearch",
            input_text=f"Search for {cfg['majors']} major and {cfg['startups']} startup jobs",
            output_text=f"Found {len(all_jobs)} jobs",
            duration=api_duration,
            tokens_used=token_usage,
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


def parse():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Deep Job Search: OpenAI powered job search tool for finding software engineering jobs in video/streaming companies"
    )
    parser.add_argument(
        "--majors", type=int, default=10, help="Number of major company jobs to find"
    )
    parser.add_argument(
        "--startups",
        type=int,
        default=10,
        help="Number of startup company jobs to find",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run with minimal settings (2 major, 2 startup jobs)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=100000, help="Max tokens to use"
    )
    parser.add_argument(
        "--budget", type=float, help="Maximum cost in USD (exit if exceeded)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Skip cost confirmation prompt"
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (default: INFO)"
    )
    parser.add_argument("--log-file", help="Log to this file in addition to console")
    parser.add_argument("--trace", action="store_true", help="Enable trace output")
    parser.add_argument(
        "--use-web-verify",
        action="store_true",
        help="Use web search for URL verification",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model to use for Responses API implementation (default: gpt-4o)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="Generate visualization diagrams (default: True)",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_false",
        dest="visualize",
        help="Disable visualization generation",
    )
    parser.add_argument(
        "--company-list-limit",
        type=int,
        default=10,
        help="Maximum number of companies to list in prompts (default: 10)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="API call timeout in seconds (default: 120)",
    )

    # No legacy arguments needed

    args = parser.parse_args()

    # Handle sample mode
    if args.sample:
        args.majors = 2
        args.startups = 2
        args.log_level = "DEBUG"

    # Convert to config dict
    cfg = {
        "majors": args.majors,
        "startups": args.startups,
        "max_tokens": args.max_tokens,
        "budget": args.budget,
        "force": args.force,
        "use_web_verify": args.use_web_verify,
        "log_level": args.log_level,
        "log_file": args.log_file,
        "trace": args.trace,
        "model": args.model,
        "company_list_limit": args.company_list_limit,
        "timeout": args.timeout,
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
    logger = setup_logger(level=cfg["log_level"], file=cfg["log_file"])

    # Check if visualization is enabled
    if cfg.get("visualize", True):
        # Initialize visualizer if not already initialized
        visuals_dir = Path("logs/visuals")
        initialize_visualizer(visuals_dir)

    # Log startup info
    logger.info(f"deep_job_search starting with Python {sys.version}")

    # Check for estimate-only mode
    if os.environ.get("JOBBOT_ESTIMATE_ONLY", ""):
        logger.info("Estimate-only mode enabled. Exiting now.")
        return 0

    # Log execution start
    logger.info("Job search execution started")

    # Run the job search
    if rows := search_with_responses(cfg, logger):
        save(rows, logger)

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
