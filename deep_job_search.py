#!/usr/bin/env python
"""
deep_job_search.py — GPT‑4.1 text version

Default agent models:
  Planner   -> gpt-4.1
  Searcher  -> gpt-4o-mini
  Verifier  -> o3
"""

import os, re, json, time, argparse, asyncio, logging, sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, ModelSettings, WebSearchTool, usage as agent_usage
# Import tracing only if available
try:
    from agents.tracing.tracer import tracer
    # Check if ConsoleSpanProcessor exists
    try:
        from agents.tracing import ConsoleSpanProcessor
        HAS_CONSOLE_PROCESSOR = True
    except ImportError:
        HAS_CONSOLE_PROCESSOR = False
    HAS_TRACING = True
except ImportError:
    HAS_TRACING = False
from pydantic import BaseModel

# Import our new modules
from logger_utils import setup_enhanced_logger, DepthContext
from api_wrapper import initialize_api_wrapper, get_api_wrapper
from agent_visualizer import initialize_visualizer, get_visualizer

load_dotenv()

OUTPUT_DIR = Path(__file__).with_suffix("")

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

# ------------- Custom function tools -------------
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

# ------------- prompts -------------
def planner_prompt()->str:
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
- Limit to 120 pairs total
- Include both major companies and startups
- Keep the JSON structure simple without extra properties

## Output format
JSON array only, no preamble or explanation.
"""

def searcher_prompt()->str:
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

def verifier_prompt()->str:
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

def processor_prompt()->str:
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

    # Estimated tokens per operation
    ESTIMATES = {
        'plan': {
            'gpt-4.1': 5000,
            'gpt-4o': 5000,
            'gpt-4o-mini': 6000,
            'gpt-3.5-turbo': 6000,
        },
        'search': {
            'gpt-4.1': 250,
            'gpt-4o': 250,
            'gpt-4o-mini': 300,
            'gpt-3.5-turbo': 350,
        },
        'process': {
            'gpt-4.1': 1000,
            'gpt-4o': 1000,
            'gpt-4o-mini': 1200,
            'gpt-3.5-turbo': 1500,
        },
        'verify': {
            'gpt-4.1': 350,
            'gpt-4o': 350,
            'gpt-4o-mini': 400,
            'gpt-3.5-turbo': 500,
            'o3': 300,
        }
    }

    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.max_tokens = cfg['max_tokens']
        self.phase_budgets = {
            'plan': int(self.max_tokens * 0.05),       # 5% for planning
            'search': int(self.max_tokens * 0.40),     # 40% for searching
            'process': int(self.max_tokens * 0.20),    # 20% for processing
            'verify': int(self.max_tokens * 0.35),     # 35% for verification
        }
        self.phase_usage = {
            'plan': 0,
            'search': 0,
            'process': 0,
            'verify': 0
        }
        self.last_total = 0
        self.warnings_shown = set()
        self.total_tokens_used = 0

        # Track API calls by phase
        self.api_calls_by_phase = {
            'plan': 0,
            'search': 0,
            'process': 0,
            'verify': 0
        }

    def get_model_rate(self, model: str) -> float:
        """Get the cost rate for a model, with fallback for unknown models"""
        base_model = model.split('-preview')[0].split('-vision')[0]
        return self.COST_PER_1K.get(base_model, 0.005)  # Default to mid-tier pricing if unknown

    def estimate_cost(self):
        """Estimate total tokens and cost based on configuration"""
        # Get models from config
        planner_model = self.cfg['planner_model']
        search_model = self.cfg['search_model']
        verifier_model = self.cfg['verifier_model']

        # Get token estimates for each phase
        plan_tokens = self.ESTIMATES['plan'].get(planner_model, 5000)

        # Calculate search phase tokens (search + process)
        searches = min(40, max(self.cfg['majors'], self.cfg['startups']) * 3)  # Approximate search count
        search_tokens_per_query = self.ESTIMATES['search'].get(search_model, 300)
        process_tokens_per_query = self.ESTIMATES['process'].get(search_model, 1200)
        search_total = searches * (search_tokens_per_query + process_tokens_per_query)

        # Calculate verification phase tokens
        verifications = min(self.cfg['majors'] + self.cfg['startups'], 200)
        verify_tokens_per_url = self.ESTIMATES['verify'].get(verifier_model, 300)
        verify_total = verifications * verify_tokens_per_url

        # Sum up all tokens
        total_tokens = plan_tokens + search_total + verify_total

        # Cap at max tokens
        total_tokens = min(total_tokens, self.max_tokens)

        # Calculate cost
        cost = (
            (plan_tokens / 1000) * self.get_model_rate(planner_model) +
            (search_total / 1000) * self.get_model_rate(search_model) +
            (verify_total / 1000) * self.get_model_rate(verifier_model)
        )

        # Log detailed estimates
        self.logger.debug(f"Token estimate breakdown:")
        self.logger.debug(f"  - Planning ({planner_model}): {plan_tokens:,} tokens (${(plan_tokens/1000)*self.get_model_rate(planner_model):.4f})")
        self.logger.debug(f"  - Search/Process ({search_model}): {search_total:,} tokens (${(search_total/1000)*self.get_model_rate(search_model):.4f})")
        self.logger.debug(f"  - Verification ({verifier_model}): {verify_total:,} tokens (${(verify_total/1000)*self.get_model_rate(verifier_model):.4f})")

        return total_tokens, cost

    def track_api_call(self, phase: str):
        """Track an API call for a specific phase"""
        if phase in self.api_calls_by_phase:
            self.api_calls_by_phase[phase] += 1

        # Also track in the visualizer
        visualizer = get_visualizer()
        if hasattr(visualizer, 'track_api_call'):
            # This is a placeholder - in a real implementation, we'd pass actual API call details
            visualizer.track_api_call(f"{phase}_call", "completion", True, 0.5)

    def check_budget(self, phase: str) -> bool:
        """Check if we're within budget for a phase"""
        # Get current token usage by summing tokens_per_model dictionary
        try:
            current_total = sum(agent_usage.tokens_per_model.values())
        except (AttributeError, TypeError):
            # If tokens_per_model doesn't exist or isn't usable, use our own tracking
            current_total = self.total_tokens_used

        phase_delta = current_total - self.last_total
        self.last_total = current_total
        self.total_tokens_used = current_total

        # Update phase usage
        self.phase_usage[phase] += phase_delta

        # Log phase usage update
        self.logger.debug(f"Phase '{phase}' token usage updated: +{phase_delta:,} tokens, now {self.phase_usage[phase]:,} tokens")

        # Check if we're over budget for this phase
        if self.phase_usage[phase] > self.phase_budgets[phase]:
            if phase not in self.warnings_shown:
                self.logger.warning(f"TOKEN LIMIT: Phase '{phase}' exceeded its budget " +
                              f"({self.phase_usage[phase]:,}/{self.phase_budgets[phase]:,} tokens)")
                self.warnings_shown.add(phase)
            return False

        # Check if we're approaching the budget (80%)
        if self.phase_usage[phase] > self.phase_budgets[phase] * 0.8 and phase not in self.warnings_shown:
            self.logger.warning(f"TOKEN WARNING: Phase '{phase}' approaching its budget " +
                          f"({self.phase_usage[phase]:,}/{self.phase_budgets[phase]:,} tokens)")
            self.warnings_shown.add(phase)

        return True

    def check_total(self) -> bool:
        """Check if we're within the overall token budget"""
        try:
            total_tokens = sum(agent_usage.tokens_per_model.values())
        except (AttributeError, TypeError):
            total_tokens = self.total_tokens_used

        if total_tokens > self.max_tokens * 0.8 and 'total' not in self.warnings_shown:
            self.logger.warning(f"TOKEN WARNING: Approaching overall token limit " +
                          f"({total_tokens:,}/{self.max_tokens:,} tokens)")
            self.warnings_shown.add('total')

        if total_tokens > self.max_tokens:
            if 'total_exceeded' not in self.warnings_shown:
                self.logger.error(f"TOKEN LIMIT EXCEEDED: {total_tokens:,}/{self.max_tokens:,} tokens")
                self.warnings_shown.add('total_exceeded')
            return False

        return True

    def get_current_cost(self) -> float:
        """Calculate the current cost based on token usage"""
        total_cost = 0.0

        try:
            for model, tokens in agent_usage.tokens_per_model.items():
                rate = self.get_model_rate(model)
                cost = (tokens / 1000) * rate
                total_cost += cost
        except (AttributeError, TypeError):
            # If we can't get the tokens per model, estimate based on total tokens
            # Use a mid-range model rate as an approximation
            total_cost = (self.total_tokens_used / 1000) * 0.005

        return total_cost

    def log_usage(self, phase=None) -> None:
        """Log the current token usage with enhanced detail"""
        if phase:
            self.logger.info(f"Phase '{phase}' token usage: {self.phase_usage[phase]:,} tokens")
            if phase in self.api_calls_by_phase and self.api_calls_by_phase[phase] > 0:
                self.logger.info(f"Phase '{phase}' API calls: {self.api_calls_by_phase[phase]}")

        try:
            self.logger.info(f"Current token usage by model:")
            token_usage_by_model = []
            total_tokens = 0
            total_cost = 0.0

            for model, tokens in agent_usage.tokens_per_model.items():
                rate = self.get_model_rate(model)
                cost = (tokens / 1000) * rate
                total_tokens += tokens
                total_cost += cost
                token_usage_by_model.append({
                    'model': model,
                    'tokens': tokens,
                    'cost': cost
                })
                self.logger.info(f"  - {model}: {tokens:,} tokens (${cost:.4f})")

            # Update the visualizer with token usage
            visualizer = get_visualizer()
            for entry in token_usage_by_model:
                if hasattr(visualizer, 'track_agent_call') and phase:
                    # This tracks token usage in the visualizer
                    visualizer.track_agent_call(
                        agent_name=phase,
                        input_text="",
                        output_text="",
                        duration=0.1,
                        tokens_used={entry['model']: entry['tokens']}
                    )

        except (AttributeError, TypeError):
            self.logger.info(f"Token usage details not available from API")
            total_tokens = self.total_tokens_used
            total_cost = self.get_current_cost()

        # Calculate usage percentages
        percent_used = (total_tokens / self.max_tokens) * 100 if self.max_tokens > 0 else 0
        self.logger.info(f"Total usage: {total_tokens:,}/{self.max_tokens:,} tokens " +
                         f"({percent_used:.1f}%, ${total_cost:.4f})")

        # Log API calls
        total_api_calls = sum(self.api_calls_by_phase.values())
        if total_api_calls > 0:
            self.logger.info(f"Total API calls: {total_api_calls}")
            for phase, count in self.api_calls_by_phase.items():
                if count > 0:
                    self.logger.debug(f"  - {phase}: {count} calls")

# ------------- gather --------------
async def gather(cfg, logger)->List[Dict[str,str]]:
    # Log function start with configuration
    logger.info(f"Job search started with configuration: {json.dumps(cfg, indent=2)}")

    # Initialize token monitor
    token_monitor = TokenMonitor(cfg, logger)
    estimated_tokens, estimated_cost = token_monitor.estimate_cost()

    logger.info(f"Estimated resource usage:")
    logger.info(f"  - Tokens: ~{estimated_tokens:,} tokens")
    logger.info(f"  - Cost: ~${estimated_cost:.4f}")

    if estimated_cost > 0.50:  # Arbitrary threshold for a "high" cost
        logger.warning(f"⚠️ COST WARNING: Estimated cost (${estimated_cost:.4f}) exceeds $0.50")
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

    # Get API wrapper
    api = get_api_wrapper()

    # Get visualizer
    visualizer = get_visualizer()

    # Initialize agents with detailed logging
    with DepthContext(logger, "Agent Initialization"):
        logger.info(f"Initializing Planner agent with model: {cfg['planner_model']}")
        planner = await api.create_agent(
            name="planner",
            instructions=planner_prompt(),
            model=cfg['planner_model'],
            model_settings=ModelSettings(temperature=0)
        )

        logger.info(f"Initializing Processor agent with model: {cfg['search_model']}")
        processor = await api.create_agent(
            name="processor",
            instructions=processor_prompt(),
            model=cfg['search_model'],
            model_settings=ModelSettings(temperature=0)
        )

        logger.info(f"Initializing Verifier agent with model: {cfg['verifier_model']}")
        # Configure verifier tools based on web search option
        verifier_tools = []
        if cfg.get('use_web_verify', False):
            logger.info("Web search verification mode enabled")
            verifier_tools.append(WebSearchTool())

        verifier = await api.create_agent(
            name="verifier",
            instructions=verifier_prompt(),
            tools=verifier_tools,
            model=cfg['verifier_model'],
            model_settings=ModelSettings(temperature=0)
        )

        # Set up the processor to hand off to verifier
        processor.handoffs = [verifier]

        # Track the handoff in the visualizer
        visualizer.track_handoff("processor", "verifier", "Job verification handoff")

        logger.info(f"Initializing Searcher agent with model: {cfg['search_model']}")
        searcher = await api.create_agent(
            name="searcher",
            instructions=searcher_prompt(),
            tools=[WebSearchTool(), extract_job_listings],
            model=cfg['search_model'],
            model_settings=ModelSettings(temperature=0),
            # Set up searcher to hand off to processor
            handoffs=[processor]
        )

        # Track the handoff in the visualizer
        visualizer.track_handoff("searcher", "processor", "Job processing handoff")

    search_plan = []
    with Timer('Planning Phase', logger):
        # Generate search plan
        logger.info("Generating job search plan...")

        # Track API call
        token_monitor.track_api_call('plan')

        # Run the planner agent
        plan_result = await api.run_agent(planner, "Generate a comprehensive job search plan covering both major companies and startups")
        plan_json = plan_result.final_output

        # Check token budget after planning
        token_monitor.check_budget('plan')
        token_monitor.log_usage('plan')

        # Log entire plan for debugging
        logger.debug(f"Raw plan output: {plan_json}")

        # Try to pretty-print JSON if possible
        try:
            # Strip markdown code blocks if present
            plan_json = re.sub(r'```(?:json)?\s*([\s\S]*?)\s*```', r'\1', plan_json)
            plan_json = plan_json.strip()

            # Try to parse the JSON
            try:
                # Direct parsing attempt
                plan_data = json.loads(plan_json)
                formatted_plan = json.dumps(plan_data, indent=2)
            except json.JSONDecodeError:
                # Try to extract JSON array if embedded in text
                match = re.search(r'\[\s*{.*}\s*\]', plan_json, re.DOTALL)
                if match:
                    extracted_json = match.group(0)
                    plan_data = json.loads(extracted_json)
                    formatted_plan = json.dumps(plan_data, indent=2)
                else:
                    # Replace single quotes with double quotes
                    sanitized_json = re.sub(r"'([^']*)'", r'"\1"', plan_json)
                    # Quote unquoted keys
                    sanitized_json = re.sub(r"(\w+):", r'"\1":', sanitized_json)
                    plan_data = json.loads(sanitized_json)
                    formatted_plan = json.dumps(plan_data, indent=2)

            logger.info(f"Plan generated:\n{formatted_plan}")
        except Exception as e:
            logger.warning(f"Could not format plan as JSON: {e}")
            logger.info(f"Plan generated (raw):\n{plan_json}")

        try:
            plan_data = json.loads(plan_json)
            # Log each search pair by company type to validate distribution
            major_pairs = [p for p in plan_data if p['company'] in MAJOR_COMPANIES]
            startup_pairs = [p for p in plan_data if p['company'] in STARTUP_COMPANIES]

            logger.info(f"Plan breakdown: {len(plan_data)} total pairs")
            logger.info(f"  - Major companies: {len(major_pairs)} pairs")
            logger.info(f"  - Startup companies: {len(startup_pairs)} pairs")

            # Convert to model objects for validation
            search_plan = [JobSearchPair(company=p['company'], keyword=p['keyword']) for p in plan_data]
            logger.info(f"Valid search pairs: {len(search_plan)}")

            # Log some examples from the plan
            if search_plan:
                sample_size = min(5, len(search_plan))
                logger.info(f"Sample search pairs: {search_plan[:sample_size]}")
        except Exception as e:
            logger.error(f"Error parsing plan: {e}")
            return []

    all_jobs = []
    majors_quota = cfg['majors']
    startups_quota = cfg['startups']
    major_jobs = []
    startup_jobs = []

    search_successes = 0
    search_failures = 0

    with Timer('Search Phase', logger):
        # Log search quotas
        logger.info(f"Search quotas - Major: {majors_quota}, Startup: {startups_quota}")

        # Limit number of searches to avoid excessive API calls
        max_searches = min(len(search_plan), 40)
        logger.info(f"Executing up to {max_searches} searches from plan with {len(search_plan)} pairs")

        for i, search_pair in enumerate(search_plan[:max_searches], 1):
            # Check if we've exceeded token budget
            if not token_monitor.check_budget('search') or not token_monitor.check_total():
                logger.warning(f"Stopping search phase early due to token budget constraints")
                break

            company = search_pair.company
            keyword = search_pair.keyword
            company_type = "Major" if company in MAJOR_COMPANIES else "Startup"

            # Skip if we already have enough jobs of this type
            if company_type == "Major" and len(major_jobs) >= majors_quota:
                logger.debug(f"Skipping {company} ({company_type}) - major quota reached")
                continue
            if company_type == "Startup" and len(startup_jobs) >= startups_quota:
                logger.debug(f"Skipping {company} ({company_type}) - startup quota reached")
                continue

            logger.info(f"Search {i}/{max_searches}: {keyword} jobs at {company} ({company_type})")

            search_query = f"{company} {keyword} jobs careers software engineering apply"
            logger.debug(f"Search query: {search_query}")

            try:
                with DepthContext(logger, f"Search: {company}-{keyword}", log_level='DEBUG'):
                    # Track API call
                    token_monitor.track_api_call('search')

                    # Perform search
                    logger.debug(f"Executing web search via {cfg['search_model']} model")
                    search_result = await api.run_agent(searcher,
                                                       f"Find {keyword} jobs at {company} ({company_type}) with web search. Search query: {search_query}")
                    search_output = search_result.final_output

                    # Log search result truncated for readability
                    truncated_output = search_output[:500] + "..." if len(search_output) > 500 else search_output
                    logger.debug(f"Search result: {truncated_output}")

                    # Track this in the visualizer
                    visualizer.track_agent_call(
                        agent_name="searcher",
                        input_text=search_query,
                        output_text=search_output,
                        duration=1.0,  # Placeholder duration
                        tokens_used=None  # We'll update this later
                    )

                    # Process results
                    token_monitor.track_api_call('process')
                    logger.debug(f"Processing search results via {cfg['search_model']} model")
                    process_result = await api.run_agent(processor,
                                                        f"Process these job search results: {search_output}")

                    processor_output = process_result.final_output
                    logger.debug(f"Processor output: {processor_output}")

                    # Track this in the visualizer
                    visualizer.track_agent_call(
                        agent_name="processor",
                        input_text=search_output[:100] + "...", # Truncated for visualizer
                        output_text=processor_output,
                        duration=0.5,  # Placeholder duration
                        tokens_used=None
                    )

                    # Try to parse the output as JSON
                    try:
                        # Check if output is empty or not valid JSON
                        if not processor_output or processor_output.strip() == '':
                            logger.warning(f"Empty processor output for {company} {keyword} search")
                            continue

                        # Strip markdown code blocks if present
                        processor_output = re.sub(r'```(?:json)?\s*([\s\S]*?)\s*```', r'\1', processor_output)
                        processor_output = processor_output.strip()

                        # Try to normalize the JSON if it's not properly formatted
                        if not processor_output.strip().startswith('['):
                            # If output doesn't start with [, try to extract JSON array
                            match = re.search(r'\[\s*{.*}\s*\]', processor_output, re.DOTALL)
                            if match:
                                processor_output = match.group(0)
                                logger.debug(f"Extracted JSON array: {processor_output[:100]}...")
                            else:
                                # Create a single-item array if output looks like a single object
                                if processor_output.strip().startswith('{'):
                                    processor_output = f"[{processor_output}]"
                                    logger.debug("Wrapped single object in array")
                                else:
                                    logger.warning(f"Could not normalize JSON output: {processor_output[:100]}...")
                                    # Try to wrap in array brackets as last resort
                                    processor_output = f"[{processor_output}]"

                        # Parse the JSON output
                        try:
                            jobs_data = json.loads(processor_output)
                        except json.JSONDecodeError as e:
                            logger.warning(f"JSON decode error: {e}")

                            # As a fallback, try to sanitize the output
                            sanitized_output = re.sub(r"'([^']*)'", r'"\1"', processor_output)  # Replace single quotes with double quotes
                            sanitized_output = re.sub(r"(\w+):", r'"\1":', sanitized_output)  # Quote unquoted keys
                            logger.debug(f"Sanitized JSON: {sanitized_output[:100]}...")

                            # Try to force it into valid JSON format by removing additional text before/after the array
                            try:
                                match = re.search(r'\[\s*{.*}\s*\]', sanitized_output, re.DOTALL)
                                if match:
                                    jobs_data = json.loads(match.group(0))
                                else:
                                    # Last resort attempt - extract array from a line-by-line basis
                                    extraction_attempt = "["
                                    in_json = False

                                    for line in sanitized_output.split("\n"):
                                        line = line.strip()
                                        if line.startswith("["):
                                            in_json = True
                                        if in_json:
                                            extraction_attempt += line
                                        if line.endswith("]"):
                                            in_json = False

                                    logger.debug(f"Final extraction attempt: {extraction_attempt[:100]}...")
                                    jobs_data = json.loads(extraction_attempt)
                            except Exception as extraction_error:
                                logger.error(f"Final JSON extraction failed: {extraction_error}")
                                # Create an empty array if all parsing attempts fail
                                jobs_data = []

                        # Ensure we have a list
                        if not isinstance(jobs_data, list):
                            jobs_data = [jobs_data]

                        logger.info(f"Found {len(jobs_data)} potential job listings for {company}")

                        for job_idx, job in enumerate(jobs_data, 1):
                            # Skip if job is not a dictionary
                            if not isinstance(job, dict):
                                logger.warning(f"Skipping non-dictionary job data: {job}")
                                continue

                            # Ensure each job has the required fields
                            if not all(field in job for field in ['title', 'company', 'url']):
                                logger.warning(f"Job {job_idx} missing required fields: {job}")
                                # Try to infer missing fields
                                if 'title' not in job and 'position' in job:
                                    job['title'] = job['position']
                                if 'company' not in job:
                                    job['company'] = company
                                if 'url' not in job and 'link' in job:
                                    job['url'] = job['link']
                                # Skip if still missing required fields
                                if not all(field in job for field in ['title', 'company', 'url']):
                                    continue

                            # Ensure each job has the type field
                            job['type'] = company_type

                            # Log each job
                            logger.debug(f"Job {job_idx}: {job.get('title', 'Unknown')} at {job.get('company', 'Unknown')}")

                            # Normalize the job listings
                            try:
                                job_listing = JobListing(**job)

                                # Add to appropriate list
                                if company_type == "Major":
                                    major_jobs.append(job_listing.model_dump())
                                else:
                                    startup_jobs.append(job_listing.model_dump())

                                logger.debug(f"Added {company_type.lower()} job: {job_listing.title}")
                            except Exception as e:
                                logger.warning(f"Invalid job listing format: {e} - {job}")

                        search_successes += 1

                    except Exception as e:
                        logger.error(f"Error parsing job results: {e}", exc_info=True)
                        search_failures += 1
                        continue

            except Exception as e:
                logger.error(f"Error searching for jobs: {e}")
                search_failures += 1
                continue

        # Log search phase results and token usage
        token_monitor.log_usage('search')
        logger.info(f"Search phase complete - Successes: {search_successes}, Failures: {search_failures}")
        logger.info(f"Jobs found - Major: {len(major_jobs)}, Startup: {len(startup_jobs)}")

    with Timer('Verification Phase', logger):
        # Verify job URLs
        logger.info(f"Verifying up to {len(major_jobs)} major jobs and {len(startup_jobs)} startup jobs")

        # Verify major and startup jobs
        verified_majors = await verify_jobs(cfg, logger, major_jobs[:majors_quota], major_type=True)
        verified_startups = await verify_jobs(cfg, logger, startup_jobs[:startups_quota], major_type=False)

        # Log verification phase results and token usage
        token_monitor.log_usage('verify')
        logger.info(f"Verification complete - Major: {len(verified_majors)}, Startup: {len(verified_startups)}")

    # Final token usage report
    logger.info("Final token usage report:")
    token_monitor.log_usage()

    # Generate visualizations
    try:
        with DepthContext(logger, "Generating visualizations"):
            flow_diagram = visualizer.generate_flow_diagram(title="Job Search Agent Flow")
            timeline_diagram = visualizer.generate_timeline_diagram(title="Job Search Timeline")
            report_file = visualizer.generate_report()

            logger.info(f"Visualizations generated:")
            logger.info(f"  - Flow diagram: {flow_diagram}")
            logger.info(f"  - Timeline: {timeline_diagram}")
            logger.info(f"  - Report: {report_file}")
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")

    # Combine and return results
    results = verified_majors[:majors_quota] + verified_startups[:startups_quota]

    # Add sequential numbering
    for i, job in enumerate(results, 1):
        job['#'] = i

    logger.info(f"Final results: {len(results)} total jobs found ({len(verified_majors)} major, {len(verified_startups)} startup)")
    return results

# ------------- main ----------------
async def verify_jobs(cfg, logger, jobs, major_type: bool = True) -> List[Dict[str, str]]:
    """
    Verify job URLs for validity.

    Args:
        cfg: Configuration dictionary
        logger: Logger instance
        jobs: List of job dictionaries to verify
        major_type: Whether these are major company jobs (for logging)

    Returns:
        List of verified job dictionaries
    """
    token_monitor = TokenMonitor(cfg, logger)
    company_type = "Major" if major_type else "Startup"
    logger.info(f"Verifying {len(jobs)} {company_type.lower()} company job URLs")

    # Get API wrapper
    api = get_api_wrapper()

    # Get visualizer
    visualizer = get_visualizer()

    verified_jobs = []
    verification_failures = 0
    verification_successes = 0

    # Configure verifier tools based on web search option
    logger.debug(f"Initializing verifier with model {cfg['verifier_model']}, web search: {cfg.get('use_web_verify', False)}")

    # Configure verifier tools based on web search option
    verifier_tools = []
    if cfg.get('use_web_verify', False):
        logger.info("Web search verification mode enabled")
        verifier_tools.append(WebSearchTool())

    # Create the verifier agent
    verifier = await api.create_agent(
        name="verifier",
        instructions=verifier_prompt(),
        tools=verifier_tools,
        model=cfg['verifier_model'],
        model_settings=ModelSettings(temperature=0)
    )

    for i, job in enumerate(jobs, 1):
        # Check if we've exceeded token budget
        if not token_monitor.check_budget('verify') or not token_monitor.check_total():
            logger.warning(f"Stopping verification phase early due to token budget constraints")
            break

        job_url = job.get('url', '')
        job_title = job.get('title', 'Unknown')
        job_company = job.get('company', 'Unknown')

        # Skip if URL is empty or not a string
        if not job_url or not isinstance(job_url, str):
            logger.warning(f"Skipping job with invalid URL: {job}")
            verification_failures += 1
            continue

        logger.info(f"Verifying {company_type} job {i}/{len(jobs)}: {job_title} at {job_company}")

        try:
            with DepthContext(logger, f"Verify: {job_company}-{job_title}", log_level='DEBUG'):
                # Track API call
                token_monitor.track_api_call('verify')

                # Check URL format with regex rather than web search
                # Common job URL patterns
                valid_patterns = [
                    r'.*\.jobs/.*',
                    r'.*careers\..*\.com/.*',
                    r'.*apply\..*\.com/.*',
                    r'.*\/jobs?\/.*',
                    r'.*\/careers?\/.*',
                    r'.*\/positions?\/.*',
                    r'.*\/opportunities?\/.*',
                    r'.*linkedin\.com\/jobs\/.*',
                    r'.*indeed\.com\/.*',
                    r'.*glassdoor\.com\/.*',
                    r'.*lever\.co\/.*',
                    r'.*greenhouse\.io\/.*',
                    r'.*workday\.com\/.*',
                ]

                # Log the URL being verified
                logger.debug(f"Verifying URL: {job_url}")

                # Check if URL matches any valid pattern for quick validation
                pattern_match = any(re.search(pattern, job_url, re.IGNORECASE) for pattern in valid_patterns)
                if pattern_match:
                    logger.debug(f"URL pattern check passed")
                else:
                    logger.debug(f"URL did not match any known job URL pattern")

                # Determine if we need to use the agent for verification
                if pattern_match and not cfg.get('use_web_verify', False):
                    # If pattern matches and web search is not enabled, consider valid
                    url_valid = True
                    logger.debug(f"URL pattern check passed for: {job_url}")
                else:
                    # Otherwise, use agent verification regardless
                    if cfg.get('use_web_verify', False):
                        verify_input = f"Verify this job URL using web search: {job_url}"
                        logger.debug(f"Using web search verification for: {job_url}")
                    else:
                        verify_input = f"Verify this job URL: {job_url}"
                        logger.debug(f"Using agent verification for: {job_url}")

                    # Run the verifier agent
                    verify_result = await api.run_agent(verifier, verify_input)
                    verify_output = verify_result.final_output.strip().lower()
                    url_valid = verify_output == 'true'

                    # Track in visualizer
                    visualizer.track_agent_call(
                        agent_name="verifier",
                        input_text=verify_input,
                        output_text=verify_output,
                        duration=0.3,  # Placeholder duration
                        tokens_used=None
                    )

                # Update job status based on verification
                if url_valid:
                    job['has_apply'] = True
                    verified_jobs.append(job)
                    verification_successes += 1
                    logger.debug(f"Verified job: {job_title} at {job_company}")
                else:
                    verification_failures += 1
                    logger.warning(f"Failed to verify job URL: {job_url}")

        except Exception as e:
            logger.error(f"Error verifying job: {e}")
            verification_failures += 1
            continue

    # Log verification results
    logger.info(f"{company_type} verification results - Success: {verification_successes}, Failed: {verification_failures}")
    return verified_jobs

def save(rows, logger):
    logger.info(f"Saving {len(rows)} job results to CSV and Markdown")

    df=pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Select and reorder columns
    columns = ['#', 'title', 'company', 'type', 'url']
    if not rows:
        df = pd.DataFrame(columns=columns)
        logger.warning("No job results to save")
    else:
        df = df[columns]  # Select only these columns

        # Log some statistics about the results
        company_counts = df['company'].value_counts().to_dict()
        logger.info(f"Company distribution in results: {company_counts}")

        type_counts = df['type'].value_counts().to_dict()
        logger.info(f"Type distribution in results: {type_counts}")

    # Save to files
    csv_path = OUTPUT_DIR/'deep_job_results.csv'
    md_path = OUTPUT_DIR/'deep_job_results.md'

    df.to_csv(csv_path, index=False)

    # Try to save markdown but handle the case where tabulate is missing
    try:
        # Test if tabulate is available
        import tabulate
        df.to_markdown(md_path, index=False)
        logger.info(f"Results saved to:")
        logger.info(f"  - CSV: {csv_path}")
        logger.info(f"  - Markdown: {md_path}")
    except ImportError:
        # Handle missing tabulate dependency
        logger.warning("Could not save markdown table - tabulate package missing")
        logger.warning("Install tabulate with: pip install tabulate==0.9.0")
        logger.info(f"Results saved to CSV: {csv_path}")

        # Save a simple text representation instead
        with open(md_path, 'w') as f:
            f.write(f"# Deep Job Search Results\n\n")
            f.write(f"Found {len(rows)} jobs.\n\n")
            for job in rows:
                f.write(f"## {job['#']}. {job['title']} at {job['company']}\n")
                f.write(f"- Type: {job['type']}\n")
                f.write(f"- URL: {job['url']}\n\n")

def parse():
    p=argparse.ArgumentParser(description="Deep Job Search - Find software jobs in video/streaming companies")
    p.add_argument('--majors',type=int,default=100, help="Number of major company jobs to find (default: 100)")
    p.add_argument('--startups',type=int,default=100, help="Number of startup jobs to find (default: 100)")
    p.add_argument('--planner-model',default='gpt-4.1', help="Model for planning (default: gpt-4.1)")
    p.add_argument('--search-model',default='gpt-4o-mini', help="Model for search/processing (default: gpt-4o-mini)")
    p.add_argument('--verifier-model',default='o3', help="Model for verification (default: o3)")
    p.add_argument('--log-level',default='INFO', help="Logging level (default: INFO)")
    p.add_argument('--log-file', help="Log to this file in addition to console")
    p.add_argument('--max-tokens',type=int,default=100000, help="Maximum tokens to use (default: 100000)")
    p.add_argument('--force', action='store_true', help="Skip cost confirmation prompt")
    p.add_argument('--budget',type=float, help="Maximum cost in USD (will estimate and exit if exceeded)")
    p.add_argument('--sample', action='store_true', help="Run with minimal settings (10 major, 10 startup jobs) for testing")
    p.add_argument('--use-web-verify', action='store_true', help="Use web search for URL verification (slower but more accurate)")
    p.add_argument('--trace', action='store_true', help="Enable detailed agent tracing for debugging")
    return p.parse_args()

def main():
    # Parse arguments
    args=parse()

    # Handle sample mode
    if args.sample:
        args.majors = 10
        args.startups = 10

    # Setup logger
    logger=setup_logger(args.log_level, args.log_file)
    logger.info(f"deep_job_search starting with Python {os.sys.version}")

    # Setup tracing if enabled
    if args.trace:
        logger.info("Agent tracing enabled")
        # Only try to initialize tracing if the necessary imports were successful
        if 'HAS_TRACING' in globals() and HAS_TRACING:
            if 'HAS_CONSOLE_PROCESSOR' in globals() and HAS_CONSOLE_PROCESSOR:
                logger.info("Initializing tracing with console processor")
                console_processor = ConsoleSpanProcessor()
                tracer.add_span_processor(console_processor)
            else:
                logger.info("Console processor not available, using basic tracing")
        else:
            logger.warning("Tracing requested but not available in this version of the agents SDK")
            logger.info("Run 'pip install openai-agents[tracing]' for tracing support")

    # Convert args to dict
    cfg=vars(args)

    # Check for budget constraint or estimate-only mode
    token_monitor = TokenMonitor(cfg, logger)
    estimated_tokens, estimated_cost = token_monitor.estimate_cost()

    logger.info(f"Estimated resource usage:")
    logger.info(f"  - Tokens: ~{estimated_tokens:,} tokens")
    logger.info(f"  - Cost: ~${estimated_cost:.4f}")

    # Check if this is estimate-only mode
    if os.environ.get('JOBBOT_ESTIMATE_ONLY', ''):
        logger.info(f"Estimate-only mode enabled. Exiting now.")
        return 0

    # Check budget constraint
    if args.budget and estimated_cost > args.budget:
        logger.error(f"Estimated cost ${estimated_cost:.4f} exceeds budget ${args.budget:.2f}")
        logger.info("Suggestions to reduce costs:")
        logger.info("  - Reduce --majors and --startups values")
        logger.info("  - Use cheaper models (--search-model gpt-4o-mini --verifier-model o3)")
        logger.info("  - Set a lower --max-tokens value")
        return 1

    # Warn about high cost and offer to abort
    if estimated_cost > 0.50 and not args.force and not os.environ.get('JOBBOT_SKIP_CONFIRM', ''):
        logger.warning(f"⚠️ COST WARNING: Estimated cost (${estimated_cost:.4f}) exceeds $0.50")
        logger.warning(f"Consider reducing job counts or using less expensive models")
        logger.info("Continue? (y/n, or set JOBBOT_SKIP_CONFIRM=1 to skip this prompt)")
        try:
            response = input().strip().lower()
            if response != 'y':
                logger.info("Aborting job search")
                return 0
        except (KeyboardInterrupt, EOFError):
            logger.info("\nAborting job search")
            return 0

    # Track execution time and token usage
    start=time.time()
    logger.info(f"Job search execution started")

    try:
        # Execute search
        rows=asyncio.run(gather(cfg,logger))

        # Save results
        save(rows,logger)

        # Log execution time
        run_time = time.time() - start
        logger.info(f"Execution completed in {run_time:.1f}s")
        logger.info(f"Found {len(rows)} roles ({cfg['majors']} major target, {cfg['startups']} startup target)")

    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        return 1

    return 0

if __name__=='__main__':
    exit_code = main()
    exit(exit_code)
