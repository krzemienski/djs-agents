#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs/visuals results

# Get timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Default values
MAJORS=10
STARTUPS=10
LOG_LEVEL="DEBUG"
USE_WEB_VERIFY=false
TRACE=false
SAMPLE=false
MODEL_PLANNER="gpt-4.1"
MODEL_SEARCH="gpt-4o-mini"
MODEL_VERIFY="o3"
HELP=false
USE_DOCKER=true
USE_RESPONSES=false
FORCE_REBUILD=false

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "⚠️ Docker not found. Will run directly with Python."
    USE_DOCKER=false
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --majors)
      MAJORS="$2"
      shift # past argument
      shift # past value
      ;;
    --startups)
      STARTUPS="$2"
      shift # past argument
      shift # past value
      ;;
    --log-level)
      LOG_LEVEL="$2"
      shift # past argument
      shift # past value
      ;;
    --use-web-verify)
      USE_WEB_VERIFY=true
      shift # past argument
      ;;
    --trace)
      TRACE=true
      shift # past argument
      ;;
    --sample)
      SAMPLE=true
      shift # past argument
      ;;
    --cheap)
      MODEL_PLANNER="gpt-4o-mini"
      MODEL_SEARCH="gpt-4o-mini"
      MODEL_VERIFY="o3"
      shift # past argument
      ;;
    --premium)
      MODEL_PLANNER="gpt-4.1"
      MODEL_SEARCH="gpt-4.1"
      MODEL_VERIFY="gpt-4o"
      shift # past argument
      ;;
    --responses)
      USE_RESPONSES=true
      shift # past argument
      ;;
    --no-docker)
      USE_DOCKER=false
      shift # past argument
      ;;
    --rebuild)
      FORCE_REBUILD=true
      shift # past argument
      ;;
    --help|-h)
      HELP=true
      shift # past argument
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Show help
if [ "$HELP" = true ]; then
  echo "Deep Job Search - Enhanced Logging Runner"
  echo ""
  echo "Usage: ./run_with_logs.sh [options]"
  echo ""
  echo "Options:"
  echo "  --majors N          Number of major company jobs to find (default: 10)"
  echo "  --startups N        Number of startup company jobs to find (default: 10)"
  echo "  --log-level LEVEL   Set logging level (default: DEBUG)"
  echo "  --use-web-verify    Enable web search for URL verification"
  echo "  --trace             Enable agent tracing (if available)"
  echo "  --sample            Run in sample mode (10 major, 10 startup jobs)"
  echo "  --cheap             Use lowest-cost model combination"
  echo "  --premium           Use highest-quality model combination"
  echo "  --responses         Use Responses API implementation instead of Agents SDK"
  echo "  --no-docker         Force running with Python directly (no Docker)"
  echo "  --rebuild           Force rebuild Docker image before running"
  echo "  --help, -h          Show this help message"
  echo ""
  echo "Examples:"
  echo "  ./run_with_logs.sh                        # Run with default settings"
  echo "  ./run_with_logs.sh --sample --cheap       # Quick, low-cost run"
  echo "  ./run_with_logs.sh --responses            # Use Responses API implementation"
  exit 0
fi

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️ ERROR: OPENAI_API_KEY environment variable is not set"
    echo "Please set your API key with: export OPENAI_API_KEY=your_api_key_here"
    exit 1
fi

if [ "$USE_DOCKER" = true ]; then
    echo "🐳 Using Docker to run the application"

    # Check if image exists or force rebuild
    if $FORCE_REBUILD || ! docker image inspect jobbot &> /dev/null; then
        echo "🔹 Building Docker image..."
        docker build -t jobbot .
        if [ $? -ne 0 ]; then
            echo "⚠️ ERROR: Docker build failed"
            exit 1
        fi
        echo "✅ Docker image built successfully"
    else
        echo "✅ Using existing Docker image"
    fi

    # Build Docker command
    DOCKER_CMD="docker run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY"
    DOCKER_CMD+=" -v $(pwd)/logs:/app/logs -v $(pwd)/results:/app/results"

    # Set entrypoint based on implementation choice
    if [ "$USE_RESPONSES" = true ]; then
        DOCKER_CMD+=" --entrypoint python jobbot responses_job_search.py"

        # Add parameters for Responses API
        DOCKER_CMD+=" --majors $MAJORS --startups $STARTUPS"

        if [ "$SAMPLE" = true ]; then
            DOCKER_CMD+=" --sample"
        fi

        # Model selection for Responses API
        if [ "$MODEL_PLANNER" = "gpt-4o-mini" ]; then
            DOCKER_CMD+=" --model gpt-4o-mini"
        elif [ "$MODEL_PLANNER" = "gpt-4.1" ]; then
            DOCKER_CMD+=" --model gpt-4.1"
        fi

        # Debug logging for Responses API
        if [ "$LOG_LEVEL" = "DEBUG" ]; then
            # Responses API doesn't have log-level, make sure output is verbose
            DOCKER_CMD+=" --max-tokens 4000"
        fi
    else
        DOCKER_CMD+=" --entrypoint python jobbot deep_job_search.py"

        # Add parameters for Agents SDK
        DOCKER_CMD+=" --majors $MAJORS --startups $STARTUPS"
        DOCKER_CMD+=" --log-level $LOG_LEVEL"
        DOCKER_CMD+=" --log-file /app/logs/jobsearch_${TIMESTAMP}.log"
        DOCKER_CMD+=" --planner-model $MODEL_PLANNER"
        DOCKER_CMD+=" --search-model $MODEL_SEARCH"
        DOCKER_CMD+=" --verifier-model $MODEL_VERIFY"

        # Optional flags
        if [ "$USE_WEB_VERIFY" = true ]; then
            DOCKER_CMD+=" --use-web-verify"
        fi

        if [ "$TRACE" = true ]; then
            DOCKER_CMD+=" --trace"
        fi

        if [ "$SAMPLE" = true ]; then
            DOCKER_CMD+=" --sample"
        fi
    fi

    # Print and run Docker command
    echo "Running: $DOCKER_CMD"
    eval $DOCKER_CMD

    # Determine file prefixes based on implementation
    if [ "$USE_RESPONSES" = true ]; then
        RESULTS_PREFIX="responses_job"
    else
        RESULTS_PREFIX="deep_job"
    fi

    # Summarize results
    echo "Results saved to:"
    echo "  - CSV: $(pwd)/results/${RESULTS_PREFIX}_results.csv"
    echo "  - Markdown: $(pwd)/results/${RESULTS_PREFIX}_results.md"

    if [ "$USE_RESPONSES" = false ]; then
        echo "  - Logs: $(pwd)/logs/jobsearch_${TIMESTAMP}.log"
        echo "  - API logs: $(pwd)/logs/jobsearch_${TIMESTAMP}.api.log"

        # Check if visualization directory has files
        if [ -d "logs/visuals" ] && [ "$(ls -A logs/visuals)" ]; then
            echo "  - Visualizations: $(pwd)/logs/visuals/"
        fi
    fi
else
    # Build Python command for direct execution
    if [ "$USE_RESPONSES" = true ]; then
        CMD="python3 responses_job_search.py"

        # Add parameters
        CMD+=" --majors $MAJORS"
        CMD+=" --startups $STARTUPS"

        if [ "$SAMPLE" = true ]; then
            CMD+=" --sample"
        fi

        # Map model choices
        if [ "$MODEL_PLANNER" = "gpt-4o-mini" ]; then
            CMD+=" --model gpt-4o-mini"
        elif [ "$MODEL_PLANNER" = "gpt-4.1" ]; then
            CMD+=" --model gpt-4.1"
        fi
    else
        CMD="python3 deep_job_search.py"

        # Add parameters
        CMD+=" --majors $MAJORS"
        CMD+=" --startups $STARTUPS"
        CMD+=" --log-level $LOG_LEVEL"
        CMD+=" --log-file logs/jobsearch_${TIMESTAMP}.log"
        CMD+=" --planner-model $MODEL_PLANNER"
        CMD+=" --search-model $MODEL_SEARCH"
        CMD+=" --verifier-model $MODEL_VERIFY"

        # Optional flags
        if [ "$USE_WEB_VERIFY" = true ]; then
            CMD+=" --use-web-verify"
        fi

        if [ "$TRACE" = true ]; then
            CMD+=" --trace"
        fi

        if [ "$SAMPLE" = true ]; then
            CMD+=" --sample"
        fi
    fi

    # Print command
    echo "Running: $CMD"

    # Run the command
    eval $CMD

    # Determine result file prefixes based on implementation
    if [ "$USE_RESPONSES" = true ]; then
        RESULTS_DIR="responses_job_search"
        RESULTS_PREFIX="responses_job"
    else
        RESULTS_DIR="deep_job_search"
        RESULTS_PREFIX="deep_job"
    fi

    # Summarize results
    echo "Results saved to:"
    echo "  - CSV: $(pwd)/$RESULTS_DIR/${RESULTS_PREFIX}_results.csv"
    echo "  - Markdown: $(pwd)/$RESULTS_DIR/${RESULTS_PREFIX}_results.md"

    if [ "$USE_RESPONSES" = false ]; then
        echo "  - Logs: $(pwd)/logs/jobsearch_${TIMESTAMP}.log"
        echo "  - API logs: $(pwd)/logs/jobsearch_${TIMESTAMP}.api.log"

        # Check if visualization directory has files
        if [ -d "logs/visuals" ] && [ "$(ls -A logs/visuals)" ]; then
            echo "Visualizations generated in logs/visuals"
            echo "Latest visualizations:"
            ls -lt logs/visuals | head -5
        fi
    fi
fi
