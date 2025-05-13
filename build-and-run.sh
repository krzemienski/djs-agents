#!/bin/bash
# ---------------------------------------------------------------
# build-and-run.sh - Build and run the Deep Job Search container
# ---------------------------------------------------------------

# Set strict error handling
set -e

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️ ERROR: OPENAI_API_KEY environment variable is not set"
    echo "Please set your API key with: export OPENAI_API_KEY=your_api_key_here"
    exit 1
fi

# Create output directories
echo "🔹 Creating output directories..."
mkdir -p results logs/visuals

# Function to display help message
show_help() {
    echo "Deep Job Search - Docker Edition"
    echo "--------------------------------"
    echo "Usage: ./build-and-run.sh [options]"
    echo ""
    echo "Options:"
    echo "  --help            Show this help message"
    echo "  --rebuild         Force rebuild the Docker image"
    echo "  --quick           Run with minimal jobs (10 major, 10 startup)"
    echo "  --debug           Enable DEBUG logging level"
    echo "  --cheap           Use lowest-cost models for all agents"
    echo "  --premium         Use highest-quality models for all agents"
    echo "  --budget VALUE    Set maximum cost in USD (e.g., --budget 0.25)"
    echo "  --estimate-only   Only display cost estimate without running search"
    echo "  --force           Skip cost confirmation prompts"
    echo "  --use-web-verify  Use web search for URL verification (slower but more accurate)"
    echo "  --trace           Enable detailed agent tracing for debugging"
    echo "  --visualize       Generate agent visualization instead of running job search"
    echo "  --responses       Use the Responses API implementation instead of Agents SDK"
    echo "  --custom \"args\"   Pass custom arguments to the container"
    echo ""
    echo "Examples:"
    echo "  ./build-and-run.sh --quick            # Fast run with few jobs"
    echo "  ./build-and-run.sh --debug            # Run with debug logging"
    echo "  ./build-and-run.sh --budget 0.25      # Set a maximum cost of $0.25"
    echo "  ./build-and-run.sh --estimate-only    # Just estimate cost and exit"
    echo "  ./build-and-run.sh --visualize        # Generate agent visualization"
    echo "  ./build-and-run.sh --responses        # Use Responses API implementation"
    exit 0
}

# Default values
FORCE_REBUILD=false
QUICK_RUN=false
DEBUG_MODE=false
CHEAP_MODE=false
PREMIUM_MODE=false
USE_WEB_VERIFY=false
TRACE_MODE=false
VISUALIZE_MODE=false
RESPONSES_MODE=false
CUSTOM_ARGS=""
BUDGET=""
ESTIMATE_ONLY=false
FORCE_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help)
            show_help
            ;;
        --rebuild)
            FORCE_REBUILD=true
            shift
            ;;
        --quick)
            QUICK_RUN=true
            shift
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        --cheap)
            CHEAP_MODE=true
            shift
            ;;
        --premium)
            PREMIUM_MODE=true
            shift
            ;;
        --budget)
            BUDGET="$2"
            shift 2
            ;;
        --estimate-only)
            ESTIMATE_ONLY=true
            shift
            ;;
        --force)
            FORCE_MODE=true
            shift
            ;;
        --use-web-verify)
            USE_WEB_VERIFY=true
            shift
            ;;
        --trace)
            TRACE_MODE=true
            CUSTOM_ARGS="$CUSTOM_ARGS --trace"
            shift
            ;;
        --visualize)
            VISUALIZE_MODE=true
            CUSTOM_ARGS="$CUSTOM_ARGS --visualize"
            shift
            ;;
        --responses)
            RESPONSES_MODE=true
            CUSTOM_ARGS="$CUSTOM_ARGS --responses"
            shift
            ;;
        --custom)
            CUSTOM_ARGS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "⚠️ ERROR: Docker is not installed or not in PATH"
    echo "Please install Docker and try again"
    exit 1
fi

# Check if image exists
if $FORCE_REBUILD || ! docker image inspect jobbot &> /dev/null; then
    echo "🔹 Building Docker image..."
    docker build -t jobbot .
    if [ $? -ne 0 ]; then
        echo "⚠️ ERROR: Docker build failed"
        exit 1
    fi
    echo "✅ Docker image built successfully"
else
    echo "✅ Using existing Docker image (use --rebuild to force rebuild)"
fi

# Prepare run command and arguments
RUN_CMD="docker run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY -e RUNNING_IN_CONTAINER=1"
RUN_CMD+=" -v $(pwd)/logs:/app/logs -v $(pwd)/results:/app/results jobbot"

# Handle special execution modes
if $VISUALIZE_MODE; then
    echo "🔹 Running in visualization mode"
    # Override the entrypoint to run the visualization script
    RUN_CMD="docker run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY -e RUNNING_IN_CONTAINER=1"
    RUN_CMD+=" -v $(pwd)/logs:/app/logs -v $(pwd)/results:/app/results"
    RUN_CMD+=" --entrypoint python jobbot visualize_agents.py"
    # Remove any custom args that might conflict
    CUSTOM_ARGS=""
elif $RESPONSES_MODE; then
    echo "🔹 Running with Responses API implementation"
    # Override the entrypoint to run the responses script
    RUN_CMD="docker run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY -e RUNNING_IN_CONTAINER=1"
    RUN_CMD+=" -v $(pwd)/logs:/app/logs -v $(pwd)/results:/app/results"
    RUN_CMD+=" --entrypoint python jobbot responses_job_search.py"
fi

# Add appropriate arguments based on options
if [ -n "$CUSTOM_ARGS" ]; then
    # Custom arguments take precedence
    RUN_CMD="$RUN_CMD $CUSTOM_ARGS"
else
    # Add budget if specified
    if [ -n "$BUDGET" ]; then
        RUN_CMD="$RUN_CMD --budget $BUDGET"
    fi

    # Add force flag if specified
    if $FORCE_MODE; then
        RUN_CMD="$RUN_CMD --force"
    fi

    # Handle estimate-only mode
    if $ESTIMATE_ONLY; then
        # Set environment variable to skip prompt and exit after estimation
        RUN_CMD="$RUN_CMD --budget 999999 --sample"
        export JOBBOT_ESTIMATE_ONLY=1
    fi

    # Handle quick run
    if $QUICK_RUN; then
        RUN_CMD="$RUN_CMD --sample"
    fi

    # Handle model selection based on mode
    if $CHEAP_MODE; then
        RUN_CMD="$RUN_CMD --planner-model gpt-4o-mini --search-model gpt-4o-mini --verifier-model o3"
    elif $PREMIUM_MODE; then
        RUN_CMD="$RUN_CMD --planner-model gpt-4.1 --search-model gpt-4.1 --verifier-model gpt-4.1"
    fi

    # Handle web verification
    if $USE_WEB_VERIFY; then
        RUN_CMD="$RUN_CMD --use-web-verify"
    fi
fi

# Add debug flag if requested
if $DEBUG_MODE && [[ "$RUN_CMD" != *"--log-level"* ]]; then
    RUN_CMD="$RUN_CMD --log-level DEBUG"
fi

# Show the command being executed
echo "🔹 Executing: $RUN_CMD"

if $ESTIMATE_ONLY; then
    echo "🔹 Running in cost estimation mode only (will not perform actual search)"
else
    echo "🔹 This may take several minutes to complete..."
fi

# Run the container
if $ESTIMATE_ONLY; then
    # For estimate-only mode, add a custom environment variable and extract just the cost estimate
    $RUN_CMD | grep -A 5 "Estimated resource usage:"
    exit_code=${PIPESTATUS[0]}
else
    # Normal execution
    $RUN_CMD
    exit_code=$?
fi

# Check exit status
if [ $exit_code -eq 0 ]; then
    if ! $ESTIMATE_ONLY; then
        if $VISUALIZE_MODE; then
            echo "✅ Agent visualization completed successfully!"
            echo "📊 Visualization saved in the results/ directory"
        elif $RESPONSES_MODE; then
            echo "✅ Responses API job search completed successfully!"
            echo "📊 Results saved in the results/ directory (responses_job_results.*)"
        else
            echo "✅ Job search completed successfully!"
            echo "📊 Results saved in the results/ directory"
        fi
    fi
else
    echo "⚠️ Job search exited with an error"
    exit 1
fi
