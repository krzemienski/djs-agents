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
    echo "  --help         Show this help message"
    echo "  --majors N     Number of major company jobs to find (default: 10)"
    echo "  --startups N   Number of startup jobs to find (default: 10)"
    echo "  --sample       Run with minimal settings (2 major, 2 startup jobs)"
    echo "  --quick        Alias for --sample"
          # --use-responses option removed as we now only use Responses API
    echo "  Other options are passed through to the container"
    echo ""
    exit 0
}

# Parse arguments
SHOW_HELP=false
DOCKER_ARGS=""
REBUILD=false
SAMPLE=false
# USE_RESPONSES variable is no longer needed

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            SHOW_HELP=true
            shift
            ;;
        --rebuild)
            REBUILD=true
            shift
            ;;
        --quick|--sample)
            SAMPLE=true
            # Explicitly set these flags for sample mode to match previous behavior
            DOCKER_ARGS="$DOCKER_ARGS --sample --majors 2 --startups 2 --log-level DEBUG"
            shift
            ;;
        # --use-responses option removed as we now only use Responses API
        *)
            # Pass through all other arguments
            DOCKER_ARGS="$DOCKER_ARGS $1"
            shift
            ;;
    esac
done

# Show help if requested
if [ "$SHOW_HELP" = true ]; then
    show_help
fi

# Build the Docker image if needed or requested
if [ "$REBUILD" = true ] || ! docker image inspect jobbot &>/dev/null; then
    echo "🔹 Building Docker image..."
    docker build -t jobbot .
    echo "✅ Docker image built successfully"
else
    echo "✅ Using existing Docker image (use --rebuild to force rebuild)"
fi

# Set default entry point
ENTRYPOINT="python deep_job_search.py"

# Run the container
echo "🔹 Executing: docker run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY -e RUNNING_IN_CONTAINER=1 -v $PWD/logs:/app/logs -v $PWD/results:/app/results jobbot $DOCKER_ARGS"
docker run --rm \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    -e RUNNING_IN_CONTAINER=1 \
    -v $PWD/logs:/app/logs \
    -v $PWD/results:/app/results \
    --entrypoint=/bin/sh \
    jobbot \
    -c "$ENTRYPOINT $DOCKER_ARGS"

echo "🔹 Job search completed"
echo "🔹 Results saved in the results directory"
