#!/bin/bash
# ---------------------------------------------------------------
# build-and-run.sh - Build and run the Deep Job Search container
# ---------------------------------------------------------------

# Set strict error handling
set -e

# Create required directories
mkdir -p logs/debug
mkdir -p results

echo "=== Deep Job Search - Build & Run ==="

# Check if we need to rebuild the container
REBUILD=false
if [ -f .rebuild-marker ]; then
  echo "Rebuild marker found (.rebuild-marker), will rebuild container"
  REBUILD=true
  rm .rebuild-marker
fi

# Also check if Dockerfile has been modified
if [ "$(find Dockerfile -newer logs/debug/.last_build 2>/dev/null)" ]; then
  echo "Dockerfile has been modified since last build, will rebuild container"
  REBUILD=true
fi

# Also check if requirements files have been modified
if [ "$(find requirements*.txt -newer logs/debug/.last_build 2>/dev/null)" ]; then
  echo "Requirements files have been modified, will rebuild container"
  REBUILD=true
fi

# Ensure we have a Docker image
if ! docker image inspect djso 2>/dev/null >/dev/null; then
  echo "Docker image 'djso' not found, will build it"
  REBUILD=true
fi

# Build the Docker image if needed
if [ "$REBUILD" = true ]; then
  echo "Building Docker image..."
  docker build --pull -t djso .
  touch logs/debug/.last_build
fi

# Check if output arg is already specified
HAS_OUTPUT=false
# Check if simplified flag is present
HAS_SIMPLIFIED=false
# Check if dry-run flag is present
IS_DRY_RUN=false
for arg in "$@"; do
  if [[ "$arg" == "--output" ]]; then
    HAS_OUTPUT=true
  fi
  if [[ "$arg" == "--simplified" ]]; then
    HAS_SIMPLIFIED=true
  fi
  if [[ "$arg" == "--dry-run" ]]; then
    IS_DRY_RUN=true
  fi
done

# Build the argument array for run.sh
ARGS=("$@")

# Add output argument if needed
if [ "$HAS_OUTPUT" = false ]; then
  ARGS+=("--output" "results/jobs.csv")
fi

# Run the tool
echo "Running Deep Job Search in Docker container..."

# Print message for dry run
if [ "$IS_DRY_RUN" = true ]; then
  echo "DRY RUN MODE: Testing pipeline without making API calls"
fi

# Prepare arguments as a string for docker command
ARG_STRING=""
for arg in "${ARGS[@]}"; do
  ARG_STRING="$ARG_STRING \"$arg\""
done

# Execute the container with the arguments
docker run --rm \
  -e OPENAI_API_KEY \
  -e RUNNING_IN_CONTAINER=1 \
  -v "$(pwd)/results:/app/results" \
  -v "$(pwd)/logs:/app/logs" \
  --entrypoint /bin/bash \
  djso -c "cd /app && ./run.sh $ARG_STRING"

# Check exit code
STATUS=$?
if [ $STATUS -eq 0 ]; then
  echo "Job search completed successfully!"
else
  echo "Job search failed with exit code $STATUS"
fi

exit $STATUS
