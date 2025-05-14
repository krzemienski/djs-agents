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
  docker build -t djso .
  touch logs/debug/.last_build
fi

# Check if output arg is already specified
HAS_OUTPUT=false
for arg in "$@"; do
  if [[ "$arg" == "--output" ]]; then
    HAS_OUTPUT=true
    break
  fi
done

# Build the command
CMD="./run.sh"
for arg in "$@"; do
  CMD="$CMD \"$arg\""
done

# Add output argument if needed
if [ "$HAS_OUTPUT" = false ]; then
  CMD="$CMD --output results/jobs.csv"
fi

# Run the tool
echo "Running Deep Job Search in Docker container..."
echo "Command: $CMD"

# Execute the container
docker run --rm \
  -e OPENAI_API_KEY \
  -e RUNNING_IN_CONTAINER=1 \
  -v "$(pwd)/results:/app/results" \
  -v "$(pwd)/logs:/app/logs" \
  djso /bin/bash -c "$CMD"

# Check exit code
STATUS=$?
if [ $STATUS -eq 0 ]; then
  echo "Job search completed successfully!"
else
  echo "Job search failed with exit code $STATUS"
fi

exit $STATUS
