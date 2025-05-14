#!/bin/bash
# ---------------------------------------------------------------
# run.sh - Simple alias to build-and-run.sh for backward compatibility
# ---------------------------------------------------------------

# Create required directories
mkdir -p logs/debug
mkdir -p results

# Set defaults
MAJOR_COUNT=10
STARTUP_COUNT=10
MODEL="gpt-4o"
COMPANY_LIST=""
WEB_VERIFY="--web-verify" # Enable web verification by default
DEBUG=""
OUTPUT="results/jobs.csv"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -m|--majors)
      MAJOR_COUNT="$2"
      shift 2
      ;;
    -s|--startups)
      STARTUP_COUNT="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --company-list)
      COMPANY_LIST="--company-list $2"
      shift 2
      ;;
    --no-web-verify)
      WEB_VERIFY=""
      shift
      ;;
    --debug)
      DEBUG="--debug"
      shift
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [-m|--majors COUNT] [-s|--startups COUNT] [--model MODEL] [--company-list FILE] [--no-web-verify] [--debug] [--output FILE]"
      exit 1
      ;;
  esac
done

echo "=== Deep Job Search ==="
echo "Starting job search with:"
echo "- Major companies: $MAJOR_COUNT"
echo "- Startups: $STARTUP_COUNT"
echo "- Model: $MODEL"
echo "- Web verification: $([ -n "$WEB_VERIFY" ] && echo "Enabled" || echo "Disabled")"
echo "- Output: $OUTPUT"
echo "========================="

# Run the job search
python deep_job_search.py \
  --majors $MAJOR_COUNT \
  --startups $STARTUP_COUNT \
  --model $MODEL \
  $COMPANY_LIST \
  $WEB_VERIFY \
  $DEBUG \
  --output $OUTPUT

# Check exit code
STATUS=$?
if [ $STATUS -eq 0 ]; then
  echo "Job search completed successfully!"

  # Count the jobs found
  if [ -f "$OUTPUT" ]; then
    JOB_COUNT=$(wc -l < "$OUTPUT")
    if [ $JOB_COUNT -gt 1 ]; then
      # Subtract header row
      JOB_COUNT=$((JOB_COUNT - 1))
      echo "Found $JOB_COUNT jobs."
      echo "Results saved to $OUTPUT"
    else
      echo "No jobs found."
    fi
  else
    echo "Output file not created."
  fi
else
  echo "Job search failed with exit code $STATUS"
fi

exit $STATUS
