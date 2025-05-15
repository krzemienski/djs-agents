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
MAX_TOKENS=""
COMPANY_LIST_LIMIT=""
LOG_LEVEL=""
LOG_FILE=""
DRY_RUN=""
VISUALIZE="--visualize" # Enable visualization by default
TRACE=""

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
    --company-list-limit)
      COMPANY_LIST_LIMIT="--company-list-limit $2"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="--max-tokens $2"
      shift 2
      ;;
    --no-web-verify)
      WEB_VERIFY=""
      shift
      ;;
    --web-verify)
      WEB_VERIFY="--web-verify"
      shift
      ;;
    --use-web-verify)
      WEB_VERIFY="--web-verify"
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
    --dry-run)
      DRY_RUN="--dry-run"
      shift
      ;;
    --log-level)
      LOG_LEVEL="--log-level $2"
      shift 2
      ;;
    --log-file)
      LOG_FILE="--log-file $2"
      shift 2
      ;;
    --visualize)
      VISUALIZE="--visualize"
      shift
      ;;
    --no-visualize)
      VISUALIZE=""
      shift
      ;;
    --trace)
      TRACE="--trace"
      shift
      ;;
    --budget|--force)
      # These arguments aren't supported by deep_job_search.py but are mentioned in README
      # Just skip them for now
      shift
      if [[ "$1" != --* && "$1" != -* && "$1" != "" ]]; then
        # If the next argument isn't a flag, it's the value for this parameter
        shift
      fi
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [-m|--majors COUNT] [-s|--startups COUNT] [--model MODEL] [--max-tokens COUNT]"
      echo "         [--company-list FILE] [--company-list-limit COUNT] [--web-verify|--no-web-verify]"
      echo "         [--debug] [--log-level LEVEL] [--log-file FILE] [--output FILE] [--dry-run]"
      echo "         [--visualize|--no-visualize] [--trace]"
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
echo "- Mode: Full multi-agent"
if [ -n "$DRY_RUN" ]; then
  echo "- Dry run: Yes (no API calls will be made)"
fi
echo "========================="

# Run the job search
python deep_job_search.py \
  --majors $MAJOR_COUNT \
  --startups $STARTUP_COUNT \
  --model $MODEL \
  $COMPANY_LIST \
  $COMPANY_LIST_LIMIT \
  $MAX_TOKENS \
  $WEB_VERIFY \
  $DEBUG \
  $LOG_LEVEL \
  $LOG_FILE \
  $DRY_RUN \
  $VISUALIZE \
  $TRACE \
  --output $OUTPUT

# Check exit code
STATUS=$?
if [ $STATUS -eq 0 ]; then
  echo "Job search completed successfully!"

  # Count the jobs found - skip for dry run
  if [ -z "$DRY_RUN" ] && [ -f "$OUTPUT" ]; then
    JOB_COUNT=$(wc -l < "$OUTPUT")
    if [ $JOB_COUNT -gt 1 ]; then
      # Subtract header row
      JOB_COUNT=$((JOB_COUNT - 1))
      echo "Found $JOB_COUNT jobs."
      echo "Results saved to $OUTPUT"
    else
      echo "No jobs found."
    fi
  elif [ -n "$DRY_RUN" ]; then
    echo "Dry run completed successfully."
  else
    echo "Output file not created."
  fi
else
  echo "Job search failed with exit code $STATUS"
fi

exit $STATUS
