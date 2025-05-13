# Docker Container Setup

This document explains how the Deep Job Search Docker container is set up and how to run both implementations within it.

## Docker Configuration

The Docker container is configured with the following components:

1. **Base Image**: Python 3.11 slim
2. **Dependencies**:
   - Python packages from requirements.txt
   - graphviz for visualization
   - git for version control

3. **Volumes**:
   - `/app/logs` - For storing logs and visualizations
   - `/app/results` - For storing job search results

4. **Environment Variables**:
   - `OPENAI_API_KEY` - Passed from host
   - `RUNNING_IN_CONTAINER=1` - Set to indicate container environment

5. **Entry Point**:
   - Default: `python deep_job_search.py`
   - Can be overridden to run other scripts

## Running the Container

There are three scripts for running the application:

1. **run.sh** - Unified script that runs either implementation in Docker (if available) or directly
   ```bash
   ./run.sh [options]
   ```

2. **build-and-run.sh** - Original Docker-specific script
   ```bash
   ./build-and-run.sh [options]
   ```

3. **run_with_logs.sh** - Enhanced logging script (used by run.sh)
   ```bash
   ./run_with_logs.sh [options]
   ```

## Implementation Selection

You can run either implementation:

1. **Deep Job Search** (default): Multi-agent implementation
   ```bash
   ./run.sh
   ```

2. **Responses Job Search**: Simpler implementation
   ```bash
   ./run.sh --responses
   ```

## File Structure and Output

- **Input Configuration**: Command-line arguments or environment variables
- **Output Files**: Saved to `/app/results` directory
  - Deep Job Search: `deep_job_results.csv` and `deep_job_results.md`
  - Responses Job Search: `responses_job_results.csv` and `responses_job_results.md`
- **Logs**: Saved to `/app/logs` directory
  - Main logs: `*.log`
  - API logs: `*.api.log`
  - Visualizations: `logs/visuals/*`

## Docker Advantages

Running in Docker provides several benefits:

1. **Consistent Environment**: Same dependencies regardless of host OS
2. **Isolated Execution**: Clean execution environment each time
3. **Simplified Distribution**: Easy to share and deploy
4. **Volume Mounting**: Persists results and logs on the host
5. **Unified Command Interface**: Same commands work on any platform

## Troubleshooting

If you encounter issues:

1. **Rebuild the image**: `./run.sh --rebuild`
2. **Enable debug logging**: `./run.sh --log-level DEBUG`
3. **Check logs**: Examine files in the `logs/` directory
4. **Verify API key**: Ensure `OPENAI_API_KEY` is properly set
5. **Check volume mounts**: Ensure `logs/` and `results/` directories exist
