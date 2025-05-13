FROM python:3.11-slim
WORKDIR /app

# Install necessary dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY *.py *.sh *.md *.csv ./

# Make shell scripts executable
RUN chmod +x *.sh

# Create directories for logs and results
RUN mkdir -p logs/visuals results

# Create volumes for persistent storage
VOLUME ["/app/logs", "/app/results"]

# Set environment variable for running in container
ENV RUNNING_IN_CONTAINER=1

# Default entrypoint runs the main implementation
ENTRYPOINT ["python", "deep_job_search.py"]

# CMD can be overridden at runtime to run other scripts
CMD ["--help"]
