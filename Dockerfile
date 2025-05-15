FROM python:3.11-slim
WORKDIR /app

# Install necessary dependencies including those for browser support
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    graphviz \
    wget \
    gnupg \
    ca-certificates \
    curl \
    libgconf-2-4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    libatspi2.0-0 \
    libgtk-3-0 \
    chromium \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Explicitly install MCP puppeteer packages if not in requirements
RUN pip install --no-cache-dir mcp-puppeteer || echo "MCP Puppeteer already installed or not available"

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
ENV PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true
ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium

# Default entrypoint runs the main implementation
ENTRYPOINT ["python", "deep_job_search.py"]

# CMD can be overridden at runtime to run other scripts
CMD ["--help"]
