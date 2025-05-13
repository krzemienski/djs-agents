FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git graphviz \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY deep_job_search.py responses_job_search.py visualize_agents.py ./
VOLUME ["/app/output"]

# Default entrypoint is still the main implementation
ENTRYPOINT ["python", "deep_job_search.py"]
