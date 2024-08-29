# Use a lighter base image
FROM python:3.11-slim-buster as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VENV="/opt/poetry-venv" \
    POETRY_CACHE_DIR="/opt/.cache" \
    PYTHONPATH="/app"

# Install system dependencies and Poetry, then clean up in a single RUN command
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add Poetry to PATH
ENV PATH="${PATH}:${POETRY_HOME}/bin"

# Set working directory
WORKDIR /app

# Copy only requirements
COPY pyproject.toml poetry.lock* /app/

# Install project dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi \
    && rm -rf ${POETRY_CACHE_DIR}

# Start a new stage for a smaller final image
FROM python:3.11-slim-buster

# Copy installed packages and application code from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Set working directory
WORKDIR /app

# Copy only the necessary application code
COPY llm_inference /app/llm_inference
COPY scripts /app/scripts

# Expose port 8000 to the host
EXPOSE 8000

# Define a build-time argument with a default value
ARG INFERENCE_SERVER=llm_inference.s3_inference_server

# Set an environment variable using the argument
ENV INFERENCE_SERVER=${INFERENCE_SERVER}

# Run the application
CMD ["sh", "-c", "poetry run python -m ${INFERENCE_SERVER}"]
