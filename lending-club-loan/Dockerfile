# Dockerfile

# ---- Builder Stage ----
# This stage is used to build the Python dependencies.
FROM python:3.12-slim AS builder

# Install build-time system dependencies
# gcc and other build tools are only needed in this stage
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install python dependencies into the virtual environment
# Using --no-cache-dir prevents pip from storing a cache, reducing image size
RUN pip install --no-cache-dir "mlflow==3.5.1" botocore boto3 psycopg2-binary


# ---- Final Stage ----
# This is the final, smaller image that will be used.
FROM python:3.12-slim

# Install only the necessary runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN groupadd -r mlflow && useradd --no-log-init -r -g mlflow mlflow
USER mlflow

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Set the environment to use the virtual environment
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Expose the port the server will run on
EXPOSE 5000

# Set the working directory
WORKDIR /mlflow