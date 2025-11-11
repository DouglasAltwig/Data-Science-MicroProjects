# =============================================================================
# Stage 1: Builder
#
# Use the full CUDA development image to build our dependencies.
# This stage has compilers (like gcc, nvcc) and development headers.
# =============================================================================
FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04 AS builder

# Prevent apt-get from asking for user input
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install prerequisites for adding new repositories and for building Python packages
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    wget \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheels
RUN python -m pip install --no-cache-dir --upgrade pip wheel

# Copy requirements and install dependencies into the venv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# =============================================================================
# Stage 2: Final Image
#
# Use the smaller CUDA runtime image. This image does NOT have compilers,
# making it more secure and much smaller.
# =============================================================================
FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN addgroup --system app && adduser --system --ingroup app app

# Install runtime dependencies for Python and OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    libgl1 \
    libglib2.0-0 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    libpython3.12-stdlib \
    # Clean up apt cache and remove software-properties-common after it's used
    && apt-get purge -y --auto-remove software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Set the PATH to use the virtual environment's executables
ENV PATH="/opt/venv/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy the application source code
# Ensure all necessary python files are in the context
COPY --chown=app:app class_names.py .
COPY --chown=app:app utils.py .
COPY --chown=app:app worker.py .

USER app

# Set the command to run the worker
CMD ["python", "worker.py"]