# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Set PYTHONPATH to the working directory. This allows imports like 'from src.model import ...'
ENV PYTHONPATH="/app"

# Copy only the necessary files for dependency installation
COPY deployment/fastapi_service/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the source code, preserving the 'src' directory structure
COPY src /app/src

# Copy the FastAPI service code into the working directory
COPY deployment/fastapi_service/ .

# Make port 8000 available
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]