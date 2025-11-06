FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Prevent apt-get from asking for user input
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including Python, pip, and libraries for OpenCV
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    git -y

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Specify the command to run on container startup
# This will execute our main Python script
CMD ["python3", "main.py"]