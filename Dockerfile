# Video-to-Video Diffusion Model Training Docker Image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    tmux \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements file
COPY requirements.txt /workspace/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /workspace/

# Create directories for outputs
RUN mkdir -p /workspace/outputs /workspace/checkpoints /workspace/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/workspace/.cache/torch
ENV HF_HOME=/workspace/.cache/huggingface

# Expose tensorboard port
EXPOSE 6006

# Default command (can be overridden)
CMD ["/bin/bash"]
