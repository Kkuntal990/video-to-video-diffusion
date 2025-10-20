#!/bin/bash
# Script to build Docker image locally using Colima

set -e

echo "=== Setting up Colima for Docker ==="

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Error: Homebrew is not installed. Install it from https://brew.sh"
    exit 1
fi

# Install Colima if not already installed
if ! command -v colima &> /dev/null; then
    echo "Installing Colima..."
    brew install colima
else
    echo "Colima is already installed"
fi

# Install Docker CLI if not already installed
if ! command -v docker &> /dev/null; then
    echo "Installing Docker CLI..."
    brew install docker
else
    echo "Docker CLI is already installed"
fi

# Check if Colima is running
if ! colima status &> /dev/null; then
    echo "Starting Colima..."
    # Start Colima with 4 CPUs, 8GB RAM, 60GB disk
    colima start --cpu 4 --memory 8 --disk 60
else
    echo "Colima is already running"
fi

# Verify Docker is working
echo ""
echo "=== Verifying Docker setup ==="
docker ps

echo ""
echo "=== Building Docker image ==="
docker build -t kkokate990/v2v-diffusion:latest .

echo ""
echo "=== Build complete! ==="
echo "Image: kkokate990/v2v-diffusion:latest"
echo ""
echo "To push to Docker Hub:"
echo "  docker login"
echo "  docker push kkokate990/v2v-diffusion:latest"
echo ""
echo "To stop Colima when done:"
echo "  colima stop"
