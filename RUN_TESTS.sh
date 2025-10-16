#!/bin/bash

# Quick Test Script for APE-Data
# Run this to verify everything works before cloud GPU training

set -e  # Exit on any error

echo "╔══════════════════════════════════════════════════════════╗"
echo "║     APE-Data Quick Test - Step by Step                  ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check if in correct directory
if [ ! -f "requirements.txt" ]; then
    echo "✗ Error: Please run this script from the project root directory"
    echo "  cd /Users/kuntalkokate/Desktop/LLM_agents_projects/LLM_agent_v2v"
    exit 1
fi

# Check if data exists
DATA_DIR="/Users/kuntalkokate/Desktop/LLM_agents_projects/dataset"
if [ ! -d "$DATA_DIR" ]; then
    echo "✗ Error: Data directory not found: $DATA_DIR"
    echo "  Please download APE-data samples first"
    exit 1
fi

echo "Step 1: Checking Python environment..."
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    echo "✓ Found: $PYTHON_VERSION"
else
    echo "✗ Python not found. Please install Python 3.8+"
    exit 1
fi

echo ""
echo "Step 2: Installing dependencies..."
echo "(This may take a few minutes on first run)"
pip install -q -r requirements.txt
echo "✓ Dependencies installed"

echo ""
echo "Step 3: Running APE-Data loading tests..."
echo "------------------------------------------------------------"
python test_ape_data_loading.py

if [ $? -eq 0 ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║                   🎉 ALL TESTS PASSED! 🎉                ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""
    echo "Your setup is working correctly!"
    echo ""
    echo "Next steps:"
    echo "1. Review QUICKSTART_APE_DATA.md for detailed testing guide"
    echo "2. Run mini training test: python test_train_ape.py (optional)"
    echo "3. When ready for cloud GPU, see CLOUD_GPU_TRAINING_APE.md"
    echo ""
else
    echo ""
    echo "✗ Tests failed. Please check the errors above."
    echo "  See QUICKSTART_APE_DATA.md for troubleshooting"
    exit 1
fi
