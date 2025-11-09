#!/bin/bash
# Quick setup script for LOCAL training

set -e  # Exit on error

echo "=========================================="
echo "LiDAR Sim2Real - Local Setup"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Check Python
echo ""
echo "Step 1: Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓${NC} Python found: $PYTHON_VERSION"
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version)
    echo -e "${GREEN}✓${NC} Python found: $PYTHON_VERSION"
    PYTHON_CMD=python
else
    echo -e "${RED}✗${NC} Python not found. Please install Python 3.10+"
    exit 1
fi

# Step 2: Install dependencies
echo ""
echo "Step 2: Installing dependencies..."
$PYTHON_CMD -m pip install -r requirements.txt
echo -e "${GREEN}✓${NC} Dependencies installed"

# Step 3: Test installation
echo ""
echo "Step 3: Testing installation..."
$PYTHON_CMD test_installation.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Installation test passed!"
else
    echo -e "${RED}✗${NC} Installation test failed. Check errors above."
    exit 1
fi

# Step 4: Create directories
echo ""
echo "Step 4: Creating directories..."
mkdir -p data/raw/SemanticKITTI
mkdir -p data/raw/SynLiDAR
mkdir -p data/processed
mkdir -p outputs
echo -e "${GREEN}✓${NC} Directories created"

# Step 5: Check for data
echo ""
echo "Step 5: Checking for datasets..."

if [ -d "data/raw/SemanticKITTI/sequences" ]; then
    NUM_SEQS=$(ls -d data/raw/SemanticKITTI/sequences/*/ 2>/dev/null | wc -l)
    echo -e "${GREEN}✓${NC} SemanticKITTI found ($NUM_SEQS sequences)"
    SEMANTIC_KITTI_READY=1
else
    echo -e "${YELLOW}⚠${NC} SemanticKITTI not found"
    echo "  Download from: http://semantic-kitti.org/"
    echo "  Extract to: data/raw/SemanticKITTI/"
    SEMANTIC_KITTI_READY=0
fi

if [ -d "data/raw/SynLiDAR" ] && [ "$(ls -A data/raw/SynLiDAR 2>/dev/null)" ]; then
    echo -e "${GREEN}✓${NC} SynLiDAR found"
    SYNLIDAR_READY=1
else
    echo -e "${YELLOW}⚠${NC} SynLiDAR not found"
    echo "  Download from: https://github.com/xiaoaoran/SynLiDAR"
    echo "  Extract to: data/raw/SynLiDAR/"
    SYNLIDAR_READY=0
fi

# Summary
echo ""
echo "=========================================="
echo "Setup Summary"
echo "=========================================="
echo -e "${GREEN}✓${NC} Python installed"
echo -e "${GREEN}✓${NC} Dependencies installed"
echo -e "${GREEN}✓${NC} Installation tests passed"
echo -e "${GREEN}✓${NC} Directories created"

if [ $SEMANTIC_KITTI_READY -eq 1 ] && [ $SYNLIDAR_READY -eq 1 ]; then
    echo -e "${GREEN}✓${NC} Datasets ready"
    echo ""
    echo "=========================================="
    echo "Next Steps:"
    echo "=========================================="
    echo "1. Preprocess data:"
    echo "   ./preprocess_data.sh"
    echo ""
    echo "2. Train model:"
    echo "   $PYTHON_CMD main.py --stage direct --config config.yaml"
else
    echo -e "${YELLOW}⚠${NC} Datasets need to be downloaded"
    echo ""
    echo "=========================================="
    echo "Next Steps:"
    echo "=========================================="
    echo "1. Download SemanticKITTI:"
    echo "   - Go to: http://semantic-kitti.org/"
    echo "   - Download velodyne + labels"
    echo "   - Extract to: data/raw/SemanticKITTI/"
    echo ""
    echo "2. Download SynLiDAR:"
    echo "   - Go to: https://github.com/xiaoaoran/SynLiDAR"
    echo "   - Follow download instructions"
    echo "   - Extract to: data/raw/SynLiDAR/"
    echo ""
    echo "3. Then run: ./preprocess_data.sh"
fi

echo "=========================================="

