#!/bin/bash
# Setup script for MODAL (cloud) training

set -e

echo "=========================================="
echo "LiDAR Sim2Real - Modal Setup"
echo "=========================================="

# Detect Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    PYTHON_CMD=python
fi

# Step 1: Install Modal
echo ""
echo "Step 1: Installing Modal..."
$PYTHON_CMD -m pip install modal
echo "✓ Modal installed"

# Step 2: Authenticate
echo ""
echo "Step 2: Authenticating with Modal..."
echo "This will open your browser. Follow instructions to authenticate."
read -p "Press Enter to continue..."
modal token new

# Step 3: Create volumes
echo ""
echo "Step 3: Creating Modal volumes..."
modal volume create lidar-data || echo "Volume lidar-data already exists"
modal volume create lidar-checkpoints || echo "Volume lidar-checkpoints already exists"
modal volume create lidar-eval || echo "Volume lidar-eval already exists"
echo "✓ Volumes created"

# Step 4: Check for preprocessed data
echo ""
echo "Step 4: Checking for preprocessed data..."

if [ ! -d "data/processed/semantickitti" ]; then
    echo "Error: Preprocessed SemanticKITTI not found!"
    echo "Please run preprocessing first:"
    echo "  ./preprocess_data.sh"
    echo ""
    echo "Or if you have raw data, run:"
    echo "  ./setup_local.sh"
    echo "  ./preprocess_data.sh"
    exit 1
fi

if [ ! -d "data/processed/synlidar" ]; then
    echo "Error: Preprocessed SynLiDAR not found!"
    echo "Please run preprocessing first:"
    echo "  ./preprocess_data.sh"
    exit 1
fi

# Step 5: Upload data to Modal
echo ""
echo "Step 5: Uploading data to Modal volumes..."
echo "This may take 10-30 minutes depending on your internet speed..."

echo "Uploading SemanticKITTI..."
modal volume put lidar-data data/processed/semantickitti /data/SemanticKITTI

echo "Uploading SynLiDAR..."
modal volume put lidar-data data/processed/synlidar /data/SynLiDAR

if [ -f "sensor_profile.json" ]; then
    echo "Uploading sensor profile..."
    modal volume put lidar-data sensor_profile.json /data/sensor_profile.json
fi

echo "✓ Data uploaded to Modal"

# Summary
echo ""
echo "=========================================="
echo "Modal Setup Complete!"
echo "=========================================="
echo "✓ Modal authenticated"
echo "✓ Volumes created"
echo "✓ Data uploaded"
echo ""
echo "=========================================="
echo "Next: Train on Modal"
echo "=========================================="
echo "Stage A (Direct UNet, ~6 hours on A100):"
echo "  modal run modal_app.py --command train --stage direct"
echo ""
echo "Monitor at: https://modal.com/"
echo ""
echo "After training completes, download checkpoint:"
echo "  modal volume get lidar-checkpoints /checkpoints/direct/checkpoints/best.pt ./outputs/"
echo ""
echo "Then evaluate locally:"
echo "  $PYTHON_CMD scripts/translate_batch.py --checkpoint ./outputs/best.pt --input_dir ./data/processed/synlidar/val --output_dir ./outputs/translated"
echo "=========================================="

