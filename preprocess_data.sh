#!/bin/bash
# Preprocess datasets script

set -e

echo "=========================================="
echo "Preprocessing Datasets"
echo "=========================================="

# Detect Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    PYTHON_CMD=python
fi

# Check for raw data
if [ ! -d "data/raw/SemanticKITTI/sequences" ]; then
    echo "Error: SemanticKITTI not found at data/raw/SemanticKITTI/"
    echo "Please download it first from: http://semantic-kitti.org/"
    exit 1
fi

if [ ! -d "data/raw/SynLiDAR" ] || [ ! "$(ls -A data/raw/SynLiDAR 2>/dev/null)" ]; then
    echo "Error: SynLiDAR not found at data/raw/SynLiDAR/"
    echo "Please download it first from: https://github.com/xiaoaoran/SynLiDAR"
    exit 1
fi

# Preprocess SemanticKITTI
echo ""
echo "Step 1: Preprocessing SemanticKITTI..."
echo "This may take 30-60 minutes..."
$PYTHON_CMD scripts/preprocess_semantickitti.py \
  --data_root ./data/raw/SemanticKITTI \
  --output_root ./data/processed \
  --config config.yaml

# Preprocess SynLiDAR
echo ""
echo "Step 2: Preprocessing SynLiDAR..."
echo "This may take 20-40 minutes..."
$PYTHON_CMD scripts/preprocess_synlidar.py \
  --data_root ./data/raw/SynLiDAR \
  --output_root ./data/processed \
  --config config.yaml

# Learn sensor profile
echo ""
echo "Step 3: Learning sensor profile..."
echo "This may take 5-10 minutes..."
$PYTHON_CMD scripts/learn_sensor_profile.py \
  --data_root ./data/processed \
  --output sensor_profile.json \
  --num_samples 500

# Summary
echo ""
echo "=========================================="
echo "Preprocessing Complete!"
echo "=========================================="
echo "✓ SemanticKITTI preprocessed"
echo "✓ SynLiDAR preprocessed"
echo "✓ Sensor profile learned"
echo ""
echo "Output files:"
echo "  - data/processed/semantickitti/"
echo "  - data/processed/synlidar/"
echo "  - sensor_profile.json"
echo "  - sensor_profile_plot.png"
echo ""
echo "=========================================="
echo "Next: Train the model"
echo "=========================================="
echo "Stage A (Direct UNet):"
echo "  $PYTHON_CMD main.py --stage direct --config config.yaml"
echo ""
echo "Monitor training:"
echo "  tensorboard --logdir ./outputs"
echo "=========================================="

