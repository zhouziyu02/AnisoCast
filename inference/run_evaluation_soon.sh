#!/bin/bash

echo "Starting SOON model evaluation..."

# Check Python environment
echo "Checking Python environment..."
python3 --version

# Check if checkpoint file exists
CHECKPOINT_PATH="${1:-./logs/soon/checkpoints/best.ckpt}"  # Use first argument or default path
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT_PATH"
    echo "Please provide the correct checkpoint path as the first argument"
    exit 1
fi

echo "Checkpoint file found: $CHECKPOINT_PATH"

# Run evaluation
echo "Running SOON model evaluation..."
python3 inference/evaluate_soon.py \
    --checkpoint_path "$CHECKPOINT_PATH"
    # --save_predictions \
    # --calculate_metrics

echo "Evaluation completed"
echo "Results saved to: ./results/soon/"
echo "Metrics included: RMSE, Bias, ACC, MS-SSIM, SpectralDiv, SpectralRes"