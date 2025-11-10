#!/bin/bash
# Activation script for the training environment
echo "Activating training environment..."
source venv_training/bin/activate
echo "✅ Training environment activated!"
echo "Python version: $(python --version)"
echo "Available packages: torch, gymnasium, stable-baselines3, onnx, etc."
echo ""
echo "To deactivate, run: deactivate"
