#!/bin/bash
# Activation script for the deployment environment
echo "Activating deployment environment..."
source venv_deployment/bin/activate
echo "✅ Deployment environment activated!"
echo "Python version: $(python --version)"
echo "Available packages: tensorflow, keras, onnx-tf, etc."
echo ""
echo "To deactivate, run: deactivate"

