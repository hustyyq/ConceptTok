#!/bin/bash

## python3.12.12
echo "Starting to install all pip packages..."

# Upgrade pip
pip install --upgrade pip

# Install PyTorch 2.5.1 + CUDA 12.4
echo "Installing PyTorch 2.5.1 + CUDA 12.4..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install all other packages
echo "Installing other packages..."
pip install -r requirements.txt 
echo "Installation completed!"

# Verify installation
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

echo "Verifying other key packages..."
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import accelerate; print(f'Accelerate version: {accelerate.__version__}')"
python -c "import diffusers; print(f'Diffusers version: {diffusers.__version__}')"
python -c 'import xformers.ops; import xformers; print(f"xformers OK: {xformers.__version__}")'
