#!/bin/bash
# set -e
# export PATH="/kaggle/working/miniconda3/bin:$PATH"
# conda init bash
# conda activate py310

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.41.1
pip install accelerate sentencepiece protobuf
pip install "numpy<2.0"
pip install einops
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"

