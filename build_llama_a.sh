#!/bin/bash
set -e

git clone https://github.com/white-and-dark/mykaggle.git

wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-0

# 1. 安装 Miniconda
echo "安装 Miniconda..."
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p /root/miniconda3

/root/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
/root/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

/root/miniconda3/bin/conda --version

# 2. 配置环境变量
echo "配置环境变量..."
export PATH="/root/miniconda3/bin:$PATH"
echo "export PATH=/root/miniconda3/bin:\$PATH" >> ~/.bashrc

ln -sf /usr/local/nvidia/lib64/libcuda.so.1 /usr/lib64/libcuda.so

# 4. 创建 Python 3.10.20 环境
echo "创建环境 py310 (Python 3.10.20)..."
conda create -n py310 python=3.10.20 -y

conda init bash
conda activate py310
python --version
