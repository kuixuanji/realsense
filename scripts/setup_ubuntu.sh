#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_name="realsense"
aliyun_pypi="https://mirrors.aliyun.com/pypi/simple/"
aliyun_torch="https://mirrors.aliyun.com/pytorch-wheels/cu121"

cd "${project_dir}"

if conda env list | awk '{print $1}' | grep -qx "${env_name}"; then
    conda env update -n "${env_name}" -f environment.yml
else
    conda env create -f environment.yml
fi

conda run -n "${env_name}" python -m pip install \
    "${aliyun_torch}/torch-2.4.1%2Bcu121-cp311-cp311-linux_x86_64.whl" \
    "${aliyun_torch}/torchvision-0.19.1%2Bcu121-cp311-cp311-linux_x86_64.whl" \
    --index-url "${aliyun_pypi}"

conda run -n "${env_name}" python -m pip install -r requirements.txt \
    --index-url "${aliyun_pypi}"

conda run -n "${env_name}" python -c \
    'import torch; device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"; print(f"PyTorch {torch.__version__}; CUDA={torch.cuda.is_available()}; device={device}")'

echo "安装完成。运行：conda activate ${env_name}"
