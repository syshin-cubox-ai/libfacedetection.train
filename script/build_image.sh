#!/bin/bash

# 값을 확인하세요.
pytorch_version="2.12.1"
cuda_version="12.6"
cudnn_version="9"
workdir="/purestorage/AILAB/AI_1/syshin/github_repository/libfacedetection.train"
wandb_host=""
wandb_key="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"
# -----------------------------------------------------------------------------

if [ -n "${wandb_host}" ]; then
    wandb_host="--host ${wandb_host}"
fi

if [ ! -f "pytorch+pytorch+${pytorch_version}-cuda${cuda_version}-cudnn${cudnn_version}-runtime.sqsh" ]; then
    enroot import docker://pytorch/pytorch:${pytorch_version}-cuda${cuda_version}-cudnn${cudnn_version}-runtime
fi
enroot create -n yunet pytorch+pytorch+${pytorch_version}-cuda${cuda_version}-cudnn${cudnn_version}-runtime.sqsh
enroot start --root --rw --mount /purestorage:/purestorage yunet bash -c "
cd ${workdir} &&
rm -rf /var/lib/apt/lists/* &&
sed -i 's|http://archive.ubuntu.com|http://kr.archive.ubuntu.com|g' /etc/apt/sources.list.d/ubuntu.sources &&
sed -i 's|http://security.ubuntu.com|http://kr.archive.ubuntu.com|g' /etc/apt/sources.list.d/ubuntu.sources &&
apt update &&
apt install build-essential curl nano pkg-config wget -y &&
apt install libgl1 libglib2.0-0t64 -y &&
pip config set global.no-cache-dir false &&
pip install -r requirements.txt --break-system-packages &&
wandb login ${wandb_host} ${wandb_key}
"
enroot export -f yunet
enroot remove -f yunet