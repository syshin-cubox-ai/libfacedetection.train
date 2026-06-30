#!/bin/bash

#SBATCH --job-name=yunet
#SBATCH --partition=hopper
#SBATCH --exclusive
#SBATCH --gpus=8
#SBATCH -o ../logs/%A.txt
#SBATCH --container-image=./yunet.sqsh
#SBATCH --container-mounts=/purestorage:/purestorage
#SBATCH --container-workdir=/purestorage/AILAB/AI_1/syshin/github_repository/libfacedetection.train
#SBATCH --container-remap-root
#SBATCH --container-writable

unset RANK
unset LOCAL_RANK

python -m yunet_train.cli.train --variant yunet_n \
--epochs 640 --batch-size 16 --workers 2 --device cuda \
--checkpoint-interval 80 --eval-interval 100 \
--work-dir work_dirs/yunet_n --grayscale-prob 0
