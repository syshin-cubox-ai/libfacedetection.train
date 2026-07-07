#!/bin/bash

#SBATCH --job-name=yunet-train
#SBATCH --partition=hopper
#SBATCH --exclusive
#SBATCH --gpus=8
#SBATCH -o ../logs/%A.txt
#SBATCH --container-image=./yunet.sqsh
#SBATCH --container-mounts=/purestorage:/purestorage
#SBATCH --container-workdir=/purestorage/AILAB/AI_1/syshin/github_repository/libfacedetection.train
#SBATCH --container-remap-root
#SBATCH --container-writable

# SLURM/pyxis가 주입한 값이 남아 있으면 _init_distributed가 오동작하므로 제거.
# torchrun이 프로세스별로 WORLD_SIZE/RANK/LOCAL_RANK를 다시 설정한다.
unset RANK
unset LOCAL_RANK
unset WORLD_SIZE

NUM_GPUS=${SLURM_GPUS_ON_NODE:-8}
RUN_NAME=$(date +%y%m%d)_yunet26_n_upbatch_lr

echo "job=${SLURM_JOB_ID} node=$(hostname) gpus=${NUM_GPUS}"

# 유효 배치 128 (16 x 8 GPU):
#   - lr 0.02: 유효 배치 8배 증가분을 일부 보상 (0.03은 warmup 직후 발산 -> NaN 확인됨,
#     MuSGD는 배치 크기에 덜 민감하므로 full linear scaling 0.08 대신 보수적으로 2배)
#   - grad-clip-norm 10: 초반 큰 obj gradient로 인한 급발진 방어
torchrun \
    --nnodes 1 \
    --nproc_per_node "${NUM_GPUS}" \
    --rdzv-backend c10d \
    --rdzv-endpoint localhost:0 \
    -m yunet_train.cli.train \
    --variant yunet_n \
    --image-size 640 \
    --epochs 640 \
    --batch-size 16 \
    --workers 8 \
    --lr 0.02 \
    --warmup-iters 500 \
    --grad-clip-norm 10 \
    --optimizer musgd \
    --use-rle \
    --device cuda \
    --eval-interval 1 \
    --work-dir "work_dirs/${RUN_NAME}" \
    --wandb-project yunet-train \
    --wandb-run-name "${RUN_NAME}"
