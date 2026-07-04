#!/bin/bash
# Concept Tokenizer note: modified from GigaTok c2i training/evaluation scripts.

set -euo pipefail

TORCH_RUN_PATH=${TORCH_RUN_PATH:-torchrun}
EVAL_PYTHON_PATH=${EVAL_PYTHON_PATH:-python}

TOK_NAME=${TOK_NAME:-textenctok_lc_bl128_vq}
TOK_CONFIG=${TOK_CONFIG:-configs/training/ConceptTok/textenctok_lc_bl128_vq.yaml}
VQ_CKPT=${VQ_CKPT:?Set VQ_CKPT to the tokenizer checkpoint directory or pytorch_model.bin}

WANDB_PROJECT=${WANDB_PROJECT:-concept-tokenizer-gpt-b}
PROJECT_ROOT=${PROJECT_ROOT:-outputs/c2i}
IMGNET_ROOT=${IMGNET_ROOT:?Set IMGNET_ROOT to an ImageNet ImageFolder root containing train/ and val/}
GT_NPZ_PATH=${GT_NPZ_PATH:?Set GT_NPZ_PATH to VIRTUAL_imagenet256_labeled.npz for gFID evaluation}

GPT_2D=${GPT_2D:-False}
LM_EXP_DIR=${LM_EXP_DIR:-GPT_B256_VQ_BL_e300}
LM_EPOCH=${LM_EPOCH:-300}
LM_BSZ=${LM_BSZ:-256}
FRACT_DECAY=${FRACT_DECAY:-0.2}
GPT_MODEL=${GPT_MODEL:-GPT-B}
WARM_ITER=${WARM_ITER:-5000}
LR=${LR:-1e-4}
PRECISION=${PRECISION:-fp16}
CFG_SCHEDULE=${CFG_SCHEDULE:-rectangular}
CFG_START_RATIO=${CFG_START_RATIO:-0.18}
EVAL_BATCH_PER_GPU=${EVAL_BATCH_PER_GPU:-32}
DATASET=${DATASET:-imagenet}
CKPT_EVERY=${CKPT_EVERY:-5000}
CODEPATH=${CODEPATH:-None}
DATAPATH=${DATAPATH:-${IMGNET_ROOT}/train}
USE_QK_NORM=${USE_QK_NORM:-False}
USE_ADALN=${USE_ADALN:-False}
USE_FLASH_ATTN_FLAG=${USE_FLASH_ATTN_FLAG:---flash-attn}

QK_NORM_FLAG=""
if [[ "${USE_QK_NORM}" == "True" ]]; then
    QK_NORM_FLAG="--qk-norm"
fi

GPT_2D_FLAG=""
if [[ "${GPT_2D}" != "False" ]]; then
    GPT_2D_FLAG="--gpt-2d"
fi

USE_ADALN_FLAG=""
if [[ "${USE_ADALN}" != "False" ]]; then
    USE_ADALN_FLAG="--adaLN"
fi

BSZ_FACTOR=${BSZ_FACTOR:-5000}
EPOCH_TO_ITER_FACTOR=$((256 * BSZ_FACTOR / LM_BSZ))
LM_ITER=$((LM_EPOCH * EPOCH_TO_ITER_FACTOR))
LM_STOP_ITER=${LM_STOP_ITER:-$LM_ITER}

printf -v LM_ITER "%07d" "$LM_ITER"
printf -v LM_STOP_ITER "%07d" "$LM_STOP_ITER"

bash scripts/train_c2i.sh   --save-path "${PROJECT_ROOT}/${TOK_NAME}/gpt"   --data-path "${DATAPATH}"   --code-path "${CODEPATH}"   --dataset "${DATASET}"   --image-size 256   --tok-config "${TOK_CONFIG}"   --mixed-precision "${PRECISION}"   --gpt-model "${GPT_MODEL}"   --vq-ckpt "${VQ_CKPT}"   --sub-exp-dir "${LM_EXP_DIR}"   --lr-scheduler wsd   --warmup "${WARM_ITER}"   --lr "${LR}"   --ckpt-every "${CKPT_EVERY}"   --global-batch-size "${LM_BSZ}"   --fract-decay "${FRACT_DECAY}"   --iterations "${LM_ITER}"   --early-stop-iter "${LM_STOP_ITER}"   --wandb-project "${WANDB_PROJECT}"   --no-compile   ${QK_NORM_FLAG}   ${GPT_2D_FLAG}   ${USE_ADALN_FLAG}

if [[ "${LM_STOP_ITER}" < "${LM_ITER}" ]]; then
    GPT_CKPT="${PROJECT_ROOT}/${TOK_NAME}/gpt/${LM_EXP_DIR}/checkpoints/${LM_STOP_ITER}.pt"
    LM_ITER="${LM_STOP_ITER}"
else
    GPT_CKPT="${PROJECT_ROOT}/${TOK_NAME}/gpt/${LM_EXP_DIR}/cd_records/cd_fract_${FRACT_DECAY}_to_${LM_ITER}/checkpoints/${LM_ITER}.pt"
fi

bash scripts/sample_c2i_search_cfg.sh   --search   --quant-way vq   --image-size 256   --sample-dir "${PROJECT_ROOT}/${TOK_NAME}/gpt/quan_eval/${LM_EXP_DIR}/${LM_ITER}"   --vq-ckpt "${VQ_CKPT}"   --tok-config "${TOK_CONFIG}"   --gpt-model "${GPT_MODEL}"   --cfg-schedule "${CFG_SCHEDULE}"   --step-start-ratio "${CFG_START_RATIO}"   --gpt-ckpt "${GPT_CKPT}"   --per-proc-batch-size "${EVAL_BATCH_PER_GPU}"   --precision "${PRECISION}"   --clear-cache   --eval-python-path "${EVAL_PYTHON_PATH}"   --gt-npz-path "${GT_NPZ_PATH}"   ${QK_NORM_FLAG}   ${GPT_2D_FLAG}   ${USE_ADALN_FLAG}   ${USE_FLASH_ATTN_FLAG}
