#!/bin/bash
# Concept Tokenizer note: adapted from REPA (https://github.com/sihyun-yu/REPA) for REPA-SAE concept guidance experiments.

set -euo pipefail

accelerate launch train.py \
  --report-to="wandb" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-L/2" \
  --enc-type="dinov2-vit-b" \
  --proj-coeff=0.2 \
  --encoder-depth=8 \
  --sae-dim=24576 \
  --sae-topk=128 \
  --output-dir="/data/repa_sae_imagenet" \
  --exp-name="linear-dinov2-b-enc8-sae-L-proj2" \
  --sae-path="/root/REPA/ckpts/dinov2-vit-b_x_norm_patchtokens.pt" \
  --data-dir="/data/repa_imagenet" \
  --batch-size=256 \
  --gradient-accumulation-steps=1

torchrun --nnodes=1 --nproc_per_node=8 generate.py \
  --model SiT-L/2 \
  --num-fid-samples 50000 \
  --ckpt /data/repa_sae_imagenet/linear-dinov2-b-enc8-sae-L-proj2/checkpoints/0400000.pt \
  --sae-path="/root/REPA/ckpts/dinov2-vit-b_x_norm_patchtokens.pt" \
  --exp-name="linear-dinov2-b-enc8-sae-L-proj2" \
  --sample-dir="/data/repa_generations/" \
  --sae-dim=24576 \
  --path-type=linear \
  --encoder-depth=8 \
  --projector-embed-dims=768 \
  --per-proc-batch-size=32 \
  --mode=sde \
  --num-steps=250 \
  --cfg-scale=1.0 \
  --guidance-high=0.7
