# Concept Tokenizer

This repository introduces **concept guidance** for visual generative modeling. We use sparse concept targets from pretrained vision features to guide representation learning in two settings:

- **Concept Tokenizer**: concept guidance is added to a text-guided 1D visual tokenizer.
- **REPA-SAE**: concept guidance is applied to REPA-style SiT training.

The codebase contains the tokenizer training pipeline, text-to-image via MaskGen, c2i via LlamaGen, and a compact REPA-SAE subproject.

## Setup

```bash
bash install_packages.sh
```

Pre-trained SAE checkpoints and the trained REPA-SAE checkpoint are released on [Hugging Face](https://huggingface.co/goodvegetable/ConceptTok). Download them into `checkpoints/` before running training or evaluation:

- `siglip-vit-base-patch16_12.pt`: SigLIP SAE for Concept-Guided Tokenizer training.
- `dinov2-vit-b_x_norm_patchtokens.pt`: DINOv2-B SAE for Concept-Guided REPA training.
- `REPA/linear-dinov2-b-enc8-sae-L-proj2/0400000.pt`: trained Concept-Guided REPA checkpoint.

## Concept Tokenizer

The base tokenizer is used for c2i evaluation. The T2I tokenizer is used for MaskGen.

Train the base tokenizer without clustering:

```bash
PYTHONPATH=. accelerate launch --num_machines=1 --num_processes=8 \
  scripts/train_textenctitok_lc.py \
  config=configs/training/ConceptTok/textenctok_lc_bl128_vq.yaml
```

Train and evaluate c2i GPT:

```bash
export VQ_CKPT=/path/to/concept-tokenizer-checkpoint
export IMGNET_ROOT=/path/to/imagenet
export GT_NPZ_PATH=/path/to/VIRTUAL_imagenet256_labeled.npz

PYTHONPATH=. bash scripts/train_c2i_and_eval_concepttok_bl128.sh
```


Train the T2I tokenizer with clustering:

```bash
PYTHONPATH=. accelerate launch --num_machines=1 --num_processes=8 \
  scripts/train_textenctitok_lc.py \
  config=configs/training/ConceptTok/textenctok_lc_bl128_vq_t2i_entropy.yaml
```

Train MaskGen with the T2I tokenizer:

```bash
PYTHONPATH=. accelerate launch --num_machines=1 --num_processes=8 \
  scripts/train_maskgen_textenctitok_lc.py \
  config=configs/training/MaskGen/maskgen_textenc_lc_bl128_t2i_vq_en_l_stage1.yaml
```

## REPA-SAE

The `repa-sae/` folder contains the REPA-SAE code for applying concept guidance to REPA. The release entry script is:

```bash
cd repa-sae
bash run_repa_sae.sh
```

Update the data, checkpoint, output, and sample paths in `repa-sae/run_repa_sae.sh` for your environment before launching.

## Attribution

This repository adapts code from:

- 1d-tokenizer / TiTok: https://github.com/bytedance/1d-tokenizer
- GigaTok: https://silentview.github.io/GigaTok/
- REPA: https://github.com/sihyun-yu/REPA

Copied or adapted source files include `Concept Tokenizer note` headers while preserving upstream notices where present.

We thank the authors of the upstream projects for releasing their code.

## License

See `LICENSE` for this repository and `repa-sae/LICENSE` for the copied REPA-SAE subproject.
