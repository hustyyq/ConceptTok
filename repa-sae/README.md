# REPA-SAE

This folder contains the REPA-SAE code used to apply concept guidance to REPA-style SiT training.

## Source

The code is adapted from REPA:

- Paper/project: https://sihyun.me/REPA
- Upstream code: https://github.com/sihyun-yu/REPA

Original upstream notices and license files are kept where present. Source files adapted from REPA include a `Concept Tokenizer note` header.

## Run

The only release run entry is:

```bash
bash run_repa_sae.sh
```

The script trains `SiT-L/2` with DINOv2-B SAE guidance and then runs generation with the resulting checkpoint. Update the data, checkpoint, output, and sample paths in `run_repa_sae.sh` for the target environment before launching.

The DINOv2-B SAE checkpoint and trained REPA-SAE checkpoint are released on [Hugging Face](https://huggingface.co/goodvegetable/ConceptTok). Download them into `checkpoints/` before running training or generation.
