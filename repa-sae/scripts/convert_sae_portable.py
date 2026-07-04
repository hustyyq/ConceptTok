#!/usr/bin/env python3
# Concept Tokenizer note: adapted from REPA (https://github.com/sihyun-yu/REPA) for REPA-SAE concept guidance experiments.
"""
Convert SAE checkpoint to portable format.
The original checkpoint has pickle dependencies (e.g. ViTSAERunnerConfig from 'src' module)
that may require adding the original ViTSAE repository to `sys.path`. This script
saves only the
tensor data so it can be loaded from any directory.
"""
import argparse
import os
import sys
import torch
from typing import Optional

def convert_sae_to_portable(
    src_path: str,
    dst_path: str,
    vitsae_root: Optional[str] = None,
):
    """Load SAE from vitsae context and save portable version."""
    if vitsae_root and not os.path.isdir(vitsae_root):
        raise FileNotFoundError(f"vitsae root not found: {vitsae_root}")
    if vitsae_root and vitsae_root not in sys.path:
        sys.path.insert(0, vitsae_root)

    ckpt = torch.load(src_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # REPA SAE expects W_enc, b_enc, b_dec (no W_dec)
    portable = {
        "W_enc": state_dict["W_enc"],
        "b_enc": state_dict["b_enc"],
        "b_dec": state_dict["b_dec"],
    }

    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    torch.save(portable, dst_path)
    print(f"Saved portable SAE to {dst_path}")
    print(f"  W_enc: {portable['W_enc'].shape}, b_enc: {portable['b_enc'].shape}, b_dec: {portable['b_dec'].shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert an SAE checkpoint to portable tensor-only format.")
    parser.add_argument("--src", required=True, help="Source SAE checkpoint path.")
    parser.add_argument("--dst", required=True, help="Destination portable checkpoint path.")
    parser.add_argument(
        "--vitsae-root",
        default=None,
        help="Optional ViTSAE repository root to add to sys.path for pickle-based checkpoints.",
    )
    args = parser.parse_args()
    convert_sae_to_portable(args.src, args.dst, args.vitsae_root)
