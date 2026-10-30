# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read VDN's released directory layout without executing remote model code."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

VDN_STAGE = "stage-dmd-step-250"


def is_vdn_checkpoint(root: Path) -> bool:
    return (root / VDN_STAGE / "diffusers" / "config.json").is_file()


def read_vdn_config(root: Path, stage: str = VDN_STAGE) -> dict:
    if stage not in (VDN_STAGE, "stage-b-step-2000"):
        raise ValueError("VDN stage must be stage-dmd-step-250 or stage-b-step-2000.")
    with (root / stage / "diffusers" / "config.json").open() as stream:
        config = json.load(stream)
    if config.get("_class_name") != "VDNMiniMaxH3Transformer3DModel":
        raise ValueError("Expected a VDNMiniMaxH3Transformer3DModel checkpoint.")
    return config["vdn"]


def merge_vdn_adapter(weights: dict[str, torch.Tensor], adapter: dict[str, torch.Tensor]) -> None:
    """Merge released unit-scale PEFT pairs before the base loader fuses projections.

    VDN's published default and turbo adapters have alpha == rank, including
    their rank-16 modulation layers. Match upstream's per-adapter BF16 rounding.
    """
    consumed = set()
    for name, a in adapter.items():
        if ".lora_A." not in name:
            continue
        b_name = name.replace(".lora_A.", ".lora_B.")
        if b_name not in adapter:
            raise ValueError(f"Incomplete VDN LoRA pair: {name}")
        target = name.split(".lora_A.")[0].replace(".attn.orig.", ".attn.") + ".weight"
        if target not in weights:
            raise ValueError(f"Unknown VDN LoRA target: {target}")
        b, weight = adapter[b_name], weights[target]
        if (
            a.ndim != 2
            or b.ndim != 2
            or b.shape[1] != a.shape[0]
            or (b.shape[0], a.shape[1]) != weight.shape
        ):
            raise ValueError(f"VDN LoRA shape mismatch: {target}")
        # The upstream model merges into BF16, even for FP32 checkpoint tensors
        # inside the trunk; time embedding and media projections are not adapted.
        weight = weight.to(torch.bfloat16)
        weight.add_((b.float() @ a.float()).to(weight.dtype))
        weights[target] = weight
        consumed.update((name, b_name))
    if consumed != set(adapter):
        raise ValueError(f"Unrecognized VDN adapter tensors: {sorted(set(adapter) - consumed)[:4]}")


def load_vdn_overlay(
    root: Path, stage: str, weights: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    spec = read_vdn_config(root, stage)
    stage_root = (root / stage).resolve()

    def artifact(relative: str) -> Path:
        path = (stage_root / relative).resolve()
        if not path.is_relative_to(stage_root):
            raise ValueError("VDN artifact paths must stay inside the stage directory.")
        return path

    for adapter in spec["adapters"]:
        merge_vdn_adapter(weights, load_file(str(artifact(adapter))))
    branch = load_file(str(artifact(spec["branch"])))
    if any(
        not name.startswith("transformer_blocks.") or ".attn." not in name or ".orig." in name
        for name in branch
    ):
        raise ValueError("VDN linear branch contains unexpected tensor names.")
    weights.update(branch)
    return weights
