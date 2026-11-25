# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pieces of the deepseek_v41 targets' shells that do not depend on the parallel layout: the text model of a
vision-bearing config, the KV pools of the engine's CSA2CacheManager, a row view of raw FP8 checkpoint bytes, and the
FP8 small-M opt-in of the engine-built DSpark draft model."""

import copy

import torch
from torch import nn

from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheRole
from tensorrt_llm._torch.model_config import ModelConfig


def text_model_config(model_config: ModelConfig) -> ModelConfig:
    """The checkpoint carries a vision tower, so the engine hands this target the composite
    config; the text model is nested under `.text_config`. Everything below reads that one."""
    cfg = model_config.pretrained_config
    text = getattr(cfg, "text_config", None)
    # A checkpoint with a vision tower (as the loader reads it: a vision config with layers) has image tokens in its
    # prompts; the Engram hash reads the image token from the text config (see `EngramHash`).
    vision = getattr(cfg, "vision_config", None)
    image = (
        getattr(cfg, "image_token_id", None)
        if getattr(vision, "num_hidden_layers", 0) > 0
        else None
    )
    (cfg if text is None else text)._v41_image_token_id = image
    if text is None:
        return model_config
    # ModelConfig is frozen except for `pretrained_config`; the path rides on the (mutable) text config.
    text._v41_checkpoint_dir = getattr(cfg, "_name_or_path", None)
    text._v41_quantization_config = getattr(
        cfg, "quantization_config", None
    )  # top-level in the checkpoint
    out = copy.copy(model_config)
    out.pretrained_config = text
    return out


class KvPools:
    """The engine's CSA2CacheManager as an allocator: page pools per (layer, role) and the manager's
    device page tables of the step (rewritten by the engine before every step and graph replay)."""

    def __init__(self, kvm, swa_tables):
        self.kvm = kvm
        self.swa_tables = swa_tables
        self._pools = {}

    def pool(self, layer: int, role: str) -> torch.Tensor:
        key = (layer, role)
        if key not in self._pools:
            self._pools[key] = self.kvm.get_buffers(layer, CSA2CacheRole[role])
        return self._pools[key]

    def step_table(self, layer: int, role: str, nreq: int) -> torch.Tensor:
        kvm = self.kvm
        if (
            role == "SWA"
        ):  # the engine converts these into the metadata's buffer (`swa_tables.engine_swa_tables`)
            return self.swa_tables[layer, :nreq]
        return kvm._device_batch_page_tables[
            kvm._batch_page_index[(layer, CSA2CacheRole[role])], :nreq
        ]


class Fp8Rows:
    """Row-sliceable view of a checkpoint tensor stored as raw bytes, reinterpreted as `dtype` (not converted)."""

    def __init__(self, rows, dtype):
        self.rows, self.dtype = rows, dtype

    def __getitem__(self, index):
        return self.rows[index].view(self.dtype)


def opt_into_small_m_gemm(model: nn.Module) -> None:
    """Stock's `DeepseekV4ForCausalLM.__init__`, after the shell is built: every `Linear` (here the engine-built DSpark
    draft model's; this target's own projections are entries) opts into the FP8 small-M kernel race
    (`use_fp8_small_m_gemm`), so its FP8 block-scale weights keep the FP32 128x128 scales (`weight_scale_block128`) the
    draft's shared expert needs for stock's full-TP piece path (`full_tp_shared_expert_forward`)."""
    from tensorrt_llm._torch.modules.linear import Linear

    for module in model.modules():
        if isinstance(module, Linear):
            module.use_fp8_small_m_gemm = True
