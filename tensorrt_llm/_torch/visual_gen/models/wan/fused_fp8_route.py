# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which Wan fused FP8 paths a deployment runs, set by ``pipeline_config["fused_fp8"]``.

- ``"off"`` (default): the standard WanBlock forward and the configured attention backend.
- ``"auto"``: the fused paths this deployment supports; the rest falls back.
- ``"require"``: both the fused block and fused attention, or a ``ValueError`` saying why not.
  This covers the load-time conditions and attn1's shape; the per-forward checks in
  fused_block.eligible() (weights, biases, temb) still fall back to WanBlock.forward.

The fused block keeps the static-FP8 math of WanBlock.forward. Fused attention replaces
cuDNN per-tensor FP8 attention, so it needs that attention config. Under Ulysses it
overlaps the all-to-all with attention when the group is on one node.
"""

from dataclasses import dataclass
from typing import List, Tuple

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

from ...config import DiffusionModelConfig
from ...modules.wan_fused_fp8 import _ext

FUSED_FP8_KEY = "fused_fp8"
FUSED_FP8_MODES = ("off", "auto", "require")


@dataclass(frozen=True)
class FusedFp8Route:
    """Load-time decision: block -> WanBlock runs fused_block.forward; attention -> attn1 runs
    ops.self_attention."""

    block: bool = False
    attention: bool = False
    # fused_fp8='require': WanBlock raises if attn1 fails ops.supports().
    required: bool = False


def _unsupported(model_config: DiffusionModelConfig, tp_size: int) -> Tuple[List[str], List[str]]:
    """Reasons the fused block and fused attention cannot run here."""
    common = []
    if get_sm_version() != 107:
        common.append(f"SM{get_sm_version()} is not Rubin (SM107)")
    if tp_size != 1:
        common.append(f"tensor parallel size {tp_size} is not 1")
    if not _ext.jit_available():
        common.append("nvcc is not available to build the fused kernels")
    vgm = model_config.visual_gen_mapping
    if model_config.parallel.async_ulysses and vgm is not None and vgm.ulysses_size > 1:
        common.append("parallel.async_ulysses is enabled (SEPARATE_QKV attn1)")

    block = list(common)
    if model_config.quant_config.quant_algo != QuantAlgo.FP8 or model_config.dynamic_weight_quant:
        block.append("the checkpoint is not static per-tensor FP8")

    attention = list(common)
    if not _ext.fmha_available():
        attention.append("the FMHA binding needs a TensorRT-LLM source build")
    attention_config = model_config.attention
    quant_attn = attention_config.quant_attention_config
    if (
        attention_config.backend != "CUDNN"
        or quant_attn is None
        or (quant_attn.qk_dtype, quant_attn.v_dtype) != ("fp8", "fp8")
    ):
        attention.append("attention is not cuDNN with fp8 qk_dtype and v_dtype")
    if attention_config.sparse_attention_config is not None:
        attention.append("sparse attention is configured")
    return block, attention


def route(model_config: DiffusionModelConfig, tp_size: int) -> FusedFp8Route:
    """Parse pipeline_config fused_fp8 and decide which fused paths run.

    Raises ValueError for an unknown mode, or for 'require' when either path is unsupported
    at load time. Per-module checks happen in WanBlock (see the module docstring).
    """
    mode = model_config.extra_attrs.get(FUSED_FP8_KEY, "off")
    if mode not in FUSED_FP8_MODES:
        raise ValueError(
            f"pipeline_config {FUSED_FP8_KEY}={mode!r}; expected one of {FUSED_FP8_MODES}."
        )
    if mode == "off":
        return FusedFp8Route()

    block_reasons, attention_reasons = _unsupported(model_config, tp_size)
    reasons = "; ".join(dict.fromkeys(block_reasons + attention_reasons))
    if mode == "require" and reasons:
        raise ValueError(f"pipeline_config {FUSED_FP8_KEY}='require', but {reasons}")
    decision = FusedFp8Route(
        block=not block_reasons, attention=not attention_reasons, required=mode == "require"
    )
    logger.info(
        f"Wan {FUSED_FP8_KEY}={mode}: fused block {'on' if decision.block else 'off'}, "
        f"fused attention {'on' if decision.attention else 'off'}"
        + (f" ({reasons})" if reasons else "")
    )
    return decision
