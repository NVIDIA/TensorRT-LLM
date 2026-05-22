# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pre-export quantization for the auto path.

Strategy: walk the Diffusers transformer BEFORE `torch.export`, replace
every ``nn.Linear`` with ``tensorrt_llm._torch.modules.Linear(quant_config=...)``,
load the BF16 weight via TRT-LLM's ``Linear.load_weights`` (which
dynamically quantizes to FP8/NVFP4 + scale at load time, mirroring
``DynamicLinearWeightLoader`` in the handwritten path). The captured graph
then contains real ``torch.ops.tensorrt_llm.*`` quant ops that
``torch.export`` traces cleanly — unlike modelopt's ``TensorQuantizer``
modules whose ``FakeTensor`` state torch.export rejects.
"""

from __future__ import annotations

import fnmatch
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from tensorrt_llm._torch.modules.linear import Linear as TRTLLMLinear
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm._torch.visual_gen.quantization.ops import quantize_nvfp4
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

if TYPE_CHECKING:
    from tensorrt_llm.models.modeling_utils import QuantConfig


def _prequantize_weight_dict(
    weight_dict: dict[str, "torch.Tensor"],
    quant_algo,
) -> dict[str, "torch.Tensor"]:
    """For storage-shape-changing quant algos (NVFP4 packs 2 4-bit values
    into 1 uint8), pre-pack the weight so `Linear.load_weights_vanilla`'s
    `copy_(src)` finds the right destination shape.

    FP8 keeps `(out, in)` shape (just narrower dtype) — PyTorch's `.copy_`
    converts BF16→FP8 inline, so no pre-quant needed.
    NVFP4 changes shape to `(out, in/2)` — pre-pack here.
    """
    if quant_algo != QuantAlgo.NVFP4:
        return weight_dict
    w = weight_dict["weight"]
    if w.device.type != "cuda":
        w = w.cuda()
    qweight, weight_scale, weight_scale_2 = quantize_nvfp4(w)
    return {
        **weight_dict,
        "weight": qweight,
        "weight_scale": weight_scale,
        "weight_scale_2": weight_scale_2,
    }


def _is_excluded(qualified_name: str, patterns: list[str] | None) -> bool:
    """Match against `quant_config.exclude_modules`-style glob patterns.

    Mirrors `QuantConfig.is_module_excluded_from_quantization` semantics:
    a Linear at qualified name `name` is excluded if any pattern matches
    `name` directly OR matches any suffix of the path (so `"*x_embedder*"`
    matches `time_guidance_embed.timestep_embedder.linear_1` via the
    bare `timestep_embedder.linear_1` tail).
    """
    if not patterns:
        return False
    parts = qualified_name.split(".")
    candidates = [qualified_name] + [".".join(parts[i:]) for i in range(1, len(parts))]
    for pat in patterns:
        for cand in candidates:
            if fnmatch.fnmatch(cand, pat):
                return True
    return False


def _tp_kwargs_for(child: nn.Linear, mapping) -> dict:
    """Return TP construction kwargs for a tagged Linear, or `{}` if untagged.

    A Linear with `_tp_role` ∈ {qkv, ff_in} → COLUMN (output-split)
    A Linear with `_tp_role` ∈ {out_proj, ff_out} → ROW (input-split, all-reduce)
    Untagged Linears stay replicated.
    """
    if mapping is None:
        return {}
    role = getattr(child, "_tp_role", None)
    if role is None:
        return {}
    if role in ("qkv", "ff_in"):
        return {"mapping": mapping, "tensor_parallel_mode": TensorParallelMode.COLUMN}
    if role in ("out_proj", "ff_out"):
        return {"mapping": mapping, "tensor_parallel_mode": TensorParallelMode.ROW}
    return {}


def replace_linear_with_trtllm(
    module: nn.Module,
    quant_config: "QuantConfig",
    dtype: torch.dtype,
    _name_prefix: str = "",
    _count: list | None = None,
    mapping=None,
) -> int:
    """Recursively swap every ``nn.Linear`` in `module` with TRT-LLM's
    quant-aware `Linear(quant_config=...)`, copying the BF16 weight and
    letting TRT-LLM's ``Linear.load_weights`` apply dynamic quantization
    (BF16 → FP8 or NVFP4 + scale at load time).

    Skips:
      - ``TRTLLMLinear`` instances (already quantized)
      - Any module whose qualified name matches `quant_config.exclude_modules`
        (glob patterns; defaults to ``[]`` if unset)

    Returns the number of modules replaced.
    """
    if _count is None:
        _count = [0, 0]  # [replaced, excluded]
        # Top-level call: union structural auto-detected patterns into the
        # user's `exclude_modules` so the auto path doesn't need per-family
        # hand-curated lists. See `auto/sensitivity.py` for the heuristic.
        from .sensitivity import find_quantization_sensitive_linears

        auto_patterns = find_quantization_sensitive_linears(module)
        if auto_patterns:
            user_patterns = list(quant_config.exclude_modules or [])
            combined = list(dict.fromkeys(user_patterns + auto_patterns))
            if combined != user_patterns:
                logger.info(
                    f"VisGen-Auto sensitivity: auto-detected {auto_patterns}; "
                    f"final exclude_modules={combined}"
                )
                quant_config.exclude_modules = combined
    exclude_patterns = getattr(quant_config, "exclude_modules", None)
    for child_name, child in list(module.named_children()):
        full_name = f"{_name_prefix}.{child_name}" if _name_prefix else child_name
        if isinstance(child, nn.Linear) and not isinstance(child, TRTLLMLinear):
            if _is_excluded(full_name, exclude_patterns):
                _count[1] += 1
                continue
            tp_kwargs = _tp_kwargs_for(child, mapping)
            new_linear = TRTLLMLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                dtype=dtype,
                quant_config=quant_config,
                skip_create_weights_in_init=False,
                # We're dynamically quantizing from BF16 weights with no
                # calibration data, so activation scales must be computed at
                # runtime. For NVFP4, `Linear.__init__` eagerly creates an
                # `input_scale=Parameter(1.0)`, and the forward path treats a
                # non-None `input_scale` as the *static* branch unless this
                # flag is set — which would produce garbage output without
                # real calibration data.
                force_dynamic_quantization=True,
                **tp_kwargs,
            ).to(child.weight.device)
            wd = {"weight": child.weight.data}
            if child.bias is not None:
                wd["bias"] = child.bias.data
            wd = _prequantize_weight_dict(wd, quant_config.quant_algo)
            new_linear.load_weights([wd])
            setattr(module, child_name, new_linear)
            _count[0] += 1
        else:
            replace_linear_with_trtllm(
                child, quant_config, dtype, full_name, _count, mapping=mapping
            )
    if not _name_prefix and _count[0]:
        msg = (
            f"VisGen-Auto quantize: replaced {_count[0]} nn.Linear modules with "
            f"TRT-LLM Linear(quant_algo={quant_config.quant_algo})"
        )
        if _count[1]:
            msg += f", excluded {_count[1]} via exclude_modules={exclude_patterns}"
        logger.info(msg)
    return _count[0]
