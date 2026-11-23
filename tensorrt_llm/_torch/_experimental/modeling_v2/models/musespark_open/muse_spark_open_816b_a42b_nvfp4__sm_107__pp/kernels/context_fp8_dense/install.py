# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``context_fp8_dense`` on a stage: the stage's dense projections of large context calls on MXFP8.

The TEP target's lever at the stage's full shapes (a lossy lever: on only when its knob sets
it). Every projection ``PROJECTIONS`` names on every layer of this stage keeps its BF16
weight and gets the TEP target's ``Fp8Projection`` copy (MXFP8, E4M3 values with one UE8M0
scale per 1x32 block); its quant method becomes the TEP target's ``ContextFp8LinearMethod``,
so a call of at least ``min_rows`` rows quantizes its BF16 activation and runs the Rubin
CuTe DSL MXFP8 GEMM, and every other call runs the BF16 GEMM. At TP1 nothing is gathered:
the attention input is quantized in place (there is no sequence-parallel all-gather).

Tactics. The TEP target's table (``TACTICS``) lists its TP4 weight shapes: a TP1 QKV
``(18432, 8192)`` and o_proj ``(8192, 8192)`` would get the runner's default tactic, which
a one-GPU sweep measured slower than the tactic below.
``TP1_TACTICS`` gives them the tactic that sweep measured fastest at 8192 rows (``_BASE_512_21``,
split-K 1, so every row count gets the same bits as with any other split-K 1 tactic). Only
8192-row calls were screened; smaller chunks run the same tactic until a sweep fills the
smaller row buckets. Pool and unpool keep their shapes at TP1 (they are not TP-sliced), so
they keep the TEP table's entries.

Install after the stage's weights are final and before the KV cache is sized and any CUDA
graph is captured. Agreed over the stages (``dispatch.agreed_install``): every stage checks
its projections and builds and compiles the copies, the outcome is agreed, and only then are
the quant methods swapped.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from tensorrt_llm.logger import logger

from ... import dispatch as _dispatch
from ..._tep import module as _tep_module

#: The log name of this path.
NAME = "context_fp8_dense"
#: The projections a copy can be installed on (the lever's ``PROJECTIONS`` choices).
PROJECTIONS = ("qkv", "o_proj", "pool", "unpool")
#: The SM version this path is written for: the GEMM is the Rubin (sm107) CuTe DSL one.
SERVED_SM = 107

#: The TEP target's ``_BASE_512_21`` (``kernels/context_fp8_dense/install.py``).
_BASE_512_21 = ("base", (512, 256, 128), (256, 256, 64), (2, 1), False, False, "static", "m", 1)

#: ``(out_features, in_features) -> ((fewest rows, tactic), ...)`` of the TP1 projection
#: shapes the TEP table does not list: the QKV with its output gate and the o_proj at all
#: 128 heads. The fastest tactic of the 8192-row sweep on one Rubin GPU (alternating
#: equal-work pairs, sustained and burst regimes).
TP1_TACTICS: Dict[Tuple[int, int], Tuple[Tuple[int, Any], ...]] = {
    (18432, 8192): ((0, _BASE_512_21),),  # QKV + output gate, 128 / 16 heads
    (8192, 8192): ((0, _BASE_512_21),),  # o_proj, 128 heads
}


def tactics_for(shape: Tuple[int, int], tep_tactics: Dict) -> Tuple[Tuple[int, Any], ...]:
    """The tactic table of a weight ``shape``: ``TP1_TACTICS``, else the TEP table's entry,
    else the runner's default tactic."""
    if shape in TP1_TACTICS:
        return TP1_TACTICS[shape]
    return tuple(tep_tactics.get(shape, ((0, -1),)))


def _modules_of(layer) -> Dict[str, Any]:
    attention, moe = layer.self_attn, layer.feed_forward
    return {
        "qkv": attention.qkv_proj,
        "o_proj": attention.o_proj,
        "pool": getattr(moe, "pooler", None),
        "unpool": getattr(moe, "unpooler", None),
    }


def install(
    text_model,
    mapping,
    compute_capability: Optional[Tuple[int, int]],
    min_rows: int,
    projections: Sequence[str] = PROJECTIONS,
) -> List[str]:
    """Give ``projections`` of every stage layer an MXFP8 copy; returns the projections installed.

    Collective over the stages (the outcome is agreed before any quant method is swapped):
    raises ValueError on every stage alike when some stage cannot serve the request (not
    sm107, no Rubin CuTe DSL GEMM, a projection that is not a bias-free BF16
    ``UnquantizedLinearMethod`` Linear, or a failing first call).
    """
    sm = _dispatch.sm_of(compute_capability)

    def build():
        from tensorrt_llm._torch.modules.linear import Linear, UnquantizedLinearMethod

        tep = _tep_module("kernels.context_fp8_dense.install")
        if sm != SERVED_SM:
            raise ValueError(f"{_dispatch.TAG} {NAME}: written for sm{SERVED_SM}, got sm{sm}")
        from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE

        if not IS_CUTLASS_DSL_RUBIN_AVAILABLE:
            raise ValueError(f"{_dispatch.TAG} {NAME}: the Rubin CuTe DSL package is not available")
        unknown = sorted(set(projections) - set(PROJECTIONS))
        if unknown or min_rows < 1:
            raise ValueError(
                f"{_dispatch.TAG} {NAME}: projections {list(projections)} of {list(PROJECTIONS)} "
                f"from {min_rows} rows"
            )
        names = [name for name in PROJECTIONS if name in projections]
        layers = _dispatch.owned_layers(text_model)
        sites = []
        for layer in layers:
            modules = _modules_of(layer)
            for name in names:
                module = modules[name]
                if (
                    not isinstance(module, Linear)
                    or type(module.quant_method) is not UnquantizedLinearMethod
                ):
                    method = type(getattr(module, "quant_method", None)).__name__
                    raise ValueError(
                        f"{_dispatch.TAG} {NAME}: layer {layer.layer_idx} {name} is not a built-in "
                        f"unquantized Linear ({method})"
                    )
                if module.bias is not None:
                    raise ValueError(
                        f"{_dispatch.TAG} {NAME}: layer {layer.layer_idx} {name} has a bias"
                    )
                sites.append(module)
        added = 0
        try:
            copies = []
            for module in sites:
                copy = tep.Fp8Projection(module.weight.data, min_rows)
                copy.tactics = tactics_for((copy.out_features, copy.in_features), tep.TACTICS)
                copies.append((module, copy))
                added += copy.nbytes()
            # Compile every listed tactic of every weight shape before any CUDA graph capture.
            seen = set()
            for module, copy in copies:
                key = (copy.out_features, copy.in_features)
                if key in seen:
                    continue
                seen.add(key)
                for fewest, tactic in copy.tactics:
                    rows = max(fewest, min_rows)
                    probe = torch.zeros(
                        (rows, copy.in_features), dtype=torch.bfloat16, device=module.weight.device
                    )
                    copy(probe, tactic)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception as error:  # noqa: BLE001 - the failure must reach the agreement
            raise RuntimeError(
                f"{_dispatch.TAG} {NAME}: could not build the MXFP8 copies: {error!r}"
            ) from error
        return tep, names, len(layers), copies, added

    tep, names, num_layers, copies, added = _dispatch.agreed_install(
        mapping, build, (ValueError, RuntimeError)
    )
    for module, copy in copies:
        fp8_method = tep.ContextFp8LinearMethod.__new__(tep.ContextFp8LinearMethod)
        fp8_method.__dict__.update(module.quant_method.__dict__)
        module.quant_method = fp8_method
        module._context_fp8 = copy
    shapes = sorted({(copy.out_features, copy.in_features) for _, copy in copies})
    logger.info(
        f"{_dispatch.TAG} {NAME}: {names} of the {num_layers} layers of stage "
        f"{text_model._pp_stage} run their calls of >= {min_rows} rows on the Rubin MXFP8 "
        f"GEMM (weight shapes {shapes}, TP1 tactics for {sorted(TP1_TACTICS)}); the copies "
        f"add {added / 2**30:.2f} GiB on this stage"
    )
    return names
