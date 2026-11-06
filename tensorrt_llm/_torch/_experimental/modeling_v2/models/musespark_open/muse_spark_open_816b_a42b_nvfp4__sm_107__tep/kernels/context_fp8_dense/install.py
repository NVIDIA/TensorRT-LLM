# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``context_fp8_dense``: the dense projections of large context calls on the Rubin MXFP8 GEMM (sm107).

A lossy lever (``levers.RULES``: on only when its knob sets it): MXFP8, E4M3 values with
one UE8M0 scale per 1x32 block of weights and of activations. The plan gives ``install``
its ``MIN_ROWS`` and ``PROJECTIONS`` parameters.

Every decoder layer's QKV projection, output projection, pooler and unpooler (the BF16
``Linear`` modules ``PROJECTIONS`` names) keeps its BF16 weight, which every decode path
and every small call still reads, and gets an MXFP8 copy of it. A call through the
module's quant method (``Linear.forward``) with at least ``min_rows`` rows -- the context
steps: the sequence-parallel arm's gathered attention rows and its shard of the MoE rows,
and the replicated forward of a large context step -- quantizes its BF16 activation
(``quant.mxfp8_quantize``, one pass at HBM speed; the same bits as
``trtllm::mxfp8_quantize``) and runs the Rubin CuTe DSL MXFP8 GEMM
(``CuteDSLMXFP8RubinLinear``, the runner behind ``trtllm::cute_dsl_mxfp8_gemm_rubin``) at
the tactic ``TACTICS`` lists for its weight shape and rows; every other call runs the BF16
GEMM unchanged. The tactics are fixed, so no server start compiles or profiles the
runner's few hundred tactics; ``install`` compiles the listed ones before any capture.
Decode steps (at most ``max_batch_size * (draft_len + 1)`` rows, far below ``min_rows``)
and every kernel that reads ``module.weight`` directly are unchanged.

The quant method stays an ``UnquantizedLinearMethod`` (a subclass), so every check that
gates a fast path on a BF16 projection still passes and still reads the BF16 weight. The
copy-engine gather of the attention input (``context_sequence_parallel.ce_gather``)
multiplies the QKV row blocks with that BF16 weight itself; where it is wired
(``SHARD_PARTS``) the QKV projection keeps BF16 on every call, so a step never mixes
the two, and the install says so.

With ``context_fp8_attn_gather`` (``gather``) the sequence-parallel arm's attention
input is quantized on the shard, before its all-gather (``Fp8AttentionInputGather``, the
decoder layer's ``_sp_attn_input_gather`` hook): the gather moves E4M3 bytes and row-major
scales instead of BF16, and the QKV projection consumes the gathered ``GatheredMxfp8``
rows. Every block of 32 lies in one row, so the values and scales are the bytes the
quantization after the gather produces.

Numerics: lossy (MXFP8 activations and weights on the context calls); gated by accuracy
runs, never on by default. The copies cost 1.03 bytes per weight element on top of the
BF16 weights (``install`` logs the total), which the KV cache sizing after the load sees.

Install after the weights are final and before the KV cache is sized and any CUDA graph
is captured. Agreed over the TP group (``agreed_install``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch.modules.linear import Linear, UnquantizedLinearMethod
from tensorrt_llm.logger import logger

from ... import dispatch as _tp_agreement
from . import quant as _quant

#: The log name of this path.
NAME = "context_fp8_dense"
#: The projections a copy can be installed on (the lever's ``PROJECTIONS`` choices).
PROJECTIONS = ("qkv", "o_proj", "pool", "unpool")
#: The SM version this path is written for: the GEMM is the Rubin (sm107) CuTe DSL one.
SERVED_SM = 107
#: The fewest rows of a call that runs FP8 by default (the lever's ``MIN_ROWS`` default).
DEFAULT_MIN_ROWS = 1024

_BASE_256_21_CLC = (
    "base",
    (256, 256, 128),
    (256, 256, 64),
    (2, 1),
    False,
    False,
    "clc_dynamic",
    "m",
    1,
)
_BASE_256_22_CLC = (
    "base",
    (256, 256, 128),
    (256, 256, 64),
    (2, 2),
    False,
    False,
    "clc_dynamic",
    "m",
    1,
)
_BASE_256_42 = ("base", (256, 256, 128), (256, 256, 64), (4, 2), False, False, "static", "m", 1)
_BASE_512_21 = ("base", (512, 256, 128), (256, 256, 64), (2, 1), False, False, "static", "m", 1)
_BASE_512_22_N = ("base", (512, 256, 128), (256, 256, 64), (2, 2), False, False, "static", "n", 1)
_MIXED_256 = (
    "mixed_clusters",
    (256, 256, 128),
    (256, 256, 64),
    (4, 2),
    (2, 1),
    False,
    False,
    "static",
    "m",
)
_MIXED_512 = (
    "mixed_clusters",
    (512, 256, 128),
    (256, 256, 64),
    (4, 2),
    (2, 1),
    False,
    False,
    "static",
    "m",
)
_BASE_512_21_CLC_N = (
    "base",
    (512, 256, 128),
    (256, 256, 64),
    (2, 1),
    False,
    False,
    "clc_dynamic",
    "n",
    1,
)
_BASE_512X192_21 = ("base", (512, 192, 128), (256, 192, 64), (2, 1), False, False, "static", "m", 1)
_BASE_512X192_21_CLC_N = (
    "base",
    (512, 192, 128),
    (256, 192, 64),
    (2, 1),
    False,
    False,
    "clc_dynamic",
    "n",
    1,
)

#: ``(out_features, in_features) -> ((fewest rows, tactic), ...)``, rows ascending: a call of M
#: rows runs the tactic of the last entry with ``fewest rows <= M``. The fastest non-swapped
#: tactic per row count of a sweep on one Rubin GPU under CUDA-graph replay (this model's TP4 shapes).
#: Between 1024 and 2048 rows (the sequence-parallel shard of pool / unpool) and between 4096
#: and 8192 rows (QKV) the buckets follow the M-tile counts: an entry starting at
#: ``256 * j + 1`` covers the rows with one more 256-row tile than ``256 * j``. Unpool keeps the
#: static ``_BASE_256_42`` from 1537 rows: its CLC twin, faster alone, is slower in serving at
#: 2048 rows. Every tactic here is split-K 1 and gives the same bits as every other at the same
#: shape.
TACTICS: Dict[Tuple[int, int], Tuple[Tuple[int, Tuple[Any, ...]], ...]] = {
    (4608, 8192): (  # QKV
        (0, _BASE_256_22_CLC),
        (1536, _BASE_512_21),
        (2048, _BASE_512_22_N),
        (3072, _MIXED_256),
        (4097, _BASE_512_22_N),
        (5121, _BASE_512_21_CLC_N),
        (5633, _BASE_512X192_21_CLC_N),
        (6657, _MIXED_256),
        (7169, _MIXED_512),
    ),
    (8192, 2048): (  # o_proj
        (0, _BASE_512_21),
        (1536, _BASE_256_42),
        (3072, _BASE_256_21_CLC),
    ),
    (4096, 8192): (  # pool
        (0, _MIXED_256),
        (1024, _BASE_256_21_CLC),
        (1281, _MIXED_256),
        (1537, _BASE_512X192_21),
        (2049, _BASE_512_22_N),
        (3072, _MIXED_512),
        (4096, _BASE_256_42),
        (6144, _BASE_512_21),
        (8192, _BASE_256_21_CLC),
    ),
    (8192, 4096): (  # unpool
        (0, _BASE_512_21),
        (1537, _BASE_256_42),
        (3072, _MIXED_512),
        (4096, _MIXED_256),
        (8192, _BASE_512_21),
    ),
}
#: Any other weight shape: the runner's default tactic.
DEFAULT_TACTIC = -1


def _modules_of(layer) -> Dict[str, Linear]:
    attention, moe = layer.self_attn, layer.feed_forward
    return {
        "qkv": attention.qkv_proj,
        "o_proj": attention.o_proj,
        "pool": getattr(moe, "pooler", None),
        "unpool": getattr(moe, "unpooler", None),
    }


def _runner():
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops

    return cute_dsl_custom_ops.CuteDSLMXFP8RubinLinear(
        output_dtype=torch.bfloat16, use_tvm_ffi=True
    ), cute_dsl_custom_ops._get_mxfp8_gemm_alpha


class Fp8Projection:
    """One projection's MXFP8 weight copy and its GEMM: ``(rows, K)`` BF16 in, ``(rows, N)`` BF16 out."""

    def __init__(self, weight: torch.Tensor, min_rows: int) -> None:
        if weight.dim() != 2 or weight.dtype != torch.bfloat16 or weight.shape[1] % _quant.BLOCK_K:
            raise ValueError(
                f"needs a 2-D BF16 weight with K a multiple of {_quant.BLOCK_K}, got "
                f"{weight.dtype} {tuple(weight.shape)}"
            )
        self.min_rows = min_rows
        self.out_features, self.in_features = weight.shape
        self.tactics: Sequence[Tuple[int, Any]] = TACTICS.get(
            (self.out_features, self.in_features), ((0, DEFAULT_TACTIC),)
        )
        self._runner, get_alpha = _runner()
        self._alpha = get_alpha(weight.device)
        with torch.no_grad():
            self.weight, self.scale = _quant.mxfp8_quantize(weight.contiguous())

    def nbytes(self) -> int:
        return self.weight.numel() * self.weight.element_size() + self.scale.numel()

    def tactic(self, rows: int) -> Any:
        chosen = self.tactics[0][1]
        for fewest, tactic in self.tactics:
            if rows >= fewest:
                chosen = tactic
        return chosen

    def takes(self, input: torch.Tensor, bias: Optional[torch.Tensor]) -> bool:
        return (
            bias is None
            and input.dim() == 2
            and input.shape[0] >= self.min_rows
            and input.shape[1] == self.in_features
            and input.dtype == torch.bfloat16
            and input.stride(1) == 1
            and not torch.compiler.is_compiling()
        )

    def __call__(self, input: torch.Tensor, tactic: Any = None) -> torch.Tensor:
        values, scale = _quant.mxfp8_quantize(input)
        return self.project(values, scale, tactic)

    def project(
        self, values: torch.Tensor, scale: torch.Tensor, tactic: Any = None
    ) -> torch.Tensor:
        """The GEMM of already quantized rows (E4M3 ``[M, K]``, 128x4-swizzled scales)."""
        if tactic is None:
            tactic = self.tactic(values.shape[0])
        return self._runner([values, self.weight, scale, self.scale, self._alpha], tactic=tactic)


class GatheredMxfp8:
    """A step's attention input rows, gathered in MXFP8 (``Fp8AttentionInputGather``).

    Its only consumer is the QKV projection whose quant method is ``ContextFp8LinearMethod``.
    """

    def __init__(self, values: torch.Tensor, scale: torch.Tensor) -> None:
        self.values = values
        self.scale = scale

    @property
    def shape(self) -> torch.Size:
        return self.values.shape


class Fp8AttentionInputGather:
    """The decoder layer's ``_sp_attn_input_gather``: quantize the shard, then all-gather it.

    Steps below ``min_rows`` rows run the NCCL all-gather of the BF16 rows the layer runs
    without the hook.
    """

    def __init__(self, mapping, min_rows: int) -> None:
        self.mapping = mapping
        self.min_rows = min_rows

    def __call__(self, attn_input: torch.Tensor, rows: int):
        from tensorrt_llm._torch.distributed import allgather

        if (
            rows < self.min_rows
            or attn_input.dim() != 2
            or attn_input.dtype != torch.bfloat16
            or attn_input.stride(1) != 1
            or attn_input.shape[1] % _quant.BLOCK_K
            or torch.compiler.is_compiling()
        ):
            return allgather(attn_input, self.mapping, dim=0)[:rows]
        values, scales = _quant.mxfp8_quantize(attn_input, swizzled=False)
        values, scales = allgather([values.view(torch.uint8), scales], self.mapping, dim=0)
        return GatheredMxfp8(
            values[:rows].view(torch.float8_e4m3fn), _quant.swizzle_scales(scales[:rows])
        )


class ContextFp8LinearMethod(UnquantizedLinearMethod):
    """``UnquantizedLinearMethod`` whose calls of at least the copy's ``min_rows`` rows run MXFP8."""

    def apply(self, module: Linear, input: torch.Tensor, bias: Optional[torch.Tensor]):
        fp8 = module.__dict__.get("_context_fp8")
        if isinstance(input, GatheredMxfp8):
            if fp8 is None or bias is not None:
                raise RuntimeError(
                    "context_fp8_dense: gathered MXFP8 rows reached a projection without an FP8 copy"
                )
            return fp8.project(input.values, input.scale)
        if fp8 is not None and fp8.takes(input, bias):
            return fp8(input)
        return super().apply(module, input, bias)


def _ce_gather_wired(sm: Optional[int]) -> bool:
    from ..context_sequence_parallel import forward as _context_sp

    return sm == _context_sp.SERVED_SM and "ce_gather" in _context_sp.SHARD_PARTS


def install(
    text_model,
    mapping,
    compute_capability: Optional[Tuple[int, int]],
    min_rows: int = DEFAULT_MIN_ROWS,
    projections: Sequence[str] = PROJECTIONS,
    gather: bool = False,
) -> List[str]:
    """Give ``projections`` of every layer an MXFP8 copy; returns the projections installed.

    Calls of ``min_rows`` rows or more run FP8; with ``gather`` (``context_fp8_attn_gather``)
    the sequence-parallel attention input is quantized before its all-gather. Collective over
    the TP group (the outcome is agreed): raises ValueError on every rank alike when some rank
    cannot serve the request (not sm107, no Rubin CuTe DSL GEMM, a projection that is not a
    bias-free BF16 ``UnquantizedLinearMethod`` Linear, or a failing first call). A call with
    LoRA parameters runs the copy as LoRA's base GEMM, as it runs the BF16 one.
    """
    sm = _tp_agreement.sm_of(compute_capability)

    def build() -> List[str]:
        if sm != SERVED_SM:
            raise ValueError(f"[musespark_open] {NAME}: written for sm{SERVED_SM}, got sm{sm}")
        from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE

        if not IS_CUTLASS_DSL_RUBIN_AVAILABLE:
            raise ValueError(
                f"[musespark_open] {NAME}: the Rubin CuTe DSL package is not available"
            )
        unknown = sorted(set(projections) - set(PROJECTIONS))
        if unknown or min_rows < 1:
            raise ValueError(
                f"[musespark_open] {NAME}: projections {list(projections)} of {list(PROJECTIONS)} "
                f"from {min_rows} rows"
            )
        names = [name for name in PROJECTIONS if name in projections]
        if "qkv" in names and _ce_gather_wired(sm):
            names.remove("qkv")
            logger.warning(
                f"[musespark_open] {NAME}: the copy-engine gather is wired on sm{sm} and multiplies "
                "the QKV row blocks with the BF16 weight, so the QKV projection keeps BF16"
            )
        sites = []
        for layer in text_model.layers:
            modules = _modules_of(layer)
            for name in names:
                module = modules[name]
                if (
                    not isinstance(module, Linear)
                    or type(module.quant_method) is not UnquantizedLinearMethod
                ):
                    raise ValueError(
                        f"[musespark_open] {NAME}: layer {layer.layer_idx} {name} is not a built-in "
                        f"unquantized Linear ({type(getattr(module, 'quant_method', None)).__name__})"
                    )
                if module.bias is not None:
                    raise ValueError(
                        f"[musespark_open] {NAME}: layer {layer.layer_idx} {name} has a bias"
                    )
                sites.append(module)
        added = 0
        try:
            copies = []
            for module in sites:
                copy = Fp8Projection(module.weight.data, min_rows)
                copies.append((module, copy))
                added += copy.nbytes()
            # Compile every listed tactic of every weight shape (and create the GEMM's alpha)
            # before any CUDA graph capture.
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
                f"[musespark_open] {NAME}: could not build the MXFP8 copies: {error!r}"
            ) from error
        for module, copy in copies:
            fp8_method = ContextFp8LinearMethod.__new__(ContextFp8LinearMethod)
            fp8_method.__dict__.update(module.quant_method.__dict__)
            module.quant_method = fp8_method
            module._context_fp8 = copy
        gathers = 0
        if "qkv" in names and gather:
            for layer in text_model.layers:
                if getattr(layer, "_sp_attn_input_gather", None) is None:
                    layer._sp_attn_input_gather = Fp8AttentionInputGather(layer.mapping, min_rows)
                    gathers += 1
        logger.info(
            f"[musespark_open] {NAME}: {names} of {len(text_model.layers)} layers run their calls of "
            f">= {min_rows} rows on the Rubin MXFP8 GEMM; the copies add {added / 2**30:.2f} GiB per rank; "
            f"{gathers} layers gather their attention input in MXFP8"
        )
        return names

    return _tp_agreement.agreed_install(mapping, build, (ValueError, RuntimeError))
