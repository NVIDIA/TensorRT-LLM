# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""fused_kv_indexer_projection merged projection bound to one native DSA ``MLA`` module.

``FusedKvProjection.run(h)`` takes the normalized hidden state ``h [R, 6144]``
that the native ``kv_a_proj_with_mqa`` GEMM and the native indexer
``wk | weights_proj`` GEMM consume, and returns ``(kva [R, 2624], kw [R,
160] or None)`` from one kernel launch, with the dtypes, shapes and strides
of the native ``dsv3_fused_a_gemm_op`` / ``F.linear`` outputs. It reads the
module's own weight tensors in place (``kv_a_proj_with_mqa.weight`` and,
with ``TRTLLM_DSA_INDEXER_BF16=1``, ``indexer._fused_wk_wp_weight``); no
packed copy exists, so the extra weight memory is zero.
"""

from dataclasses import dataclass, fields, replace
from typing import Optional, Tuple

import torch

_K = 6144
_N_KVA = 2624
_N_KW = 160
MAX_ROWS = 8
# K slices of the native fused-A GEMM (one per MMA warp): the order fused_kv_indexer_projection
# keeps.
NATIVE_NSPLIT = 4


@dataclass(frozen=True)
class FusedKvProjConfig:
    """Launch configuration of the projection kernel (no numerical effect).

    ``ctas = 0`` means one CTA per SM. ``stage_before_wait`` copies the
    weights into shared memory before the PDL dependency wait; ``False``
    copies them after the wait (attribution arm). ``nsub``: sub-chunks per
    warp slice of the staging order; ``b_dist`` / ``a_dist``: K steps of
    activation / weight fragments loaded ahead; ``producer_warp``: dedicated
    copy warp(s) (else MMA warp 0 issues the bulk copies); ``copy_method``:
    ``bulk`` (1-D ``cp.async.bulk``) or ``cpasync`` (16-byte ``cp.async``
    over ``producer_warps`` warps); ``chunks``: unequal staging chunks of a
    warp slice in K steps, ``-``-separated (overrides ``nsub``; empty means
    ``nsub`` equal chunks); ``b_load``: cache operator of the activation
    loads (``cg``: L2 only, ``ca``: L1 and L2); ``copy_method=tma``: 3-D
    tensor copies of 16 rows x ``tma_groups`` K groups of 64 (``tma_groups``
    1 or 3); ``prewait_chunks``: chunks staged before the dependency wait
    (``-1``: all; the rest is staged after the wait).
    """

    ctas: int = 0
    smem_rows: int = 18
    nsub: int = 4
    trigger: str = "start"
    stage_before_wait: bool = True
    l2_prefetch: bool = True
    b_dist: int = 32
    a_dist: int = 2
    producer_warp: bool = True
    copy_method: str = "bulk"
    producer_warps: int = 1
    chunks: str = "48-24-12-12"
    b_load: str = "ca"
    tma_groups: int = 1
    prewait_chunks: int = -1
    # Tests only: stage the weights and exit without computing (streaming floor).
    copy_only: bool = False
    # Benchmarks only: no activation loads (meaningless output; chain cost).
    diag_no_act: bool = False

    def chunk_steps(self) -> tuple:
        return tuple(int(c) for c in self.chunks.split("-")) if self.chunks else ()

    @classmethod
    def parse(cls, text: str, base: Optional["FusedKvProjConfig"] = None) -> "FusedKvProjConfig":
        """``base`` (default: the served configuration) with ``key=value,...``
        applied; an ``nsub`` without ``chunks`` selects ``nsub`` equal chunks."""
        kwargs = {}
        types = {f.name: f.type for f in fields(cls)}
        for item in filter(None, text.split(",")):
            key, value = item.split("=")
            typ = types[key]
            if typ in (bool, "bool"):
                kwargs[key] = value.lower() in ("1", "true")
            elif typ in (int, "int"):
                kwargs[key] = int(value)
            else:
                kwargs[key] = value
        if "nsub" in kwargs and "chunks" not in kwargs:
            kwargs["chunks"] = ""
        return replace(base or cls(), **kwargs)


def _sm_count(device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def kernel_config(
    config: FusedKvProjConfig,
    n0: int,
    n1: int,
    ctas: int,
    use_pdl: bool,
    nsplit: int = NATIVE_NSPLIT,
    warps: int = NATIVE_NSPLIT,
    reduce: str = "seq",
    ftz: bool = False,
    debug_ts: bool = False,
) -> dict:
    """Keyword arguments of ``FusedKvProjKernel`` / ``compiled_projection``."""
    return dict(
        k=_K,
        n0=n0,
        n1=n1,
        ctas=ctas,
        warps=warps,
        smem_rows=config.smem_rows,
        nsplit=nsplit,
        nsub=config.nsub,
        use_pdl=use_pdl,
        trigger=config.trigger,
        stage_before_wait=config.stage_before_wait,
        l2_prefetch=config.l2_prefetch,
        b_dist=config.b_dist,
        a_dist=config.a_dist,
        reduce=reduce,
        ftz=ftz,
        producer_warp=config.producer_warp,
        copy_method=config.copy_method,
        producer_warps=config.producer_warps,
        copy_only=config.copy_only,
        debug_ts=debug_ts,
        chunks=config.chunk_steps(),
        b_load=config.b_load,
        diag_no_act=config.diag_no_act,
        tma_groups=config.tma_groups,
        prewait_chunks=config.prewait_chunks,
    )


def _tensor_map_words(w: torch.Tensor, groups: int) -> list:
    """128-byte ``CUtensorMap`` (16 signed 64-bit words) of the row-major BF16
    weight ``w [N, K]`` viewed as ``(64, K / 64, N)`` (elements, K groups,
    rows), box ``(64, groups, 16)``, 128-byte swizzle."""
    import ctypes

    from cuda.bindings import driver as cu

    from .proj_kernel import TMA_GROUP_K, TMA_ROWS

    n, k = w.shape
    dims = [TMA_GROUP_K, k // TMA_GROUP_K, n]
    strides = [TMA_GROUP_K * 2, k * 2]
    box = [TMA_GROUP_K, groups, TMA_ROWS]
    err, tmap = cu.cuTensorMapEncodeTiled(
        cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        3,
        w.data_ptr(),
        [cu.cuuint64_t(d) for d in dims],
        [cu.cuuint64_t(d) for d in strides],
        [cu.cuuint32_t(d) for d in box],
        [cu.cuuint32_t(1)] * 3,
        cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed: {err}")
    if hasattr(tmap, "getPtr"):
        raw = ctypes.string_at(int(tmap.getPtr()), 128)
        return [int.from_bytes(raw[8 * i : 8 * i + 8], "little", signed=True) for i in range(16)]
    words = [int(v) for v in tmap.opaque]
    return [v - (1 << 64) if v >= (1 << 63) else v for v in words]


def projection_eligibility(mla) -> Tuple[bool, str, Optional[torch.Tensor]]:
    """Static check of one MLA module; returns ``(ok, reason, kw_weight)``.

    ``kw_weight`` is the indexer's BF16 fused ``wk | weights_proj`` weight on
    a full-indexer layer (``None`` on a shared-indexer layer).
    """
    from tensorrt_llm._utils import get_sm_version

    if get_sm_version() not in (100, 103):
        return False, f"sm{get_sm_version()} not validated", None
    lin = getattr(mla, "kv_a_proj_with_mqa", None)
    if lin is None or not getattr(mla, "fuse_qkv_a_proj", False):
        return False, "needs the fused [q_a | kv_a] projection", None
    w = lin.weight
    if (
        lin.has_any_quant
        or lin.bias is not None
        or w.dtype != torch.bfloat16
        or tuple(w.shape) != (_N_KVA, _K)
        or not w.is_contiguous()
    ):
        return False, "kv_a_proj_with_mqa must be an unquantized BF16 [2624, 6144] weight", None
    if getattr(lin, "use_cute_dsl_bf16_gemm", False):
        return False, "kv_a_proj_with_mqa runs the CuTe DSL BF16 GEMM, not fused-A", None
    kw = None
    indexer = mla.mqa.indexer if getattr(mla, "mqa", None) is not None else None
    if indexer is not None:
        kw = getattr(indexer, "_fused_wk_wp_weight", None)
        if (
            not getattr(indexer, "_indexer_bf16", False)
            or kw is None
            or kw.dtype != torch.bfloat16
            or tuple(kw.shape) != (_N_KW, _K)
            or not kw.is_contiguous()
            or getattr(indexer, "use_fp4", False)
        ):
            return False, "full-indexer layer needs the BF16 fused wk|weights_proj GEMM", None
    return True, "", kw


class FusedKvProjection:
    """Merged projection bound to one MLA module (see the module docstring)."""

    def __init__(
        self,
        mla,
        config: Optional[FusedKvProjConfig] = None,
        kw_weight: Optional[torch.Tensor] = None,
    ):
        from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

        self.mla = mla
        self.config = config or FusedKvProjConfig()
        self.w0 = mla.kv_a_proj_with_mqa.weight
        self.w1 = kw_weight
        self.n1 = 0 if kw_weight is None else kw_weight.shape[0]
        self.use_pdl = get_env_enable_pdl()
        self.ctas = self.config.ctas or _sm_count(self.w0.device)
        # Tensor maps of the weights (``copy_method=tma``), built by precompile().
        self.tmaps = None

    def kernel_config(self, debug_ts: bool = False) -> dict:
        return kernel_config(
            self.config, _N_KVA, self.n1, self.ctas, self.use_pdl, debug_ts=debug_ts
        )

    def supported(self) -> Optional[str]:
        from .proj_kernel import projection_supported

        cfg = self.kernel_config()
        if self.config.b_load not in ("cg", "ca"):
            return f"activation load {self.config.b_load!r} is a benchmark-only mode (not served)"
        if self.config.copy_only or self.config.diag_no_act:
            return "copy_only / diag_no_act is a benchmark-only mode (not served)"
        return projection_supported(
            cfg["k"],
            cfg["n0"] + cfg["n1"],
            cfg["ctas"],
            cfg["warps"],
            cfg["smem_rows"],
            cfg["nsplit"],
            cfg["nsub"],
            cfg["chunks"],
            cfg["n1"],
            cfg["copy_method"],
            cfg["tma_groups"],
        )

    def precompile(self) -> None:
        """Compile the served kernel and, for ``copy_method=tma``, build the
        weights' tensor maps (one 256-byte device buffer). Errors propagate."""
        from .proj_kernel import compiled_projection

        compiled_projection(**self.kernel_config())
        if self.config.copy_method == "tma" and self.tmaps is None:
            words = _tensor_map_words(self.w0, self.config.tma_groups)
            if self.w1 is not None:
                words += _tensor_map_words(self.w1, self.config.tma_groups)
            self.tmaps = torch.tensor(words, dtype=torch.int64).to(self.w0.device)

    def run(
        self, h: torch.Tensor, ts: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Launch on the current stream; ``h`` is the native normalized hidden state."""
        from .proj_kernel import compiled_projection

        rows = h.shape[0]
        assert h.dim() == 2 and h.shape[1] == _K and 1 <= rows <= MAX_ROWS
        assert h.is_contiguous() and h.dtype == torch.bfloat16
        kva = h.new_empty(rows, _N_KVA)
        kw = h.new_empty(rows, self.n1) if self.n1 else None
        compiled = compiled_projection(**self.kernel_config(debug_ts=ts is not None))
        if self.config.copy_method == "tma" and self.tmaps is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "fused_kv_indexer_projection tensor maps must be built (precompile) before capture"
                )
            self.precompile()
        compiled(
            h.data_ptr(),
            self.w0.data_ptr(),
            0 if self.w1 is None else self.w1.data_ptr(),
            kva.data_ptr(),
            0 if kw is None else kw.data_ptr(),
            0 if ts is None else ts.data_ptr(),
            0 if self.tmaps is None else self.tmaps.data_ptr(),
            kva.view(torch.int32).view(-1),
            rows,
        )
        return kva, kw
