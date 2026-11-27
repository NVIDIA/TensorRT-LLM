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
"""deferred_norm next-layer projection: the fused-A GEMM (and the indexer wk|weights_proj
GEMM on full-indexer layers) on a deferred-norm packet, with staged weights.

``DeferredNormProjection`` is bound to one native DSA ``MLA`` module. It reads the
module's own weight tensors in place (``kv_a_proj_with_mqa.weight`` and, with
``TRTLLM_DSA_INDEXER_BF16=1``, ``indexer._fused_wk_wp_weight``); no packed or
gain-folded copy exists, so the extra weight memory is zero. ``project``
returns ``(kva [R, 2624], kw [R, 160] or None)`` with the dtypes and layouts
of the native ``dsv3_fused_a_gemm_op`` / ``F.linear`` outputs.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .packet import DeferredNormHidden

_K = 6144
_N_KVA = 2624
_N_KW = 160
MAX_ROWS = 8


@dataclass(frozen=True)
class DeferredNormProjConfig:
    """Launch configuration of the projection kernel.

    ``ctas = 0`` means one CTA per SM. ``stage_before_wait`` copies the
    weights into shared memory before the PDL dependency wait (the deferred_norm
    schedule); ``False`` copies them after the wait (attribution arm).
    ``trigger`` is when the kernel releases its own PDL dependents.
    ``dist`` is how many K rounds ahead each warp loads its weight fragments
    from shared memory.
    """

    ctas: int = 0
    threads: int = 512
    smem_rows: int = 18
    nchunk: int = 4
    trigger: str = "start"
    stage_before_wait: bool = True
    l2_prefetch: bool = True
    dist: int = 4
    copy_threads: int = 1
    copy_method: str = "bulk"
    wait_staged_first: bool = False
    prefetch_next: bool = False


def _sm_count(device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _kernel_config(
    config: DeferredNormProjConfig,
    n1: int,
    ctas: int,
    use_pdl: bool,
    deferred: bool,
    debug_ts: bool,
) -> dict:
    return dict(
        k=_K,
        n0=_N_KVA,
        n1=n1,
        ctas=ctas,
        threads=config.threads,
        smem_rows=config.smem_rows,
        nchunk=config.nchunk,
        deferred=deferred,
        partials_per_row=24,
        use_pdl=use_pdl,
        trigger=config.trigger,
        stage_before_wait=config.stage_before_wait,
        l2_prefetch=config.l2_prefetch,
        dist=config.dist,
        copy_threads=config.copy_threads,
        copy_method=config.copy_method,
        wait_staged_first=config.wait_staged_first,
        prefetch_next=config.prefetch_next,
        debug_ts=debug_ts,
    )


# Indexer rows appended to the fused-A rows: shared-indexer layers (none) and
# full-indexer layers (the BF16 wk | weights_proj rows).
SERVED_N1 = (0, _N_KW)


def served_kernel_configs(device) -> list:
    """Kernel configurations the deferred_norm consumer launches in serving.

    One per layer kind (``SERVED_N1``): deferred mode, fragment-layout
    activation, no debug stamps; the default launch configuration
    (``DeferredNormProjConfig()``) and the PDL policy from ``TRTLLM_ENABLE_PDL``,
    exactly as :meth:`DeferredNormProjection.served_config` builds them.
    """
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    config = DeferredNormProjConfig()
    ctas = config.ctas or _sm_count(device)
    use_pdl = get_env_enable_pdl()
    out = []
    for n1 in SERVED_N1:
        cfg = _kernel_config(config, n1, ctas, use_pdl, deferred=True, debug_ts=False)
        cfg["b_layout"] = "frag"
        out.append(cfg)
    return out


def precompile_served(device) -> Optional[str]:
    """Compile every served projection kernel (no GPU work).

    Called by ``DeferredNormHop`` before its cross-rank agreement, so a
    deferred-norm packet exists only when every rank compiled both kernels
    and the consumer's compile is a cache hit. Returns why a served shape is
    not supported (rank-identical static check), or ``None``; a compile
    error propagates to the caller.
    """
    from .proj_kernel import compiled_projection, projection_supported

    for cfg in served_kernel_configs(device):
        why = projection_supported(
            cfg["k"],
            cfg["n0"] + cfg["n1"],
            cfg["ctas"],
            cfg["threads"],
            cfg["smem_rows"],
            cfg["nchunk"],
        )
        if why is not None:
            return f"projection n1={cfg['n1']}: {why}"
    for cfg in served_kernel_configs(device):
        compiled_projection(**cfg)
    return None


def projection_eligibility(mla) -> Tuple[bool, str, Optional[torch.Tensor]]:
    """Static check of one MLA module; returns ``(ok, reason, kw_weight)``.

    Under the served ``shared`` scope a full-indexer module is declined: its
    producing hop is post_moe_hop's in that scope, and a packet that reached it
    anyway is materialized (post_moe_hop's normalized output) for the native path.
    """
    from tensorrt_llm._utils import get_sm_version

    from .scope import SCOPE_LABEL, selected_scope

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
    kw = None
    indexer = mla.mqa.indexer if getattr(mla, "mqa", None) is not None else None
    if indexer is not None:
        scope = selected_scope()
        if scope == "shared":
            return (
                False,
                f"{SCOPE_LABEL}={scope}: full-indexer layer keeps its native projection",
                None,
            )
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


class DeferredNormProjection:
    """Staged projection bound to one MLA module (see the module docstring)."""

    def __init__(
        self,
        mla,
        config: Optional[DeferredNormProjConfig] = None,
        kw_weight: Optional[torch.Tensor] = None,
    ):
        from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

        self.mla = mla
        self.config = config or DeferredNormProjConfig()
        self.w0 = mla.kv_a_proj_with_mqa.weight
        self.w1 = kw_weight
        self.n1 = 0 if kw_weight is None else kw_weight.shape[0]
        self.use_pdl = get_env_enable_pdl()
        self.ctas = self.config.ctas or _sm_count(self.w0.device)
        self.debug_ts = None
        # (address, bytes) of up to three tensors the next kernels read
        # (L2-prefetched before the dependency wait with ``prefetch_next``).
        self.prefetch = []

    def set_prefetch(self, tensors) -> None:
        """Tensors the following kernels read (weights only; L2 hint)."""
        self.prefetch = [t for t in tensors if t is not None and t.is_contiguous()][:3]

    def kernel_config(self, deferred: bool = True, debug_ts: bool = False) -> dict:
        return _kernel_config(self.config, self.n1, self.ctas, self.use_pdl, deferred, debug_ts)

    def served_config(self) -> dict:
        """Configuration :meth:`project` launches (one of ``served_kernel_configs``
        when the module's config comes from the environment)."""
        cfg = self.kernel_config()
        cfg["b_layout"] = "frag"
        return cfg

    def supported(self) -> Optional[str]:
        from .proj_kernel import projection_supported

        cfg = self.config
        return projection_supported(
            _K, _N_KVA + self.n1, self.ctas, cfg.threads, cfg.smem_rows, cfg.nchunk
        )

    def precompile(self) -> None:
        """Compile the served kernel; a cache hit after ``precompile_served``.
        A compile error propagates."""
        from .proj_kernel import compiled_projection

        compiled_projection(**self.served_config())

    def run(
        self,
        act: torch.Tensor,
        sumsq: Optional[torch.Tensor],
        eps: float,
        rows: Optional[int] = None,
        ts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Launch on the current stream.

        ``act`` 1-D: the ``a_frag`` buffer of a packet (``rows`` token rows);
        2-D ``[R, K]``: a row-major activation. ``sumsq`` given: deferred mode
        (``act`` holds ``bf16(x * gamma)``); ``None``: native-input mode
        (``act`` is a normalized hidden state, ``r = 1``).
        """
        from .proj_kernel import compiled_projection

        frag = act.dim() == 1
        if not frag:
            rows = act.shape[0]
        assert rows is not None and 1 <= rows <= MAX_ROWS
        assert act.is_contiguous() and act.dtype == torch.bfloat16
        kva = act.new_empty(rows, _N_KVA)
        kw = act.new_empty(rows, self.n1) if self.n1 else None
        deferred = sumsq is not None
        cfg = self.kernel_config(deferred, ts is not None)
        cfg["b_layout"] = "frag" if frag else "plain"
        compiled = compiled_projection(**cfg)
        pf = [(t.data_ptr(), t.numel() * t.element_size()) for t in self.prefetch[:3]]
        pf += [(0, 0)] * (3 - len(pf))
        compiled(
            act.data_ptr(),
            sumsq.data_ptr() if deferred else 0,
            self.w0.data_ptr(),
            0 if self.w1 is None else self.w1.data_ptr(),
            kva.data_ptr(),
            0 if kw is None else kw.data_ptr(),
            0 if ts is None else ts.data_ptr(),
            pf[0][0],
            pf[0][1],
            pf[1][0],
            pf[1][1],
            pf[2][0],
            pf[2][1],
            kva.view(torch.int32).view(-1),
            rows,
            float(eps),
        )
        return kva, kw

    def project(self, packet: DeferredNormHidden) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert packet.sumsq.shape[1] == 24 and packet.sumsq.is_contiguous()
        return self.run(packet.a_frag, packet.sumsq, packet.eps, rows=packet.num_rows)
