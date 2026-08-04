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
"""Isolation harness for modularized attention perf tests.

One harness, one forward, all signals. The harness builds a TrtllmAttention
backend in isolation (mirroring tests/unittest/_torch/attention/test_attention.py's
``produce_outputs`` recipe), runs a single-layer forward in a pinned config, and
returns BOTH the discrete structural signals (pre-merge gate) and the continuous
gpu_time (post-merge detector) in one shot.

Design refs (see trtllm-perf-modulized-test-notes.md):
  - §11.5 two independent detection paths (discrete gate + continuous detector)
  - §12.2 isolation recipe / §12.3 what to collect / §12.4 what to pin

Signal status:
  - use_paged_context_fmha : REAL  (config-deterministic, off metadata)
  - gpu_time_median_ms     : REAL  (CUDA events + warmup + median, optional clock lock)
  - launch_count           : REAL if the optional ``cupti`` package is present, else None
  - nvrtc_compile_count    : HOOK  -> None until a dev-exposed counter exists
  - cubin_present          : HOOK  -> None until a dev-exposed registry probe exists
The two HOOKs are exactly the "discrete-observability" gap flagged in the design
notes: TRT-LLM does not yet expose these counters from Python. Cases that depend
on them skip (bootstrap) rather than fake a value.
"""

from __future__ import annotations

import statistics
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Optional

import torch

import tensorrt_llm
from tensorrt_llm._torch.attention_backend import TrtllmAttention
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionRuntimeFeatures,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

__all__ = ["AttnCase", "Signals", "collect_signals", "sm_arch"]

# --------------------------------------------------------------------------- #
# arch key — golden is stored per (case_id, arch). See notes §10.4.
# --------------------------------------------------------------------------- #


def sm_arch() -> str:
    """Return ``smNN`` for the current device, e.g. ``sm90`` / ``sm120``."""
    if not torch.cuda.is_available():
        return "cpu"
    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"


# --------------------------------------------------------------------------- #
# pinned config — every adaptive mechanism must be nailed down (notes §9.9 axis③)
# --------------------------------------------------------------------------- #

_DTYPE_TO_BINDING = {
    torch.float16: tensorrt_llm.bindings.DataType.HALF,
    torch.bfloat16: tensorrt_llm.bindings.DataType.BF16,
    torch.float8_e4m3fn: tensorrt_llm.bindings.DataType.FP8,
}


@dataclass(kw_only=True)
class AttnCase:
    """A pinned attention scenario. ``phase`` selects context vs generation."""

    case_id: str
    phase: str  # "context" | "generation"
    dtype: torch.dtype = torch.bfloat16
    num_heads: int = 64
    num_kv_heads: int = 16
    head_dim: int = 128
    batch_size: int = 7
    page_size: int = 256
    # context: seq_len tokens per request, no cache. generation: 1 new token + cache.
    seq_len: int = 32
    num_cached_tokens: int = 0
    seed: int = 0
    # runtime_features knobs drive use_paged_context_fmha (config-deterministic).
    chunked_prefill: bool = False
    cache_reuse: bool = False

    def __post_init__(self) -> None:
        if self.phase not in ("context", "generation"):
            raise ValueError(
                f"AttnCase.phase must be 'context' or 'generation', got {self.phase!r}"
            )

    @property
    def num_pages(self) -> int:
        total = self.seq_len + self.num_cached_tokens
        return max(1, (total + self.page_size - 1) // self.page_size)


@dataclass
class Signals:
    """All signals from one forward. None = not observable on this host/build."""

    case_id: str
    arch: str
    phase: str
    use_paged_context_fmha: Optional[bool] = None
    launch_count: Optional[int] = None
    nvrtc_compile_count: Optional[int] = None  # HOOK
    cubin_present: Optional[bool] = None  # HOOK
    gpu_time_median_ms: Optional[float] = None
    gpu_time_p99_ms: Optional[float] = None
    raw_times_ms: List[float] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# optional dependencies — graceful fallback (no hard requirement)
# --------------------------------------------------------------------------- #


def _try_clock_lock(gpu_id: int = 0):
    """Return an active GPUClockLock or None. Locking kills DVFS (notes §9.4)."""
    try:
        from defs.perf.gpu_clock_lock import GPUClockLock  # type: ignore
    except ImportError:
        return None
    try:
        lock = GPUClockLock(gpu_id=str(gpu_id), interval_ms=100.0, enable_clock_locking=True)
        return lock
    except Exception:
        # Construction can fail at runtime (no NVML perms, locking unsupported);
        # degrade to an unlocked run rather than fail the test.
        return None


def _l2_flush_buffer(device: torch.device) -> torch.Tensor:
    """A 2x-L2 buffer; ``.zero_()`` between iters clears L2 (notes §9.4).

    Prefer bench_moe's implementation (single source of truth) but fall back to
    an inline copy if its eager-import chain is unavailable on this host.
    """
    try:
        from bench_moe.timing import _l2_flush_buffer as _moe_l2  # type: ignore
    except ImportError:
        l2_size = torch.cuda.get_device_properties(device).L2_cache_size
        return torch.empty((l2_size * 2) // 4, dtype=torch.int32, device=device)
    return _moe_l2(device)


@contextmanager
def _cupti_kernel_counter():
    """Yield a callable returning kernels executed since entry, or None.

    Uses the bench_moe CUPTI helpers when the optional ``cupti`` package is
    installed; otherwise yields None and launch_count stays unobserved.
    """
    try:
        from bench_moe.timing.cupti import _try_init_cupti  # type: ignore
    except ImportError:
        yield None
        return
    ctx = _try_init_cupti()
    if not ctx.ok:
        yield None
        return

    def _count() -> int:
        ctx.module.activity_flush_all(0)
        return len(ctx.kernels)

    ctx.kernels.clear()
    yield _count


# --------------------------------------------------------------------------- #
# isolation recipe — build KV cache + metadata + backend (notes §12.2)
# --------------------------------------------------------------------------- #


def _build_kv_cache_manager(case: AttnCase, num_layers: int = 1) -> KVCacheManager:
    num_blocks = case.batch_size * case.num_pages
    tokens_per_block = case.page_size
    max_seq_len = num_blocks * tokens_per_block
    kv_cache_config = KvCacheConfig(max_tokens=num_blocks * tokens_per_block)
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    cache_type = tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF

    mgr = KVCacheManager(
        kv_cache_config,
        cache_type,
        num_layers=num_layers,
        num_kv_heads=case.num_kv_heads,
        head_dim=case.head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=case.batch_size,
        mapping=mapping,
        dtype=_DTYPE_TO_BINDING[case.dtype],
    )
    request_ids = list(range(case.batch_size))
    token_nums = [case.seq_len + case.num_cached_tokens] * case.batch_size
    mgr.add_dummy_requests(request_ids, token_nums)
    return mgr


def _build_metadata(case: AttnCase, kv_mgr: KVCacheManager):
    request_ids = list(range(case.batch_size))
    num_cached = [case.num_cached_tokens] * case.batch_size
    kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=num_cached)
    seq_lens = torch.tensor([case.seq_len] * case.batch_size, dtype=torch.int)
    prompt_lens = [case.seq_len + case.num_cached_tokens] * case.batch_size
    num_contexts = case.batch_size if case.phase == "context" else 0

    # runtime_features drive use_paged_context_fmha — see trtllm.py __post_init__.
    runtime_features = AttentionRuntimeFeatures(
        chunked_prefill=case.chunked_prefill, cache_reuse=case.cache_reuse
    )

    metadata = TrtllmAttention.Metadata(
        num_contexts=num_contexts,
        kv_cache_params=kv_cache_params,
        seq_lens=seq_lens,
        max_num_requests=case.batch_size,
        max_num_tokens=8192,
        kv_cache_manager=kv_mgr,
        request_ids=request_ids,
        prompt_lens=prompt_lens,
        runtime_features=runtime_features,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
    )
    metadata.prepare()
    return metadata


def _make_qkv(case: AttnCase, device: torch.device):
    """Return a fused QKV tensor [nnz, (num_heads + 2*num_kv_heads)*head_dim].

    TrtllmAttention's context path requires fused QKV (separate q/k/v is only
    allowed with SageAttention; see attentionOp.cpp). This mirrors
    modules/attention.py's ``torch.concat([q, k, v], dim=-1)``; k/v are passed
    as None and forward splits the fused tensor internally.
    """
    torch.manual_seed(case.seed)
    nnz = case.batch_size * (case.seq_len if case.phase == "context" else 1)
    qkv_dim = (case.num_heads + 2 * case.num_kv_heads) * case.head_dim
    return torch.randn(nnz, qkv_dim, device=device, dtype=case.dtype)


# --------------------------------------------------------------------------- #
# the one entry point — run one case, return all signals
# --------------------------------------------------------------------------- #


def collect_signals(
    case: AttnCase, *, warmup: int = 10, iters: int = 50, lock_clock: bool = False
) -> Signals:
    """Run one pinned forward and return discrete + continuous signals.

    Discrete signals are cheap and gate-able pre-merge; gpu_time is the
    independent continuous detector (notes §11.5). Both come from one forward.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("attention perf harness requires a CUDA device")
    device = torch.device("cuda")
    sig = Signals(case_id=case.case_id, arch=sm_arch(), phase=case.phase)

    kv_mgr = _build_kv_cache_manager(case)
    try:
        metadata = _build_metadata(case, kv_mgr)
        # ---- discrete: config-deterministic, no GPU noise ----
        sig.use_paged_context_fmha = bool(getattr(metadata, "use_paged_context_fmha", None))

        backend = TrtllmAttention(
            layer_idx=0,
            num_heads=case.num_heads,
            num_kv_heads=case.num_kv_heads,
            head_dim=case.head_dim,
        )
        qkv = _make_qkv(case, device)
        mask = PredefinedAttentionMask.CAUSAL

        def _fwd():
            return backend.forward(qkv, None, None, metadata, attention_mask=mask)

        # warmup (drops JIT / autotune / capture — notes §22.2)
        with torch.no_grad():
            for _ in range(warmup):
                _fwd()
        torch.cuda.synchronize()

        # ---- discrete: launch_count via CUPTI (one captured iteration) ----
        with _cupti_kernel_counter() as count_kernels:
            if count_kernels is not None:
                with torch.no_grad():
                    _fwd()
                torch.cuda.synchronize()
                sig.launch_count = count_kernels()

        # ---- continuous: gpu_time (CUDA events, median) ----
        # Reuse bench_moe's REAL L2 flush (a 2xL2 buffer zeroed between iters),
        # NOT torch.cuda.empty_cache() — empty_cache returns allocator blocks to
        # the driver (slow, synchronous, jittery) and does NOT clear L2. Mirror
        # the MoE eager recipe: record all iters back-to-back, sync ONCE at the
        # end (no per-iter synchronize, which would serialize and add noise).
        clock = _try_clock_lock() if lock_clock else None
        l2_buffer = _l2_flush_buffer(device)
        try:
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            with torch.no_grad():
                for i in range(iters):
                    l2_buffer.zero_()  # flush L2 before the timed region
                    starts[i].record()
                    _fwd()
                    ends[i].record()
            torch.cuda.synchronize()
            times = [starts[i].elapsed_time(ends[i]) for i in range(iters)]
            sig.raw_times_ms = times
            sig.gpu_time_median_ms = statistics.median(times)
            sig.gpu_time_p99_ms = sorted(times)[int(0.99 * (len(times) - 1))]
        finally:
            if clock is not None:
                try:
                    clock.teardown()
                except Exception:
                    pass

        # ---- HOOKs: need a dev-exposed counter (discrete-observability gap) ----
        sig.nvrtc_compile_count = _probe_nvrtc_compile_count()  # -> None for now
        sig.cubin_present = _probe_cubin_present(case)  # -> None for now
    finally:
        kv_mgr.shutdown()
    return sig


def _probe_nvrtc_compile_count() -> Optional[int]:
    """HOOK: return # of NVRTC compilations during the forward, or None.

    No public counter exists yet. When TRT-LLM exposes one (the proposal's
    discrete-observability task), wire it here and the no-JIT assert turns on.
    """
    return None


def _probe_cubin_present(case: AttnCase) -> Optional[bool]:
    """HOOK: return whether the target FMHA/XQA cubin is registered, or None."""
    return None
