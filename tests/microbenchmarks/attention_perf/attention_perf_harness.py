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

import dataclasses
import json
import math
import os
import statistics
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Optional

import torch

import tensorrt_llm
from tensorrt_llm._torch.attention_backend import TrtllmAttention
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionInputType,
    AttentionRuntimeFeatures,
    MLAParams,
    PositionalEmbeddingParams,
    PredefinedAttentionMask,
    RopeParams,
)
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

# DSA (DeepSeek-V3.2 sparse attention) imports are done lazily inside the DSA
# builders below: they pull in DeepGEMM (sm90/sm100/sm103 only) at import time,
# so importing them unconditionally would break the GQA/MLA paths on sm120.

__all__ = ["AttnCase", "Signals", "collect_signals", "sm_arch", "device_name"]

# --------------------------------------------------------------------------- #
# GPU key — golden is stored per (case_id, device name). The DEVICE NAME, not
# sm_arch: one arch spans multiple GPUs (sm90=H100/H200/H20) with different perf.
# See notes §10.4.
# --------------------------------------------------------------------------- #


def sm_arch() -> str:
    """Return ``smNN`` for the current device, e.g. ``sm90`` / ``sm120``."""
    if not torch.cuda.is_available():
        return "cpu"
    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"


# PCI device-id -> friendly name, for SKUs the driver reports as the generic
# "NVIDIA Graphics Device" (engineering / pre-release cards, e.g. B300). Mirrors
# trt_jenkins src/com/nvidia/dlswqa/Utils.groovy map_device_id_gpu_name so the
# module pipeline agrees with the cluster tooling on what a card is.
_PCI_DEVICE_ID_TO_NAME = {
    "0X31BF10DE": "NVIDIA B300",
    "0X2BB510DE": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "0X2E2A10DE": "NVIDIA B200",
    "0X292410DE": "NVIDIA B100",
    "0X292010DE": "NVIDIA B100",
    "0X233F10DE": "NVIDIA H100",
}


def _pci_device_id(gpu_id: int = 0) -> str:
    """PCI device id (e.g. ``0X31BF10DE``) from nvidia-smi, '' on failure.

    Same source the groovy uses: ``nvidia-smi -q | grep 'Device Id'``.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "-q", "-i", str(gpu_id)],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    for line in out.splitlines():
        # "    Device Id                           : 0x31BF10DE"
        if "device id" in line.lower():
            return line.split(":")[-1].strip().upper()
    return ""


def device_name(gpu_id: int = 0) -> str:
    """GPU identity used to key golden: the device NAME (not sm_arch).

    For SKUs the driver can't name (it reports the generic "NVIDIA Graphics
    Device", e.g. B300) resolve via the PCI device id, mirroring the cluster
    groovy. Falls back to the raw torch name if resolution fails — never raises.
    """
    if not torch.cuda.is_available():
        return "cpu"
    name = torch.cuda.get_device_name(gpu_id)
    if "graphics device" in name.lower():
        resolved = _PCI_DEVICE_ID_TO_NAME.get(_pci_device_id(gpu_id))
        if resolved:
            return resolved
    return name


# --------------------------------------------------------------------------- #
# pinned config — every adaptive mechanism must be nailed down (notes §9.9 axis③)
# --------------------------------------------------------------------------- #

_DTYPE_TO_BINDING = {
    torch.float16: tensorrt_llm.bindings.DataType.HALF,
    torch.bfloat16: tensorrt_llm.bindings.DataType.BF16,
    torch.float8_e4m3fn: tensorrt_llm.bindings.DataType.FP8,
}

# Short dtype label for the human-readable shape tag (see AttnCase.shape_tag).
_DTYPE_LABEL = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float8_e4m3fn: "fp8",
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
    # MLA (DeepSeek-style latent attention). When True the harness uses the MLA
    # isolation path (latent KV cache + two-call decode forward) instead of GQA.
    # Defaults are DeepSeek-V3/R1 dims. head_dim/num_kv_heads above are ignored
    # for MLA (derived from the lora ranks). See _build_mla_* below.
    is_mla: bool = False
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # FP8 KV cache for MLA — the attention-side quant of the "fp4" DeepSeek models
    # (fp4 is weight-only and never touches the attention kernel; the MLA decode's
    # quant knob is the fp8 KV cache). Exercises the fp8 MLA decode kernel path.
    kv_cache_fp8: bool = False
    # DSA (DeepSeek-V3.2 sparse attention). When True the harness uses the DSA
    # isolation path: an MLA module whose backend is DSATrtllmAttention (MLA + a
    # sparse Indexer that top-k-selects KV). is_dsa implies MLA semantics for the
    # lora dims above; hidden_size / max_position_embeddings / the index_* fields
    # below are DSA-only. NOTE: the DSA indexer's DeepGEMM MQA-logits kernels are
    # built for sm90/sm100/sm103 only — DSA cases DO NOT run on sm120 (they skip).
    is_dsa: bool = False
    hidden_size: int = 7168  # DeepSeek-V3.2 model hidden size
    max_position_embeddings: int = 8192  # rope cache size; must exceed seq_len
    index_n_heads: int = 64  # indexer heads (DeepSeek-V3.2)
    index_head_dim: int = 128  # indexer head dim
    index_topk: int = 2048  # indexer top-k KV tokens

    def __post_init__(self) -> None:
        if self.phase not in ("context", "generation"):
            raise ValueError(
                f"AttnCase.phase must be 'context' or 'generation', got {self.phase!r}"
            )
        if self.is_dsa and self.is_mla:
            raise ValueError("is_dsa and is_mla are mutually exclusive")

    @property
    def num_pages(self) -> int:
        total = self.seq_len + self.num_cached_tokens
        return max(1, (total + self.page_size - 1) // self.page_size)

    def shape_tag(self) -> str:
        """Compact, human-readable encoding of the pinned inputs for this case.

        Goes into the CSV/DB next to case_id so a dashboard row says WHICH shape
        was measured — case_id alone (e.g. ``attn_l70b_decode_xqa_time``) carries
        the model+phase+metric but not the batch/seq/head numbers. Purely derived
        from the pinned fields; it NEVER keys golden/history (case_id does), so it
        is safe to enrich without breaking baselines. Examples:
          ``ctx·b1·s2048·h32/8·d128·bf16``
          ``dec·b256·q1+c1024·h64/8·d128·bf16``
          ``dec·b7·q1+c1024·mla-fp8·h128·q1536kv512·bf16``
          ``ctx·b1·s4096·dsa·h128·topk2048·bf16``
        """
        ph = "ctx" if self.phase == "context" else "dec"
        dt = _DTYPE_LABEL.get(self.dtype, str(self.dtype).rsplit(".", 1)[-1])
        toks = (
            f"s{self.seq_len}"
            if self.phase == "context"
            else f"q{self.seq_len}+c{self.num_cached_tokens}"
        )
        if self.is_dsa:
            fam = f"dsa·h{self.num_heads}·topk{self.index_topk}"
        elif self.is_mla:
            mla = "mla-fp8" if self.kv_cache_fp8 else "mla"
            fam = f"{mla}·h{self.num_heads}·q{self.q_lora_rank}kv{self.kv_lora_rank}"
        else:
            fam = f"h{self.num_heads}/{self.num_kv_heads}·d{self.head_dim}"
        return f"{ph}·b{self.batch_size}·{toks}·{fam}·{dt}"


@dataclass
class Signals:
    """All signals from one forward. None = not observable on this host/build."""

    case_id: str
    arch: str
    phase: str
    # Human-readable pinned-input tag (AttnCase.shape_tag); recorded next to
    # case_id so the DB/dashboard shows what shape a row measured.
    shape: Optional[str] = None
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
    except (RuntimeError, OSError, ValueError) as e:
        # Construction can fail at runtime (no NVML perms, locking unsupported,
        # bad gpu_id); degrade to an unlocked run rather than fail the test.
        print(f"[warn] GPU clock lock unavailable, running unlocked: {e}")
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
        # Bound the window on the trailing edge: finish the forward, then deliver
        # ITS records before counting.
        torch.cuda.synchronize()
        ctx.module.activity_flush_all(0)
        return len(ctx.kernels)

    # Reset the counter PROPERLY. clear() only empties the Python list, not CUPTI's
    # async activity buffer — records from warmup / prior cases (the CUPTI ctx is a
    # process singleton shared by every case) sit there and would be flushed into
    # THIS measurement at _count(), inflating the count (49 -> 59 -> 219 as more ran
    # before). So first sync (finish all prior GPU work) + flush (drain the buffer
    # into the list) + clear (empty the list) → buffer AND list are truly empty.
    torch.cuda.synchronize()
    ctx.module.activity_flush_all(0)
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
# MLA (Multi-head Latent Attention, DeepSeek-style) isolation path.
# Mirrors tests/unittest/_torch/attention/test_attention_mla.py::_run_test_for_backend
# (generation/decode phase). MLA differs structurally from GQA: a compressed
# latent KV cache (CacheType.SELFKONLY, num_kv_heads=1,
# head_dim = kv_lora_rank + qk_rope_head_dim), a yarn-RoPE positional embedding,
# and a TWO-CALL decode forward (mla_rope_generation -> forward). head_dim /
# num_kv_heads from AttnCase are NOT used here.
# --------------------------------------------------------------------------- #


@dataclass(kw_only=True, frozen=True)
class _RopeConfig:
    """DeepSeek-V3/R1 yarn RoPE config (from test_attention_mla.py:276)."""

    hidden_size: int = 7168
    num_attention_heads: int = 128
    rope_scaling: dict = field(
        default_factory=lambda: {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "yarn",
        }
    )
    max_position_embeddings: int = 163840
    rope_theta: float = 10000.0
    qk_rope_head_dim: int = 64
    model_type: str = "deepseek_v3"


def _mla_kv_head_dim(case: AttnCase) -> int:
    return case.kv_lora_rank + case.qk_rope_head_dim  # 576 for DeepSeek-V3/R1


def _mla_pos_embd_and_scaling(case: AttnCase):
    """Yarn RoPE PositionalEmbeddingParams + q_scaling (mscale correction)."""
    from tensorrt_llm.functional import PositionEmbeddingType

    rope_config = _RopeConfig(
        hidden_size=case.hidden_size,
        num_attention_heads=case.num_heads,
        qk_rope_head_dim=case.qk_rope_head_dim,
    )
    pos = PositionalEmbeddingParams(
        type=PositionEmbeddingType.yarn,
        rope=RopeParams.from_config(rope_config),
        is_neox=False,
    )

    def yarn_get_mscale(scale=1.0, mscale=1.0):
        return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0

    mscale = yarn_get_mscale(pos.rope.scale, pos.rope.mscale_all_dim)
    return pos, 1.0 / (mscale * mscale)


def _mla_params(case: AttnCase) -> MLAParams:
    return MLAParams(
        q_lora_rank=case.q_lora_rank,
        kv_lora_rank=case.kv_lora_rank,
        qk_rope_head_dim=case.qk_rope_head_dim,
        qk_nope_head_dim=case.qk_nope_head_dim,
        v_head_dim=case.v_head_dim,
        predicted_tokens_per_seq=1,  # no MTP — simplest decode
    )


def _mla_quant_config(case: AttnCase):
    """QuantConfig for fp8 KV cache (attention-side quant of fp4 models), else None."""
    if not case.kv_cache_fp8:
        return None
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    return QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8.value)


def _mla_tokens_per_block() -> int:
    # FlashMLA is block-size sensitive: 32 on Blackwell (sm>=100), else 64.
    major, _ = torch.cuda.get_device_capability()
    return 32 if major >= 10 else 64


def _build_mla_kv_cache_manager(case: AttnCase, num_layers: int = 1) -> KVCacheManager:
    tokens_per_block = _mla_tokens_per_block()
    head_dim = _mla_kv_head_dim(case)
    per_seq = case.num_cached_tokens + case.seq_len
    max_seq_len = per_seq + tokens_per_block
    max_tokens = (
        (max_seq_len + tokens_per_block - 1)
        // tokens_per_block
        * tokens_per_block
        * case.batch_size
    )
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    mgr = KVCacheManager(
        KvCacheConfig(max_tokens=max_tokens, enable_block_reuse=False),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=case.batch_size,
        mapping=mapping,
        dtype=(
            _DTYPE_TO_BINDING[torch.float8_e4m3fn]
            if case.kv_cache_fp8
            else _DTYPE_TO_BINDING[case.dtype]
        ),
    )
    request_ids = list(range(case.batch_size))
    token_nums = [per_seq] * case.batch_size
    mgr.add_dummy_requests(request_ids, token_nums)
    return mgr


def _build_mla_gen_metadata(case: AttnCase, kv_mgr: KVCacheManager):
    request_ids = list(range(case.batch_size))
    kv_cache_params = KVCacheParams(
        use_cache=True, num_cached_tokens_per_seq=[case.num_cached_tokens] * case.batch_size
    )
    seq_lens = torch.tensor([case.seq_len] * case.batch_size, dtype=torch.int)
    kwargs = dict(
        seq_lens=seq_lens,
        request_ids=request_ids,
        max_num_requests=case.batch_size,
        num_contexts=0,  # generation phase
        prompt_lens=[case.num_cached_tokens + case.seq_len] * case.batch_size,
        max_num_tokens=8192,
        kv_cache_manager=kv_mgr,
        kv_cache_params=kv_cache_params,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        # FlashMLA decode kernel is sm90-only — that is where the DeepSeek-R1
        # gen regressions live; on other archs this routes to the generic kernel.
        enable_flash_mla=(torch.cuda.get_device_capability() == (9, 0)),
    )
    metadata = TrtllmAttention.Metadata(**kwargs)
    metadata.prepare()
    return metadata


def _make_mla_gen_inputs(case: AttnCase, device: torch.device):
    """Return (fused_q, q_pe, latent_cache, scratch) for one MLA decode step.

    No fused QKV: MLA decode takes separately-projected fused_q + q_pe +
    compressed latent (see test_attention_mla.py:648-697, 984-1015). bf16 KV ->
    the FP8 scale/quant buffers are None.
    """
    torch.manual_seed(case.seed)
    nnz = case.batch_size * case.seq_len  # seq_len = generation_seq_len_q (1)
    kvr, rope = case.kv_lora_rank, case.qk_rope_head_dim
    fused_q = torch.empty(
        nnz, case.num_heads * (kvr + rope), dtype=case.dtype, device=device
    ).uniform_(-1, 1)
    q_pe = torch.empty(nnz, case.num_heads, rope, dtype=case.dtype, device=device).uniform_(-1, 1)
    compressed_kv = torch.empty(nnz, kvr, dtype=case.dtype, device=device).uniform_(-1, 1)
    k_pe = torch.empty(nnz, rope, dtype=case.dtype, device=device).uniform_(-1, 1)
    latent_cache = torch.cat([compressed_kv, k_pe], dim=-1)  # (nnz, 576)
    n = case.batch_size
    scratch = dict(
        cu_q_seqlens=torch.empty(n + 1, dtype=torch.int32, device=device),
        cu_kv_seqlens=torch.empty(n + 1, dtype=torch.int32, device=device),
        fmha_scheduler_counter=torch.empty(1, dtype=torch.uint32, device=device),
    )
    if case.kv_cache_fp8:
        # fp8 KV path needs the bmm dequant scales + a uint8 quantized-q scratch.
        scratch["mla_bmm1_scale"] = torch.empty(2, dtype=torch.float32, device=device)
        scratch["mla_bmm2_scale"] = torch.empty(1, dtype=torch.float32, device=device)
        scratch["quant_q_buffer"] = torch.empty(
            nnz, case.num_heads * (kvr + rope), dtype=torch.uint8, device=device
        )
    else:
        scratch["mla_bmm1_scale"] = None
        scratch["mla_bmm2_scale"] = None
        scratch["quant_q_buffer"] = None
    return fused_q, q_pe, latent_cache, scratch


# --------------------------------------------------------------------------- #
# DSA (DeepSeek-V3.2 Sparse Attention) isolation path — CONTEXT/prefill phase.
# Mirrors tests/unittest/_torch/attention/sparse/test_sparse_mla_forward.py
# ::test_forward_sparse_mla_unified (the pure-context branch). DSA = MLA + a
# sparse Indexer: the module builds DSATrtllmAttention (subclass of TrtllmAttention)
# which owns an Indexer submodule; the forward is two-stage:
#   topk = mla.mqa.indexer(qr, hidden_states, metadata, position_ids)
#   mla.forward_context_dsa(q, compressed_kv, k_pe, metadata, output,
#                           latent_cache, topk_indices=topk, position_ids=...)
# Unlike the GQA/MLA paths this needs the FULL MLA module (weights + indexer),
# a DSACacheManager (indexer-K pool), and DSAtrtllmAttentionMetadata.
#
# ARCH: the indexer's DeepGEMM MQA-logits kernels exist only for sm90/sm100/sm103.
# On sm120 (and any arch without DeepGEMM) construction/forward raises; the caller
# converts that into a bootstrap-skip. seq_len is pinned > index_topk so the
# indexer actually selects a sparse subset (else it degenerates to dense MHA).
# --------------------------------------------------------------------------- #


def _dsa_sparse_config(case: AttnCase):
    from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig

    return DeepSeekSparseAttentionConfig(
        index_n_heads=case.index_n_heads,
        index_head_dim=case.index_head_dim,
        index_topk=case.index_topk,
    )


def _dsa_model_config(sparse_config):
    from types import SimpleNamespace

    from tensorrt_llm._torch.model_config import ModelConfig

    return ModelConfig(
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        sparse_attention_config=sparse_config,
        pretrained_config=SimpleNamespace(rms_norm_eps=1e-6),
    )


def _build_dsa_mla_module(case: AttnCase, model_config, device: torch.device):
    """Build a single-layer MLA module with a DSA (sparse) backend + init weights.

    Mirrors test_sparse_mla_forward.py's per-layer construction + weight init.
    Random-normal weights (std 0.02) are fine: the microbench measures kernel
    structure/time, not numerical correctness.
    """
    from tensorrt_llm._torch.modules.mla import MLA

    pos_embd_params, _ = _mla_pos_embd_and_scaling(case)
    mla = MLA(
        hidden_size=case.hidden_size,
        num_attention_heads=case.num_heads,
        num_key_value_heads=1,
        qk_nope_head_dim=case.qk_nope_head_dim,
        qk_rope_head_dim=case.qk_rope_head_dim,
        v_head_dim=case.v_head_dim,
        q_lora_rank=case.q_lora_rank,
        kv_lora_rank=case.kv_lora_rank,
        predicted_tokens_per_seq=1,
        max_position_embeddings=case.max_position_embeddings,
        bias=False,
        pos_embd_params=pos_embd_params,
        layer_idx=0,
        dtype=case.dtype,
        config=model_config,
    ).to(device)

    std = 0.02
    with torch.no_grad():
        mla.kv_b_proj.weight.normal_(mean=0.0, std=std)
        kv_b = mla.kv_b_proj.weight.data.view(
            case.num_heads, case.qk_nope_head_dim + case.v_head_dim, case.kv_lora_rank
        )
        mla.v_b_proj.data = kv_b[:, case.qk_nope_head_dim :, :].contiguous()
        mla.k_b_proj_trans.data = kv_b[:, : case.qk_nope_head_dim, :].transpose(1, 2).contiguous()
        mla.mqa.indexer.wq_b.weight.normal_(mean=0.0, std=std)
        mla.mqa.indexer.wk.weight.normal_(mean=0.0, std=std)
        mla.mqa.indexer.weights_proj.weight.normal_(mean=0.0, std=std)
        mla.mqa.indexer.post_load_weights()  # fuse wk + weights_proj
    return mla


def _build_dsa_ctx(case: AttnCase, device: torch.device):
    """Return (mla, metadata, fwd_inputs, kv_mgr) for a pure-context DSA forward.

    Pure prefill: batch_size requests of seq_len new tokens, no cached tokens.
    """
    from tensorrt_llm._torch.attention_backend.sparse.dsa import DSACacheManager
    from tensorrt_llm._torch.attention_backend.utils import get_attention_backend

    sparse_config = _dsa_sparse_config(case)
    model_config = _dsa_model_config(sparse_config)
    mla = _build_dsa_mla_module(case, model_config, device)

    qk_head_dim = case.qk_nope_head_dim + case.qk_rope_head_dim
    head_size = case.kv_lora_rank + case.qk_rope_head_dim  # MLA latent dim (576)
    tokens_per_block = _mla_tokens_per_block()
    per_seq = case.seq_len  # context: seq_len tokens, no cache
    max_seq_len = (per_seq + tokens_per_block - 1) // tokens_per_block * tokens_per_block

    kv_mgr = DSACacheManager(
        KvCacheConfig(max_tokens=max_seq_len * case.batch_size, enable_block_reuse=False),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=head_size,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=case.batch_size,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=_DTYPE_TO_BINDING[case.dtype],
        sparse_attn_config=sparse_config,
        model_config=model_config,
    )
    request_ids = list(range(case.batch_size))
    kv_mgr.add_dummy_requests(
        request_ids=request_ids,
        token_nums=[case.seq_len] * case.batch_size,
        is_gen=False,
        prepare_resource=True,
    )

    AttentionCls = get_attention_backend("TRTLLM", sparse_config)
    metadata = AttentionCls.Metadata(
        seq_lens=torch.tensor([case.seq_len] * case.batch_size, dtype=torch.int),
        request_ids=request_ids,
        max_num_requests=case.batch_size,
        num_contexts=case.batch_size,
        prompt_lens=[case.seq_len] * case.batch_size,
        max_num_tokens=case.seq_len * case.batch_size,
        kv_cache_manager=kv_mgr,
        kv_cache_params=KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=[0] * case.batch_size
        ),
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        sparse_attention_config=sparse_config,
    )
    metadata.prepare()

    torch.manual_seed(case.seed)
    total = case.batch_size * case.seq_len
    q = torch.randn(total, case.num_heads * qk_head_dim, dtype=case.dtype, device=device)
    compressed_kv = torch.randn(total, case.kv_lora_rank, dtype=case.dtype, device=device)
    k_pe = torch.randn(total, case.qk_rope_head_dim, dtype=case.dtype, device=device)
    hidden_states = torch.randn(total, case.hidden_size, dtype=case.dtype, device=device)
    qr = torch.randn(total, case.q_lora_rank, dtype=case.dtype, device=device)
    latent_cache = torch.cat([compressed_kv, k_pe], dim=-1)
    position_ids = torch.cat(
        [
            torch.arange(0, case.seq_len, device=device, dtype=torch.int32)
            for _ in range(case.batch_size)
        ]
    )
    output = torch.empty(total, case.num_heads * case.v_head_dim, dtype=case.dtype, device=device)
    fwd = dict(
        q=q,
        compressed_kv=compressed_kv,
        k_pe=k_pe,
        hidden_states=hidden_states,
        qr=qr,
        latent_cache=latent_cache,
        position_ids=position_ids,
        output=output,
    )
    return mla, metadata, fwd, kv_mgr


# --------------------------------------------------------------------------- #
# the one entry point — run one case, return all signals
# --------------------------------------------------------------------------- #


def collect_signals(
    case: AttnCase, *, warmup: int = 10, iters: int = 50, lock_clock: bool = False
) -> Signals:
    """Run one case in a FRESH subprocess and return its signals.

    Per-case process isolation is mandatory for the discrete metrics: a process
    accumulates CUPTI activity records, autotuner/JIT tactic caches, and CUDA-graph
    capture state, so launch_count measured in a shared process depends on which
    cases ran before (observed: 49 -> 59 -> 219). A fresh process makes the count a
    pure function of this case's code + config. ATTN_INPROC=1 (set in the child)
    short-circuits to the in-process path to avoid infinite re-spawn.
    """
    if os.environ.get("ATTN_INPROC") == "1":
        return _collect_signals_inproc(case, warmup=warmup, iters=iters, lock_clock=lock_clock)
    case_d = dataclasses.asdict(case)
    # torch.dtype isn't JSON-serializable — round-trip it by name (e.g. bfloat16).
    case_d["dtype"] = str(case.dtype).rsplit(".", 1)[-1]
    payload = json.dumps(
        {
            "case": case_d,
            "warmup": warmup,
            "iters": iters,
            "lock_clock": lock_clock,
        }
    )
    env = {
        **os.environ,
        "ATTN_INPROC": "1",
        "PYTHONPATH": os.pathsep.join(p for p in sys.path if p),
    }
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), payload],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    marker = "ATTN_SIGNALS_JSON:"
    line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith(marker)), None)
    if line is None:
        raise RuntimeError(
            f"isolated case {case.case_id} produced no signals (rc={proc.returncode}).\n"
            f"--- child stderr (tail) ---\n{os.linesep.join(proc.stderr.splitlines()[-25:])}"
        )
    return Signals(**json.loads(line[len(marker) :]))


def _collect_signals_inproc(
    case: AttnCase, *, warmup: int = 10, iters: int = 50, lock_clock: bool = False
) -> Signals:
    """Run one pinned forward and return discrete + continuous signals.

    Discrete signals are cheap and gate-able pre-merge; gpu_time is the
    independent continuous detector (notes §11.5). Both come from one forward.
    Always invoked inside the isolated child (see collect_signals).
    """
    if not torch.cuda.is_available():
        raise RuntimeError("attention perf harness requires a CUDA device")
    device = torch.device("cuda")
    sig = Signals(case_id=case.case_id, arch=sm_arch(), phase=case.phase, shape=case.shape_tag())

    if case.is_dsa:
        mla, metadata, fwd, kv_mgr = _build_dsa_ctx(case, device)
    else:
        kv_mgr = _build_mla_kv_cache_manager(case) if case.is_mla else _build_kv_cache_manager(case)
    try:
        if case.is_dsa:
            # DSA context: indexer (sparse top-k select) + sparse MLA forward.
            # use_paged_context_fmha is recorded for parity (MLA-family forces it
            # off); the DSA tripwire is launch_count (indexer + sparse-FMHA chain).
            sig.use_paged_context_fmha = bool(getattr(metadata, "use_paged_context_fmha", None))

            def _fwd():
                topk = mla.mqa.indexer(
                    fwd["qr"], fwd["hidden_states"], metadata, fwd["position_ids"]
                )
                return mla.forward_context_dsa(
                    q=fwd["q"],
                    compressed_kv=fwd["compressed_kv"],
                    k_pe=fwd["k_pe"],
                    attn_metadata=metadata,
                    output=fwd["output"],
                    latent_cache=fwd["latent_cache"],
                    topk_indices=topk,
                    position_ids=fwd["position_ids"],
                )
        elif case.is_mla:
            # MLA decode: latent KV + yarn RoPE + two-call forward. num_kv_heads
            # / head_dim from AttnCase are ignored (derived from lora ranks).
            metadata = _build_mla_gen_metadata(case, kv_mgr)
            # use_paged_context_fmha is force-set False for MLA (trtllm.py) — the
            # signal is not a meaningful tripwire here, but record it for parity.
            sig.use_paged_context_fmha = bool(getattr(metadata, "use_paged_context_fmha", None))
            pos_embd_params, q_scaling = _mla_pos_embd_and_scaling(case)
            backend = TrtllmAttention(
                layer_idx=0,
                num_heads=case.num_heads,
                head_dim=_mla_kv_head_dim(case),
                num_kv_heads=1,
                quant_config=_mla_quant_config(case),
                q_scaling=q_scaling,
                pos_embd_params=pos_embd_params,
                mla_params=_mla_params(case),
            )
            fused_q, q_pe, latent_cache, scratch = _make_mla_gen_inputs(case, device)

            def _fwd():
                # mla_rope_generation mutates fused_q/latent_cache and appends to
                # the paged cache in place; timing the pair is stable (same shapes
                # / same slot each iter). Both calls = the "MLA decode" structure.
                backend.mla_rope_generation(
                    fused_q,
                    q_pe,
                    latent_cache,
                    metadata,
                    scratch["cu_q_seqlens"],
                    scratch["cu_kv_seqlens"],
                    scratch["fmha_scheduler_counter"],
                    scratch["mla_bmm1_scale"],
                    scratch["mla_bmm2_scale"],
                    scratch["quant_q_buffer"],
                )
                return backend.forward(
                    fused_q,
                    None,
                    None,
                    metadata,
                    attention_input_type=AttentionInputType.generation_only,
                    latent_cache=latent_cache,
                    q_pe=q_pe,
                    cu_q_seqlens=scratch["cu_q_seqlens"],
                    cu_kv_seqlens=scratch["cu_kv_seqlens"],
                    fmha_scheduler_counter=scratch["fmha_scheduler_counter"],
                    mla_bmm1_scale=scratch["mla_bmm1_scale"],
                    mla_bmm2_scale=scratch["mla_bmm2_scale"],
                    quant_q_buffer=scratch["quant_q_buffer"],
                )
        else:
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
                except Exception as e:  # noqa: BLE001 - teardown must not fail the test
                    # Keep the measurement, but surface driver/NVML teardown
                    # errors instead of hiding them (helps diagnose flaky locks).
                    print(f"[warn] clock teardown failed: {e}")

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


# --------------------------------------------------------------------------- #
# child entry point: `python attention_perf_harness.py '<json payload>'`
# Runs ONE case in this fresh process and emits its signals as JSON. Invoked by
# collect_signals() for per-case isolation; never call this path directly.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    os.environ["ATTN_INPROC"] = "1"  # guard against re-spawn
    _payload = json.loads(sys.argv[1])
    _cd = _payload["case"]
    _cd["dtype"] = getattr(torch, _cd["dtype"])  # name -> torch.dtype
    _case = AttnCase(**_cd)
    _sig = _collect_signals_inproc(
        _case, warmup=_payload["warmup"], iters=_payload["iters"], lock_clock=_payload["lock_clock"]
    )
    print("ATTN_SIGNALS_JSON:" + json.dumps(dataclasses.asdict(_sig)))
