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

"""
FlashInfer TRTLLM-Gen FMHA

This module implements attention computation using flashinfer's trtllm-gen kernels.
It provides a TRT-LLM attention FMHA library for trtllm-gen kernels
(Blackwell architecture: SM100/SM103). Enable or disable it through
``TLLM_FMHA_LIBS``.

Architecture:
    - QKV preprocessing & RoPE: C++ kernels via tensorrt_llm.bindings.internal.thop,
      same as thop.attention. Writes K/V to paged KV cache via pool pointers.
    - Attention: flashinfer trtllm-gen FMHA kernels, reading KV cache through
      the paged KV cache fields carried by FmhaParams.

Entry points:
    FlashInferTrtllmGenFmha.is_available() - Check if this FMHA library can be instantiated.
    FlashInferTrtllmGenFmha.is_supported() - Check if trtllm-gen can handle the given request.
    FlashInferTrtllmGenFmha.forward()      - Main attention method.

Example:
    fmha = FlashInferTrtllmGenFmha(attn=...)
    if fmha.is_supported(q, k, v, metadata, forward_args):
        fmha.forward(q, k, v, metadata, forward_args)
"""

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE, get_env_enable_pdl

if IS_FLASHINFER_AVAILABLE:
    import flashinfer

from tensorrt_llm._torch import q35_env
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm._torch.attention.backends.sparse.skip_softmax import SkipSoftmaxParams
from tensorrt_llm._utils import get_sm_version, torch_dtype_to_binding
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantMode

from .interface import FmhaPhase
from .phased import FmhaParams, PhasedFmha
from .q35_cp import q35_cp_decode_ll
from .q35_fused_decode import (
    q35_fused_decode,
    q35_fused_decode_fold,
    q35_fused_decode_fold_supported,
    q35_fused_decode_supported,
    q35_fused_workspace_floats,
)
from .q35_segmented import q35_segmented_decode, q35_segmented_supported
from .utils import (
    get_attention_chunk_size,
    get_bmm1_scale,
    get_multi_processor_count_for_device,
    get_trtllm_gen_context_workspace_size,
)

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


def _install_flashinfer_mla_decode_tuning_config_cache() -> None:
    """Work around an autotuner cache pathology on flashinfer 0.6.15's MLA decode path.

    ``flashinfer.mla._core._build_mla_decode_tuning_config`` builds a fresh
    ``TuningConfig`` — with fresh initializer closures — on every
    ``trtllm_batch_decode_with_kv_cache_mla`` call. ``DynamicTensorSpec``'s
    custom ``__hash__`` skips ``tensor_initializers``, but its
    dataclass-generated ``__eq__`` compares them by value, so each call's
    config is hash-equal but eq-unequal to all previous ones: the
    ``lru_cache`` on ``AutoTuner._find_nearest_profile`` accumulates
    colliding keys up to the cache cap, and every autotuner cache probe on
    the MLA decode path then walks the entire collision chain in Python
    ``__eq__`` (tens of milliseconds per attention call, per layer).

    The workaround memoizes the config builder on the scalars that fully
    determine its output (kv-cache paging geometry, workspace size, runner
    set, head/rank dims, max_seq_len, device), so repeated calls reuse the
    same ``TuningConfig`` object. Cache hits then resolve through
    flashinfer's own unmodified hash/eq — no equality semantics change
    anywhere, and the scope is exactly the MLA decode path that exhibits the
    problem.

    MLA decode is the only path that needs this: every other ``TuningConfig``
    construction site in flashinfer 0.6.15 already reuses its config across
    calls (module-level constants in ``gemm``, instance-level configs in the
    cute-dsl MoE tuner, ``functools.cache``-wrapped builders in
    ``mla/_sparse_mla_sm120.py``), so those specs hit the autotuner caches by
    object identity and never accumulate collision chains. Memoizing the one
    per-call builder therefore covers the whole pathology; extending it to
    other ops would add nothing.

    The bug is upstream in flashinfer; the patch is applied only when a
    behavioral probe shows the hash/eq inconsistency and the builder still
    has the expected signature, so a fixed or refactored flashinfer release
    degrades this to a no-op.
    """
    try:
        import inspect

        from flashinfer.autotuner.autotuner import DynamicTensorSpec
        from flashinfer.mla import _core as _flashinfer_mla_core

        def _probe_spec() -> DynamicTensorSpec:
            return DynamicTensorSpec(
                input_idx=(0,),
                dim_idx=(0,),
                gen_tuning_buckets=(1,),
                map_to_tuning_buckets=_probe_spec,  # any stable callable
                tensor_initializers=[lambda shapes, dtype, device: None],
            )

        a, b = _probe_spec(), _probe_spec()
        if not (hash(a) == hash(b) and a != b):
            return  # flashinfer without the inconsistency; nothing to do

        orig_build = _flashinfer_mla_core._build_mla_decode_tuning_config
        if getattr(orig_build, "_trtllm_mla_tuning_config_cache", False):
            return  # already installed

        expected_params = (
            "kv_cache",
            "block_tables",
            "workspace_buffer",
            "runner_names",
            "q_len",
            "num_heads",
            "kv_lora_rank",
            "max_seq_len",
            "device",
        )
        signature = inspect.signature(orig_build)
        if tuple(signature.parameters) != expected_params:
            return  # builder was refactored; memo key may no longer be complete
        try:
            signature.bind(**dict.fromkeys(expected_params))
        except TypeError:
            return  # parameters went positional-only; the keyword call below would fail

        cache: dict = {}

        def _cached_build(
            kv_cache,
            block_tables,
            workspace_buffer,
            runner_names,
            q_len,
            num_heads,
            kv_lora_rank,
            max_seq_len,
            device,
        ):
            # Everything the builder (and _compute_mla_decode_buckets) reads
            # from its tensor arguments reduces to these scalars.
            key = (
                kv_cache.shape[0],  # num_pages, captured by init_block_tables
                kv_cache.shape[-2],  # page_size, feeds profile_seq_len
                block_tables.shape[-1],  # pages per sequence, feeds profile_seq_len
                workspace_buffer.numel() * workspace_buffer.element_size(),  # cute-dsl bucket cap
                tuple(runner_names),
                q_len,
                num_heads,
                kv_lora_rank,
                max_seq_len,
                device,
            )
            config = cache.get(key)
            if config is None:
                if len(cache) >= 256:  # backstop; a serving process sees a handful of keys
                    cache.clear()
                config = orig_build(
                    kv_cache=kv_cache,
                    block_tables=block_tables,
                    workspace_buffer=workspace_buffer,
                    runner_names=runner_names,
                    q_len=q_len,
                    num_heads=num_heads,
                    kv_lora_rank=kv_lora_rank,
                    max_seq_len=max_seq_len,
                    device=device,
                )
                cache[key] = config
            return config

        _cached_build._trtllm_mla_tuning_config_cache = True
        _flashinfer_mla_core._build_mla_decode_tuning_config = _cached_build
        logger.debug(
            "Memoized flashinfer _build_mla_decode_tuning_config "
            "(autotuner cache collision-chain workaround)."
        )
    except (ImportError, AttributeError, TypeError, ValueError):
        # A future flashinfer refactor (moved module, changed fields) must not
        # break attention; it just loses the workaround.
        # tensorrt_llm's logger.debug(*msg) takes no exc_info/kwargs.
        logger.debug("Skipping flashinfer MLA decode tuning-config cache workaround.")


if IS_FLASHINFER_AVAILABLE:
    _install_flashinfer_mla_decode_tuning_config_cache()


_MULTI_CTAS_KV_COUNTER_ALIGNMENT = 8

# Qwen3.5 decode paths of run_generation (generation-only, one KV head per rank, FP8 KV, uniform q_len), in order of
# precedence: CP-lite (q35_cp.py, KV head replicated over attn.q35_cp_group), fused decode (q35_fused_decode.py),
# segmented (q35_segmented.py). Thresholds tuned for Qwen3.5-397B on SM 10.3; a layout above batch 64 raises the
# CP-lite bound in its run file (TRTLLM_Q35_ATTN_CP_MAX_BATCH, TEP8 b128: 128).
_Q35_ATTN_CP_MIN_BATCH = 2
_Q35_ATTN_CP_MAX_BATCH = int(os.environ.get("TRTLLM_Q35_ATTN_CP_MAX_BATCH", "64"))
_Q35_ATTN_FUSED_MAX_BATCH = 4
_Q35_ATTN_SEGMENTS = 16
# At b4 the trtllm-gen kernel is already bandwidth-bound and segmenting loses ~3 us per call.
_Q35_ATTN_SEG_MAX_BATCH = 2
# Counter slots for segmented / CP-lite calls: >= max(b * segments * folded heads, #SMs). segments * folded heads
# is <= 128 here (MTP-6 verify: 1 x 7 x 16 = 112), so 128 slots per request up to the CP-lite batch cap (>= 64):
# TEP8 b80 / b128 need 8960 / 14336 slots, more than the 64 * 128 that capped CP-lite at b73.
_Q35_COUNTER_SLOTS = max(64, _Q35_ATTN_CP_MAX_BATCH) * 128


def _q35_cp_segments(num_requests: int, q_len: int) -> int:
    """Local segments per CP rank of a CP-lite call."""
    return 2 if (num_requests <= 2 or q_len == 1) else 1


@dataclass
class Q35FoldInputs:
    """Raw inputs of the qk-norm / RoPE / gate / KV-append prologue of a folded call (see q35_fold_inputs)."""

    # Raw QKV projection [num_tokens, (2 * Hq + 2 * Hkv) * D] bf16 ((q, gate) per head, K, V).
    qkv: torch.Tensor
    q_weight: torch.Tensor
    k_weight: torch.Tensor
    cos_sin: torch.Tensor
    positions: torch.Tensor  # [T] or [3, T] (interleaved mrope), contiguous
    eps: float
    rotary_dim: int
    mrope_section: Optional[tuple]


@dataclass
class Q35KvAppendPlan:
    """Paged FP8 destination of a generation-only call whose FP8 query / K / V the attention module writes.

    The module fills q_fp8 / scales and hands the plan over through ``attn.q35_kv_appended``.
    """

    layer_idx: int
    num_tokens: int
    num_requests: int
    q_len: int
    kv_pool: torch.Tensor  # FP8 [blocks, num_kv_heads, tokens_per_block, head_dim]
    block_tables: torch.Tensor  # int32 [num_requests, 2 (K, V), max_pages]
    seq_lens: torch.Tensor  # int32, KV length including the new tokens
    kv_scale_orig_quant: torch.Tensor
    kv_scale_quant_orig: torch.Tensor
    host_bmm1_scale: float
    q_fp8: Optional[torch.Tensor] = None  # [num_tokens, num_heads, head_dim]
    # float32 [8]: bmm1, bmm1 * log2e at [0:2]; bmm2 at [4] (both views 16-byte aligned)
    scales: Optional[torch.Tensor] = None
    # Set when the module skipped the prologue kernel; run_generation then runs the folded kernel or the prologue.
    fold: Optional[Q35FoldInputs] = None


# Shared by every attention layer (the calls are serialized on the decode stream): device -> (workspace, counters).
_Q35_FUSED_BUFFERS: dict = {}


def _q35_fused_buffers(device: torch.device, sm_count: int, capturing: bool):
    key = device.index if device.index is not None else torch.cuda.current_device()
    bufs = _Q35_FUSED_BUFFERS.get(key)
    if bufs is None:
        if capturing:
            raise RuntimeError(
                "The Q35 fused-decode workspace must be allocated before CUDA graph capture."
            )
        ws = torch.zeros(q35_fused_workspace_floats(sm_count), dtype=torch.float32, device=device)
        # grid-barrier epoch counters: row b serves batch b; zero once, never reset (the kernel keeps them consistent)
        cnt = torch.zeros(_Q35_ATTN_FUSED_MAX_BATCH + 1, 8, dtype=torch.int64, device=device)
        bufs = (ws, cnt)
        _Q35_FUSED_BUFFERS[key] = bufs
    return bufs


def _get_multi_ctas_kv_counter_size(
    num_heads: int,
    max_num_sequences: int,
    multi_processor_count: int,
) -> int:
    num_counters = max(num_heads * max_num_sequences, multi_processor_count)
    aligned_num_counters = (
        (num_counters + _MULTI_CTAS_KV_COUNTER_ALIGNMENT - 1)
        // _MULTI_CTAS_KV_COUNTER_ALIGNMENT
        * _MULTI_CTAS_KV_COUNTER_ALIGNMENT
    )
    return aligned_num_counters * torch.int32.itemsize


def _get_bmm1_scale_log2(bmm1_scale: torch.Tensor) -> torch.Tensor:
    if bmm1_scale.numel() < 2:
        raise RuntimeError("trtllm-gen bmm1_scale workspace must contain raw and log2 scales.")
    return bmm1_scale.narrow(0, 1, 1)


@lru_cache(maxsize=128)
def _get_generation_workspace_layout(
    dtype: torch.dtype,
    batch_beam: int,
    num_tokens: int,
    num_heads: int,
    head_size: int,
    num_kv_heads: int,
    rotary_embedding_dim: int,
) -> dict[str, int]:
    return thop.get_trtllm_gen_generation_workspace_layout(
        dtype,
        batch_beam,
        num_tokens,
        num_heads,
        head_size,
        rotary_embedding_dim,
        num_kv_heads,
    )


@lru_cache(maxsize=128)
def _get_generation_workspace_size(
    dtype: torch.dtype,
    max_num_seq: int,
    max_num_tokens: int,
    num_heads: int,
    head_size: int,
    num_kv_heads: int,
    rotary_embedding_dim: int,
) -> int:
    if max_num_tokens == 0:
        return 0
    if num_kv_heads <= 0:
        num_kv_heads = num_heads
    layout = _get_generation_workspace_layout(
        dtype,
        max_num_seq,
        max_num_tokens,
        num_heads,
        head_size,
        num_kv_heads,
        rotary_embedding_dim,
    )
    return int(layout["total_size"])


@lru_cache(maxsize=128)
def _get_workspace_size(
    dtype: torch.dtype,
    num_tokens: int,
    num_gen_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    max_num_requests: int,
    rotary_embedding_dim: int,
    fp8_context_fmha: bool,
) -> int:
    context_size = get_trtllm_gen_context_workspace_size(
        dtype,
        max_num_requests,
        num_tokens,
        num_heads,
        head_size,
        rotary_embedding_dim,
        fp8_context_fmha,
    )
    generation_size = _get_generation_workspace_size(
        dtype,
        max_num_requests,
        num_gen_tokens,
        num_heads,
        head_size,
        num_kv_heads,
        rotary_embedding_dim,
    )
    return max(context_size, generation_size)


class FlashInferTrtllmGenFmha(PhasedFmha):
    """
    An attention backend using pure trtllm-gen kernels from flashinfer.
    """

    # Default KV layout for flashinfer
    # HND = [max_num_pages, kv_factor, num_kv_heads, page_size, head_dim]
    DEFAULT_KV_LAYOUT = "HND"
    # Keep shared paged indices disabled to match the current TensorRT-LLM
    # block-table layout used by the fused preprocessing path.
    USE_SHARED_PAGED_KV_IDX = False

    # Supported data types
    SUPPORTED_INPUT_DTYPES = {torch.float16, torch.bfloat16, torch.float8_e4m3fn}
    SUPPORTED_KV_CACHE_DTYPES = {DataType.HALF, DataType.BF16, DataType.FP8, DataType.NVFP4}
    SUPPORTED_OUT_DTYPES = {torch.float16, torch.bfloat16, torch.float8_e4m3fn}

    # Supported Q:KV:O dtype combinations for trtllm-gen kernels
    # Format: (q_dtype: torch.dtype, kv_dtype: DataType, o_dtype: torch.dtype)
    SUPPORTED_DTYPE_COMBOS_CONTEXT = {
        (torch.float8_e4m3fn, DataType.FP8, torch.float8_e4m3fn),
        (torch.float16, DataType.HALF, torch.float16),
        (torch.bfloat16, DataType.BF16, torch.bfloat16),
        (torch.float8_e4m3fn, DataType.FP8, torch.float16),
        (torch.float8_e4m3fn, DataType.FP8, torch.bfloat16),
        # e4m3:nvfp4:*
        (torch.float8_e4m3fn, DataType.NVFP4, torch.float8_e4m3fn),
        (torch.float8_e4m3fn, DataType.NVFP4, torch.float16),
        (torch.float8_e4m3fn, DataType.NVFP4, torch.bfloat16),
    }
    SUPPORTED_DTYPE_COMBOS_GENERATION = {
        (torch.float8_e4m3fn, DataType.FP8, torch.float8_e4m3fn),
        (torch.float16, DataType.HALF, torch.float16),
        (torch.bfloat16, DataType.BF16, torch.bfloat16),
        (torch.float8_e4m3fn, DataType.FP8, torch.float16),
        (torch.float8_e4m3fn, DataType.FP8, torch.bfloat16),
        (torch.bfloat16, DataType.FP8, torch.bfloat16),
        (torch.float16, DataType.FP8, torch.float16),
        # e4m3:nvfp4:*
        (torch.float8_e4m3fn, DataType.NVFP4, torch.float8_e4m3fn),
        (torch.float8_e4m3fn, DataType.NVFP4, torch.float16),
        (torch.float8_e4m3fn, DataType.NVFP4, torch.bfloat16),
    }

    # 96 is excluded because trtllm-gen does not ship context kernels for it.
    UNSUPPORTED_HEAD_SIZES_CONTEXT = {72, 80, 96}
    MAX_HEADS_RATIO_GENERATION = 32
    MIN_TOKENS_PER_BLOCK = 8
    SUPPORTED_TOKENS_PER_BLOCK = {16, 32, 64}
    # FlashInfer narrows key_cache.size(0) to int before constructing its TMA descriptor.
    MAX_NUM_PAGES_IN_MEM_POOL = (1 << 31) - 1
    SUPPORTED_MLA_GENERATION_HEAD_DIMS = {
        (320, 256),
        (576, 512),
    }
    # (headDimQk, headDimV, tokens_per_block) whose trtllm-gen MLA decode kernel is
    # slower than the thop.attention fallback, so this backend declines them.
    SLOWER_MLA_GENERATION_KERNELS = {
        (576, 512, 32),
    }

    def __init__(self, attn: "TrtllmAttention") -> None:
        super().__init__(attn)
        self._layout = self.DEFAULT_KV_LAYOUT
        # Read once so the hot path is not sensitive to later environment changes.
        self._enable_pdl = get_env_enable_pdl()

        # Lazily set on the first forward() call from the query device.
        self._multi_processor_count: Optional[int] = None
        self._multi_ctas_kv_counter_buffer: Optional[torch.Tensor] = None
        # Qwen3.5 segmented / CP-lite calls need max(b * segments * heads, SMs) counters, more than the regular
        # buffer holds. Zero once; the kernel resets its counters at the end of every launch.
        self._q35_counter_buffer: Optional[torch.Tensor] = None

    @classmethod
    def is_environment_available(cls) -> bool:
        """Check dependencies and architecture before constructing an attention layer."""
        if not IS_FLASHINFER_AVAILABLE:
            logger.debug("FlashInfer TRTLLM-Gen FMHA is unavailable: flashinfer is not installed.")
            return False

        missing_ops = cls._missing_fused_nanobind_ops()
        if missing_ops:
            logger.debug(
                "FlashInfer TRTLLM-Gen FMHA is unavailable: missing fused "
                f"nanobind ops: {', '.join(missing_ops)}."
            )
            return False

        sm = get_sm_version()
        if sm not in (100, 103):
            logger.debug(
                f"FlashInfer TRTLLM-Gen FMHA is unavailable: requires SM100 or SM103, got SM{sm}."
            )
            return False

        return True

    @classmethod
    def _is_available(cls, attn: "TrtllmAttention") -> bool:
        if not cls.is_environment_available():
            return False

        has_skip_softmax = isinstance(attn.sparse_params, SkipSoftmaxParams)
        if has_skip_softmax:
            logger.debug(
                "FlashInfer TRTLLM-Gen FMHA is unavailable: skip-softmax attention is enabled."
            )
            return False

        if attn.num_heads <= 0 or attn.num_kv_heads <= 0:
            logger.debug(
                "FlashInfer TRTLLM-Gen FMHA is unavailable: "
                f"num_heads={attn.num_heads}, num_kv_heads={attn.num_kv_heads}."
            )
            return False

        if attn.num_heads % attn.num_kv_heads != 0:
            logger.debug(
                "FlashInfer TRTLLM-Gen FMHA is unavailable: "
                f"num_heads ({attn.num_heads}) must be divisible by "
                f"num_kv_heads ({attn.num_kv_heads})."
            )
            return False

        return True

    @property
    def layout(self) -> str:
        """KV cache layout."""
        return self._layout

    def _get_kv_scale_params(
        self,
        forward_args: AttentionForwardArgs,
        quant_mode: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        kv_cache_quant_mode = QuantMode(quant_mode)
        kv_scale_orig_quant = forward_args.kv_scale_orig_quant
        kv_scale_quant_orig = forward_args.kv_scale_quant_orig
        if (
            not kv_cache_quant_mode.has_kv_cache_quant()
            or kv_scale_orig_quant is None
            or kv_scale_quant_orig is None
        ):
            return None, None

        if kv_cache_quant_mode.has_fp4_kv_cache():
            assert kv_scale_orig_quant.size(0) == 3, (
                f"kv_scale_orig_quant must have size(0)==3 for FP4, got {kv_scale_orig_quant.size(0)}"
            )
            assert kv_scale_quant_orig.size(0) == 3, (
                f"kv_scale_quant_orig must have size(0)==3 for FP4, got {kv_scale_quant_orig.size(0)}"
            )

        return kv_scale_orig_quant, kv_scale_quant_orig

    @staticmethod
    def _get_kv_cache_dtype(
        meta: "TrtllmAttentionMetadata",
    ) -> Optional[DataType]:
        kv_cache_manager = meta.kv_cache_manager
        if kv_cache_manager is not None:
            return kv_cache_manager.dtype
        return None

    @classmethod
    def _check_mla_generation_support(
        cls,
        head_size: int,
        num_heads: int,
        tokens_per_block: int,
        kv_lora_rank: Optional[int],
        qk_rope_head_dim: Optional[int],
    ) -> Tuple[bool, str]:
        missing_params = [
            name
            for name, value in (
                ("kv_lora_rank", kv_lora_rank),
                ("qk_rope_head_dim", qk_rope_head_dim),
            )
            if value is None or value <= 0
        ]
        if missing_params:
            return (
                False,
                "[Generation][MLA] missing required MLA parameter(s): "
                f"{', '.join(missing_params)}.",
            )

        kv_rank = int(kv_lora_rank)
        qk_rope_dim = int(qk_rope_head_dim)
        head_dim_qk = kv_rank + qk_rope_dim
        head_dim_v = kv_rank
        if head_size != head_dim_qk:
            return (
                False,
                f"[Generation][MLA] head_size ({head_size}) that does not match "
                f"kv_lora_rank + qk_rope_head_dim ({head_dim_qk}).",
            )

        if (head_dim_qk, head_dim_v) not in cls.SUPPORTED_MLA_GENERATION_HEAD_DIMS:
            supported = sorted(cls.SUPPORTED_MLA_GENERATION_HEAD_DIMS)
            return (
                False,
                f"[Generation][MLA] head dimensions "
                f"headDimQk={head_dim_qk}, headDimV={head_dim_v}. Supported: {supported}.",
            )

        # FlashInfer's TRTLLM-Gen MLA decode does not support 64 < H < 128.
        # Let selection fall through to the legacy TRTLLM-Gen backend.
        if 64 < num_heads < 128:
            return (
                False,
                "[Generation][MLA] FlashInfer TRTLLM-GEN backend is disabled "
                f"for num_heads={num_heads} (64 < num_heads < 128) "
                "as it may select unsupported kernels.",
            )

        if (head_dim_qk, head_dim_v, tokens_per_block) in cls.SLOWER_MLA_GENERATION_KERNELS:
            return (
                False,
                f"[Generation][MLA] slower TRTLLM-GEN decode kernel for "
                f"headDimQk={head_dim_qk}, headDimV={head_dim_v}, "
                f"tokens_per_block={tokens_per_block}.",
            )

        return True, ""

    def _is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        supported, reason = self._is_supported_with_reason(
            q,
            k,
            v,
            self.attn,
            metadata,
            forward_args,
            phase=phase,
        )
        if not supported:
            logger.debug(f"FlashInfer trtllm-gen fmha library does not support request: {reason}")
        return supported

    def _is_supported_with_reason(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        attn: "TrtllmAttention",
        meta: "TrtllmAttentionMetadata",
        fwd: AttentionForwardArgs,
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> Tuple[bool, str]:
        is_mla_enable = attn.is_mla_enable
        if phase is None:
            has_context_phase = fwd.attention_input_type != AttentionInputType.generation_only
            has_generation_phase = fwd.attention_input_type != AttentionInputType.context_only
        elif phase == FmhaPhase.CONTEXT:
            has_context_phase = True
            has_generation_phase = False
        elif phase == FmhaPhase.GENERATION:
            has_context_phase = False
            has_generation_phase = True
        else:
            return False, f"invalid FMHA phase: {phase}."

        has_fused_qkv = False
        has_q_only = False
        if not is_mla_enable and not meta.is_cross:
            q_hidden_size = attn.num_heads * attn.head_dim
            qkv_hidden_size = q_hidden_size + 2 * attn.num_kv_heads * attn.head_dim
            has_fused_qkv = (
                fwd.is_fused_qkv and k is None and v is None and q.size(-1) == qkv_hidden_size
            )
            has_q_only = (
                not fwd.is_fused_qkv
                and not fwd.update_kv_cache
                and k is None
                and v is None
                and q.size(-1) == q_hidden_size
            )

        kv_cache_dtype = self._get_kv_cache_dtype(meta)
        # FP8/NVFP4 KV caches use a low-precision context kernel even when the
        # projection output is FP16/BF16, so the FP16/BF16 fallbacks do not apply.
        has_low_precision_kv_cache = kv_cache_dtype in (DataType.FP8, DataType.NVFP4)

        if (
            phase in (None, FmhaPhase.CONTEXT)
            and not has_low_precision_kv_cache
            and q.dtype == torch.bfloat16
            and 0 < meta.num_contexts <= 4
            and attn.head_dim != 512
            and not has_q_only
            and meta.num_generations == 0
            and not (
                meta.use_paged_context_fmha
                and meta.tokens_per_block not in self.SUPPORTED_TOKENS_PER_BLOCK
            )
        ):
            # NVBug 6579626: the per-layer host overhead of the FlashInfer
            # TRTLLM-Gen context path regresses TTFT for small BF16 batches.
            # Let the FMHA selector choose the fallback implementation. H512
            # and Q-only cached-KV requests cannot use that fallback. Neither
            # can paged prefill with model-specific page sizes such as M3's 128.
            # Keep mixed batches on FlashInfer: the monolithic fallback also
            # switches decoding to C++ kernels that generation-only warmup
            # does not cover, causing first-use JIT stalls (NVBug 6716104).
            return False, (
                "small-batch BF16 context attention uses the fallback FMHA for "
                "performance because the FlashInfer TRTLLM-Gen context path "
                "regresses TTFT due to per-layer host overhead."
            )

        sparse_params = attn.sparse_params
        has_skip_softmax = sparse_params is not None and sparse_params.algorithm == "skip_softmax"
        has_sparse_attention = sparse_params is not None and not has_skip_softmax
        if fwd.enable_dsv4_epilogue_fusion:
            return False, "trtllm-gen does not support DSv4 epilogue fusion."
        if (
            fwd.sage_attn_num_elts_per_blk_q > 0
            or fwd.sage_attn_num_elts_per_blk_k > 0
            or fwd.sage_attn_num_elts_per_blk_v > 0
        ):
            return False, "sage attention."
        if meta.helix_position_offsets is not None:
            return False, "helix parallelism."
        sparse_kv_indices = fwd.sparse_runtime_params.sparse_kv_indices
        sparse_attn_indices = fwd.sparse_runtime_params.sparse_attn_indices
        if (
            (sparse_kv_indices is not None and sparse_kv_indices.numel() > 0)
            or (sparse_attn_indices is not None and sparse_attn_indices.numel() > 0)
            or meta.num_sparse_topk > 0
            or has_sparse_attention
        ):
            return False, "sparse attention."
        if has_skip_softmax:
            return False, "skip-softmax attention."
        if fwd.relative_attention_bias is not None:
            return False, "relative attention bias."
        if meta.use_spec_decoding and meta.is_spec_dec_tree:
            return False, "spec-dec tree/custom masks."
        if is_mla_enable and has_context_phase:
            return False, "MLA context attention."

        if meta.kv_cache_block_offsets is None:
            return False, "non-paged KV cache; paged KV cache is required."

        num_pages_in_mem_pool = self._get_total_num_blocks(meta)
        if num_pages_in_mem_pool > self.MAX_NUM_PAGES_IN_MEM_POOL:
            return False, (
                f"more than {self.MAX_NUM_PAGES_IN_MEM_POOL} flattened KV-cache "
                f"pages, but this pool requires {num_pages_in_mem_pool}."
            )

        output = fwd.output
        if output is None:
            return False, "a missing output tensor; output is required."

        tokens_per_block = meta.tokens_per_block
        if tokens_per_block is None:
            tokens_per_block = 0

        if not is_mla_enable and not meta.is_cross:
            if not has_fused_qkv and not has_q_only:
                return False, "self attention requires fused QKV or Q-only cached-KV input."
            if has_q_only and q.dtype == torch.float8_e4m3fn:
                return False, "Q-only cached-KV preprocessing supports FP16 or BF16 input."

        q_dtype = q.dtype
        o_dtype = output.dtype

        if q_dtype not in self.SUPPORTED_INPUT_DTYPES:
            return False, f"input dtype {q_dtype}. Supported: FP16, BF16, FP8 (E4M3)."

        if kv_cache_dtype is None:
            kv_cache_dtype = torch_dtype_to_binding(q_dtype)
        if meta.is_cross:
            if kv_cache_dtype == DataType.NVFP4:
                return False, "cross attention with NVFP4 KV cache."
            if is_mla_enable:
                return False, "cross attention with MLA."
            if meta.is_spec_decoding_enabled or meta.use_spec_decoding:
                return False, "cross attention with speculative decoding."
            if fwd.update_kv_cache and fwd.cross_kv is None:
                return False, "cross attention with a missing cross_kv when update_kv_cache=True."

        is_fp8_out = output.dtype == torch.float8_e4m3fn
        is_fp4_out = output.dtype == torch.uint8
        has_fp8_kv = kv_cache_dtype == DataType.FP8
        has_fp4_kv = kv_cache_dtype == DataType.NVFP4
        fp8_context_fmha = (
            is_fp8_out or is_fp4_out or has_fp4_kv or (has_fp8_kv and has_context_phase)
        )
        if has_fp4_kv or fp8_context_fmha:
            q_dtype = torch.float8_e4m3fn

        if kv_cache_dtype not in self.SUPPORTED_KV_CACHE_DTYPES:
            return False, f"KV cache dtype {kv_cache_dtype}. Supported: FP16, BF16, FP8, NVFP4."
        if o_dtype not in self.SUPPORTED_OUT_DTYPES:
            return False, f"output dtype {o_dtype}. Supported: FP16, BF16, FP8."

        has_alibi = attn.position_embedding_type in (4, 5)
        check_context_phase = has_context_phase and not is_mla_enable
        if check_context_phase:
            if attn.head_dim in self.UNSUPPORTED_HEAD_SIZES_CONTEXT:
                return False, f"[Context] head size {attn.head_dim}."
            try:
                if AttentionMaskType(fwd.mask_type) == AttentionMaskType.custom_mask:
                    return False, "[Context] custom mask."
            except ValueError:
                return False, f"[Context] unsupported attention mask: {fwd.attention_mask!r}."
            if has_alibi:
                return False, "[Context] ALiBi."
            if (q_dtype, kv_cache_dtype, o_dtype) not in self.SUPPORTED_DTYPE_COMBOS_CONTEXT:
                return False, (
                    f"[Context] dtype combination: Q={q_dtype}, KV={kv_cache_dtype}, O={o_dtype}."
                )

        if has_generation_phase:
            if meta.beam_width != 1 and not meta.is_cross:
                return False, (
                    f"[Generation] beam search (beam_width={meta.beam_width}); must be 1."
                )
            sink_token_length = 0
            if sink_token_length != 0:
                return False, (
                    f"[Generation] StreamingLLM (sink_token_length={sink_token_length})."
                )
            if tokens_per_block < self.MIN_TOKENS_PER_BLOCK:
                return False, (
                    f"[Generation] tokens_per_block ({tokens_per_block}); "
                    f"must be >= {self.MIN_TOKENS_PER_BLOCK}."
                )
            heads_ratio = attn.num_heads // attn.num_kv_heads
            if not is_mla_enable and heads_ratio > self.MAX_HEADS_RATIO_GENERATION:
                return False, (
                    f"[Generation] heads ratio ({heads_ratio}) exceeding maximum "
                    f"({self.MAX_HEADS_RATIO_GENERATION})."
                )
            if has_alibi:
                return False, "[Generation] ALiBi."
            if (q_dtype, kv_cache_dtype, o_dtype) not in self.SUPPORTED_DTYPE_COMBOS_GENERATION:
                return False, (
                    f"[Generation] dtype combination: "
                    f"Q={q_dtype}, KV={kv_cache_dtype}, O={o_dtype}."
                )
            if is_mla_enable:
                supported, reason = self._check_mla_generation_support(
                    head_size=attn.head_dim,
                    num_heads=attn.num_heads,
                    tokens_per_block=tokens_per_block,
                    kv_lora_rank=attn.kv_lora_rank,
                    qk_rope_head_dim=attn.qk_rope_head_dim,
                )
                if not supported:
                    return False, reason

        if tokens_per_block <= 0:
            return False, f"non-positive tokens_per_block ({tokens_per_block})."
        if tokens_per_block & (tokens_per_block - 1) != 0:
            return False, f"tokens_per_block ({tokens_per_block}) that is not a power of 2."
        # A KV cache manager may allow extra page sizes, e.g. MiniMax-M3 adds 128.
        supported_tokens_per_block = self.SUPPORTED_TOKENS_PER_BLOCK | set(
            getattr(meta.kv_cache_manager, "trtllm_gen_extra_tokens_per_block", ())
        )
        if tokens_per_block not in supported_tokens_per_block:
            supported = sorted(supported_tokens_per_block)
            return False, f"tokens_per_block ({tokens_per_block}). Supported: {supported}."

        return True, ""

    def _get_multi_processor_count(self, device: torch.device) -> int:
        device = torch.device(device)
        if device.type != "cuda":
            raise RuntimeError("trtllm-gen requires CUDA tensors.")
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        return get_multi_processor_count_for_device(device_index)

    def _use_fp8_context_fmha(
        self,
        output: torch.Tensor,
        attention_input_type: AttentionInputType,
    ) -> bool:
        kv_cache_quant_mode = QuantMode(self.attn.quant_mode)
        return (
            output.dtype == torch.float8_e4m3fn
            or output.dtype == torch.uint8
            or kv_cache_quant_mode.has_fp4_kv_cache()
            or (
                kv_cache_quant_mode.has_fp8_kv_cache()
                and attention_input_type != AttentionInputType.generation_only
            )
        )

    def prepare_workspace(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        workspace: torch.Tensor,
    ) -> None:
        attn = self.attn
        # Lazily cache the SM count from the first query tensor's device.
        if self._multi_processor_count is None:
            self._multi_processor_count = self._get_multi_processor_count(q.device)

        # One counter per head per decoder sequence; beam search expands each
        # request into ``beam_width`` sequences.
        required_counter_size = _get_multi_ctas_kv_counter_size(
            attn.num_heads,
            metadata.max_num_sequences or metadata.max_num_requests,
            self._multi_processor_count,
        )
        counter_buffer = self._multi_ctas_kv_counter_buffer
        if (
            counter_buffer is None
            or counter_buffer.device != q.device
            or counter_buffer.numel() * counter_buffer.element_size() < required_counter_size
        ):
            if metadata.is_cuda_graph and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "The trtllm-gen multi-CTA KV counter buffer must be allocated "
                    "before CUDA graph capture."
                )
            self._multi_ctas_kv_counter_buffer = torch.zeros(
                required_counter_size,
                dtype=torch.uint8,
                device=q.device,
            )

        if q35_env.fast_paths_enabled():
            capturing = metadata.is_cuda_graph and torch.cuda.is_current_stream_capturing()
            if self._q35_counter_buffer is None or self._q35_counter_buffer.device != q.device:
                if capturing:
                    raise RuntimeError(
                        "The Q35 segmented-attention counter buffer must be allocated before capture."
                    )
                self._q35_counter_buffer = torch.zeros(
                    _Q35_COUNTER_SLOTS, dtype=torch.int32, device=q.device
                )
            _q35_fused_buffers(q.device, self._multi_processor_count, capturing)

        num_tokens = q.size(0)
        attention_input_type = forward_args.attention_input_type
        is_gen_only = attention_input_type == AttentionInputType.generation_only
        num_gen_tokens = num_tokens if is_gen_only else num_tokens - metadata.num_ctx_tokens
        output = forward_args.output
        if output is None:
            raise RuntimeError(f"{type(self).__name__} requires output.")
        fp8_context_fmha = self._use_fp8_context_fmha(output, attention_input_type)

        workspace_max_tokens = max(num_tokens, metadata.max_context_length)
        workspace_max_gen_tokens = max(num_gen_tokens, metadata.max_num_requests)
        required_workspace_size = _get_workspace_size(
            dtype=q.dtype,
            num_tokens=workspace_max_tokens,
            num_gen_tokens=workspace_max_gen_tokens,
            num_heads=attn.num_heads,
            num_kv_heads=attn.num_kv_heads,
            head_size=attn.head_dim,
            max_num_requests=metadata.max_num_requests,
            rotary_embedding_dim=attn.rope_dim,
            fp8_context_fmha=fp8_context_fmha,
        )

        current_workspace_size = workspace.numel() * workspace.element_size()
        if current_workspace_size < required_workspace_size:
            if metadata.is_cuda_graph and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Attention CUDA graph workspace is smaller than the "
                    "required size for trtllm-gen."
                )
            required_workspace_numel = math.ceil(required_workspace_size / workspace.element_size())
            workspace.resize_((required_workspace_numel,))

    def _get_multi_ctas_kv_counter_buffer(self) -> torch.Tensor:
        counter_buffer = self._multi_ctas_kv_counter_buffer
        if counter_buffer is None:
            raise RuntimeError("The trtllm-gen multi-CTA KV counter buffer is not initialized.")
        return counter_buffer

    @staticmethod
    def _compute_window_left(
        cyclic_attention_window_size: int,
        max_kv_length: int,
        attention_chunk_size: int,
    ) -> int:
        """Compute flashinfer window_left with TRTLLM semantics alignment.

        TRTLLM's attention_window_size is exclusive, while flashinfer's
        window_left is inclusive. Keep the same user-visible behavior by
        subtracting 1 when sliding-window attention is enabled.
        """
        if attention_chunk_size != 0 and cyclic_attention_window_size < max_kv_length:
            raise ValueError(
                "Chunked-attention and sliding-window-attention should not "
                "be enabled at the same time."
            )
        if 0 < cyclic_attention_window_size < max_kv_length:
            return cyclic_attention_window_size - 1
        return -1

    @staticmethod
    def _missing_fused_nanobind_ops() -> List[str]:
        required_ops = (
            "get_trtllm_gen_context_workspace_layout",
            "get_trtllm_gen_generation_workspace_layout",
            "trtllm_gen_context_preprocess",
            "trtllm_gen_context_postprocess",
            "trtllm_gen_generation_preprocess",
            "build_trtllm_gen_kv_cache_metadata",
        )
        return [op for op in required_ops if not hasattr(thop, op)]

    def run_context(
        self,
        params: FmhaParams,
    ) -> None:
        attention_input = params.qkv_input if params.qkv_input is not None else params.query_input
        if attention_input is None:
            raise RuntimeError("trtllm-gen context requires QKV or query input.")
        attn = params.attn
        meta = params.meta
        fwd = params.fwd
        rope_params = attn.rope_params
        bmm1_scale_static = get_bmm1_scale(attn)
        attention_chunk_size = get_attention_chunk_size(attn)
        output = fwd.output
        fp8_context_fmha = self._use_fp8_context_fmha(output, fwd.attention_input_type)
        (
            q_processed,
            kv_pool,
            block_tables,
            kv_scale_pool,
            bmm1_scale,
            bmm2_scale,
            fmha_workspace,
            cu_q_seqlens,
            cu_kv_seqlens,
            max_q_len,
            max_kv_len,
            window_left,
        ) = thop.trtllm_gen_context_preprocess(
            attention_input,  # qkv_input (also accepts Q-only input for cached KV)
            params.workspace,  # workspace
            params.sequence_lengths,  # sequence_lengths
            params.context_lengths,  # context_lengths
            meta.kv_cache_block_offsets,  # kv_cache_block_offsets
            meta.host_kv_cache_pool_pointers,  # host_kv_cache_pool_pointers
            meta.host_kv_cache_pool_mapping,  # host_kv_cache_pool_mapping
            fwd.kv_scale_orig_quant,  # kv_scale_orig_quant
            fwd.kv_scale_quant_orig,  # kv_scale_quant_orig
            fwd.out_scale,  # attention_output_orig_quant
            attn.rotary_inv_freq,  # rotary_inv_freq
            attn.rotary_cos_sin,  # rotary_cos_sin
            fwd.mrope_rotary_cos_sin,  # mrope_rotary_cos_sin
            attn.local_layer_idx,  # layer_idx
            attn.num_heads,  # num_heads
            attn.num_kv_heads,  # num_kv_heads
            attn.head_dim,  # head_size
            params.tokens_per_block,  # tokens_per_block
            fwd.mask_type,  # mask_type
            attn.quant_mode,  # kv_cache_quant_mode
            params.max_attention_window_size,  # max_attention_window_size
            params.cyclic_attention_window_size,  # cyclic_attention_window_size
            params.num_tokens,  # num_tokens
            params.batch_size,  # batch_size
            params.input_seq_length,  # input_seq_length
            params.max_past_kv_length,  # max_past_kv_length
            rope_params.dim,  # rotary_embedding_dim
            rope_params.theta,  # rotary_embedding_base
            int(rope_params.scale_type),  # rotary_embedding_scale_type
            rope_params.scale,  # rotary_embedding_scale
            rope_params.max_positions,  # rotary_embedding_max_positions
            attn.position_embedding_type,  # position_embedding_type
            bmm1_scale_static,  # bmm1_scale
            1.0,  # bmm2_scale
            attention_chunk_size,  # attention_chunk_size
            fp8_context_fmha,  # fp8_context_fmha
            meta.use_paged_context_fmha,  # paged_context_fmha
            attn.is_mla_enable,  # is_mla_enable
            self._multi_processor_count,  # multi_processor_count
            params.total_num_blocks,  # total_num_blocks
            params.kv_factor,  # kv_factor
            True,  # need_build_kv_cache_metadata
            fwd.cross_kv,  # cross_kv
            params.is_cross,  # is_cross
        )

        has_fp4_kv = QuantMode(attn.quant_mode).has_fp4_kv_cache()
        if has_fp4_kv and kv_scale_pool is None:
            raise RuntimeError("trtllm-gen FP4 KV cache requires KV scale pool.")
        if has_fp4_kv or fp8_context_fmha:
            q_processed = (
                q_processed.view(torch.uint8)
                .flatten()[: params.num_tokens * attn.num_heads * attn.head_dim]
                .view(torch.float8_e4m3fn)
                .view(params.num_tokens, attn.num_heads, attn.head_dim)
            )
        ctx_bmm1_scale = bmm1_scale_static
        if fp8_context_fmha and bmm1_scale is not None:
            ctx_bmm1_scale = bmm1_scale.narrow(0, 0, 1)
        ctx_bmm2_scale = bmm2_scale if fp8_context_fmha and bmm2_scale is not None else 1.0
        causal = (
            False
            if params.is_cross
            else AttentionMaskType(fwd.mask_type) == AttentionMaskType.causal
        )
        flashinfer.prefill.trtllm_batch_context_with_kv_cache(
            query=q_processed,
            kv_cache=(kv_pool, kv_pool),
            workspace_buffer=fmha_workspace,
            block_tables=block_tables,
            seq_lens=params.sequence_lengths,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            bmm1_scale=ctx_bmm1_scale,
            bmm2_scale=ctx_bmm2_scale,
            batch_size=params.batch_size,
            cum_seq_lens_q=cu_q_seqlens,
            cum_seq_lens_kv=cu_kv_seqlens,
            window_left=window_left,
            out=params.output,
            kv_layout=self._layout,
            enable_pdl=self._enable_pdl,
            sinks=fwd.attention_sinks,
            kv_cache_sf=(kv_scale_pool, kv_scale_pool) if kv_scale_pool is not None else None,
            uses_shared_paged_kv_idx=self.USE_SHARED_PAGED_KV_IDX,
            causal=causal,
            multi_ctas_kv_counter_buffer=self._get_multi_ctas_kv_counter_buffer(),
        )

        if params.is_cross:
            return

        thop.trtllm_gen_context_postprocess(
            attention_input,  # qkv_input
            params.workspace,  # workspace
            params.sequence_lengths,  # sequence_lengths
            params.context_lengths,  # context_lengths
            meta.kv_cache_block_offsets,  # kv_cache_block_offsets
            meta.host_kv_cache_pool_pointers,  # host_kv_cache_pool_pointers
            meta.host_kv_cache_pool_mapping,  # host_kv_cache_pool_mapping
            fwd.kv_scale_orig_quant,  # kv_scale_orig_quant
            fwd.kv_scale_quant_orig,  # kv_scale_quant_orig
            fwd.out_scale,  # attention_output_orig_quant
            attn.rotary_cos_sin,  # rotary_cos_sin
            fwd.mrope_rotary_cos_sin,  # mrope_rotary_cos_sin
            attn.local_layer_idx,  # layer_idx
            attn.num_heads,  # num_heads
            attn.num_kv_heads,  # num_kv_heads
            attn.head_dim,  # head_size
            params.tokens_per_block,  # tokens_per_block
            fwd.mask_type,  # mask_type
            attn.quant_mode,  # kv_cache_quant_mode
            params.max_attention_window_size,  # max_attention_window_size
            params.cyclic_attention_window_size,  # cyclic_attention_window_size
            params.num_tokens,  # num_tokens
            params.batch_size,  # batch_size
            params.input_seq_length,  # input_seq_length
            params.max_past_kv_length,  # max_past_kv_length
            rope_params.dim,  # rotary_embedding_dim
            rope_params.theta,  # rotary_embedding_base
            int(rope_params.scale_type),  # rotary_embedding_scale_type
            rope_params.scale,  # rotary_embedding_scale
            rope_params.max_positions,  # rotary_embedding_max_positions
            attn.position_embedding_type,  # position_embedding_type
            bmm1_scale_static,  # bmm1_scale
            fp8_context_fmha,  # fp8_context_fmha
            meta.use_paged_context_fmha,  # paged_context_fmha
            attn.is_mla_enable,  # is_mla_enable
            attention_chunk_size,  # attention_chunk_size
            self._multi_processor_count,  # multi_processor_count
        )

    def run_generation(
        self,
        params: FmhaParams,
    ) -> None:
        attention_input = params.qkv_input if params.qkv_input is not None else params.query_input
        if attention_input is None:
            raise RuntimeError("trtllm-gen generation requires QKV or query input.")
        attn = params.attn
        fwd = params.fwd
        bmm1_scale_static = get_bmm1_scale(attn)
        attention_chunk_size = get_attention_chunk_size(attn)
        output = fwd.output
        fp8_context_fmha = self._use_fp8_context_fmha(output, fwd.attention_input_type)
        batch_beam = params.batch_size
        # The attention module may already have written this call's FP8 query / K / V (q35_kv_append_plan).
        preprocessed = None
        q35_plan = getattr(attn, "q35_kv_appended", None)
        if q35_plan is not None:
            attn.q35_kv_appended = None
            if q35_plan.fold is not None:
                if self._q35_fold_generation(params, q35_plan, fp8_context_fmha):
                    return
                attention_input = self._q35_fold_fallback(params, q35_plan)
            preprocessed = self._q35_appended_preprocess(
                params, q35_plan, batch_beam, attention_chunk_size, fp8_context_fmha
            )
        if preprocessed is None:
            preprocessed = self._run_generation_preprocess(
                params,
                attention_input,
                batch_beam,
                bmm1_scale_static,
                attention_chunk_size,
                fp8_context_fmha,
            )
        (
            q_processed,
            kv_pool,
            block_tables,
            kv_scale_pool,
            bmm1_scale,
            bmm2_scale,
            fmha_workspace,
            cu_seqlens,
            max_q_len,
            max_kv_len,
            window_left,
            is_multi_token_gen,
        ) = preprocessed

        q_len_per_req = None if is_multi_token_gen else params.input_seq_length
        decode_max_q_len = max_q_len if is_multi_token_gen else None
        decode_cu_seqlens = cu_seqlens if is_multi_token_gen else None

        has_fp4_kv = QuantMode(attn.quant_mode).has_fp4_kv_cache()
        if has_fp4_kv and kv_scale_pool is None:
            raise RuntimeError("trtllm-gen FP4 KV cache requires KV scale pool.")
        if has_fp4_kv or fp8_context_fmha:
            q_processed = (
                q_processed.view(torch.uint8)
                .flatten()[: params.num_tokens * attn.num_heads * attn.head_dim]
                .view(torch.float8_e4m3fn)
                .view(params.num_tokens, attn.num_heads, attn.head_dim)
            )
        gen_bmm1_scale = (
            bmm1_scale if fp8_context_fmha and bmm1_scale is not None else bmm1_scale_static
        )
        gen_bmm2_scale = bmm2_scale if fp8_context_fmha and bmm2_scale is not None else 1.0

        if q35_env.fast_paths_enabled() and self._q35_run_generation(
            params,
            q_processed,
            kv_pool,
            block_tables,
            kv_scale_pool,
            max_kv_len,
            window_left,
            is_multi_token_gen,
            has_fp4_kv,
            gen_bmm1_scale,
            gen_bmm2_scale,
            fmha_workspace,
        ):
            return

        flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=q_processed,
            kv_cache=(kv_pool, kv_pool),
            workspace_buffer=fmha_workspace,
            block_tables=block_tables,
            seq_lens=params.sequence_lengths,
            max_seq_len=max_kv_len,
            bmm1_scale=gen_bmm1_scale,
            bmm2_scale=gen_bmm2_scale,
            window_left=window_left,
            out=params.output,
            sinks=fwd.attention_sinks,
            kv_layout=self._layout,
            enable_pdl=self._enable_pdl,
            backend="trtllm-gen",
            q_len_per_req=q_len_per_req,
            max_q_len=decode_max_q_len,
            cum_seq_lens_q=decode_cu_seqlens,
            kv_cache_sf=(kv_scale_pool, kv_scale_pool) if kv_scale_pool is not None else None,
            uses_shared_paged_kv_idx=self.USE_SHARED_PAGED_KV_IDX,
            bmm1_scale_log2=(
                _get_bmm1_scale_log2(gen_bmm1_scale)
                if isinstance(gen_bmm1_scale, torch.Tensor)
                else None
            ),
            multi_ctas_kv_counter_buffer=self._get_multi_ctas_kv_counter_buffer(),
        )

    def _run_generation_preprocess(
        self,
        params: FmhaParams,
        attention_input: torch.Tensor,
        batch_beam: int,
        bmm1_scale_static: float,
        attention_chunk_size: int,
        fp8_context_fmha: bool,
    ) -> tuple:
        """C++ generation preprocess: query, K / V append into the paged cache, FMHA scales and cache views."""
        attn = params.attn
        meta = params.meta
        fwd = params.fwd
        rope_params = attn.rope_params
        return thop.trtllm_gen_generation_preprocess(
            attention_input,  # qkv_input (also accepts Q-only input for cached KV)
            params.workspace,  # workspace
            params.sequence_lengths,  # sequence_lengths
            params.spec_decoding_generation_lengths,  # spec_decoding_generation_lengths
            params.spec_decoding_position_offsets,  # spec_decoding_position_offsets
            meta.kv_cache_block_offsets,  # kv_cache_block_offsets
            meta.host_kv_cache_pool_pointers,  # host_kv_cache_pool_pointers
            meta.host_kv_cache_pool_mapping,  # host_kv_cache_pool_mapping
            fwd.kv_scale_orig_quant,  # kv_scale_orig_quant
            fwd.kv_scale_quant_orig,  # kv_scale_quant_orig
            fwd.out_scale,  # attention_output_orig_quant
            attn.rotary_inv_freq,  # rotary_inv_freq
            attn.rotary_cos_sin,  # rotary_cos_sin
            fwd.mrope_position_deltas,  # mrope_position_deltas
            attn.local_layer_idx,  # layer_idx
            params.seq_offset,  # seq_offset
            attn.num_heads,  # num_heads
            attn.num_kv_heads,  # num_kv_heads
            attn.head_dim,  # head_size
            params.tokens_per_block,  # tokens_per_block
            attn.quant_mode,  # kv_cache_quant_mode
            params.max_attention_window_size,  # max_attention_window_size
            params.cyclic_attention_window_size,  # cyclic_attention_window_size
            params.num_tokens,  # num_tokens
            batch_beam,  # batch_beam
            params.input_seq_length,  # input_seq_length
            params.max_past_kv_length,  # max_past_kv_length
            rope_params.dim,  # rotary_embedding_dim
            rope_params.theta,  # rotary_embedding_base
            int(rope_params.scale_type),  # rotary_embedding_scale_type
            rope_params.scale,  # rotary_embedding_scale
            rope_params.max_positions,  # rotary_embedding_max_positions
            attn.position_embedding_type,  # position_embedding_type
            bmm1_scale_static,  # bmm1_scale
            1.0,  # bmm2_scale
            fp8_context_fmha,  # fp8_context_fmha
            attn.predicted_tokens_per_seq,  # predicted_tokens_per_seq
            attention_chunk_size,  # attention_chunk_size
            self._multi_processor_count,  # multi_processor_count
            params.total_num_blocks,  # total_num_blocks
            params.kv_factor,  # kv_factor
            True,  # need_build_kv_cache_metadata
            params.is_cross,  # is_cross
        )

    def _q35_run_generation(
        self,
        params: FmhaParams,
        q_processed: torch.Tensor,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        kv_scale_pool: Optional[torch.Tensor],
        max_kv_len: int,
        window_left: int,
        is_multi_token_gen: bool,
        has_fp4_kv: bool,
        gen_bmm1_scale,
        gen_bmm2_scale,
        fmha_workspace: torch.Tensor,
    ) -> bool:
        """Run a generation call on a Qwen3.5 decode path; False when none applies."""
        attn = params.attn
        meta = params.meta
        fwd = params.fwd
        if not (
            not is_multi_token_gen
            and not has_fp4_kv
            and meta.beam_width == 1
            and not params.is_cross
            and q_processed.dim() == 3
            and q_processed.is_contiguous()
            # These trtllm-gen launches take an FP8 query with an FP8 KV cache.
            and (q_processed.dtype == torch.float8_e4m3fn or q_processed.dtype == kv_pool.dtype)
            and self._q35_counter_buffer is not None
        ):
            return False
        num_requests = params.num_requests
        # Pure-generation calls only: the gate the attention module offers covers every token of the forward.
        gate = getattr(attn, "q35_pending_gate", None)
        if gate is not None:
            gate = (
                gate[: params.num_tokens]
                if meta.num_contexts == 0 and gate.shape[0] >= params.num_tokens
                else None
            )
        generation_only = (
            meta.num_contexts == 0
            and kv_scale_pool is None
            and window_left < 0
            and fwd.attention_sinks is None
        )

        cp_group = getattr(attn, "q35_cp_group", None)
        if (
            generation_only
            and cp_group is not None
            and attn.num_kv_heads == 1
            and _Q35_ATTN_CP_MIN_BATCH <= num_requests <= _Q35_ATTN_CP_MAX_BATCH
        ):
            exchange = self._q35_cp_exchange(attn, len(cp_group))
            # The q all-gather and the partials of one call must fit the exchange's per-pair payload.
            if (
                exchange is not None
                and params.num_tokens * attn.num_heads * (attn.head_dim * 2 + 4)
                <= exchange.max_words * 4
            ):
                self._q35_cp_generation(
                    params,
                    exchange,
                    q_processed,
                    kv_pool,
                    block_tables,
                    max_kv_len,
                    gen_bmm1_scale,
                    gen_bmm2_scale,
                    fmha_workspace,
                    gate,
                )
                return True

        if (
            generation_only
            and num_requests <= _Q35_ATTN_FUSED_MAX_BATCH
            and kv_pool.dim() == 4
            and kv_pool.is_contiguous()
            and q35_fused_decode_supported(
                attn.num_heads,
                attn.num_kv_heads,
                attn.head_dim,
                params.input_seq_length,
                num_requests,
                window_left,
                False,
                params.tokens_per_block,
                q_processed.dtype,
                params.output.dtype,
                kv_pool.dtype,
                self._multi_processor_count,
            )
        ):
            ws, cnt = _q35_fused_buffers(q_processed.device, self._multi_processor_count, False)
            q35_fused_decode(
                q_processed,
                kv_pool,
                block_tables,
                params.sequence_lengths,
                params.input_seq_length,
                gen_bmm1_scale,
                gen_bmm2_scale,
                params.output,
                ws,
                cnt,
                self._multi_processor_count,
                self._enable_pdl,
                gate=gate,
            )
            if gate is not None:
                attn.q35_gate_applied = True
            return True

        if num_requests <= _Q35_ATTN_SEG_MAX_BATCH and q35_segmented_supported(
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            params.input_seq_length,
            num_requests,
            _Q35_ATTN_SEGMENTS,
            window_left,
            fwd.attention_sinks is not None,
            params.tokens_per_block,
            q_processed.dtype,
            params.output.dtype,
        ):
            q35_segmented_decode(
                flashinfer.decode.get_trtllm_gen_fmha_module().trtllm_paged_attention_decode,
                q_processed,
                kv_pool,
                block_tables,
                params.sequence_lengths,
                max_kv_len,
                params.input_seq_length,
                attn.num_kv_heads,
                _Q35_ATTN_SEGMENTS,
                gen_bmm1_scale,
                _get_bmm1_scale_log2(gen_bmm1_scale)
                if isinstance(gen_bmm1_scale, torch.Tensor)
                else gen_bmm1_scale,
                gen_bmm2_scale,
                fmha_workspace,
                self._q35_counter_buffer,
                params.output,
                flashinfer.decode.get_device_sm_count(q_processed.device),
                kv_scale_pool,
                self._enable_pdl,
                gate=gate,
            )
            if gate is not None:
                attn.q35_gate_applied = True
            return True
        return False

    def _q35_fold_cp_applies(self, attn: "TrtllmAttention", num_requests: int) -> bool:
        """Whether CP-lite may take a generation call of num_requests (it has precedence over the fused path)."""
        group = getattr(attn, "q35_cp_group", None)
        return (
            group is not None
            and attn.num_kv_heads == 1
            and _Q35_ATTN_CP_MIN_BATCH <= num_requests <= _Q35_ATTN_CP_MAX_BATCH
        )

    def _q35_fold_generation(
        self, params: FmhaParams, plan: Q35KvAppendPlan, fp8_context_fmha: bool
    ) -> bool:
        """Run the folded fused decode when the call meets the conditions of the fused branch of _q35_run_generation
        and of _q35_appended_preprocess; False otherwise."""
        attn = params.attn
        meta = params.meta
        fwd = params.fwd
        f = plan.fold
        if not (
            getattr(attn, "q35_pending_gate", None) is not None
            and fp8_context_fmha
            and meta.num_contexts == 0
            and meta.beam_width == 1
            and not params.is_cross
            and params.seq_offset == 0
            and params.spec_decoding_generation_lengths is None
            and params.num_requests <= _Q35_ATTN_FUSED_MAX_BATCH
            and not self._q35_fold_cp_applies(attn, params.num_requests)
            and plan.layer_idx == attn.local_layer_idx
            and plan.num_tokens == params.num_tokens
            and plan.num_requests == params.num_requests
            and plan.q_len == params.input_seq_length
            and fwd.out_scale is None
            and fwd.attention_sinks is None
            and fwd.kv_scale_orig_quant is plan.kv_scale_orig_quant
            and fwd.kv_scale_quant_orig is plan.kv_scale_quant_orig
            and params.sequence_lengths is not None
            and params.sequence_lengths.data_ptr() == plan.seq_lens.data_ptr()
            and params.output is not None
            and params.output.dtype == torch.bfloat16
            and not QuantMode(attn.quant_mode).has_fp4_kv_cache()
            and self._multi_processor_count is not None
        ):
            return False
        window_left = self._compute_window_left(
            params.cyclic_attention_window_size,
            params.max_past_kv_length,
            get_attention_chunk_size(attn),
        )
        if window_left >= 0 or not q35_fused_decode_supported(
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            params.input_seq_length,
            params.num_requests,
            window_left,
            False,
            params.tokens_per_block,
            torch.float8_e4m3fn,
            params.output.dtype,
            plan.kv_pool.dtype,
            self._multi_processor_count,
        ):
            return False
        ws, cnt = _q35_fused_buffers(params.output.device, self._multi_processor_count, False)
        q35_fused_decode_fold(
            f.qkv[: params.num_tokens],
            f.q_weight,
            f.k_weight,
            f.cos_sin,
            f.positions,
            f.eps,
            f.rotary_dim,
            f.mrope_section,
            plan.kv_pool,
            plan.block_tables,
            params.sequence_lengths,
            params.input_seq_length,
            plan.kv_scale_orig_quant,
            plan.kv_scale_quant_orig,
            plan.host_bmm1_scale,
            params.output,
            ws,
            cnt,
            self._multi_processor_count,
            self._enable_pdl,
        )
        attn.q35_gate_applied = True
        return True

    @staticmethod
    def _q35_fold_fallback(params: FmhaParams, plan: Q35KvAppendPlan) -> torch.Tensor:
        """Run the prologue kernel the module skipped for a folded call that misses the fused path.

        Returns the packed QKV of this call; the contiguous gate replaces attn.q35_pending_gate.
        """
        from ...kernels.fused_qk_norm_rope_gate import q35_fused_qkv_rope_gate_append

        attn = params.attn
        f = plan.fold
        plan.fold = None
        qkv_out, gate_out, plan.q_fp8, plan.scales = q35_fused_qkv_rope_gate_append(
            f.qkv,
            f.q_weight,
            f.k_weight,
            f.cos_sin,
            f.positions,
            f.eps,
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            f.rotary_dim,
            f.mrope_section,
            plan,
        )
        params.qkv_input = qkv_out[: params.num_tokens]
        if getattr(attn, "q35_pending_gate", None) is not None:
            attn.q35_pending_gate = gate_out
        return params.qkv_input

    def _q35_appended_preprocess(
        self,
        params: FmhaParams,
        plan: Q35KvAppendPlan,
        batch_beam: int,
        attention_chunk_size: int,
        fp8_context_fmha: bool,
    ) -> Optional[tuple]:
        """trtllm_gen_generation_preprocess outputs for a call whose FP8 query / K / V the module wrote.

        None when the call does not match the plan; the C++ path then rewrites the same bytes from the bf16 QKV.
        """
        attn = params.attn
        fwd = params.fwd
        if not (
            fp8_context_fmha
            and plan.q_fp8 is not None
            and plan.scales is not None
            and plan.layer_idx == attn.local_layer_idx
            and plan.num_tokens == params.num_tokens
            and plan.num_requests == batch_beam
            and plan.q_len == params.input_seq_length
            and params.seq_offset == 0
            and not params.is_cross
            and params.spec_decoding_generation_lengths is None
            and fwd.out_scale is None
            and fwd.kv_scale_orig_quant is plan.kv_scale_orig_quant
            and fwd.kv_scale_quant_orig is plan.kv_scale_quant_orig
            and params.sequence_lengths is not None
            and params.sequence_lengths.data_ptr() == plan.seq_lens.data_ptr()
            and params.qkv_input is not None
            and params.qkv_input.dtype == torch.bfloat16
        ):
            return None
        layout = _get_generation_workspace_layout(
            params.qkv_input.dtype,
            batch_beam,
            params.num_tokens,
            attn.num_heads,
            attn.head_dim,
            attn.num_kv_heads,
            attn.rope_params.dim,
        )
        offset = int(layout["trtllm_gen_workspace_offset"])
        size = int(layout["trtllm_gen_workspace_size"])
        workspace = params.workspace.view(-1).view(torch.uint8)
        if size <= 0 or offset + size > workspace.numel():
            return None
        window_left = self._compute_window_left(
            params.cyclic_attention_window_size,
            params.max_past_kv_length,
            attention_chunk_size,
        )
        return (
            plan.q_fp8,
            plan.kv_pool,
            plan.block_tables,
            None,
            plan.scales.narrow(0, 0, 2),
            plan.scales.narrow(0, 4, 1),
            workspace.narrow(0, offset, size),
            None,
            params.input_seq_length,
            params.max_past_kv_length,
            window_left,
            False,
        )

    @staticmethod
    def _q35_cp_exchange(attn: "TrtllmAttention", cp: int):
        """The LL exchange of this rank's CP-lite group, or None (CP-lite is then skipped).

        Created lazily on the first eager call; a graph captured before it exists keeps the other paths.
        """
        mapping = getattr(attn, "q35_cp_mapping", None)
        if mapping is None:
            return None
        from tensorrt_llm._torch.distributed.ll_exchange import get_tp_group_exchange

        exchange = get_tp_group_exchange(mapping, group_size=cp)
        if exchange is not None and not (
            exchange.group_size == cp and exchange.grank == attn.q35_cp_rank
        ):
            raise RuntimeError("TPGroupExchange group does not match the CP-lite KV group.")
        return exchange

    def _q35_cp_generation(
        self,
        params: FmhaParams,
        exchange,
        q_processed: torch.Tensor,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        max_kv_len: int,
        bmm1_scale,
        bmm2_scale,
        fmha_workspace: torch.Tensor,
        gate: Optional[torch.Tensor],
    ) -> None:
        """CP-lite generation attention (fmha/q35_cp.py) for the rows of this rank's own heads."""
        rows, hq, d = q_processed.shape
        q35_cp_decode_ll(
            exchange,
            flashinfer.decode.get_trtllm_gen_fmha_module().trtllm_paged_attention_decode,
            q_processed,
            kv_pool,
            block_tables,
            params.sequence_lengths,
            max_kv_len,
            params.input_seq_length,
            _q35_cp_segments(params.num_requests, params.input_seq_length),
            bmm1_scale,
            _get_bmm1_scale_log2(bmm1_scale)
            if isinstance(bmm1_scale, torch.Tensor)
            else bmm1_scale,
            bmm2_scale,
            fmha_workspace,
            self._q35_counter_buffer,
            flashinfer.decode.get_device_sm_count(q_processed.device),
            self._enable_pdl,
            params.output.view(rows, hq, d),
            gate=gate,
        )
        if gate is not None:
            params.attn.q35_gate_applied = True

    def run_mla_generation(
        self,
        params: FmhaParams,
    ) -> None:
        """MLA generation decode using flashinfer MLA kernel."""
        attn = params.attn
        meta = params.meta
        fwd = params.fwd
        if 0 < params.cyclic_attention_window_size < params.max_past_kv_length:
            raise NotImplementedError(
                "Sliding-window attention is not supported by MLA decode path."
            )
        if get_attention_chunk_size(attn) != 0:
            raise NotImplementedError("Chunked-attention is not supported by MLA decode path.")

        batch_beam = params.batch_size
        if params.query_input is None:
            raise RuntimeError("MLA generation requires query_input.")
        kv_cache, block_tables, _kv_scale_pool = thop.build_trtllm_gen_kv_cache_metadata(
            meta.host_kv_cache_pool_pointers,  # host_kv_cache_pool_pointers
            meta.host_kv_cache_pool_mapping,  # host_kv_cache_pool_mapping
            meta.kv_cache_block_offsets,  # kv_cache_block_offsets
            attn.local_layer_idx,  # layer_idx
            attn.num_kv_heads,  # num_kv_heads
            params.tokens_per_block,  # tokens_per_block
            attn.head_dim,  # head_dim
            params.kv_factor,  # kv_factor
            params.total_num_blocks,  # total_num_blocks
            attn.quant_mode,  # kv_cache_quant_mode
            params.seq_offset,  # batch_start
            batch_beam,  # batch_size
            params.query_input.dtype,  # dtype
        )

        kv_lora_rank = attn.kv_lora_rank or 0
        qk_nope_head_dim = attn.qk_nope_head_dim or 0
        qk_rope_head_dim = attn.qk_rope_head_dim or 0
        mla_head_dim_qk = kv_lora_rank + qk_rope_head_dim
        q_len_per_req = params.num_tokens // batch_beam if batch_beam > 0 else 1

        if QuantMode(attn.quant_mode).has_fp8_kv_cache():
            quant_q_buffer = fwd.quant_q_buffer
            bmm1_scale_buffer = fwd.mla_bmm1_scale
            bmm2_scale_buffer = fwd.mla_bmm2_scale
            if quant_q_buffer is None or bmm1_scale_buffer is None or bmm2_scale_buffer is None:
                raise RuntimeError(
                    "FP8 MLA generation requires quant_q_buffer, "
                    "mla_bmm1_scale, and mla_bmm2_scale."
                )

            expected_q_elements = params.num_tokens * attn.num_heads * mla_head_dim_qk
            if quant_q_buffer.numel() < expected_q_elements:
                raise RuntimeError(
                    f"FP8 MLA quant_q_buffer has {quant_q_buffer.numel()} elements; "
                    f"expected at least {expected_q_elements}."
                )
            if bmm1_scale_buffer.dtype != torch.float32 or bmm1_scale_buffer.numel() < 1:
                raise RuntimeError("FP8 MLA bmm1 scale must contain a float32 value.")
            if bmm2_scale_buffer.dtype != torch.float32 or bmm2_scale_buffer.numel() < 1:
                raise RuntimeError("FP8 MLA bmm2 scale must contain a float32 value.")

            query = (
                quant_q_buffer.view(torch.uint8)
                .flatten()[:expected_q_elements]
                .view(torch.float8_e4m3fn)
                .view(batch_beam, q_len_per_req, attn.num_heads, mla_head_dim_qk)
            )
            # FlashInfer converts tensor BMM1 scales to log2 internally. The
            # producer stores the regular scale at index 0 and log2 at index 1.
            bmm1_scale = bmm1_scale_buffer.flatten()[:1]
            bmm2_scale = bmm2_scale_buffer.flatten()[:1]
        else:
            query = params.query_input.view(
                batch_beam, q_len_per_req, attn.num_heads, mla_head_dim_qk
            )
            bmm1_scale = 1.0 / (attn.q_scaling * math.sqrt(qk_nope_head_dim + qk_rope_head_dim))
            bmm2_scale = 1.0
        workspace_buffer = params.workspace.view(-1, 4)

        flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
            query,  # query
            kv_cache,  # kv_cache
            workspace_buffer,  # workspace_buffer
            qk_nope_head_dim,  # qk_nope_head_dim
            kv_lora_rank,  # kv_lora_rank
            qk_rope_head_dim,  # qk_rope_head_dim
            block_tables,  # block_tables
            params.sequence_lengths,  # seq_lens
            params.max_past_kv_length,  # max_seq_len
            0,  # sparse_mla_top_k
            params.output.view(batch_beam, q_len_per_req, attn.num_heads, kv_lora_rank),  # out
            bmm1_scale,  # bmm1_scale
            bmm2_scale,  # bmm2_scale
            fwd.attention_sinks,  # sinks
            None,  # skip_softmax_threshold_scale_factor
            self._enable_pdl,  # enable_pdl
            backend="trtllm-gen",
            is_var_seq=True,
            uses_shared_paged_kv_idx=self.USE_SHARED_PAGED_KV_IDX,
            multi_ctas_kv_counter_buffer=self._get_multi_ctas_kv_counter_buffer(),
        )


def _q35_trtllm_gen_lib(attn: "TrtllmAttention") -> Optional[FlashInferTrtllmGenFmha]:
    """The FlashInferTrtllmGenFmha instance that serves this layer's non-MLA generation calls, or None.

    Only context-only libraries (PhasedFmha without its own run_generation) may precede it in selection order.
    """
    manager = getattr(attn, "_fmha_manager", None)
    for lib in getattr(manager, "fmha_libs", ()):
        if isinstance(lib, FlashInferTrtllmGenFmha):
            return lib
        if not (
            isinstance(lib, PhasedFmha) and type(lib).run_generation is PhasedFmha.run_generation
        ):
            return None
    return None


def q35_kv_append_plan(
    attn: "TrtllmAttention",
    metadata: "TrtllmAttentionMetadata",
    num_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> Optional[Q35KvAppendPlan]:
    """Destination for the module-side FP8 q / K / V write of a generation-only step, or None (stock path).

    Mirrors the conditions under which the C++ generation preprocess (applyBiasRopeUpdateKVCacheV2) writes FP8 K / V
    and an FP8 query with a uniform q_len; run_generation re-checks the call before skipping its preprocess.
    """
    if num_tokens <= 0:
        return None
    lib = _q35_trtllm_gen_lib(attn)
    if lib is None or FlashInferTrtllmGenFmha._missing_fused_nanobind_ops():
        return None
    if (
        attn.is_mla_enable
        or attn.sparse_params is not None
        or attn.num_heads != num_heads
        or attn.num_kv_heads != num_kv_heads
        or attn.head_dim != head_dim
        or head_dim % 16 != 0
    ):
        return None
    quant_mode = QuantMode(attn.quant_mode)
    if not quant_mode.has_fp8_kv_cache() or quant_mode.has_fp4_kv_cache():
        return None
    if (
        metadata.kv_cache_block_offsets is None
        or metadata.host_kv_cache_pool_pointers is None
        or metadata.host_kv_cache_pool_mapping is None
        or metadata.kv_cache_manager is None
    ):
        return None
    num_seqs = metadata.num_generations
    if (
        metadata.num_contexts != 0
        or num_seqs <= 0
        or metadata.beam_width != 1
        or metadata.is_cross
        or num_tokens % num_seqs != 0
    ):
        return None
    if metadata.is_spec_decoding_enabled and attn.predicted_tokens_per_seq > 1:
        return None
    seq_lens = metadata.kv_lens_cuda_runtime
    if seq_lens is None or seq_lens.dtype != torch.int32 or not seq_lens.is_contiguous():
        return None
    kv_scale_orig_quant = attn.kv_scale_orig_quant
    kv_scale_quant_orig = attn.kv_scale_quant_orig
    if not (
        isinstance(kv_scale_orig_quant, torch.Tensor)
        and isinstance(kv_scale_quant_orig, torch.Tensor)
        and kv_scale_orig_quant.dtype == torch.float32
        and kv_scale_quant_orig.dtype == torch.float32
    ):
        return None
    tokens_per_block = metadata.tokens_per_block if metadata.tokens_per_block is not None else 64
    if tokens_per_block <= 0 or tokens_per_block & (tokens_per_block - 1):
        return None
    layer_idx = attn.get_local_layer_idx(metadata)
    kv_pool, block_tables, kv_scale_pool = thop.build_trtllm_gen_kv_cache_metadata(
        metadata.host_kv_cache_pool_pointers,  # host_kv_cache_pool_pointers
        metadata.host_kv_cache_pool_mapping,  # host_kv_cache_pool_mapping
        metadata.kv_cache_block_offsets,  # kv_cache_block_offsets
        layer_idx,  # layer_idx
        num_kv_heads,  # num_kv_heads
        tokens_per_block,  # tokens_per_block
        head_dim,  # head_dim
        lib.kv_factor,  # kv_factor
        lib._get_total_num_blocks(metadata),  # total_num_blocks
        attn.quant_mode,  # kv_cache_quant_mode
        0,  # batch_start
        num_seqs,  # batch_size
        torch.bfloat16,  # dtype
    )
    if (
        kv_scale_pool is not None
        or kv_pool.dtype != torch.float8_e4m3fn
        or block_tables.dtype != torch.int32
    ):
        return None
    return Q35KvAppendPlan(
        layer_idx=layer_idx,
        num_tokens=num_tokens,
        num_requests=num_seqs,
        q_len=num_tokens // num_seqs,
        kv_pool=kv_pool,
        block_tables=block_tables,
        seq_lens=seq_lens,
        kv_scale_orig_quant=kv_scale_orig_quant,
        kv_scale_quant_orig=kv_scale_quant_orig,
        host_bmm1_scale=get_bmm1_scale(attn),
    )


def q35_fold_inputs(
    attn: "TrtllmAttention",
    plan: Q35KvAppendPlan,
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    rotary_dim: int,
    mrope_section: Optional[tuple],
) -> Optional[Q35FoldInputs]:
    """Fold inputs when a call with KV-append ``plan`` will take the fused decode path, else None.

    The module then skips its prologue kernel; run_generation runs it itself if the call goes elsewhere after all.
    """
    lib = _q35_trtllm_gen_lib(attn)
    if lib is None:
        return None
    if lib._multi_processor_count is None:
        lib._multi_processor_count = lib._get_multi_processor_count(qkv.device)
    if (
        plan.num_requests > _Q35_ATTN_FUSED_MAX_BATCH
        or lib._q35_fold_cp_applies(attn, plan.num_requests)
        or plan.num_tokens != qkv.shape[0]
        or plan.num_tokens != plan.num_requests * plan.q_len
    ):
        return None
    if mrope_section is not None and positions.numel() != 3 * qkv.shape[0]:
        return None
    if not q35_fused_decode_supported(
        attn.num_heads,
        attn.num_kv_heads,
        attn.head_dim,
        plan.q_len,
        plan.num_requests,
        -1,
        False,
        plan.kv_pool.shape[2],
        torch.float8_e4m3fn,
        torch.bfloat16,
        plan.kv_pool.dtype,
        lib._multi_processor_count,
    ) or not q35_fused_decode_fold_supported(
        attn.num_heads,
        attn.num_kv_heads,
        attn.head_dim,
        rotary_dim,
        plan.q_len,
        qkv,
        positions,
        cos_sin,
    ):
        return None
    if not (
        q_weight.dtype == torch.bfloat16
        and k_weight.dtype == torch.bfloat16
        and q_weight.is_contiguous()
        and k_weight.is_contiguous()
        and q_weight.numel() == attn.head_dim
        and k_weight.numel() == attn.head_dim
    ):
        return None
    return Q35FoldInputs(
        qkv=qkv,
        q_weight=q_weight,
        k_weight=k_weight,
        cos_sin=cos_sin,
        positions=positions,
        eps=float(eps),
        rotary_dim=rotary_dim,
        mrope_section=tuple(mrope_section) if mrope_section is not None else None,
    )
