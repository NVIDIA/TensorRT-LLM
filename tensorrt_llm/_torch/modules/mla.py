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

import atexit
import functools
import math
import os
import threading
import time
import weakref
from typing import Dict, Optional, cast

import torch
from torch import nn

import tensorrt_llm.quantization.utils.fp8_utils as fp8_utils
from tensorrt_llm._utils import get_sm_version, is_sm_100f, maybe_pin_memory
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.math_utils import pad_up
from tensorrt_llm.quantization.utils.fp4_utils import NVFP4_SF_VEC_SIZE

from ..attention_backend import (
    AttentionForwardArgs,
    AttentionInputType,
    AttentionMetadata,
    FlashInferAttentionMetadata,
    TrtllmAttention,
    TrtllmAttentionMetadata,
)
from ..attention_backend.interface import (
    AttentionBackend,
    PositionalEmbeddingParams,
    PredefinedAttentionMask,
)
from ..attention_backend.sparse.hooks import get_sparse_mla_hooks
from ..attention_backend.utils import create_attention
from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE, IS_CUTLASS_DSL_INTERNAL_AVAILABLE
from ..distributed import AllReduceParams
from ..model_config import ModelConfig
from ..pyexecutor.breakable_cuda_graph import eager_on_graph
from ..ugpu.policy import UgpuExecutionPlanner, UgpuPolicy
from ..ugpu.runtime import UgpuRuntime
from ..utils import (
    AuxStreamType,
    Fp4QuantizedTensor,
    compute_swizzled_sf_shape,
    maybe_compiled_cat,
    maybe_compiled_copy_,
)
from .attention import (
    _helix_cp_allgather_input,
    _helix_cp_output_projection,
    _helix_post_process,
    _helix_zero_kv_mask,
    extract_extra_attrs,
)
from .linear import Linear, NVFP4LinearMethod, TensorParallelMode, is_static_nvfp4_input_eligible
from .multi_stream_utils import maybe_execute_in_parallel
from .rms_norm import RMSNorm
from .rotary_embedding import RotaryEmbedding


def _slice_hidden_states_to_num_tokens(hidden_states, num_tokens: int):
    """Drop CUDA-graph padding by slicing hidden_states to num_tokens rows.

    For a plain tensor this is the usual row slice. For an Fp4QuantizedTensor
    (produced when the previous layer's boundary fusion pre-quantized this
    layer's input) the slice must act on the packed FP4 form:
      - fp4_tensor: row-major [m, k//2], so [:num_tokens] is a plain row slice;
      - scaling_factor: 1D swizzled buffer of padUp(m,128)*padUp(cols,4). NVFP4
        quant is per-row independent and the swizzle is laid out in independent
        128-row tiles, so the leading padUp(num_tokens,128)*padUp(cols,4) bytes
        are exactly the scale factors for the first num_tokens rows (verified by
        tests/unittest/_torch/modules/test_fp4_num_tokens_slice.py).
      - unquantized_hidden_states (when present): plain row slice.
    """
    if not isinstance(hidden_states, Fp4QuantizedTensor):
        return hidden_states[:num_tokens, ...]

    fp4 = hidden_states.fp4_tensor
    if fp4.shape[0] == num_tokens:
        return hidden_states
    if not hidden_states.is_sf_swizzled:
        raise ValueError(
            "_slice_hidden_states_to_num_tokens only supports swizzled FP4 scaling factors"
        )
    sf = hidden_states.scaling_factor
    sf_cols = fp4.shape[-1] * 2 // NVFP4_SF_VEC_SIZE
    padded_rows, padded_cols = compute_swizzled_sf_shape(num_tokens, sf_cols)
    sf_len = padded_rows * padded_cols
    sliced_unquant = (
        hidden_states.unquantized_hidden_states[:num_tokens]
        if hidden_states.unquantized_hidden_states is not None
        else None
    )
    return Fp4QuantizedTensor(
        fp4_tensor=fp4[:num_tokens].contiguous(),
        scaling_factor=sf.view(-1)[:sf_len].contiguous(),
        is_sf_swizzled=True,
        unquantized_hidden_states=sliced_unquant,
    )


def _extract_mla_extra_attrs(layer_idx: str):
    metadata, mla_layer = extract_extra_attrs(layer_idx, "mla")
    assert isinstance(mla_layer, MLA), "MLA layer must be a subclass of MLA or an instance of MLA"
    return metadata, mla_layer


def create_mla_outputs_impl(hidden_states: torch.Tensor, layer_idx: str) -> torch.Tensor:
    metadata, mla_layer = _extract_mla_extra_attrs(layer_idx)
    outputs = mla_layer._create_outputs(hidden_states, metadata)
    if len(outputs) != 1:
        raise RuntimeError("MLA custom ops require exactly one output tensor.")
    return outputs[0]


@torch.library.custom_op("trtllm::create_mla_outputs", mutates_args=())
def create_mla_outputs(hidden_states: torch.Tensor, layer_idx: str) -> torch.Tensor:
    return create_mla_outputs_impl(hidden_states, layer_idx)


@create_mla_outputs.register_fake
def _create_mla_outputs_fake(hidden_states: torch.Tensor, layer_idx: str) -> torch.Tensor:
    return create_mla_outputs_impl(hidden_states, layer_idx)


@torch.library.custom_op(
    "trtllm::mla_custom_op_inplace",
    mutates_args=("output",),
)
def mla_custom_op_inplace(
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
    output: torch.Tensor,
    latent_cache_gen: Optional[torch.Tensor],
    hidden_states_fp4: Optional[torch.Tensor] = None,
    hidden_states_sf: Optional[torch.Tensor] = None,
) -> None:
    metadata, mla_layer = _extract_mla_extra_attrs(layer_idx)
    if hidden_states_fp4 is not None or hidden_states_sf is not None:
        assert hidden_states_fp4 is not None and hidden_states_sf is not None, (
            "hidden_states_fp4 and hidden_states_sf must be passed together"
        )
        hidden_states = Fp4QuantizedTensor(
            fp4_tensor=hidden_states_fp4,
            scaling_factor=hidden_states_sf,
            unquantized_hidden_states=hidden_states,
        )
    mla_layer.forward_impl(
        position_ids,
        hidden_states,
        metadata,
        attn_output=[output],
        latent_cache_gen=latent_cache_gen,
    )


maybe_bcg_mla_custom_op_inplace = eager_on_graph(mla_custom_op_inplace)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)) or default)
    except ValueError:
        return default


# NVFP4-out epilogues for the absorbed-MLA generation BMMs (Rubin SM107, BF16 W_UK/W_UV, FP4 MLA KV; default 0 =
# off; set on the generation role only). The absorbed bmm_o / bmm_q become a CuTe DSL batched GEMM
# (cute_dsl_kernels/rubin/dense_gemm_nvfp4_out.py) whose epilogue writes the NVFP4 bytes the BF16 GEMM + the
# consumer's quantizer would have produced, so the BF16 round trip and the standalone quantizer go:
#   bit 1: bmm_o writes o_proj's NVFP4 input (packed E2M1 [tokens, heads * v_head_dim / 2] + 128x4-swizzled E4M3
#          scales, fp4_quantize's bytes); pure-generation batches of >= TRTLLM_MLA_NVFP4_EPILOGUE_MIN_TOKENS.
#   bit 2: bmm_q writes the FP4 MLA Q prefix (attention_backend/fp4_mla.fp4_mla_bmm_q_fp4_prefix) and the fused
#          RoPE/Q/KV launch drops its Q-prefix programs; one token per sequence, >= ..._MIN_TOKENS_Q tokens.
#   bit 4: with bit 2, that RoPE/Q-tail/KV launch (now independent of bmm_q) runs beside bmm_q on the aux stream.
# TRTLLM_MLA_NVFP4_EPILOGUE_VERIFY (verification only): recompute the BF16-path bytes and count mismatches;
# 1 = eager forwards only, logged per call (synchronizes); 2 = every forward incl. CUDA-graph replays, counted on
# the device without host syncs (VerifyMismatchCounter) and logged by a host thread.
MLA_NVFP4_EPILOGUE = _env_int("TRTLLM_MLA_NVFP4_EPILOGUE", 0)
MLA_NVFP4_EPILOGUE_MIN_TOKENS = max(_env_int("TRTLLM_MLA_NVFP4_EPILOGUE_MIN_TOKENS", 1), 1)
MLA_NVFP4_EPILOGUE_MIN_TOKENS_Q = max(_env_int("TRTLLM_MLA_NVFP4_EPILOGUE_MIN_TOKENS_Q", 16), 1)
MLA_NVFP4_EPILOGUE_VERIFY = _env_int("TRTLLM_MLA_NVFP4_EPILOGUE_VERIFY", 0)
_MLA_NVFP4_EPILOGUE_VERIFY_STATS: Dict[str, Dict[str, int]] = {}
if MLA_NVFP4_EPILOGUE:
    logger.info(
        f"[TRTLLM_MLA_NVFP4_EPILOGUE] gate {MLA_NVFP4_EPILOGUE}: bmm_o -> o_proj NVFP4 input "
        f"{bool(MLA_NVFP4_EPILOGUE & 1)} (min tokens {MLA_NVFP4_EPILOGUE_MIN_TOKENS}), bmm_q -> FP4 Q prefix "
        f"{bool(MLA_NVFP4_EPILOGUE & 2)} (min tokens {MLA_NVFP4_EPILOGUE_MIN_TOKENS_Q}), RoPE beside bmm_q "
        f"{(MLA_NVFP4_EPILOGUE & 6) == 6}, verify {MLA_NVFP4_EPILOGUE_VERIFY}"
    )


def _mla_nvfp4_epilogue_verify_record(
    kind: str, tokens: int, mismatches: Dict[str, torch.Tensor]
) -> None:
    """Accumulate TRTLLM_MLA_NVFP4_EPILOGUE_VERIFY=1 counts (eager; synchronizes); log the first calls, every bad
    one and every 1024th."""
    counts = {k: int(v) for k, v in mismatches.items()}
    stats = _MLA_NVFP4_EPILOGUE_VERIFY_STATS.setdefault(
        kind, {"calls": 0, "tokens": 0, "bad_calls": 0, "mismatches": 0}
    )
    bad = sum(counts.values())
    stats["calls"] += 1
    stats["tokens"] += tokens
    stats["mismatches"] += bad
    stats["bad_calls"] += int(bad > 0)
    if stats["calls"] <= 3 or (bad and stats["bad_calls"] <= 10) or stats["calls"] % 1024 == 0:
        logger.info(
            f"[TRTLLM_MLA_NVFP4_EPILOGUE] VERIFY {kind}: call {stats['calls']} tokens {tokens} "
            f"mismatches {counts}; total calls {stats['calls']} tokens {stats['tokens']} "
            f"bad calls {stats['bad_calls']} mismatched bytes {stats['mismatches']}"
        )


class VerifyMismatchCounter:
    """Device-side mismatch counters that also work inside CUDA graphs (verification modes only).

    ``add`` accumulates 0-d device counts and the number of checks into a device tensor and copies it to pinned
    host memory with a (capturable) async copy, so every replay of a captured graph updates the host copy without
    a host sync. A daemon thread logs the host copy every ``period`` seconds when it changed, and once at exit.
    Counters are created on first use, which must be an eager forward (the warmup before graph capture).
    """

    _instances: list = []
    _reporter: Optional[threading.Thread] = None
    _lock = threading.Lock()

    def __init__(self, tag: str, fields: tuple, period: float = 30.0):
        self.tag = tag
        self.fields = tuple(fields)
        self.device = torch.zeros(len(self.fields) + 1, dtype=torch.int64, device="cuda")
        self.host = maybe_pin_memory(torch.zeros(len(self.fields) + 1, dtype=torch.int64))
        self._last = None
        with VerifyMismatchCounter._lock:
            VerifyMismatchCounter._instances.append(self)
            if VerifyMismatchCounter._reporter is None:
                VerifyMismatchCounter._reporter = threading.Thread(
                    target=VerifyMismatchCounter._report_loop, args=(period,), daemon=True
                )
                VerifyMismatchCounter._reporter.start()
                atexit.register(VerifyMismatchCounter.report_all, True)

    def add(self, mismatches: Dict[str, torch.Tensor]) -> None:
        for i, field in enumerate(self.fields):
            self.device[i : i + 1].add_(mismatches[field].reshape(1).to(torch.int64))
        self.device[-1:].add_(1)
        if self.host.is_pinned():  # a pageable D2H copy cannot be captured
            self.host.copy_(self.device, non_blocking=True)

    def report(self, final: bool = False) -> None:
        values = self.host.tolist()  # pinned host memory: no CUDA call
        if values != self._last or final:
            self._last = values
            counts = dict(zip(self.fields, values[:-1]))
            logger.info(
                f"{self.tag} VERIFY(device) checks {values[-1]} mismatches {counts}"
                f"{' (final)' if final else ''}"
            )

    @classmethod
    def report_all(cls, final: bool = False) -> None:
        for counter in list(cls._instances):
            counter.report(final)

    @classmethod
    def _report_loop(cls, period: float) -> None:
        while True:
            time.sleep(period)
            cls.report_all()


_MLA_NVFP4_EPILOGUE_VERIFY_COUNTERS: Dict[str, VerifyMismatchCounter] = {}


def _mla_nvfp4_epilogue_verify(kind: str, tokens: int, compute_mismatches) -> None:
    """Run ``compute_mismatches()`` (0-d device counts) for TRTLLM_MLA_NVFP4_EPILOGUE_VERIFY: 1 = eager forwards
    only, synchronizing and logging per call; 2 = every forward incl. graph capture, device counters."""
    capturing = torch.cuda.is_current_stream_capturing()
    if MLA_NVFP4_EPILOGUE_VERIFY == 1 and not capturing:
        _mla_nvfp4_epilogue_verify_record(kind, tokens, compute_mismatches())
    elif MLA_NVFP4_EPILOGUE_VERIFY >= 2:
        counter = _MLA_NVFP4_EPILOGUE_VERIFY_COUNTERS.get(kind)
        if counter is None:
            if capturing:
                return  # first use must be eager (graph warmup runs before capture)
            mismatches = compute_mismatches()
            counter = VerifyMismatchCounter(
                f"[TRTLLM_MLA_NVFP4_EPILOGUE] {kind}", tuple(sorted(mismatches))
            )
            _MLA_NVFP4_EPILOGUE_VERIFY_COUNTERS[kind] = counter
            counter.add(mismatches)
            return
        counter.add(compute_mismatches())


@torch.library.custom_op(
    "trtllm::mla_custom_op_inplace_nvfp4_out",
    mutates_args=("output", "output_sf"),
)
def mla_custom_op_inplace_nvfp4_out(
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
    output: torch.Tensor,
    output_sf: torch.Tensor,
    latent_cache_gen: Optional[torch.Tensor],
    hidden_states_fp4: Optional[torch.Tensor] = None,
    hidden_states_sf: Optional[torch.Tensor] = None,
) -> None:
    """mla_custom_op_inplace for a pure-generation batch whose bmm_o writes o_proj's NVFP4 input
    (TRTLLM_MLA_NVFP4_EPILOGUE bit 1): ``output`` is the packed E2M1 buffer and ``output_sf`` its swizzled scales."""
    metadata, mla_layer = _extract_mla_extra_attrs(layer_idx)
    if hidden_states_fp4 is not None or hidden_states_sf is not None:
        assert hidden_states_fp4 is not None and hidden_states_sf is not None, (
            "hidden_states_fp4 and hidden_states_sf must be passed together"
        )
        hidden_states = Fp4QuantizedTensor(
            fp4_tensor=hidden_states_fp4,
            scaling_factor=hidden_states_sf,
            unquantized_hidden_states=hidden_states,
        )
    mla_layer.forward_impl(
        position_ids,
        hidden_states,
        metadata,
        attn_output=[output],
        latent_cache_gen=latent_cache_gen,
        output_sf=output_sf,
    )


maybe_bcg_mla_custom_op_inplace_nvfp4_out = eager_on_graph(mla_custom_op_inplace_nvfp4_out)


def _is_cute_dsl_fp8_bmm_available(sm_version: Optional[int] = None) -> bool:
    if sm_version is None:
        sm_version = get_sm_version()
    if sm_version == 107:
        return IS_CUTLASS_DSL_INTERNAL_AVAILABLE
    return sm_version in (100, 103) and IS_CUTLASS_DSL_AVAILABLE


def _cute_dsl_fp8_bmm_out(
    mat1_fp8: torch.Tensor,
    mat2_fp8: torch.Tensor,
    mat1_scale: torch.Tensor,
    mat2_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    sm_version = get_sm_version()
    if sm_version == 107:
        torch.ops.trtllm.cute_dsl_fp8_bmm_rubin(mat1_fp8, mat2_fp8, mat1_scale, mat2_scale, out)
    elif sm_version in (100, 103):
        torch.ops.trtllm.cute_dsl_fp8_bmm_blackwell(mat1_fp8, mat2_fp8, mat1_scale, mat2_scale, out)
    else:
        raise NotImplementedError(
            f"CuTe DSL FP8 block-scaled BMM is not available on SM{sm_version}"
        )


def fp8_block_scaling_bmm_out(
    mat1: torch.Tensor,
    mat2_fp8: torch.Tensor,
    mat2_scale: torch.Tensor,
    out: torch.Tensor,
    mat2_dequant: Optional[torch.Tensor] = None,
    use_cute_dsl_blockscaling_bmm: bool = False,
) -> torch.Tensor:
    sm_version = get_sm_version()
    if sm_version == 90 or sm_version == 89:
        mat1_fp8, mat1_scale = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(mat1)

        output = out.new_empty(out.shape, dtype=out.dtype, device=out.device)
        torch.ops.trtllm.fp8_block_scaling_bmm_out(
            mat1_fp8, mat2_fp8, mat1_scale, mat2_scale, output
        )
        out.copy_(output)
    elif sm_version == 120:
        mat1_fp8, mat1_scale = fp8_utils.per_token_quant_and_transform(mat1, need_permute102=True)
        output = out.new_empty(out.shape, dtype=out.dtype, device=out.device)
        torch.ops.trtllm.fp8_block_scaling_bmm_out(
            mat1_fp8, mat2_fp8, mat1_scale, mat2_scale, output
        )
        out.copy_(output)
    elif is_sm_100f(sm_version):
        if use_cute_dsl_blockscaling_bmm:
            mat1_fp8, mat1_scale = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(mat1)
            _cute_dsl_fp8_bmm_out(mat1_fp8, mat2_fp8, mat1_scale, mat2_scale, out)
            mat1_scale = None
        else:
            torch.bmm(mat1.transpose(0, 1), mat2_dequant.transpose(1, 2), out=out)
    else:
        raise NotImplementedError(f"SM{sm_version} is not supported")


class MLA(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        predicted_tokens_per_seq: int,
        max_position_embeddings: int,
        bias: bool,
        aux_stream_dict: Optional[Dict[AuxStreamType, torch.cuda.Stream]] = None,
        pos_embd_params: Optional[PositionalEmbeddingParams] = None,
        layer_idx: Optional[int] = None,
        dtype: torch.dtype = None,
        dense_bias: Optional[bool] = None,
        config: Optional[ModelConfig] = None,
        mapping_with_cp: Optional[Mapping] = None,
        reduce_output: bool = True,
        num_groups: int = 1,
        o_lora_rank: int = 1024,
        fuse_qkv_a_proj: bool = True,
        rms_norm_eps: Optional[float] = None,
        enable_ugpu_bf16_linear: bool = False,
        flashinfer_mla_backend: Optional[str] = None,
    ) -> None:
        """
        Initialize the MLA module.

        Args:
            hidden_size (int): The size of the hidden dimension.
            num_attention_heads (int): The number of attention heads.
            num_key_value_heads (int): The number of key value heads.
            qk_nope_head_dim (int): The dimension of the query and key without Rope.
            qk_rope_head_dim (int): The dimension of the Rope of query and key.
            v_head_dim (int): The dimension of the value.
            q_lora_rank (int): The dimension of the compressed query.
            kv_lora_rank (int): The dimension of the compressed key and value.
            predicted_tokens_per_seq (int): The number of predicted tokens per sequence.
            max_position_embeddings (int): The maximum position embeddings.
            bias (bool): Whether to use bias in the linear layers.
            aux_stream_dict (Optional[Dict[AuxStreamType, torch.cuda.Stream]]): Auxiliary
                CUDA streams by role.
            pos_embd_params (PositionalEmbeddingParams): The positional embedding parameters.
            layer_idx (int): The layer index.
            dtype (torch.dtype): The data type.
            dense_bias (bool): Whether to use bias in the output projection layer.
            config (ModelConfig): The model configuration.
            num_groups (int): The number of groups.
            o_lora_rank (int): The dimension of the compressed output.
            fuse_qkv_a_proj (bool): Whether q_a and kv_a share one fused
                projection. Set to ``False`` for checkpoints that store a
                separate ``q_a_proj``.
            rms_norm_eps (Optional[float]): Override the RMSNorm epsilon from
                the pretrained config. If neither source provides a value
                (e.g. config.pretrained_config is None), falls back to 1e-6.
            enable_ugpu_bf16_linear (bool): Enable localized BF16 MLA projections.
            flashinfer_mla_backend (Optional[str]): Generation backend for the
                FlashInfer/TRTLLM-Gen MLA dispatcher. ``None`` preserves the
                attention backend default.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_idx_str = str(layer_idx)
        self.dtype = dtype
        self._weights_transformed = False

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.predicted_tokens_per_seq = predicted_tokens_per_seq
        self.max_position_embeddings = max_position_embeddings
        self.pos_embd_params = pos_embd_params
        self.dense_bias = dense_bias
        self.num_groups = num_groups
        self.o_lora_rank = o_lora_rank
        self.fuse_qkv_a_proj = fuse_qkv_a_proj
        if dense_bias is None:
            self.dense_bias = bias

        if self.q_lora_rank is None:
            self.q_lora_rank = hidden_size
            self.is_lite = True
        else:
            self.is_lite = False

        assert pos_embd_params is not None, "pos_embd_params must be provided in MLA"

        self.register_to_config = False
        if config is not None:
            if "mla_layers" not in config.extra_attrs:
                config.extra_attrs["mla_layers"] = {}
            suffix = 0
            # ``layer_idx`` is local to an attention stack, while this registry is shared
            # by target and draft modules in one-model speculative decoding. Keep the first
            # registration under ``"<layer_idx>"`` and suffix later collisions as
            # ``"<layer_idx>_<n>"`` so custom ops resolve the originating module without
            # overwriting another stack's weak reference.
            while self.layer_idx_str in config.extra_attrs["mla_layers"]:
                self.layer_idx_str = str(layer_idx) + f"_{suffix}"
                suffix += 1
            config.extra_attrs["mla_layers"][self.layer_idx_str] = weakref.ref(self)
            self.register_to_config = True

        config = config or ModelConfig()
        sparse_attn_cfg = config.sparse_attention_config
        sparse_params = (
            sparse_attn_cfg.to_sparse_params(
                pretrained_config=config.pretrained_config,
                layer_idx=self.layer_idx,
            )
            if sparse_attn_cfg is not None
            else None
        )
        self.sparse_params = sparse_params
        self.sparse_attn_hooks = get_sparse_mla_hooks(self)
        if self.sparse_attn_hooks and not fuse_qkv_a_proj:
            # forward_dsa_proj assumes the fused [q_a | kv_a | k_pe]
            # projection layout; the separate q_a_proj layout (Kimi K3 MLA)
            # is not wired into the sparse MLA paths.
            raise NotImplementedError(
                "Sparse MLA requires "
                "fuse_qkv_a_proj=True; the separate q_a_proj layout is not "
                "supported with sparse MLA."
            )

        # Fold the residual-less q_a_layernorm -> q_b_proj NVFP4 input
        # quantization into one fused RMSNorm + FP4-quantize kernel. Resolve
        # lazily because q_b_proj.input_scale is finalized after weight loading.
        self._qa_fused_scale = None

        # tensor parallel
        if mapping_with_cp is not None:
            logger.warning_once(
                "[MLA::__init__] Overriding mapping with CP detected.",
                key="mla_init_mapping_with_cp",
            )
            self.mapping = mapping_with_cp
        else:
            self.mapping = config.mapping
        tp_size = self.mapping.tp_size
        pp_size = self.mapping.pp_size
        cp_size = self.mapping.cp_size
        dp_size = 1
        if self.mapping.enable_attention_dp:
            dp_size = tp_size
            tp_size = 1
        if self.mapping.has_cp_ulysses():
            raise NotImplementedError("MLA doesn't support CP Ulysses yet")
        if self.mapping.cp_size > 1:
            assert self.mapping.has_cp_helix(), (
                f"CP type must be HELIX for MLA, but got {self.mapping.cp_config['cp_type']}."
            )

        mapping = Mapping(
            world_size=pp_size * dp_size * tp_size * cp_size,
            tp_size=tp_size,
            pp_size=pp_size * dp_size,
            cp_size=cp_size,
            cp_config=self.mapping.cp_config,
            rank=self.mapping.rank,
            gpus_per_node=self.mapping.gpus_per_node,
            enable_attention_dp=self.mapping.enable_attention_dp,
        )

        assert self.num_heads % (tp_size * cp_size) == 0
        self.num_heads_tp = self.num_heads // tp_size
        self.num_heads_tp_cp = self.num_heads_tp // cp_size
        self.num_key_value_heads_tp = (self.num_key_value_heads + tp_size - 1) // tp_size

        if rms_norm_eps is None:
            rms_norm_eps = getattr(config.pretrained_config, "rms_norm_eps", None)
        if rms_norm_eps is None:
            # No explicit value and pretrained_config is None or lacks
            # rms_norm_eps (e.g. unit tests constructing MLA directly):
            # keep the historical default.
            rms_norm_eps = 1e-6
        quant_config = config.get_quant_config()
        self.quant_config = quant_config
        # Construction settings used by sparse module hooks.
        self.rms_norm_eps = rms_norm_eps
        self.reduce_output = reduce_output
        self.skip_create_weights_in_init = config.skip_create_weights_in_init
        self.allreduce_strategy = config.allreduce_strategy
        self.force_dynamic_quantization = config.force_dynamic_quantization
        self.num_hidden_layers = getattr(config.pretrained_config, "num_hidden_layers", None)
        self.pp_mapping = config.mapping

        self.use_cute_dsl_blockscaling_mm = config.use_cute_dsl_blockscaling_mm
        self.use_cute_dsl_blockscaling_bmm = config.use_cute_dsl_blockscaling_bmm
        self.use_cute_dsl_bf16_bmm = config.use_cute_dsl_bf16_bmm
        self.use_cute_dsl_bf16_gemm = config.use_cute_dsl_bf16_gemm
        ugpu_policy = config.extra_attrs.get("ugpu_policy")
        self._ugpu_policy = ugpu_policy if ugpu_policy is not None else UgpuPolicy()
        self.enable_ugpu_bf16_linear = enable_ugpu_bf16_linear
        self._ugpu_runtime: Optional[UgpuRuntime] = None
        self._ugpu_k_b_proj_trans_shards: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._ugpu_v_b_proj_shards: Optional[tuple[torch.Tensor, torch.Tensor]] = None

        if not self.is_lite:
            kv_a_out_features = self.kv_lora_rank + self.qk_rope_head_dim
            if self.fuse_qkv_a_proj:
                kv_a_out_features += self.q_lora_rank
            self.kv_a_proj_with_mqa = Linear(
                hidden_size,
                kv_a_out_features,
                bias=bias,
                dtype=dtype,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                use_custom_cublas_mm=True,
                force_dynamic_quantization=config.force_dynamic_quantization,
                use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
                use_cute_dsl_bf16_gemm=self.use_cute_dsl_bf16_gemm,
            )

            if not self.fuse_qkv_a_proj:
                self.q_a_proj = Linear(
                    hidden_size,
                    self.q_lora_rank,
                    bias=bias,
                    dtype=dtype,
                    quant_config=quant_config,
                    skip_create_weights_in_init=config.skip_create_weights_in_init,
                    use_custom_cublas_mm=True,
                    force_dynamic_quantization=config.force_dynamic_quantization,
                    use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
                    use_cute_dsl_bf16_gemm=self.use_cute_dsl_bf16_gemm,
                )

            self.q_a_layernorm = RMSNorm(
                hidden_size=self.q_lora_rank, eps=rms_norm_eps, dtype=dtype
            )

            self.q_b_proj = Linear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=bias,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                allreduce_strategy=config.allreduce_strategy,
                force_dynamic_quantization=config.force_dynamic_quantization,
                use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
                use_cute_dsl_bf16_gemm=self.use_cute_dsl_bf16_gemm,
                enable_ugpu_bf16_linear=enable_ugpu_bf16_linear,
                ugpu_policy=self._ugpu_policy,
            )
        else:
            self.kv_a_proj_with_mqa = Linear(
                hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=bias,
                dtype=dtype,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                use_custom_cublas_mm=True,
                force_dynamic_quantization=config.force_dynamic_quantization,
                use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
                use_cute_dsl_bf16_gemm=self.use_cute_dsl_bf16_gemm,
            )

            self.q_proj = Linear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=bias,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                allreduce_strategy=config.allreduce_strategy,
                force_dynamic_quantization=config.force_dynamic_quantization,
                use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
                use_cute_dsl_bf16_gemm=self.use_cute_dsl_bf16_gemm,
                enable_ugpu_bf16_linear=enable_ugpu_bf16_linear,
                ugpu_policy=self._ugpu_policy,
            )
            self.q_b_proj = self.q_proj

        mapping_o = Mapping(
            world_size=pp_size * dp_size * tp_size * cp_size,
            tp_size=tp_size * cp_size,
            pp_size=pp_size * dp_size,
            cp_size=1,
            rank=self.mapping.rank,
            gpus_per_node=self.mapping.gpus_per_node,
            enable_attention_dp=self.mapping.enable_attention_dp,
        )
        self.mapping_o = mapping_o

        def yarn_get_mscale(scale=1, mscale=1):
            if scale <= 1:
                return 1.0
            return 0.1 * mscale * math.log(scale) + 1.0

        mscale_all_dim = pos_embd_params.rope.mscale_all_dim
        scaling_factor = pos_embd_params.rope.scale
        mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
        q_scaling = 1.0 / (mscale * mscale)

        self.aux_stream_dict = aux_stream_dict or {}
        self.aux_stream = self.aux_stream_dict.get(AuxStreamType.Attention)
        mqa_aux_stream = (
            self.sparse_attn_hooks.get_mqa_aux_stream(self)
            if self.sparse_attn_hooks is not None
            else self.aux_stream
        )
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]
        need_absorption = self.sparse_attn_hooks is None or self.sparse_attn_hooks.need_absorption
        if need_absorption:
            self.kv_a_layernorm = RMSNorm(
                hidden_size=self.kv_lora_rank, dtype=dtype, eps=rms_norm_eps
            )
            self.kv_b_proj = Linear(
                self.kv_lora_rank,
                self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
                bias=bias,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                allreduce_strategy=config.allreduce_strategy,
                force_dynamic_quantization=config.force_dynamic_quantization,
                use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
                use_cute_dsl_bf16_gemm=self.use_cute_dsl_bf16_gemm,
                enable_ugpu_bf16_linear=enable_ugpu_bf16_linear,
                ugpu_policy=self._ugpu_policy,
            )
            self.v_b_proj = nn.Parameter(
                torch.empty(
                    (self.num_heads_tp_cp, self.v_head_dim, self.kv_lora_rank),
                    dtype=dtype,
                ),
                requires_grad=False,
            )

        self.mqa = create_attention(
            config.attn_backend,
            self.layer_idx,
            self.num_heads_tp,
            head_dim=self.kv_lora_rank + self.qk_rope_head_dim,
            num_kv_heads=1,
            pos_embd_params=self.pos_embd_params,
            quant_config=quant_config,
            q_scaling=q_scaling,
            is_mla_enable=True,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            hidden_size=self.hidden_size,
            predicted_tokens_per_seq=self.predicted_tokens_per_seq,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            v_head_dim=self.kv_lora_rank,
            sparse_params=self.sparse_params,
            dtype=dtype,
            aux_stream=mqa_aux_stream,
            rope_append=True,
            flashinfer_mla_backend=flashinfer_mla_backend,
        )
        if self.mqa is None:
            raise RuntimeError("MLA requires a non-null MQA attention backend")

        if self.sparse_attn_hooks is not None:
            self.sparse_attn_hooks.initialize(self)

        create_dense_mha = self.sparse_attn_hooks is None or self.sparse_attn_hooks.need_dense_mha
        if create_dense_mha:
            mha_sparse_params = self.sparse_params if self.sparse_attn_hooks is None else None
            self.mha = create_attention(
                config.attn_backend,
                self.layer_idx,
                self.num_heads_tp,
                head_dim=self.qk_head_dim,
                num_kv_heads=self.num_key_value_heads_tp,
                pos_embd_params=self.pos_embd_params,
                quant_config=quant_config,
                q_scaling=q_scaling,
                is_mla_enable=True,
                q_lora_rank=self.q_lora_rank,
                kv_lora_rank=self.kv_lora_rank,
                qk_nope_head_dim=self.qk_nope_head_dim,
                qk_rope_head_dim=self.qk_rope_head_dim,
                v_head_dim=self.v_head_dim,
                predicted_tokens_per_seq=self.predicted_tokens_per_seq,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                sparse_params=mha_sparse_params,
            )
        else:
            self.mha = None

        need_default_o_proj = (
            self.sparse_attn_hooks is None or self.sparse_attn_hooks.need_default_o_proj
        )
        if need_default_o_proj:
            self.o_proj = Linear(
                self.num_key_value_heads * self.v_head_dim,
                self.hidden_size,
                bias=self.dense_bias,
                dtype=dtype,
                mapping=mapping_o,
                tensor_parallel_mode=TensorParallelMode.ROW,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                reduce_output=reduce_output,
                allreduce_strategy=config.allreduce_strategy,
                force_dynamic_quantization=config.force_dynamic_quantization,
                use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
                use_cute_dsl_bf16_gemm=self.use_cute_dsl_bf16_gemm,
                enable_ugpu_bf16_linear=enable_ugpu_bf16_linear,
                ugpu_policy=self._ugpu_policy,
            )
            self.attention_output_hidden_size = self.o_proj.in_features

        self.softmax_scale = 1.0 / (math.sqrt(self.qk_head_dim) * q_scaling)

        self.rope_fusion = self.mqa.support_fused_rope()
        self.rotary_emb = None
        self.apply_rotary_emb = not self.rope_fusion
        if self.apply_rotary_emb:
            self.rotary_emb = RotaryEmbedding(
                pos_embd_params.rope,
                head_dim=self.qk_rope_head_dim,
                is_neox=pos_embd_params.is_neox,
            )

        self.llama_4_scaling = False
        if hasattr(config.pretrained_config, "llama_4_scaling"):
            self.llama_4_scaling = True
            self.floor_scale = getattr(
                config.pretrained_config.llama_4_scaling,
                "original_max_position_embeddings",
                8192,
            )
            self.attn_scale = getattr(config.pretrained_config.llama_4_scaling, "beta", 0.1)

        if not config.skip_create_weights_in_init:
            self.create_weights()

    def create_weights(self):
        # self.mha/mqa has no weights but has states that are related to
        # quant_config, which could be modified after __init__.
        if self.mha is not None:
            self.mha.update_quant_config(self.quant_config)
        self.mqa.update_quant_config(self.quant_config)

        # Although we use FP8 MLA for context/generation phase, the output is still in BF16
        self.out_scale = None
        if self.sparse_attn_hooks is not None:
            self.sparse_attn_hooks.create_weights(self)
        if self.sparse_attn_hooks is not None and not self.sparse_attn_hooks.need_absorption:
            self._weights_transformed = False
            return

        # k_b_proj_trans's dtype must be consistent with self.kv_b_proj,
        # which can be modified after __init__
        has_fp8_block_scales = bool(
            self.kv_b_proj.quant_config
            and self.kv_b_proj.quant_config.quant_mode.has_fp8_block_scales()
        )
        mla_weight_dtype = torch.float8_e4m3fn if has_fp8_block_scales else self.dtype
        self.k_b_proj_trans = nn.Parameter(
            torch.empty(
                (self.num_heads_tp, self.kv_lora_rank, self.qk_nope_head_dim),
                dtype=mla_weight_dtype,
            ),
            requires_grad=False,
        )

        self.k_b_proj_trans_dequant = None
        self.v_b_proj_dequant = None
        if has_fp8_block_scales:
            self.k_b_proj_trans_scale = nn.Parameter(
                torch.empty(
                    (
                        self.num_heads_tp,
                        self.kv_lora_rank // 128,
                        self.qk_nope_head_dim // 128,
                    ),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            # This parameter will view into self.kv_b_proj.weight_scale after loading weights.
            # For dummy weight initialization, this parameter is initialized with empty tensor.
            self.v_b_proj_scale = nn.Parameter(
                torch.empty(
                    (
                        self.num_heads_tp_cp,
                        self.v_head_dim // 128,
                        self.kv_lora_rank // 128,
                    ),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            if is_sm_100f() and not self.use_cute_dsl_blockscaling_bmm:
                assert self.dtype == torch.bfloat16
                self.k_b_proj_trans_dequant = nn.Parameter(
                    torch.empty(
                        (self.num_heads_tp, self.kv_lora_rank, self.qk_nope_head_dim),
                        dtype=self.dtype,
                    ),
                    requires_grad=False,
                )
                self.v_b_proj_dequant = nn.Parameter(
                    torch.empty(
                        (self.num_heads_tp_cp, self.v_head_dim, self.kv_lora_rank),
                        dtype=self.dtype,
                    ),
                    requires_grad=False,
                )
        else:
            self.k_b_proj_trans_scale = None
            self.v_b_proj_scale = None
        self._ugpu_k_b_proj_trans_shards = None
        self._ugpu_v_b_proj_shards = None
        self._kvbproj_head_weights_cache = None
        self._weights_transformed = False

    def _bind_v_b_proj_weight(self, weight: torch.Tensor) -> None:
        """Bind the V absorption weight without retaining a released full weight."""
        plan = self.kv_b_proj.partition_plan
        if plan.enabled and plan.op_kind == "bf16_linear":
            weight = weight.clone(memory_format=torch.contiguous_format)
        self.v_b_proj = nn.Parameter(weight, requires_grad=False)

    def apply_rope(
        self,
        q: torch.Tensor,
        k_pe: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        q = q.view(-1, self.num_heads_tp, self.qk_head_dim)
        q_pe = q[..., self.qk_nope_head_dim :].reshape(
            -1, self.num_heads_tp * self.qk_rope_head_dim
        )
        q_pe, k_pe = self.rotary_emb(position_ids, [q_pe, k_pe])
        q[..., self.qk_nope_head_dim :] = q_pe.view(-1, self.num_heads_tp, self.qk_rope_head_dim)
        return k_pe

    def _attn_forward_gen(
        self,
        attn_backend: AttentionBackend,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        **kwargs,
    ):
        if self.mapping.has_cp_helix():
            # partial_o: [num_tokens, num_heads_tp * kv_lora_rank]
            # softmax_stats: [num_tokens, num_heads_tp, 2]
            softmax_stats = torch.empty(
                (q.shape[0], self.num_heads_tp, 2), device=q.device, dtype=torch.float32
            )
            kwargs["softmax_stats_tensor"] = softmax_stats
            partial_o = attn_backend.forward(
                q,
                k,
                v,
                attn_metadata,
                forward_args=AttentionForwardArgs(**kwargs),
            )
            kv_lora_rank = partial_o.shape[-1] // self.num_heads_tp
            assert self.kv_lora_rank == kv_lora_rank

            helix_kv_bounds = getattr(attn_metadata, "helix_kv_bounds", None)
            if helix_kv_bounds is not None and getattr(
                attn_metadata, "_helix_spec_tokens_valid", False
            ):
                # Speculative verify groups: KV ownership is per-TOKEN. A rank
                # owning only the tail page of a group has zero visible KV for
                # the group's leading tokens while its per-sequence kv_len is
                # nonzero, so the per-sequence mask above misses those rows.
                # Their decode rows are fully masked with a finite sentinel,
                # making partial_o an average over (possibly uninitialized)
                # pool values; the combine multiplies by corr = 0 and
                # 0 * NaN would poison the token on every CP rank — sanitize
                # by the per-token bound instead.
                zero_kv_mask = helix_kv_bounds[: partial_o.shape[0]] == 0
            else:
                # MLA processes only the generation token slice here, so build
                # the mask from [num_contexts, num_seqs). Skip this expansion
                # when the more precise per-token bounds above are valid.
                zero_kv_mask = _helix_zero_kv_mask(
                    attn_metadata,
                    partial_o.shape[0],
                    seq_start=attn_metadata.num_contexts,
                    num_seqs=attn_metadata.num_generations,
                )
            return _helix_post_process(
                partial_o,
                softmax_stats,
                self.mapping,
                self.num_heads_tp_cp,
                kv_lora_rank,
                self.aux_stream,
                self.ln_events,
                zero_kv_mask=zero_kv_mask,
            )
        else:
            attn_output = attn_backend.forward(
                q,
                k,
                v,
                attn_metadata,
                forward_args=AttentionForwardArgs(**kwargs),
            )
            return attn_output

    def create_output(self, hidden_states: torch.Tensor, num_contexts: int):
        if isinstance(hidden_states, Fp4QuantizedTensor):
            assert hidden_states.unquantized_hidden_states is not None, (
                "MLA.create_output received an Fp4QuantizedTensor without a "
                "unquantized_hidden_states view; the producing fusion must use "
                "return_norm_out=True"
            )
            hidden_states = hidden_states.unquantized_hidden_states
        num_tokens = hidden_states.shape[0]
        return hidden_states.new_empty(
            [num_tokens, self.attention_output_hidden_size], dtype=hidden_states.dtype
        )

    def _create_outputs(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> list[torch.Tensor]:
        """Create the standard output and any algorithm-specific output buffers."""
        if self.sparse_attn_hooks is not None:
            outputs = self.sparse_attn_hooks.prepare_outputs(self, hidden_states, attn_metadata)
            if outputs is not None:
                return outputs
        return [self.create_output(hidden_states, attn_metadata.num_contexts)]

    def _attention_scaling(self, q, position_ids):
        def _get_attn_scale(position_ids: torch.Tensor) -> torch.Tensor:
            positions = position_ids.view(-1)
            floor = torch.floor((positions + 1.0) / self.floor_scale)
            attn_scale = torch.log(floor + 1.0) * self.attn_scale + 1.0
            return attn_scale.unsqueeze(-1)

        attn_scale = _get_attn_scale(position_ids)
        q = (q * attn_scale).to(q.dtype)
        return q

    def _resolve_qa_fused_scale(self):
        """Resolve whether q_a_layernorm can fuse q_b_proj NVFP4 input quantization."""
        if self._qa_fused_scale is not None:
            return self._qa_fused_scale if self._qa_fused_scale is not False else None
        eligible = False
        if (
            not self.is_lite
            and getattr(self, "q_a_layernorm", None) is not None
            and getattr(self, "q_b_proj", None) is not None
        ):
            qb = self.q_b_proj
            qa = self.q_a_layernorm
            if is_static_nvfp4_input_eligible(qb) and not qa.use_gemma and not qa.is_nvfp4:
                eligible = True
        if eligible:
            self.q_a_layernorm.is_nvfp4 = True
            self.q_a_layernorm.nvfp4_scale = qb.input_scale
        self._qa_fused_scale = qb.input_scale if eligible else False
        return self._qa_fused_scale if eligible else None

    def _q_a_layernorm_maybe_fused(self, q: torch.Tensor, return_norm_out: bool = False):
        """Apply q_a_layernorm and fuse static NVFP4 input quantization when eligible."""
        scale = self._resolve_qa_fused_scale()
        if scale is None:
            normed = self.q_a_layernorm(q)
            return (normed, normed) if return_norm_out else normed
        return self.q_a_layernorm(q, return_norm_out=return_norm_out)

    def forward_impl(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        attn_output: list[torch.Tensor],
        latent_cache_gen: Optional[torch.Tensor] = None,
        output_sf: Optional[torch.Tensor] = None,
    ) -> None:
        """Run the dense or sparse implementation of the shared MLA module.

        ``output_sf`` (TRTLLM_MLA_NVFP4_EPILOGUE bit 1, pure-generation batches only): ``attn_output[0]`` is
        o_proj's packed NVFP4 input and this its swizzled scale buffer."""
        if output_sf is not None and (
            self.sparse_attn_hooks is not None or attn_metadata.num_contexts > 0
        ):
            raise RuntimeError(
                "MLA NVFP4 output (TRTLLM_MLA_NVFP4_EPILOGUE) is for dense pure-generation batches only."
            )
        if self.sparse_attn_hooks is not None:
            self.sparse_attn_hooks.forward(
                self,
                position_ids,
                hidden_states,
                attn_metadata,
                attn_output,
            )
            return

        self._forward_impl(
            position_ids,
            hidden_states,
            attn_metadata,
            attn_output[0],
            latent_cache_gen=latent_cache_gen,
            output_sf=output_sf,
        )

    def _forward_impl(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        latent_cache_gen: Optional[torch.Tensor] = None,
        output_sf: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Forward pass for the MLA module. Writes result into output tensor in-place.

        Args:
            position_ids (Optional[torch.IntTensor]): The position IDs.
            hidden_states (torch.Tensor): The hidden states.
            attn_metadata (AttentionMetadata): The attention metadata.
            output (torch.Tensor): The output tensor to write results into.
            latent_cache_gen (Optional[torch.Tensor]): The latent cache used in generation.
        """
        # split q, k, v into context and gen batches
        num_contexts = attn_metadata.num_contexts
        num_generations = attn_metadata.num_generations
        num_ctx_tokens = attn_metadata.num_ctx_tokens
        num_tokens = attn_metadata.num_tokens

        hidden_states = _slice_hidden_states_to_num_tokens(hidden_states, num_tokens)
        if position_ids is not None:
            position_ids = position_ids[..., :num_tokens]

        if self.is_lite:
            compressed_kv, k_pe = self.kv_a_proj_with_mqa(hidden_states).split(
                [self.kv_lora_rank, self.qk_rope_head_dim], -1
            )
            compressed_kv = self.kv_a_layernorm(compressed_kv)
            q = hidden_states
        else:
            if self.fuse_qkv_a_proj:
                q, compressed_kv, k_pe = self.kv_a_proj_with_mqa(hidden_states).split(
                    [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], -1
                )
            else:
                q = self.q_a_proj(hidden_states)
                compressed_kv, k_pe = self.kv_a_proj_with_mqa(hidden_states).split(
                    [self.kv_lora_rank, self.qk_rope_head_dim], -1
                )

            q, compressed_kv = maybe_execute_in_parallel(
                lambda: self._q_a_layernorm_maybe_fused(q),
                lambda: self.kv_a_layernorm(compressed_kv),
                self.ln_events[0],
                self.ln_events[1],
                self.aux_stream,
            )

        q, latent_cache = maybe_execute_in_parallel(
            lambda: self.q_b_proj(q),
            lambda: torch.concat([compressed_kv, k_pe], dim=-1),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )

        assert q.shape[0] == num_tokens, (
            f"Expect q.shape[0] to be {num_tokens}, but got {q.shape[0]}"
        )

        assert output is not None, "output must be provided"

        if num_contexts > 0:
            q_ctx = q[:num_ctx_tokens, ...]
            compressed_kv_ctx = compressed_kv[:num_ctx_tokens, ...]
            k_pe_ctx = k_pe[:num_ctx_tokens, ...]
            latent_cache_ctx = latent_cache[:num_ctx_tokens, ...]
            if self.apply_rotary_emb:
                assert position_ids is not None
                # position_ids spans [ctx..., gen...] in mixed batches; slice to
                # match q_ctx/k_pe_ctx so external RoPE uses ctx positions.
                ctx_position_ids = position_ids[..., :num_ctx_tokens]
                k_pe_ctx = self.apply_rope(q_ctx, k_pe_ctx, ctx_position_ids)
                # External RoPE is only used by backends that do not handle
                # fused RoPE internally, so keep latent_cache in sync.
                latent_cache_ctx = torch.cat([compressed_kv_ctx, k_pe_ctx], dim=-1)

            if self.llama_4_scaling:
                q_ctx = self._attention_scaling(q_ctx, position_ids[..., :num_ctx_tokens])

            self.forward_context(
                q_ctx,
                compressed_kv_ctx,
                k_pe_ctx,
                position_ids,
                attn_metadata,
                output[:num_ctx_tokens, :],
                latent_cache_ctx,
            )

        if num_generations > 0:
            q_gen = q[num_ctx_tokens:, ...]
            compressed_kv_gen = compressed_kv[num_ctx_tokens:, ...]
            k_pe_gen = k_pe[num_ctx_tokens:, ...]
            if latent_cache_gen is None:
                latent_cache_gen = latent_cache[num_ctx_tokens:, ...]

            if self.llama_4_scaling:
                q_gen = self._attention_scaling(q_gen, position_ids[..., num_ctx_tokens:])

            self.forward_absorption_generation(
                q_gen,
                compressed_kv_gen,
                k_pe_gen,
                attn_metadata,
                output[num_ctx_tokens:num_tokens, :],
                position_ids=position_ids,
                latent_cache=latent_cache_gen,
                output_sf=output_sf,
            )

    def _use_kvbproj_strided_out(self) -> bool:
        """Whether context KV expansion can write directly into strided outputs.

        This requires an SM100-family GPU and an unquantized BF16 kv_b_proj
        without bias. Other configurations retain the wide-GEMM path.
        """
        if not is_sm_100f():
            return False
        w = getattr(self.kv_b_proj, "weight", None)
        expected_shape = (
            self.num_heads_tp * (self.qk_nope_head_dim + self.v_head_dim),
            self.kv_lora_rank,
        )
        expected_numel = math.prod(expected_shape)
        return (
            w is not None
            and w.dtype == torch.bfloat16
            and tuple(w.shape) == expected_shape
            and w.numel() == expected_numel
            and getattr(self.kv_b_proj, "bias", None) is None
        )

    def _get_kvbproj_head_weights(self):
        """Return cached per-head [H, N, K] views of kv_b_proj.weight."""
        cached = getattr(self, "_kvbproj_head_weights_cache", None)
        weight = self.kv_b_proj.weight
        if cached is not None and cached[0] is weight:
            return cached[1], cached[2]

        nope_rows = self.num_heads_tp * self.qk_nope_head_dim
        weight_k = weight[:nope_rows].view(
            self.num_heads_tp,
            self.qk_nope_head_dim,
            self.kv_lora_rank,
        )
        weight_v = weight[nope_rows:].view(
            self.num_heads_tp,
            self.v_head_dim,
            self.kv_lora_rank,
        )
        self._kvbproj_head_weights_cache = (weight, weight_k, weight_v)
        return weight_k, weight_v

    def forward_context_default(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        latent_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Dense MHA context path: expand KV via kv_b_proj and run attention."""
        k = torch.empty_like(q).view(-1, self.num_heads_tp, self.qk_head_dim)
        if self._use_kvbproj_strided_out():
            num_tokens = compressed_kv.shape[0]
            weight_k, weight_v = self._get_kvbproj_head_weights()
            compressed_kv_batched = compressed_kv.unsqueeze(0).expand(
                self.num_heads_tp,
                num_tokens,
                self.kv_lora_rank,
            )

            k_nope_out = k[..., : self.qk_nope_head_dim].transpose(0, 1)
            nope_cols = self.num_heads_tp * self.qk_nope_head_dim
            v_cols = self.num_heads_tp * self.v_head_dim
            kv_buffer = torch.empty(
                num_tokens,
                nope_cols + v_cols,
                dtype=q.dtype,
                device=q.device,
            )
            v = kv_buffer[:, nope_cols:]
            v_out = v.unflatten(
                -1,
                (self.num_heads_tp, self.v_head_dim),
            ).transpose(0, 1)

            torch.ops.trtllm.bmm_out(
                compressed_kv_batched,
                weight_k.transpose(1, 2),
                k_nope_out,
            )
            torch.ops.trtllm.bmm_out(
                compressed_kv_batched,
                weight_v.transpose(1, 2),
                v_out,
            )
        else:
            kv = self.kv_b_proj(compressed_kv)
            k_nope, v = kv.split(
                [
                    self.num_heads_tp * self.qk_nope_head_dim,
                    self.num_heads_tp * self.v_head_dim,
                ],
                -1,
            )
            maybe_compiled_copy_(
                k[..., : self.qk_nope_head_dim],
                k_nope.view(-1, self.num_heads_tp, self.qk_nope_head_dim),
            )
        # When rope_fusion=True (apply_rotary_emb=False), the rope portion
        # of k is left uninitialized here; the fused attention kernel
        # handles k_pe RoPE via latent_cache instead.
        if self.apply_rotary_emb:
            k[..., self.qk_nope_head_dim :] = k_pe.view(-1, 1, self.qk_rope_head_dim)
        k = k.view(-1, self.num_heads_tp * self.qk_head_dim)

        attn_output = self.mha.forward(
            q,
            k,
            v,
            attn_metadata,
            forward_args=AttentionForwardArgs(
                attention_input_type=AttentionInputType.context_only,
                latent_cache=latent_cache,
                out_scale=self.out_scale,
                output=output,
            ),
        )

        return attn_output

    def forward_context_with_cached_kv(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        assert latent_cache is not None
        trtllm_attention = cast(TrtllmAttention, self.mha)

        # apply RoPE, append compressed_kv + k_pe to paged kv cache and assign q_pe to q
        trtllm_attention.mla_rope_append_paged_kv_assign_q(q, latent_cache, attn_metadata)

        # copy full_compressed_kv and full_k_pe from paged kv cache
        full_compressed_kv, full_k_pe = trtllm_attention.load_paged_kv_cache_for_mla(
            attn_metadata, q.dtype
        )
        assert (
            full_compressed_kv.shape[0]
            == attn_metadata.num_ctx_cached_tokens + attn_metadata.num_ctx_tokens
        )
        assert full_compressed_kv.shape[1] == self.kv_lora_rank
        assert (
            full_k_pe.shape[0] == attn_metadata.num_ctx_cached_tokens + attn_metadata.num_ctx_tokens
        )
        assert full_k_pe.shape[1] == self.qk_rope_head_dim
        assert full_compressed_kv.is_contiguous()
        assert full_k_pe.is_contiguous()

        # compute full_k_nope and full_v from full_compressed_kv
        full_kv = self.kv_b_proj(full_compressed_kv)
        full_k_nope, full_v = full_kv.split(
            [
                self.num_heads_tp * self.qk_nope_head_dim,
                self.num_heads_tp * self.v_head_dim,
            ],
            -1,
        )

        full_k_nope = full_k_nope.view(-1, self.num_heads_tp, self.qk_nope_head_dim)
        full_k_pe = full_k_pe.view(-1, 1, self.qk_rope_head_dim)
        full_k = maybe_compiled_cat(
            (full_k_nope, full_k_pe.expand(-1, self.num_heads_tp, -1)), dim=-1
        )
        full_k = full_k.view(-1, self.num_heads_tp * self.qk_head_dim)

        # release pytorch activation memory
        full_compressed_kv = None
        full_k_pe = None
        full_kv = None
        full_k_nope = None

        # latent_cache must be None to differentiate from normal context phase,
        # so that we can skip applying RoPE and appending KV cache inside attention op
        attn_output = self.mha.forward(
            q,
            full_k,
            full_v,
            attn_metadata,
            forward_args=AttentionForwardArgs(
                attention_input_type=AttentionInputType.context_only,
                latent_cache=None,
                out_scale=self.out_scale,
                output=output,
            ),
        )

        return attn_output

    def forward_context_with_chunked_prefill_fp4(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        latent_cache: torch.Tensor,
        attn_metadata: TrtllmAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Run dense FP4 MLA chunked prefill through FP8 context partitions."""
        trtllm_attention = cast(TrtllmAttention, self.mha)
        if latent_cache is None:
            raise RuntimeError("FP4 MLA chunked prefill requires latent_cache.")

        # Match the FP8 KV path: fuse Q/K RoPE with cache append, then consume
        # the rotated current K directly from latent_cache.
        trtllm_attention.append_fp4_mla_context_cache(q, latent_cache, attn_metadata)
        current_k_pe = latent_cache.reshape(
            -1,
            self.kv_lora_rank + self.qk_rope_head_dim,
        )[:, self.kv_lora_rank :]

        self.softmax_stats_tensor = torch.empty(
            (attn_metadata.num_ctx_tokens, self.num_heads_tp, 2),
            dtype=torch.float,
            device=q.device,
        )
        self.temp_softmax_stats_tensor = torch.empty_like(self.softmax_stats_tensor)
        attn_output = output
        temp_attn_output = q.new_empty(
            (q.size(0), self.num_heads_tp * self.v_head_dim),
        )

        def expand_kv(
            partition_compressed_kv: torch.Tensor,
            partition_k_pe: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            partition_kv = self.kv_b_proj(partition_compressed_kv)
            partition_k_nope, partition_v = partition_kv.split(
                [
                    self.num_heads_tp * self.qk_nope_head_dim,
                    self.num_heads_tp * self.v_head_dim,
                ],
                -1,
            )
            partition_k_nope = partition_k_nope.view(
                -1,
                self.num_heads_tp,
                self.qk_nope_head_dim,
            )
            partition_k_pe = partition_k_pe.view(-1, 1, self.qk_rope_head_dim)
            partition_k = maybe_compiled_cat(
                (
                    partition_k_nope,
                    partition_k_pe.expand(-1, self.num_heads_tp, -1),
                ),
                dim=-1,
            )
            return partition_k.view(-1, self.num_heads_tp * self.qk_head_dim), partition_v

        def run_partition(
            partition_k: torch.Tensor,
            partition_v: torch.Tensor,
            attention_mask: PredefinedAttentionMask,
            kv_lens_cuda: torch.Tensor,
            kv_lens_cpu: torch.Tensor,
            merge_op: torch.Tensor,
        ) -> None:
            trtllm_attention.forward_fp4_mla_context_partition(
                q,
                partition_k,
                partition_v,
                attn_metadata,
                AttentionForwardArgs(
                    attention_input_type=AttentionInputType.context_only,
                    latent_cache=None,
                    out_scale=self.out_scale,
                    attention_mask=attention_mask,
                    softmax_stats_tensor=self.temp_softmax_stats_tensor,
                    chunked_prefill_buffer_batch_size=(
                        attn_metadata.runtime_features.chunked_prefill_buffer_batch_size
                    ),
                    output=temp_attn_output,
                ),
                kv_lens_cuda=kv_lens_cuda,
                kv_lens_cpu=kv_lens_cpu,
            )
            trtllm_attention.merge_attention_for_mla(
                attn_output,
                temp_attn_output,
                self.softmax_stats_tensor,
                self.temp_softmax_stats_tensor,
                merge_op,
                attn_metadata,
            )

        for loop_idx in range(attn_metadata.chunked_loop_num):
            total_chunked_tokens = int(
                attn_metadata.host_cu_chunked_seq_len[
                    loop_idx,
                    attn_metadata.num_contexts,
                ]
            )
            chunked_compressed_kv, chunked_k_pe = trtllm_attention.load_chunked_kv_cache_for_mla(
                metadata=attn_metadata,
                num_ctx_cached_tokens=total_chunked_tokens,
                cu_chunked_seq_len=attn_metadata.cu_chunked_seq_len[loop_idx],
                chunked_global_offset=attn_metadata.chunked_global_offset[loop_idx],
                chunked_max_seq_len=int(attn_metadata.max_chunk_len_per_loop[loop_idx]),
                out_dtype=q.dtype,
            )
            chunked_k, chunked_v = expand_kv(chunked_compressed_kv, chunked_k_pe)
            run_partition(
                chunked_k,
                chunked_v,
                PredefinedAttentionMask.FULL,
                attn_metadata.chunked_seq_len[loop_idx],
                attn_metadata.host_chunked_seq_len[loop_idx],
                attn_metadata.merge_op_tensor[loop_idx],
            )

        current_k, current_v = expand_kv(compressed_kv, current_k_pe)
        run_partition(
            current_k,
            current_v,
            PredefinedAttentionMask.CAUSAL,
            attn_metadata.prompt_lens_cuda_runtime,
            attn_metadata.prompt_lens_cpu_runtime,
            attn_metadata.merge_op_tensor[attn_metadata.chunked_loop_num],
        )
        return attn_output

    def forward_context_with_chunked_prefill(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        latent_cache: torch.Tensor,  # compressed_kv + k_pe [context_tokens, 1, lora_size + rope_size]
        attn_metadata: TrtllmAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        trtllm_attention = cast(TrtllmAttention, self.mha)
        # apply RoPE, append compressed_kv + k_pe to paged kv cache and assign q_pe to q
        trtllm_attention.mla_rope_append_paged_kv_assign_q(q, latent_cache, attn_metadata)

        # determine the number of loop
        # currently we assume that the chunk size is the same as the max_num_tokens
        chunked_loop_num = attn_metadata.chunked_loop_num

        # [total_token_q, num_heads, 2] -> [total_token_q, num_heads] float2
        self.softmax_stats_tensor = torch.empty(
            (attn_metadata.num_ctx_tokens, self.num_heads_tp, 2),
            dtype=torch.float,
            device="cuda",
        )
        self.temp_softmax_stats_tensor = torch.empty(
            (attn_metadata.num_ctx_tokens, self.num_heads_tp, 2),
            dtype=torch.float,
            device="cuda",
        )

        attn_output = output
        temp_attn_output = q.new_empty(
            (q.size(0), self.num_heads_tp * self.v_head_dim), dtype=q.dtype
        )

        # use fake cached_cu_seq_len for chunked loop
        origin_kv_lens_cuda_runtime = attn_metadata.kv_lens_cuda_runtime
        origin_kv_lens_runtime = attn_metadata.kv_lens_runtime
        origin_ctx_total_kv_len = attn_metadata.host_total_kv_lens[0]

        for loop_idx in range(chunked_loop_num):
            # {b, chunked_unit_size, h, kv_lora_rank + qk_rope_head_dim} zero padded
            # fetch `loop_idx` chunk from kv cache
            temp_cu_chunked_seq_len = attn_metadata.cu_chunked_seq_len[loop_idx]
            total_ctx_chunked_tokens = attn_metadata.host_cu_chunked_seq_len[
                loop_idx, attn_metadata.num_contexts
            ]
            chunked_global_offset = attn_metadata.chunked_global_offset[loop_idx]
            chunked_max_seq_len = attn_metadata.max_chunk_len_per_loop[loop_idx]
            chunked_compressed_kv, chunked_k_pe = trtllm_attention.load_chunked_kv_cache_for_mla(
                metadata=attn_metadata,
                num_ctx_cached_tokens=total_ctx_chunked_tokens,
                cu_chunked_seq_len=temp_cu_chunked_seq_len,
                chunked_global_offset=chunked_global_offset,
                chunked_max_seq_len=chunked_max_seq_len,
                out_dtype=q.dtype,
            )

            # up proj to uncompressed kv
            # [tokens, 2, h, kv_dim], without rope_dim
            chunked_kv = self.kv_b_proj(chunked_compressed_kv)
            chunked_k_nope, chunked_v = chunked_kv.split(
                [
                    self.num_heads_tp * self.qk_nope_head_dim,
                    self.num_heads_tp * self.v_head_dim,
                ],
                -1,
            )

            chunked_k_nope = chunked_k_nope.view(-1, self.num_heads_tp, self.qk_nope_head_dim)
            chunked_k_pe = chunked_k_pe.view(-1, 1, self.qk_rope_head_dim)
            chunked_k = maybe_compiled_cat(
                (chunked_k_nope, chunked_k_pe.expand(-1, self.num_heads_tp, -1)), dim=-1
            )
            chunked_k = chunked_k.view(-1, self.num_heads_tp * self.qk_head_dim)

            # release pytorch activation memory
            chunked_compressed_kv = None
            chunked_k_pe = None
            chunked_kv = None
            chunked_k_nope = None

            # copy chunked_seq_len to replace kv_lens_runtime
            attn_metadata.kv_lens_runtime = attn_metadata.host_chunked_seq_len[loop_idx]
            attn_metadata.kv_lens_cuda_runtime = attn_metadata.chunked_seq_len[loop_idx]
            attn_metadata.host_total_kv_lens[0] = total_ctx_chunked_tokens

            # do not apply mask for attention within loop
            # latent_cache must be None to differentiate from normal context phase,
            # so that we can skip applying RoPE and appending KV cache inside attention op
            temp_attn_output = self.mha.forward(
                q,
                chunked_k,
                chunked_v,
                attn_metadata,
                forward_args=AttentionForwardArgs(
                    attention_input_type=AttentionInputType.context_only,
                    latent_cache=None,
                    out_scale=self.out_scale,
                    attention_mask=PredefinedAttentionMask.FULL,
                    softmax_stats_tensor=self.temp_softmax_stats_tensor,
                    chunked_prefill_buffer_batch_size=attn_metadata.runtime_features.chunked_prefill_buffer_batch_size,
                    output=temp_attn_output,
                ),
            )
            # merge attn result
            temp_merge_op = attn_metadata.merge_op_tensor[loop_idx]
            trtllm_attention.merge_attention_for_mla(
                attn_output,
                temp_attn_output,
                self.softmax_stats_tensor,
                self.temp_softmax_stats_tensor,
                temp_merge_op,
                attn_metadata,
            )

        # deal with the uncached kv
        kv = self.kv_b_proj(compressed_kv)
        _, k_pe = latent_cache.view([-1, self.kv_lora_rank + self.qk_rope_head_dim]).split(
            [self.kv_lora_rank, self.qk_rope_head_dim], -1
        )
        # final round of attention

        k_nope, v = kv.split(
            [
                self.num_heads_tp * self.qk_nope_head_dim,
                self.num_heads_tp * self.v_head_dim,
            ],
            -1,
        )

        k_nope = k_nope.view(-1, self.num_heads_tp, self.qk_nope_head_dim)
        k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim)
        k = maybe_compiled_cat((k_nope, k_pe.expand(-1, self.num_heads_tp, -1)), dim=-1)
        k = k.view(-1, self.num_heads_tp * self.qk_head_dim)

        # copy q_lens to replace kv_lens_runtime
        attn_metadata.kv_lens_runtime = attn_metadata.prompt_lens_cpu_runtime
        attn_metadata.kv_lens_cuda_runtime = attn_metadata.prompt_lens_cuda_runtime
        attn_metadata.host_total_kv_lens[0] = (
            attn_metadata.prompt_lens_cpu_runtime[: attn_metadata.num_contexts].sum().item()
        )

        # latent_cache must be None to differentiate from normal context phase,
        # so that we can skip applying RoPE and appending KV cache inside attention op
        temp_attn_output = self.mha.forward(
            q,
            k,
            v,
            attn_metadata,
            forward_args=AttentionForwardArgs(
                attention_input_type=AttentionInputType.context_only,
                latent_cache=None,
                out_scale=self.out_scale,
                softmax_stats_tensor=self.temp_softmax_stats_tensor,
                chunked_prefill_buffer_batch_size=attn_metadata.runtime_features.chunked_prefill_buffer_batch_size,
                output=temp_attn_output,
            ),
        )
        temp_merge_op = attn_metadata.merge_op_tensor[chunked_loop_num]
        trtllm_attention.merge_attention_for_mla(
            attn_output,
            temp_attn_output,
            self.softmax_stats_tensor,
            self.temp_softmax_stats_tensor,
            temp_merge_op,
            attn_metadata,
        )
        # copy back kv_lens_runtime and kv_lens_cuda_runtime
        attn_metadata.kv_lens_runtime = origin_kv_lens_runtime
        attn_metadata.kv_lens_cuda_runtime = origin_kv_lens_cuda_runtime
        attn_metadata.host_total_kv_lens[0] = origin_ctx_total_kv_len

        return attn_output

    @staticmethod
    @functools.cache
    def cached_warmup_forward_context_with_cached_kv(
        num_heads_tp,
        qk_nope_head_dim,
        qk_rope_head_dim,
        kv_lora_rank,
        v_head_dim,
        dtype,
        device,
    ):
        """Warmup torch.compile for cat operations with different tensor layouts.

        Tensors are marked with torch._dynamo.maybe_mark_dynamic(..., 0) on the
        num_tokens dimension, so for num_tokens != 1 a single warmup run is
        enough and the compiled kernel generalizes across varying num_tokens at
        runtime. num_tokens=1 still triggers recompile (torch.compile specializes
        for it), so it is warmed up separately. Do not use torch.compile with
        dynamic=True here because it completely ignores tensor layout/stride
        information, resulting in significantly degraded performance.
        """

        def warmup(num_tokens):
            chunked_k_nope = k_nope = torch.empty(
                num_tokens,
                num_heads_tp * (qk_nope_head_dim + v_head_dim),
                dtype=dtype,
                device=device,
            )[:, : num_heads_tp * qk_nope_head_dim].view(num_tokens, num_heads_tp, qk_nope_head_dim)
            chunked_k_pe = torch.empty(
                num_tokens, 1, qk_rope_head_dim, dtype=dtype, device=device
            ).expand(-1, num_heads_tp, -1)
            k_pe = torch.empty(
                num_tokens,
                1,
                kv_lora_rank + qk_rope_head_dim,
                dtype=dtype,
                device=device,
            )[:, :, -qk_rope_head_dim:].expand(-1, num_heads_tp, -1)
            torch._dynamo.maybe_mark_dynamic(chunked_k_nope, 0)
            torch._dynamo.maybe_mark_dynamic(chunked_k_pe, 0)
            torch._dynamo.maybe_mark_dynamic(k_pe, 0)
            maybe_compiled_cat((chunked_k_nope, chunked_k_pe), dim=-1)
            maybe_compiled_cat((k_nope, k_pe), dim=-1)

        # With dim 0 (num_tokens) marked dynamic, one warmup suffices for all
        # num_tokens != 1 at runtime.
        warmup(2)

        # num_tokens=1 still triggers recompile; warm it separately.
        warmup(1)

    def forward_context(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        latent_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(attn_metadata, FlashInferAttentionMetadata):
            if (
                attn_metadata.enable_context_mla_with_cached_kv
                and attn_metadata.num_ctx_cached_tokens > 0
            ):
                return self.forward_absorption_context(
                    q,
                    compressed_kv,
                    k_pe,
                    attn_metadata,
                    output,
                    position_ids=position_ids,
                    latent_cache=latent_cache,
                )

        if isinstance(self.mha, TrtllmAttention):
            assert isinstance(attn_metadata, TrtllmAttentionMetadata)
            trtllm_attention = cast(TrtllmAttention, self.mha)
            # Warm up maybe_compiled_cat for both the chunked-prefill path and
            # the cached-kv path; without this, the cached-kv prefill
            # (block-reuse without chunked_prefill) recompiles per shape and
            # may stall inside inductor's compile worker.
            if trtllm_attention.has_cached_kv_for_mla_context_warmup(attn_metadata):
                self.cached_warmup_forward_context_with_cached_kv(
                    self.num_heads_tp,
                    self.qk_nope_head_dim,
                    self.qk_rope_head_dim,
                    self.kv_lora_rank,
                    self.v_head_dim,
                    q.dtype,
                    q.device,
                )
            if (
                trtllm_attention.is_chunked_prefill_for_mla_context(attn_metadata)
                and get_sm_version() >= 100
            ):
                if trtllm_attention.uses_fp4_mla_attention:
                    return self.forward_context_with_chunked_prefill_fp4(
                        q,
                        compressed_kv,
                        latent_cache,
                        attn_metadata,
                        output,
                    )
                return self.forward_context_with_chunked_prefill(
                    q, compressed_kv, latent_cache, attn_metadata, output
                )
            elif trtllm_attention.has_cached_kv_for_mla_context(
                attn_metadata
            ) or trtllm_attention.is_chunked_prefill_for_mla_context(attn_metadata):
                return self.forward_context_with_cached_kv(q, latent_cache, attn_metadata, output)
        return self.forward_context_default(
            q, compressed_kv, k_pe, position_ids, attn_metadata, output, latent_cache
        )

    def _bmm_bf16_out(
        self,
        a: torch.Tensor,
        b_no_transpose: torch.Tensor,
        b_transposed: torch.Tensor,
        output: torch.Tensor,
        ugpu_weight_shards: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> None:
        """BMM with optional CuTe DSL bf16 acceleration on Blackwell/Rubin."""
        if not self.use_cute_dsl_bf16_bmm:
            torch.ops.trtllm.bmm_out(a, b_transposed, output)
            return

        sm_version = get_sm_version()
        if sm_version == 107 and ugpu_weight_shards is not None:
            weight_0, weight_1 = ugpu_weight_shards
            torch.ops.trtllm.cute_dsl_bf16_bmm_ugpu_inplace_rubin(a, weight_0, weight_1, output)
        elif is_sm_100f(sm_version):
            bf16_bmm_op = (
                torch.ops.trtllm.cute_dsl_bf16_bmm_rubin
                if sm_version == 107
                else torch.ops.trtllm.cute_dsl_bf16_bmm_blackwell
            )
            bf16_bmm_op(a, b_no_transpose, output)
        else:
            torch.ops.trtllm.bmm_out(a, b_transposed, output)

    def _has_full_absorption_weights(self) -> bool:
        if getattr(self, "kv_lora_rank", None) != 512 or getattr(self, "v_head_dim", None) != 128:
            return False
        # The Rubin BF16 BMM requires its K dimension to be 16-byte aligned.
        # qk_nope_head_dim is K for the absorbed K projection.
        if getattr(self, "qk_nope_head_dim", 0) % 8 != 0:
            return False

        k_weight = getattr(self, "k_b_proj_trans", None)
        v_weight = getattr(self, "v_b_proj", None)
        if not isinstance(k_weight, torch.Tensor) or not isinstance(v_weight, torch.Tensor):
            return False
        if k_weight.is_meta or v_weight.is_meta:
            return False

        expected_k_shape = (
            self.num_heads_tp,
            self.kv_lora_rank,
            self.qk_nope_head_dim,
        )
        expected_v_shape = (
            self.num_heads_tp_cp,
            self.v_head_dim,
            self.kv_lora_rank,
        )
        return (
            k_weight.dtype == v_weight.dtype
            and tuple(k_weight.shape) == expected_k_shape
            and tuple(v_weight.shape) == expected_v_shape
            and k_weight.numel() == math.prod(expected_k_shape)
            and v_weight.numel() == math.prod(expected_v_shape)
        )

    def _copy_absorption_weight_shards(
        self, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._ugpu_runtime is not None
        num_partitions = self._ugpu_runtime.num_partitions
        assert weight.shape[1] % num_partitions == 0

        shard_n = weight.shape[1] // num_partitions
        shards = []
        for partition_id in range(num_partitions):
            weight_slice = weight[
                :,
                partition_id * shard_n : (partition_id + 1) * shard_n,
                :,
            ]
            with self._ugpu_runtime.partition_weight_context(partition_id):
                shard = torch.empty_like(
                    weight_slice,
                    memory_format=torch.contiguous_format,
                )
                shard.copy_(weight_slice)
            shards.append(shard)
        assert len(shards) == 2
        return shards[0], shards[1]

    def _rebuild_ugpu_bf16_bmm_weight_shards(self) -> None:
        """Rebuild localized BF16 absorption weights from the full parameters."""
        self._ugpu_k_b_proj_trans_shards = None
        self._ugpu_v_b_proj_shards = None

        if getattr(self, "is_deepseek_v4", False) or not self._has_full_absorption_weights():
            return

        ugpu_policy = getattr(self, "_ugpu_policy", None)
        if ugpu_policy is None:
            return
        plan = UgpuExecutionPlanner(ugpu_policy).plan_bf16_bmm(
            dtype=self.k_b_proj_trans.dtype,
            use_cute_dsl_bf16_bmm=getattr(self, "use_cute_dsl_bf16_bmm", False),
        )
        if not plan.enabled:
            return

        if getattr(self, "_ugpu_runtime", None) is None:
            self._ugpu_runtime = UgpuRuntime(plan.num_partitions)

        k_shards = self._copy_absorption_weight_shards(self.k_b_proj_trans)
        v_shards = self._copy_absorption_weight_shards(self.v_b_proj)
        self._ugpu_runtime.prepare_for_capture(plan)
        self._ugpu_k_b_proj_trans_shards = k_shards
        self._ugpu_v_b_proj_shards = v_shards

    def _mla_nvfp4_bmm_o_scale(self) -> Optional[torch.Tensor]:
        """o_proj's static NVFP4 input_scale when bmm_o may write o_proj's NVFP4 input
        (TRTLLM_MLA_NVFP4_EPILOGUE bit 1), else None. Decided once per module."""
        cached = self.__dict__.get("_mla_nvfp4_bmm_o_scale_cache")
        if cached is None:
            o_proj = getattr(self, "o_proj", None)
            quant_method = getattr(o_proj, "quant_method", None)
            input_scale = getattr(o_proj, "input_scale", None)
            v_b_proj = getattr(self, "v_b_proj", None)
            eligible = (
                get_sm_version() == 107
                and IS_CUTLASS_DSL_AVAILABLE
                and self.sparse_attn_hooks is None
                and not self.mapping.has_cp_helix()
                and self.num_heads_tp_cp == self.num_heads_tp
                and self._ugpu_v_b_proj_shards is None
                and isinstance(v_b_proj, torch.Tensor)
                and v_b_proj.dtype == torch.bfloat16
                and v_b_proj.dim() == 3
                and v_b_proj.stride(2) == 1
                and self.v_head_dim % 64 == 0
                and self.kv_lora_rank % 8 == 0
                and type(quant_method) is NVFP4LinearMethod
                and is_static_nvfp4_input_eligible(o_proj)
                and getattr(o_proj, "scaling_vector_size", None) == 16
                and isinstance(input_scale, torch.Tensor)
                and input_scale.dtype == torch.float32
                and input_scale.numel() == 1
                and o_proj.in_features == self.num_heads_tp * self.v_head_dim
            )
            cached = input_scale if eligible else False
            self.__dict__["_mla_nvfp4_bmm_o_scale_cache"] = cached
            logger.info_once(
                f"[TRTLLM_MLA_NVFP4_EPILOGUE] bmm_o -> o_proj NVFP4 input eligible: {eligible} "
                f"(layer {self.layer_idx}, sm {get_sm_version()}, o_proj method "
                f"{type(quant_method).__name__})",
                key=f"mla_nvfp4_bmm_o_eligible_{eligible}",
            )
        return cached if cached is not False else None

    def _mla_nvfp4_output_buffers(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """o_proj's NVFP4 input buffers (packed E2M1 [tokens, hidden / 2], swizzled scales
        [pad_up(tokens, 128) * pad_up(hidden / 16, 4)]: the shapes fp4_quantize returns) when bmm_o writes them
        (TRTLLM_MLA_NVFP4_EPILOGUE bit 1, dense pure-generation batches), else None."""
        if not (MLA_NVFP4_EPILOGUE & 1) or attn_metadata.num_contexts != 0:
            return None
        if isinstance(hidden_states, Fp4QuantizedTensor):
            hidden_states = hidden_states.unquantized_hidden_states
            if hidden_states is None:
                return None
        num_tokens = hidden_states.shape[0]
        if (
            num_tokens < MLA_NVFP4_EPILOGUE_MIN_TOKENS
            or num_tokens != attn_metadata.num_tokens
            or torch.compiler.is_compiling()
            or self._mla_nvfp4_bmm_o_scale() is None
        ):
            return None
        hidden = self.num_heads_tp * self.v_head_dim
        output = hidden_states.new_empty([num_tokens, hidden // 2], dtype=torch.uint8)
        output_sf = hidden_states.new_empty(
            [pad_up(num_tokens, 128) * pad_up(hidden // 16, 4)], dtype=torch.uint8
        )
        return output, output_sf

    def _mla_nvfp4_bmm_o(
        self, attn_out_latent: torch.Tensor, output: torch.Tensor, output_sf: torch.Tensor
    ) -> None:
        """bmm_o (attn_latent^T x W_UV) whose epilogue writes o_proj's NVFP4 input: ``fp4_quantize``'s bytes for
        the BF16 [tokens, heads * v_head_dim] product with o_proj's static input_scale (swizzled scales)."""
        from ..cute_dsl_kernels.rubin.dense_gemm_nvfp4_out import MODE_O, Sm107Nvfp4OutBmm

        num_tokens = attn_out_latent.shape[0]
        input_scale = self._mla_nvfp4_bmm_o_scale()
        hidden = self.num_heads_tp * self.v_head_dim
        if (
            input_scale is None
            or output.dtype != torch.uint8
            or tuple(output.shape) != (num_tokens, hidden // 2)
            or output_sf.dtype != torch.uint8
            or output_sf.numel() < pad_up(num_tokens, 128) * pad_up(hidden // 16, 4)
        ):
            raise RuntimeError(
                "MLA NVFP4 bmm_o output received without an eligible o_proj or with wrong buffers: "
                f"{tuple(output.shape)} {output.dtype}, {output_sf.numel()} scale bytes."
            )
        tactic = Sm107Nvfp4OutBmm.run(
            MODE_O,
            attn_out_latent.transpose(0, 1),
            self.v_b_proj,
            output,
            output_sf,
            input_scale,
            out_stride_m=hidden // 2,
            out_stride_l=self.v_head_dim // 2,
            sf_row_m=1,
            sf_row_l=0,
            sf_col_l=self.v_head_dim // 16,
            sf_cols_padded=pad_up(hidden // 16, 4),
            fp4_rows=num_tokens,
            sf_rows=pad_up(num_tokens, 128),
        )
        logger.info_once(
            f"[TRTLLM_MLA_NVFP4_EPILOGUE] bmm_o writes o_proj's NVFP4 input in its epilogue (CuTe DSL SM107, "
            f"tactic {tactic}); first call {num_tokens} tokens",
            key="mla_nvfp4_bmm_o_active",
        )
        if MLA_NVFP4_EPILOGUE_VERIFY:

            def bmm_o_mismatches() -> Dict[str, torch.Tensor]:
                from .fused_ops.nvfp4_quantize_triton import fp4_quantize

                ref = torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device=output.device)
                torch.ops.trtllm.bmm_out(
                    attn_out_latent.transpose(0, 1),
                    self.v_b_proj.transpose(1, 2),
                    ref.view(num_tokens, self.num_heads_tp, self.v_head_dim).transpose(0, 1),
                )
                ref_fp4, ref_sf = fp4_quantize(ref, input_scale, 16, False, True)
                return {
                    "fp4_bytes": (output != ref_fp4.view(torch.uint8)).sum(),
                    "scale_bytes": (output_sf[: ref_sf.numel()] != ref_sf.view(torch.uint8)).sum(),
                }

            _mla_nvfp4_epilogue_verify("o", num_tokens, bmm_o_mismatches)

    def _mla_nvfp4_bmm_q_prefix_eligible(
        self, num_tokens: int, attn_metadata: AttentionMetadata
    ) -> bool:
        """TRTLLM_MLA_NVFP4_EPILOGUE bit 2 for this generation slice: one token per sequence, at least
        ..._MIN_TOKENS_Q tokens, BF16 W_UK on the FP4 MLA fused RoPE/Q-quant path, SM107."""
        if not (MLA_NVFP4_EPILOGUE & 2) or num_tokens < MLA_NVFP4_EPILOGUE_MIN_TOKENS_Q:
            return False
        if num_tokens != attn_metadata.num_seqs - attn_metadata.num_contexts:
            return False
        eligible = self.__dict__.get("_mla_nvfp4_bmm_q_eligible_cache")
        if eligible is None:
            k_weight = getattr(self, "k_b_proj_trans", None)
            eligible = bool(
                get_sm_version() == 107
                and IS_CUTLASS_DSL_AVAILABLE
                and not self.apply_rotary_emb
                and self.sparse_attn_hooks is None
                and not self.mapping.has_cp_helix()
                and self.num_heads_tp == 128
                and self.kv_lora_rank == 512
                and self.qk_nope_head_dim % 8 == 0
                and self._ugpu_k_b_proj_trans_shards is None
                and isinstance(k_weight, torch.Tensor)
                and k_weight.dtype == torch.bfloat16
                and k_weight.is_contiguous()
                and tuple(k_weight.shape)
                == (self.num_heads_tp, self.kv_lora_rank, self.qk_nope_head_dim)
            )
            self.__dict__["_mla_nvfp4_bmm_q_eligible_cache"] = eligible
            logger.info_once(
                f"[TRTLLM_MLA_NVFP4_EPILOGUE] bmm_q -> FP4 Q prefix eligible: {eligible} "
                f"(layer {self.layer_idx}, heads {self.num_heads_tp}, sm {get_sm_version()})",
                key=f"mla_nvfp4_bmm_q_eligible_{eligible}",
            )
        return eligible

    def _mla_nvfp4_bmm_q_prefix(
        self, attn_metadata: AttentionMetadata, q_nope_t: torch.Tensor, overlap: bool
    ) -> None:
        """bmm_q into the FP4 MLA Q buffers (the packed Q prefix and its scale columns)."""
        from ..attention_backend.fp4_mla import fp4_mla_bmm_q_fp4_prefix

        tactic = fp4_mla_bmm_q_fp4_prefix(
            attn_metadata, q_nope_t, self.k_b_proj_trans, overlap=overlap
        )
        logger.info_once(
            f"[TRTLLM_MLA_NVFP4_EPILOGUE] bmm_q writes the NVFP4 Q prefix in its epilogue (CuTe DSL SM107, "
            f"tactic {tactic}); the fused RoPE/Q/KV launch runs without Q-prefix programs"
            f"{' beside it on the aux stream' if overlap else ''}; first call {q_nope_t.shape[1]} tokens",
            key=f"mla_nvfp4_bmm_q_active_{overlap}",
        )

    def forward_absorption_generation(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        latent_cache: Optional[torch.Tensor] = None,
        output_sf: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        q_nope, q_pe = q.view([-1, self.num_heads_tp, self.qk_head_dim]).split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # fused_q contains 1) the result of the following bmm with shape [num_tokens, num_heads, kv_lora_rank]
        # 2) rope(q_pe) with shape [num_tokens, num_heads, qk_rope_head_dim]. rope is applied inside AttentionOp
        num_seqs = attn_metadata.num_seqs

        cu_q_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=q.device)
        cu_kv_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=q.device)
        fmha_scheduler_counter = torch.empty(1, dtype=torch.uint32, device=q.device)
        use_fp8_mla = getattr(self.mqa, "has_fp8_kv_cache", False) or getattr(
            self.mqa, "has_fp4_kv_cache", False
        )

        mla_bmm1_scale = None
        mla_bmm2_scale = None
        quant_q_buffer = None
        if use_fp8_mla:
            mla_bmm1_scale = torch.empty(2, dtype=torch.float32, device=q.device)
            mla_bmm2_scale = torch.empty(1, dtype=torch.float32, device=q.device)
            quant_q_buffer = torch.empty(
                num_tokens,
                self.num_heads_tp,
                (self.kv_lora_rank + self.qk_rope_head_dim),
                dtype=torch.uint8,
                device=q.device,
            )

        if hasattr(self, "k_b_proj_trans"):
            fused_q = torch.empty(
                [
                    num_tokens,
                    self.num_heads_tp,
                    (self.kv_lora_rank + self.qk_rope_head_dim),
                ],
                dtype=q.dtype,
                device=q.device,
            )

            fp4_mla = isinstance(self.mqa, TrtllmAttention) and self.mqa.uses_fp4_mla_attention
            if fp4_mla and latent_cache is None:
                raise RuntimeError(
                    "FP4 MLA generation requires fused Q quantization, RoPE, "
                    "and cache update on the TRT-LLM backend, which needs a "
                    "latent cache to append into."
                )
            # The absorption weight dtype is deliberately not constrained here.
            # It follows kv_b_proj: an FP8 block-scaled checkpoint yields
            # float8_e4m3fn, anything else the module dtype. Both have a BMM
            # below that writes q_nope into the same fused_q slice, and an
            # unsupported dtype is rejected there. Requiring bfloat16 at this
            # point rejected every FP8-attention Kimi K3 checkpoint outright.
            fuse_fp4_q_quant = fp4_mla
            fp4_rope_kwargs = {"fuse_fp4_q_quant": True} if fuse_fp4_q_quant else {}

            def _mla_gen_rope():
                if self.apply_rotary_emb:
                    # Non-fused backends (Vanilla / FlashInfer) do not fuse RoPE
                    # in the attention kernel. Reuse apply_rope, which rotates
                    # q's q_pe slice in place, then copy rotated k_pe into the
                    # latent cache that the backend appends.
                    assert position_ids is not None
                    assert latent_cache is not None
                    gen_position_ids = position_ids[..., attn_metadata.num_ctx_tokens :]
                    k_pe_rope = self.apply_rope(q, k_pe, gen_position_ids)
                    fused_q[..., self.kv_lora_rank :] = q_pe
                    latent_cache[..., self.kv_lora_rank :] = k_pe_rope
                else:
                    # Fused backend (TRTLLM): RoPE, latent-cache append and the
                    # trtllm-gen scheduler buffers are all produced in-kernel.
                    self.mqa.mla_rope_generation(
                        fused_q,
                        q_pe,
                        latent_cache,
                        attn_metadata,
                        cu_q_seqlens,
                        cu_kv_seqlens,
                        fmha_scheduler_counter,
                        mla_bmm1_scale,
                        mla_bmm2_scale,
                        quant_q_buffer,
                        **fp4_rope_kwargs,
                    )

            rope_stream = self.aux_stream if not use_fp8_mla else None
            # TRTLLM_MLA_NVFP4_EPILOGUE bit 2: bmm_q writes the FP4 MLA Q prefix itself; the fused RoPE/Q/KV
            # launch then writes only the Q tails, K and the latent, and nothing reads fused_q's BF16 prefix
            # (the FP4 MLA decode consumes the prequantized Q only).
            fp4_q_prefix = fuse_fp4_q_quant and self._mla_nvfp4_bmm_q_prefix_eligible(
                num_tokens, attn_metadata
            )
            fp4_q_prefix_overlap = False
            if fp4_q_prefix:
                fp4_rope_kwargs["fp4_q_prefix_external"] = True
                # bit 4: that launch no longer reads bmm_q's output, so it may run beside it
                if (MLA_NVFP4_EPILOGUE & 4) and self.aux_stream is not None:
                    rope_stream = self.aux_stream
                    fp4_q_prefix_overlap = True
            if self.k_b_proj_trans.dtype == torch.bfloat16:
                # [num_heads, num_tokens, self.qk_nope_head_dim]
                q_nope_t = q_nope.transpose(0, 1)
                # [num_heads, num_tokens, self.kv_lora_rank]
                q_nope_out = fused_q[..., : self.kv_lora_rank].transpose(0, 1)

                # [num_heads, num_tokens, self.qk_nope_head_dim] x [num_heads, kv_lora_rank, qk_nope_head_dim]
                # -> [num_heads, num_tokens, kv_lora_rank] -> [num_tokens, num_heads, kv_lora_rank]
                # The output of bmm is written directly into fused_q
                if fp4_q_prefix:
                    bmm_q = functools.partial(
                        self._mla_nvfp4_bmm_q_prefix,
                        attn_metadata,
                        q_nope_t,
                        fp4_q_prefix_overlap,
                    )
                else:
                    bmm_q = functools.partial(
                        self._bmm_bf16_out,
                        q_nope_t,
                        self.k_b_proj_trans,
                        self.k_b_proj_trans.transpose(1, 2),
                        q_nope_out,
                        self._ugpu_k_b_proj_trans_shards,
                    )
                maybe_execute_in_parallel(
                    bmm_q,
                    _mla_gen_rope,
                    self.ln_events[0],
                    self.ln_events[1],
                    rope_stream,
                )
                if fp4_q_prefix and MLA_NVFP4_EPILOGUE_VERIFY:
                    from ..attention_backend.fp4_mla import fp4_mla_verify_q_prefix

                    _mla_nvfp4_epilogue_verify(
                        "q",
                        num_tokens,
                        functools.partial(
                            fp4_mla_verify_q_prefix, attn_metadata, q_nope_t, self.k_b_proj_trans
                        ),
                    )

            elif self.k_b_proj_trans.dtype == torch.float8_e4m3fn:
                # [num_heads, num_tokens, self.kv_lora_rank]
                q_nope_out = fused_q[..., : self.kv_lora_rank].transpose(0, 1)

                maybe_execute_in_parallel(
                    lambda: fp8_block_scaling_bmm_out(
                        q_nope,
                        self.k_b_proj_trans,
                        self.k_b_proj_trans_scale,
                        q_nope_out,
                        self.k_b_proj_trans_dequant,
                        self.use_cute_dsl_blockscaling_bmm,
                    ),
                    _mla_gen_rope,
                    self.ln_events[0],
                    self.ln_events[1],
                    rope_stream,
                )
            else:
                raise NotImplementedError(
                    f"Missing bmm impl for dtype: {self.k_b_proj_trans.dtype}."
                )

            fused_q = fused_q.view(
                [
                    num_tokens,
                    self.num_heads_tp * (self.kv_lora_rank + self.qk_rope_head_dim),
                ]
            )
        else:
            raise RuntimeError("MLA absorption requires k_b_proj_trans")

        attn_out_latent = self._attn_forward_gen(
            self.mqa,
            fused_q,
            None,
            None,
            position_ids,
            attn_metadata,
            attention_input_type=AttentionInputType.generation_only,
            out_scale=self.out_scale,
            latent_cache=latent_cache,  # kvcache and k_pe
            q_pe=q_pe,  # used by `invokeMLARopeGeneration`
            cu_q_seqlens=cu_q_seqlens,  # used by `mlaGeneration`
            cu_kv_seqlens=cu_kv_seqlens,  # used by `mlaGeneration`
            fmha_scheduler_counter=fmha_scheduler_counter,  # used by `mlaGeneration`
            mla_bmm1_scale=mla_bmm1_scale,  # used by `mlaGeneration`
            mla_bmm2_scale=mla_bmm2_scale,  # used by `mlaGeneration`
            quant_q_buffer=quant_q_buffer,  # used by `mlaGeneration`
            **kwargs,
        )
        fused_q = None

        # note: if we do not have CP, then num_heads_tp_cp == num_heads_tp
        assert (
            attn_out_latent.shape[0] == q.shape[0]
            and attn_out_latent.shape[1] == self.num_heads_tp_cp * self.kv_lora_rank
        )

        # [seq, num_heads, kv_lora_rank]
        attn_out_latent = attn_out_latent.view([-1, self.num_heads_tp_cp, self.kv_lora_rank])

        if output_sf is not None:
            # TRTLLM_MLA_NVFP4_EPILOGUE bit 1: output is o_proj's packed NVFP4 input, output_sf its scales.
            self._mla_nvfp4_bmm_o(attn_out_latent, output, output_sf)
            return output

        attn_output = output.view([num_tokens, self.num_heads_tp_cp, self.v_head_dim])

        if self.v_b_proj.dtype == torch.bfloat16:
            # [num_heads, seq, kv_lora_rank] x [num_heads, kv_lora_rank, v_head_dim]
            # -> [num_heads, seq, v_head_dim]
            self._bmm_bf16_out(
                attn_out_latent.transpose(0, 1),
                self.v_b_proj,
                self.v_b_proj.transpose(1, 2),
                attn_output.transpose(0, 1),
                self._ugpu_v_b_proj_shards,
            )
        elif self.v_b_proj.dtype == torch.float8_e4m3fn:
            fp8_block_scaling_bmm_out(
                attn_out_latent,
                self.v_b_proj,
                self.v_b_proj_scale,
                attn_output.transpose(0, 1),
                self.v_b_proj_dequant,
                self.use_cute_dsl_blockscaling_bmm,
            )
        else:
            raise NotImplementedError(f"Missing bmm impl for dtype: {self.v_b_proj.dtype}.")

        return output

    def forward_absorption_context(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        latent_cache: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        q_rope_applied = kwargs.pop("q_rope_applied", False)

        q_nope, q_pe = q.view([-1, self.num_heads_tp, self.qk_head_dim]).split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        if hasattr(self, "k_b_proj_trans"):
            # fused_q contains 1) the result of the following bmm with shape [num_tokens, num_heads, kv_lora_rank]
            # 2) rope(q_pe) with shape [num_tokens, num_heads, qk_rope_head_dim]. rope is applied inside AttentionOp
            fused_q = torch.empty(
                [
                    num_tokens,
                    self.num_heads_tp,
                    (self.kv_lora_rank + self.qk_rope_head_dim),
                ],
                dtype=q.dtype,
                device=q.device,
            )

            if self.k_b_proj_trans.dtype == torch.bfloat16:
                # [num_heads, num_tokens, self.qk_nope_head_dim]
                q_nope_t = q_nope.transpose(0, 1)
                # [num_heads, num_tokens, self.kv_lora_rank]
                q_nope_out = fused_q[..., : self.kv_lora_rank].transpose(0, 1)

                # [num_heads, num_tokens, self.qk_nope_head_dim] x [num_heads, kv_lora_rank, qk_nope_head_dim]
                # -> [num_heads, num_tokens, kv_lora_rank] -> [num_tokens, num_heads, kv_lora_rank]
                # The output of bmm is written directly into fused_q
                self._bmm_bf16_out(
                    q_nope_t,
                    self.k_b_proj_trans,
                    self.k_b_proj_trans.transpose(1, 2),
                    q_nope_out,
                    self._ugpu_k_b_proj_trans_shards,
                )
            elif self.k_b_proj_trans.dtype == torch.float8_e4m3fn:
                # [num_heads, num_tokens, self.kv_lora_rank]
                q_nope_out = fused_q[..., : self.kv_lora_rank].transpose(0, 1)

                fp8_block_scaling_bmm_out(
                    q_nope,
                    self.k_b_proj_trans,
                    self.k_b_proj_trans_scale,
                    q_nope_out,
                    self.k_b_proj_trans_dequant,
                    self.use_cute_dsl_blockscaling_bmm,
                )
            else:
                raise NotImplementedError(
                    f"Missing bmm impl for dtype: {self.k_b_proj_trans.dtype}."
                )

            if self.apply_rotary_emb or q_rope_applied:
                fused_q[..., self.kv_lora_rank :] = q_pe
            fused_q = fused_q.view(
                [
                    num_tokens,
                    self.num_heads_tp * (self.kv_lora_rank + self.qk_rope_head_dim),
                ]
            )
        else:
            raise RuntimeError("MLA absorption requires k_b_proj_trans")

        attn_out_latent = self._attn_forward_gen(
            self.mqa,
            fused_q,
            None,
            None,
            position_ids,
            attn_metadata,
            attention_input_type=AttentionInputType.context_only,
            out_scale=self.out_scale,
            latent_cache=latent_cache,  # kvcache and k_pe
            q_pe=q_pe,  # used by applyMLARopeAndAssignQKVKernelOptContext
            **kwargs,
        )
        fused_q = None

        # note: if we do not have CP, then num_heads_tp_cp == num_heads_tp
        assert (
            attn_out_latent.shape[0] == q.shape[0]
            and attn_out_latent.shape[1] == self.num_heads_tp_cp * self.kv_lora_rank
        )

        # [seq, num_heads, kv_lora_rank]
        attn_out_latent = attn_out_latent.view([-1, self.num_heads_tp_cp, self.kv_lora_rank])

        attn_output = output.view([num_tokens, self.num_heads_tp_cp, self.v_head_dim])

        if self.v_b_proj.dtype == torch.bfloat16:
            # [num_heads, seq, kv_lora_rank] x [num_heads, kv_lora_rank, v_head_dim]
            # -> [num_heads, seq, v_head_dim]
            self._bmm_bf16_out(
                attn_out_latent.transpose(0, 1),
                self.v_b_proj,
                self.v_b_proj.transpose(1, 2),
                attn_output.transpose(0, 1),
                self._ugpu_v_b_proj_shards,
            )
        elif self.v_b_proj.dtype == torch.float8_e4m3fn:
            fp8_block_scaling_bmm_out(
                attn_out_latent,
                self.v_b_proj,
                self.v_b_proj_scale,
                attn_output.transpose(0, 1),
                self.v_b_proj_dequant,
                self.use_cute_dsl_blockscaling_bmm,
            )
        else:
            raise NotImplementedError(f"Missing bmm impl for dtype: {self.v_b_proj.dtype}.")

        return output

    def _forward_custom_op(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attn_output: list[torch.Tensor],
        latent_cache_gen: Optional[torch.Tensor],
    ) -> None:
        """Run the dense or sparse registered custom-op implementation."""
        if self.sparse_attn_hooks is not None and self.sparse_attn_hooks.forward_custom_op(
            self,
            hidden_states,
            position_ids,
            attn_output,
            latent_cache_gen,
        ):
            return

        output = attn_output[0]
        if len(attn_output) != 1:
            raise RuntimeError("MLA custom ops require exactly one output tensor.")

        if isinstance(hidden_states, Fp4QuantizedTensor):
            maybe_bcg_mla_custom_op_inplace(
                hidden_states.unquantized_hidden_states,
                position_ids,
                self.layer_idx_str,
                output,
                latent_cache_gen,
                hidden_states.fp4_tensor,
                hidden_states.scaling_factor,
            )
        else:
            maybe_bcg_mla_custom_op_inplace(
                hidden_states,
                position_ids,
                self.layer_idx_str,
                output,
                latent_cache_gen,
            )

    def _project_output(
        self,
        attn_output: list[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams],
    ) -> torch.Tensor:
        """Apply the MLA output projection."""
        if self.sparse_attn_hooks is not None:
            output = self.sparse_attn_hooks.project_output(
                self,
                attn_output,
                position_ids,
                attn_metadata,
                all_reduce_params,
            )
            if output is not None:
                return output

        return self._project_output_impl(
            attn_output[0], position_ids, attn_metadata, all_reduce_params
        )

    def _project_output_impl(
        self,
        attn_output: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams],
    ) -> torch.Tensor:
        """Apply the default MLA output projection implementation."""
        return _helix_cp_output_projection(
            self.o_proj,
            attn_output,
            attn_metadata,
            all_reduce_params,
            self.mapping,
            self.mapping_o,
            self.layer_idx,
        )

    def forward(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams] = None,
        latent_cache_gen: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = _helix_cp_allgather_input(
            hidden_states, attn_metadata, self.mapping, self.layer_idx
        )

        nvfp4_output = self._mla_nvfp4_output_buffers(hidden_states, attn_metadata)
        if nvfp4_output is not None:
            # TRTLLM_MLA_NVFP4_EPILOGUE bit 1: bmm_o writes o_proj's NVFP4 input; o_proj skips its quantizer.
            output_fp4, output_sf = nvfp4_output
            if self.register_to_config:
                if isinstance(hidden_states, Fp4QuantizedTensor):
                    maybe_bcg_mla_custom_op_inplace_nvfp4_out(
                        hidden_states.unquantized_hidden_states,
                        position_ids,
                        self.layer_idx_str,
                        output_fp4,
                        output_sf,
                        latent_cache_gen,
                        hidden_states.fp4_tensor,
                        hidden_states.scaling_factor,
                    )
                else:
                    maybe_bcg_mla_custom_op_inplace_nvfp4_out(
                        hidden_states,
                        position_ids,
                        self.layer_idx_str,
                        output_fp4,
                        output_sf,
                        latent_cache_gen,
                    )
            else:
                self.forward_impl(
                    position_ids,
                    hidden_states,
                    attn_metadata,
                    attn_output=[output_fp4],
                    latent_cache_gen=latent_cache_gen,
                    output_sf=output_sf,
                )
            attn_output = [Fp4QuantizedTensor(output_fp4, output_sf)]
        elif self.register_to_config:
            output_hidden_states = hidden_states
            if isinstance(hidden_states, Fp4QuantizedTensor):
                assert hidden_states.unquantized_hidden_states is not None, (
                    "MLA.forward received an Fp4QuantizedTensor without a "
                    "unquantized_hidden_states view"
                )
                output_hidden_states = hidden_states.unquantized_hidden_states
            attn_output = [
                torch.ops.trtllm.create_mla_outputs(output_hidden_states, self.layer_idx_str)
            ]
            self._forward_custom_op(
                hidden_states,
                position_ids,
                attn_output,
                latent_cache_gen,
            )
        else:
            attn_output = self._create_outputs(hidden_states, attn_metadata)
            self.forward_impl(
                position_ids,
                hidden_states,
                attn_metadata,
                attn_output=attn_output,
                latent_cache_gen=latent_cache_gen,
            )

        return self._project_output(attn_output, position_ids, attn_metadata, all_reduce_params)

    def resmooth_parameters(self, module_weight, module_weight_scale, recipe=(1, 128, 128)):
        weight, weight_scale = fp8_utils.resmooth_to_fp8_e8m0(module_weight, module_weight_scale)

        transfromed_scale = fp8_utils.transform_sf_into_required_layout(
            weight_scale,
            mn=weight.shape[1],
            k=weight.shape[2],
            recipe=recipe,
            num_groups=weight.shape[0],
            is_sfa=False,
        )

        weight_param = torch.nn.Parameter(weight, requires_grad=False)
        scale_param = torch.nn.Parameter(transfromed_scale, requires_grad=False)

        return weight_param, scale_param

    def transform_weights(self) -> None:
        if self._weights_transformed:
            return
        sparse_attn_hooks = getattr(self, "sparse_attn_hooks", None)
        if sparse_attn_hooks is not None:
            sparse_attn_hooks.transform_weights(self)
        if sparse_attn_hooks is not None and not sparse_attn_hooks.need_absorption:
            self._weights_transformed = True
            return

        has_fp8_block_scales = bool(
            self.kv_b_proj.quant_config
            and self.kv_b_proj.quant_config.quant_mode.has_fp8_block_scales()
        )
        if get_sm_version() == 120 and has_fp8_block_scales:
            self.k_b_proj_trans, self.k_b_proj_trans_scale = self.resmooth_parameters(
                self.k_b_proj_trans, self.k_b_proj_trans_scale, recipe=(1, 128, 128)
            )

            self.v_b_proj, self.v_b_proj_scale = self.resmooth_parameters(
                self.v_b_proj, self.v_b_proj_scale, recipe=(1, 128, 128)
            )
        self._weights_transformed = True

    def cache_derived_state(self) -> None:
        self._kvbproj_head_weights_cache = None
        self._rebuild_ugpu_bf16_bmm_weight_shards()
        self._weights_transformed = True

    def post_load_weights(self) -> None:
        self.transform_weights()
        self.cache_derived_state()
