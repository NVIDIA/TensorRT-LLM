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
"""The state of one attention layer that the FMHA libraries and the native attention op read."""

from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantMode

from .interface import MLAParams, PositionalEmbeddingParams, RopeParams
from .sparse.params import SparseParams
from .sparse.skip_softmax import SkipSoftmaxParams

if TYPE_CHECKING:
    from .fmha.gqa_decode import GqaDecodeSetup
    from .trtllm import TrtllmAttentionMetadata

_SKIP_CORRECTION_SUPPORTED_SMS = frozenset((100, 103))


def _resolve_skip_correction_threshold(
    threshold: float, sm_version: int, *, is_mla: bool = True
) -> float:
    if not is_mla:
        return 0.0
    threshold = float(threshold)
    if threshold <= 0.0:
        return 0.0
    if sm_version in _SKIP_CORRECTION_SUPPORTED_SMS:
        return threshold
    logger.warning_once(
        f"Skip-correction is supported only on SM100 and SM103; disabling it on SM{sm_version}.",
        key="skip_correction_unsupported_sm",
    )
    return 0.0


def _resolve_uses_spcompress(sparse_params: Optional[SparseParams], sm_version: int) -> bool:
    uses_spcompress = bool(
        isinstance(sparse_params, SkipSoftmaxParams) and sparse_params.uses_spcompress
    )
    if not uses_spcompress:
        return False
    if sm_version == 107:
        return True
    logger.warning_once(
        f"spcompress is supported only on SM107; disabling it on SM{sm_version}.",
        key="uses_spcompress_unsupported_sm",
    )
    return False


class AttentionLayer:
    """Layer state that the FMHA libraries and the native attention op read.

    ``TrtllmAttention`` is one; a model that drives an ``FmhaManager`` itself
    constructs one directly.
    """

    # FMHA library classes this layer tries before the registry's libraries.
    prepended_fmha_lib_classes: tuple = ()
    # Head dimension per K/V stage of FlashInferTrtllmGenFmha's FP8 paged causal context attention, run through
    # trtllm::fmha_paged_context_fp8; 0 keeps FlashInfer's kernel.
    context_fmha_head_dim_per_stage_kv: int = 0
    # In a context-only step, FlashInferTrtllmGenFmha computes only each request's last output row (the others are
    # zero); every row's K / V is still written.
    context_last_rows_only: bool = False

    # Layer state the native attention op reads through FmhaParams.attn. The
    # generator (scripts/generate_fmha_params.py) collects these annotations;
    # attributes assigned from an annotated __init__ parameter need none.
    layer_idx: int
    num_heads: int
    head_dim: int
    num_kv_heads: int
    is_mla_enable: bool
    v_head_dim: int
    rope_params: RopeParams
    q_scaling: float
    predicted_tokens_per_seq: int
    skip_correction_threshold: float
    uses_spcompress: bool
    q_lora_rank: Optional[int]
    kv_lora_rank: Optional[int]
    qk_nope_head_dim: Optional[int]
    qk_rope_head_dim: Optional[int]
    rope_append: Optional[bool]
    rotary_inv_freq: Optional[torch.Tensor]
    rotary_cos_sin: Optional[torch.Tensor]
    position_embedding_type: PositionEmbeddingType
    skip_softmax_stat: torch.Tensor
    quant_mode: QuantMode

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        quant_config: Optional[QuantConfig] = None,
        q_scaling: Optional[float] = None,
        pos_embd_params: Optional[PositionalEmbeddingParams] = None,
        mla_params: Optional[MLAParams] = None,
        attention_chunk_size: Optional[int] = None,
        sparse_params: Optional[SparseParams] = None,
        kv_cache_dtype: str = "auto",
        skip_correction_threshold: float = 0.0,
        gqa_decode: Optional["GqaDecodeSetup"] = None,
        context_fmha_head_dim_per_stage_kv: int = 0,
        context_last_rows_only: bool = False,
        local_layer_idx: Optional[int] = None,
    ) -> None:
        """
        Args:
            local_layer_idx (int): The layer's row in the KV-cache pool mapping. None
                resolves it from the KV cache manager on first use.

        See ``TrtllmAttention`` for the other arguments.
        """
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.sparse_params = sparse_params
        self.kv_cache_dtype = kv_cache_dtype
        self.use_fp8_ds_mla: bool = kv_cache_dtype == "fp8_ds_mla"
        self.is_mla_enable = mla_params is not None
        self.mla_params = mla_params or MLAParams()
        self.v_head_dim = self.mla_params.v_head_dim if self.is_mla_enable else head_dim

        rope_params = None
        if pos_embd_params is not None:
            rope_params = pos_embd_params.rope
        self.rope_params = rope_params or RopeParams()

        self.q_scaling = q_scaling or 1.0
        self.predicted_tokens_per_seq = self.mla_params.predicted_tokens_per_seq
        self.attention_chunk_size = attention_chunk_size
        self.skip_correction_threshold = _resolve_skip_correction_threshold(
            skip_correction_threshold, get_sm_version(), is_mla=self.is_mla_enable
        )
        self.uses_spcompress = _resolve_uses_spcompress(sparse_params, get_sm_version())

        if self.is_mla_enable:
            self.q_lora_rank = self.mla_params.q_lora_rank
            self.kv_lora_rank = self.mla_params.kv_lora_rank
            self.qk_nope_head_dim = self.mla_params.qk_nope_head_dim
            self.qk_rope_head_dim = self.mla_params.qk_rope_head_dim
            self.rope_append = self.mla_params.rope_append
        else:
            self.q_lora_rank = None
            self.kv_lora_rank = None
            self.qk_nope_head_dim = None
            self.qk_rope_head_dim = None
            self.rope_append = None

        self.rotary_inv_freq, self.rotary_cos_sin = self.rope_params.create_rope_const_params()
        self.position_embedding_type = (
            pos_embd_params.type
            if pos_embd_params is not None
            else PositionEmbeddingType.learned_absolute
        )
        self.skip_softmax_stat = torch.zeros(2, dtype=torch.uint32, device="cuda")

        # Layer-level fp8 KV-cache scales. Stay at 1.0 (no-op) for the
        # PyTorch backend, which never overrides them. They guarantee the
        # kernel always receives a valid pointer, since several non-MLA
        # XQA kernels (cpp/kernels/xqa/mha.cu, mha_sm90.cu) deref
        # `kvCacheScale[0]` whenever `isKVCacheQuantized` is true and
        # do not check for nullptr. `attention/attention.py` only assigns
        # `forward_args.kv_scale_*` for fp4 KV cache, so without this
        # fallback the kernel takes nullptr on fp8-KV models → illegal
        # memory access.
        self.kv_cache_scaling_factor = torch.ones(1, dtype=torch.float32, device="cuda")
        self.kv_scale_quant_orig = self.kv_cache_scaling_factor
        self.kv_scale_orig_quant = 1.0 / self.kv_cache_scaling_factor

        self.local_layer_idx: Optional[int] = local_layer_idx
        self.gqa_decode = gqa_decode
        if context_fmha_head_dim_per_stage_kv < 0 or (
            context_fmha_head_dim_per_stage_kv > 0
            and head_dim % context_fmha_head_dim_per_stage_kv != 0
        ):
            raise ValueError(
                f"context_fmha_head_dim_per_stage_kv="
                f"{context_fmha_head_dim_per_stage_kv} must be 0 or divide "
                f"head_dim={head_dim}."
            )
        self.context_fmha_head_dim_per_stage_kv = context_fmha_head_dim_per_stage_kv
        self.context_last_rows_only = context_last_rows_only
        self.set_layer_quant_config(quant_config)

    def set_layer_quant_config(self, quant_config: Optional[QuantConfig]) -> None:
        """Set the quantization state the libraries read from ``quant_config``."""
        self.quant_config = quant_config or QuantConfig()
        self.quant_mode = self.quant_config.layer_quant_mode
        quant_mode = self.quant_mode
        self.has_fp8_kv_cache = quant_mode.has_fp8_kv_cache()
        self.has_fp4_kv_cache = quant_mode.has_fp4_kv_cache()
        self.has_fp8_qdq = quant_mode.has_fp8_qdq()
        self.has_fp8_block_wise = quant_mode.has_fp8_block_scales()
        self.has_fp8_rowwise = quant_mode.has_fp8_rowwise()
        self.has_nvfp4 = quant_mode.has_nvfp4()
        self.has_w4a8_nvfp4_fp8 = quant_mode.has_w4a8_nvfp4_fp8()

    @property
    def uses_dsa_sparse_mla(self) -> bool:
        return getattr(self.sparse_params, "algorithm", None) in ("deepseek_v4", "dsa")

    @property
    def uses_fp4_mla_attention(self) -> bool:
        """Whether this layer executes dense FP4 MLA, not sparse NVFP4 storage."""
        return self.is_mla_enable and self.has_fp4_kv_cache and self.sparse_params is None

    def get_local_layer_idx(self, metadata: "TrtllmAttentionMetadata") -> int:
        if self.local_layer_idx is not None:
            return self.local_layer_idx
        if metadata.kv_cache_manager is None:
            # Uncached: recomputed each call until a cache manager appears.
            return self.layer_idx
        self.local_layer_idx = metadata.kv_cache_manager.layer_offsets[self.layer_idx]
        return self.local_layer_idx

    def out_head_size(self, is_gen_only: bool) -> int:
        """Per-head width of the attention output for this phase.

        Single source of truth for the output buffer `create_output` allocates and
        the view a phased FMHA library takes over it; the two must not drift. MLA
        generation writes the latent (plus the rope part when it is not appended in
        place), while MLA context writes the already-projected V head.
        """
        if not self.is_mla_enable:
            return self.head_dim
        if not is_gen_only:
            return self.v_head_dim
        return self.kv_lora_rank if self.rope_append else self.kv_lora_rank + self.qk_rope_head_dim

    def _ensure_rope_table_size(self, required_max_positions: int) -> None:
        floats_per_position = self.rope_params.dim * (2 if self.rope_params.duplicate_data else 1)
        table_max_positions = (
            self.rotary_cos_sin.numel() // floats_per_position
            if self.rotary_cos_sin is not None
            else 0
        )
        self.rope_params.max_positions = max(
            self.rope_params.max_positions, table_max_positions, required_max_positions
        )
        if required_max_positions > table_max_positions:
            self.rotary_inv_freq, self.rotary_cos_sin = self.rope_params.create_rope_const_params()

    @property
    def rope_dim(self) -> int:
        return self.rope_params.dim

    @property
    def rope_base(self) -> float:
        return self.rope_params.theta

    @property
    def rope_scale_type(self) -> RotaryScalingType:
        return self.rope_params.scale_type

    @property
    def rope_scale(self) -> float:
        return self.rope_params.scale

    @property
    def rope_short_m_scale(self) -> float:
        return self.rope_params.short_m_scale

    @property
    def rope_long_m_scale(self) -> float:
        return self.rope_params.long_m_scale

    @property
    def rope_max_positions(self) -> int:
        return self.rope_params.max_positions

    @property
    def rope_original_max_positions(self) -> int:
        return self.rope_params.original_max_positions
