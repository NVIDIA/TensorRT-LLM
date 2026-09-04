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

"""VisualGen TRTLLM-first attention backends for Video Sparse Attention."""

from typing import Optional

import torch
import torch.nn.functional as F

from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.visual_gen.args import QuantAttentionConfig

from .....attention_backend.fmha.prims_ts_block_sparse import PrimsTSBlockSparseFmha
from .....attention_backend.interface import PredefinedAttentionMask
from ...cute_dsl import CuTeDSLAttention
from ...trtllm import SparseForwardInputs, TrtllmAttention
from .metadata import VSA_BLOCK_SIZE, VSAMetadata
from .predictor import VSAForwardInputs, VSAPredictor, vsa_post_process

_vsa_import_error = None
try:
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.video_sparse_attention import (
        block_sparse_attn_from_indices_cute,
        is_cute_supported,
    )
except (ImportError, OSError) as error:
    block_sparse_attn_from_indices_cute = None
    is_cute_supported = None
    _vsa_import_error = error


VSA_KERNEL_MAX_CUBES: int = 4 * 1024


def _normalize_qkv_inputs(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize separate BSHD or Ulysses-packed BSH3HD inputs."""

    if k is not None and v is not None:
        return q, k, v
    if k is not None or v is not None:
        raise ValueError("VSA requires complete separate Q/K/V or one packed QKV tensor.")
    if q.ndim != 5 or q.shape[2] != 3:
        raise ValueError("VSA packed QKV must have shape [B, S, 3, H, D].")
    return q.unbind(dim=2)


def _get_unsupported_primts_reason(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    metadata: VSAMetadata,
) -> str | None:
    if q.shape != k.shape or q.shape != v.shape:
        return "VSA PrimTS requires matching MHA Q/K/V shapes"
    if q.device.type != "cuda":
        return f"VSA PrimTS requires CUDA tensors, got {q.device}"
    if q.dtype not in (torch.float16, torch.bfloat16):
        return f"VSA PrimTS requires FP16 or BF16 tensors, got {q.dtype}"
    batch_size, seq_len, num_heads, head_dim = map(int, q.shape)
    if min(batch_size, seq_len, num_heads, metadata.num_cubes) <= 0:
        return "VSA PrimTS requires positive batch, sequence, head, and cube extents"
    if head_dim != 128:
        return f"VSA PrimTS requires head_dim=128, got {head_dim}"
    if metadata.padded_seq_length != metadata.num_cubes * VSA_BLOCK_SIZE:
        return "VSA tiled sequence length must match its 64-token cube count"
    if batch_size > 65535 or num_heads > 65535:
        return "VSA PrimTS batch and head dimensions must fit the CUDA grid"
    return None


class VSATrtllmAttention(TrtllmAttention):
    """TRTLLM VSA backend using the generic block-sparse forward lifecycle."""

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 64,
        num_kv_heads: Optional[int] = None,
        quant_config: Optional[QuantConfig] = None,
        dtype: Optional[torch.dtype] = None,
        max_batch_size: int = 16,
        max_seq_len: int = 4096,
        quant_attention_config: Optional[QuantAttentionConfig] = None,
        attention_metadata_state: Optional[dict] = None,
    ) -> None:
        num_kv_heads = num_kv_heads or num_heads
        super().__init__(
            layer_idx=layer_idx,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            quant_config=quant_config,
            dtype=dtype,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            quant_attention_config=quant_attention_config,
            attention_metadata_state=attention_metadata_state,
            sparse_params=None,
        )
        assert attention_metadata_state is not None
        predictor_cache = attention_metadata_state.setdefault("sparse_predictors", {})
        predictor_key = ("vsa", num_heads, num_kv_heads)
        predictor = predictor_cache.get(predictor_key)
        if predictor is None:
            predictor = VSAPredictor(
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
            )
            predictor_cache[predictor_key] = predictor
        elif not isinstance(predictor, VSAPredictor):
            raise TypeError("model-scoped VSA predictor cache contains an invalid value")
        self.predictor = predictor

    def block_sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        *,
        batch_size: int,
        seq_len: int,
        seq_len_kv: int,
        attention_mask: PredefinedAttentionMask,
        forward_kwargs: dict[str, object],
    ) -> VSAForwardInputs:
        """Build effective QKV and block inputs for the normal TRTLLM forward."""

        q, k, v = _normalize_qkv_inputs(q, k, v)
        metadata = self.predictor.get_metadata()
        use_primts = any(
            isinstance(fmha, PrimsTSBlockSparseFmha) for fmha in self._fmha_manager.fmha_libs
        )
        unsupported_reason = _get_unsupported_primts_reason(q, k, v, metadata)
        if self.quant_attention_config is not None:
            unsupported_reason = "VSA PrimTS does not support quant_attention_config"
        if not use_primts:
            logger.warning_once(
                "TRTLLM VSA cannot use PrimTS block-sparse attention because the "
                "prims_ts_block_sparse FMHA library is unavailable; using the compact "
                "dense TRTLLM fine stage.",
                key="trtllm_vsa_primts_unavailable",
            )
        elif unsupported_reason is not None:
            logger.warning_once(
                "TRTLLM VSA cannot use PrimTS block-sparse attention: "
                f"{unsupported_reason}; using the compact dense TRTLLM fine stage.",
                key=("trtllm_vsa_primts_unsupported_envelope", unsupported_reason),
            )
        use_sparse_fine = use_primts and unsupported_reason is None

        remaining_kwargs = dict(forward_kwargs)
        gate_compress = remaining_kwargs.pop("gate_compress", None)
        gate_fine = remaining_kwargs.pop("gate_fine", None)
        return self.predictor.predict(
            q,
            k,
            v,
            batch_size=batch_size,
            seq_len=seq_len,
            seq_len_kv=seq_len_kv,
            attention_mask=attention_mask,
            gate_compress=gate_compress,
            gate_fine=gate_fine,
            use_sparse_fine=use_sparse_fine,
            produce_block_sparse_inputs=use_sparse_fine,
            forward_kwargs=remaining_kwargs,
            metadata=metadata,
        )

    def sparse_post_process(
        self,
        output: torch.Tensor,
        sparse_inputs: SparseForwardInputs,
    ) -> torch.Tensor:
        """Combine fine/coarse outputs and restore the compact TRTLLM layout."""

        if not isinstance(sparse_inputs, VSAForwardInputs):
            raise TypeError("VSA sparse post-processing requires VSAForwardInputs")
        combined = vsa_post_process(output, sparse_inputs)
        return combined.reshape(combined.shape[0], combined.shape[1], -1)

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return True


class VSACuTeDSLAttention(CuTeDSLAttention):
    """CuTe DSL VSA backend reusing TRTLLM's predictor and post-processing."""

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 128,
        num_kv_heads: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            layer_idx=layer_idx,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            dtype=dtype,
            **kwargs,
        )
        self.predictor = VSAPredictor(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        *,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        **kwargs,
    ) -> torch.Tensor:
        q, k, v = _normalize_qkv_inputs(q, k, v)
        gate_compress = kwargs.pop("gate_compress", None)
        gate_fine = kwargs.pop("gate_fine", None)
        expected_extents = {
            "batch_size": int(q.shape[0]),
            "seq_len": int(q.shape[1]),
            "seq_len_kv": int(k.shape[1]),
        }
        for name, expected in expected_extents.items():
            actual = kwargs.pop(name, expected)
            if not isinstance(actual, int) or isinstance(actual, bool) or actual != expected:
                raise ValueError(f"VSA {name}={actual!r} does not match Q/K/V extent {expected}")
        kwargs.pop("timestep", None)
        if kwargs:
            unexpected_names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected CuTeDSL VSA forward keyword arguments: {unexpected_names}")

        metadata = self.predictor.get_metadata()
        # The CuTe kernel's fixed launch topology is bounded by the number of
        # VSA cubes; larger shapes retain identical VSA math via dense SDPA.
        use_cute = (
            _vsa_import_error is None
            and is_cute_supported is not None
            and is_cute_supported(q)
            and q.dtype == k.dtype == v.dtype
            and metadata.num_cubes <= VSA_KERNEL_MAX_CUBES
        )
        inputs = self.predictor.predict(
            q,
            k,
            v,
            batch_size=int(q.shape[0]),
            seq_len=int(q.shape[1]),
            seq_len_kv=int(k.shape[1]),
            attention_mask=attention_mask,
            gate_compress=gate_compress,
            gate_fine=gate_fine,
            use_sparse_fine=use_cute,
            produce_block_sparse_inputs=False,
            forward_kwargs={},
            metadata=metadata,
        )
        if use_cute:
            fine_output = self._execute_sparse_fine(inputs)
        else:
            fine_output = F.scaled_dot_product_attention(
                inputs.q.transpose(1, 2),
                inputs.k.transpose(1, 2),
                inputs.v.transpose(1, 2),
            ).transpose(1, 2)
        return vsa_post_process(fine_output, inputs)

    def _execute_sparse_fine(self, inputs: VSAForwardInputs) -> torch.Tensor:
        """Execute only the CuTe-specific VSA fine kernel."""

        q_hnd = inputs.q.transpose(1, 2).contiguous()
        k_hnd = inputs.k.transpose(1, 2).contiguous()
        v_hnd = inputs.v.transpose(1, 2).contiguous()
        q2k_num = torch.full(
            (inputs.batch_size, q_hnd.shape[1], inputs.num_cubes),
            inputs.cur_topk,
            dtype=torch.int32,
            device=inputs.q.device,
        )
        output_hnd, _lse = block_sparse_attn_from_indices_cute(
            q_hnd,
            k_hnd,
            v_hnd,
            q2k_idx=inputs.topk_indices.contiguous(),
            q2k_num=q2k_num,
            variable_block_sizes=inputs.variable_block_sizes.to(torch.int32),
        )
        return output_hnd.transpose(1, 2)

    def forward_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("CuTe DSL VSA does not support LSE output.")

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return True

    @classmethod
    def support_lse(cls) -> bool:
        return False


__all__ = [
    "VSACuTeDSLAttention",
    "VSATrtllmAttention",
    "VSA_KERNEL_MAX_CUBES",
]
