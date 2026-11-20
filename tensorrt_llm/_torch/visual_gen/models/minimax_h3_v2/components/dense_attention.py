# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import struct

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.attention.backends.interface import AttentionTensorLayout
from tensorrt_llm._torch.visual_gen.cuda_graph_runner import resolved_extra_key
from tensorrt_llm.visual_gen.sparse_attention import SkipSoftmaxAttentionConfig, SolAttentionConfig

from .skip_softmax_threshold import (
    H3SkipSoftmax,
    dense_prefix,
    resolve_skip_softmax_threshold_scale_factor,
)


def _cute_output(q, k, v, scale, *, recipe=None, threshold=None):
    from ....catalog.attention.cute_fmha import cute_fmha

    dtype = q.dtype
    if dtype not in (torch.float16, torch.bfloat16):
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    out, _ = cute_fmha(
        q,
        k,
        v,
        sm_scale=scale,
        qk_dtype=recipe.qk_dtype if recipe else None,
        v_block_size=recipe.v_block_size if recipe else None,
        skip_softmax_threshold_scale_factor=threshold,
    )
    return out.to(dtype)


class H3DenseAttention:
    def __init__(self, config, backend, num_heads, head_dim, layer_idx, sparse_params):
        self.backend = backend
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.recipe = config.attention.quant_attention_config
        self.preferred_layout = (
            AttentionTensorLayout.HND if backend == "VANILLA" else AttentionTensorLayout.NHD
        )
        self.runner = None
        self.sparse_params = sparse_params
        self.sol = None
        if backend == "CUDNN":
            from .cudnn_runner import CuDNNSdpaRunner

            if config.attention_metadata_state is None:
                config.attention_metadata_state = {}
            self.runner = CuDNNSdpaRunner(
                num_heads=num_heads,
                head_dim=head_dim,
                dtype=config.torch_dtype,
                quant_attention_config=self.recipe,
                attention_metadata_state=config.attention_metadata_state,
            )
        elif backend == "FLASHINFER":
            from ....catalog.attention.flashinfer_prefill import FlashInferPrefillRunner

            if config.attention_metadata_state is None:
                raise ValueError("FLASHINFER requires shared attention metadata state.")
            self.runner = FlashInferPrefillRunner(
                num_heads=num_heads,
                head_dim=head_dim,
                plan_cache=config.attention_metadata_state,
            )
        elif backend == "CUTEDSL" and getattr(sparse_params, "algorithm", None) == "sol_attn":
            from ....catalog.attention.cute_fmha import is_cute_fmha_supported
            from ..kernels.sol_attention import run_sol_attn_bthd

            self.sol = run_sol_attn_bthd
            self.sol_params = sparse_params
            self.sparse_params = None
            self.sol_dense_layer = layer_idx in (sparse_params.dense_layers or ())
            self.cute_dense_available = is_cute_fmha_supported()

    @staticmethod
    def support_fused_qkv():
        return False

    @torch.compiler.disable
    def _sol_dense_by_step(self, timestep):
        phase = resolved_extra_key("sparse_attn_phase")
        if phase is not None:
            return phase == 0
        return dense_prefix(timestep, self.sol_params.disabled_until_timestep)

    def forward(self, q, k, v, *, key_padding_mask=None, timestep=None, **kwargs):
        if self.sol is not None:

            def dense(a, b, c):
                if key_padding_mask is None and self.cute_dense_available and a.is_cuda:
                    return _cute_output(a, b, c, self.scale)
                mask = None if key_padding_mask is None else key_padding_mask[:, None, None, :]
                return F.scaled_dot_product_attention(
                    a.transpose(1, 2), b.transpose(1, 2), c.transpose(1, 2), attn_mask=mask
                ).transpose(1, 2)

            if (
                key_padding_mask is not None
                or self.sol_dense_layer
                or (
                    self.sol_params.disabled_until_timestep is not None
                    and self._sol_dense_by_step(timestep)
                )
            ):
                return dense(q, k, v)
            return self.sol(
                q,
                k,
                v,
                tau=self.sol_params.tau,
                thresh_type=self.sol_params.thresh_type,
                dense_fn=dense,
            )
        if self.backend == "VANILLA":
            mask = key_padding_mask[:, None, None, :] if key_padding_mask is not None else None
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)
        if self.backend == "FA4":
            from ....catalog.attention.flash_attention_4 import flash_attention_4

            lengths = (
                key_padding_mask.sum(dim=1).to(torch.int32)
                if key_padding_mask is not None
                else None
            )
            dtype = q.dtype
            if dtype not in (torch.float16, torch.bfloat16):
                q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
            output, _ = flash_attention_4(q, k, v, softmax_scale=self.scale, seqused_k=lengths)
            return output.to(dtype)
        if self.backend == "FLASHINFER":
            from ....catalog.attention.flashinfer_prefill import flashinfer_prefill

            output, _ = flashinfer_prefill(
                q,
                k,
                v,
                runner=self.runner,
                qk_dtype=self.recipe.qk_dtype if self.recipe is not None else None,
                v_dtype=self.recipe.v_dtype if self.recipe is not None else None,
                key_padding_mask=key_padding_mask,
            )
            return output
        if key_padding_mask is not None:
            raise NotImplementedError(f"{self.backend} requires an unpadded H3 sequence")
        if self.backend == "CUTEDSL":
            threshold = resolve_skip_softmax_threshold_scale_factor(
                None, self.sparse_params, timestep
            )
            return _cute_output(q, k, v, self.scale, recipe=self.recipe, threshold=threshold)
        if self.backend == "CUDNN":
            return self.runner.run_dense(q, k, v)
        raise ValueError(f"Unsupported H3 attention backend {self.backend}")


def create_h3_attention(config, *, num_heads, head_dim, layer_idx, module_name):
    backend = config.attention.backend
    sparse = config.attention.sparse_attention_config
    params = None
    if backend in ("TRTLLM", "CUTEDSL"):
        if isinstance(sparse, SkipSoftmaxAttentionConfig):
            prefix = config.component_name
            qualified = (
                f"{prefix}.{module_name}"
                if prefix and not module_name.startswith(prefix + ".")
                else module_name
            )
            params = H3SkipSoftmax.from_config(sparse, qualified, config.pretrained_config)
        elif isinstance(sparse, SolAttentionConfig):
            try:
                tau = struct.unpack("=f", struct.pack("=f", sparse.tau))[0]
            except (OverflowError, ValueError, struct.error) as error:
                raise ValueError("SOL tau must be representable as float32") from error
            if not math.isfinite(tau):
                raise ValueError("SOL tau must be finite")
            params = sparse.model_copy(
                update={"tau": tau, "dense_layers": list(sparse.dense_layers or ())}
            )
    if backend == "TRTLLM":
        from .trtllm_attention import H3TrtllmAttention

        return H3TrtllmAttention(
            layer_idx=layer_idx,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            quant_config=config.quant_config,
            dtype=config.torch_dtype,
            attention_config=config.attention,
            attention_metadata_state=config.attention_metadata_state,
            sparse_params=params,
        )
    return H3DenseAttention(config, backend, num_heads, head_dim, layer_idx, params)
