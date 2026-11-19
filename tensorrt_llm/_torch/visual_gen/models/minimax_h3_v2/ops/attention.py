# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.visual_gen.attention.backends.interface import AttentionTensorLayout
from tensorrt_llm.visual_gen.sparse_attention import SkipSoftmaxAttentionConfig, SolAttentionConfig


@torch.compiler.disable
def _fa4(q, k, v, scale, prefix_lengths):
    from tensorrt_llm._torch.visual_gen.attention.backends.flash_attn4 import (
        _flash_attn_fwd,
        _flash_attn_fwd_import_error,
    )

    if _flash_attn_fwd is None:
        raise ImportError("FlashAttention 4 is unavailable") from _flash_attn_fwd_import_error
    output, *_ = _flash_attn_fwd(
        q,
        k,
        v,
        seqused_k=prefix_lengths,
        softmax_scale=scale,
        causal=False,
        window_size_left=None,
        window_size_right=None,
        learnable_sink=None,
        softcap=0.0,
        pack_gqa=None,
        mask_mod=None,
        block_sparse_tensors=None,
        return_lse=True,
        num_splits=0,
    )
    return output


def _flashinfer(q, k, v, mask, recipe, runner):
    if recipe is None:
        if q.shape[0] == 1 and torch.cuda.get_device_capability(q.device)[0] != 10:
            return runner._run_single_prefill(q, k, v, is_causal=False, key_padding_mask=mask)[0]
        return runner._run_batch_prefill(q, k, v, is_causal=False, key_padding_mask=mask)[0]
    if mask is not None:
        raise ValueError("FlashInfer quantized attention does not support key padding masks.")
    capability = torch.cuda.get_device_capability(q.device)
    if (
        capability in ((10, 0), (10, 3))
        and recipe.qk_dtype in ("mxfp8", "nvfp4")
        and recipe.v_dtype == "fp8"
    ):
        return runner._run_blockscaled_sm10x(q, k, v, qk_mode=recipe.qk_dtype, is_causal=False)[0]
    if capability[0] == 12 and recipe.qk_dtype == recipe.v_dtype == "nvfp4":
        return runner._run_nvfp4_sm12x(q, k, v, is_causal=False)[0]
    raise RuntimeError(f"Unsupported FlashInfer attention recipe for SM{capability}: {recipe}")


def _cute(q, k, v, scale, recipe, sparse_params, timestep):
    from tensorrt_llm._torch.visual_gen.attention.backends.cute_dsl.fmha import (
        _check_cute_runtime_available,
        _quantize_blockscaled_one,
        _quantize_fp8,
        cute_dsl_fmha_fwd,
        cutlass,
    )

    _check_cute_runtime_available()
    dtype = q.dtype
    if dtype not in (torch.float16, torch.bfloat16):
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    lse = torch.empty(q.shape[:3], device=q.device, dtype=torch.float32)
    qsf = ksf = storage_type = channel_scales = None
    vec = 0
    vs = 1.0
    if recipe is not None:
        if recipe.qk_dtype == "fp8":
            q, qs, _ = _quantize_fp8(q)
            k, ks, _ = _quantize_fp8(k)
            scale = scale * qs * ks
        elif recipe.qk_dtype in ("mxfp8", "nvfp4"):
            # Q/K use reciprocal channel scales. Subtracting the key mean adds
            # a row-wise constant to the logits and preserves softmax.
            qmax = q.abs().amax(dim=(0, 1), keepdim=True).float().clamp_min(1e-4)
            kmax = k.abs().amax(dim=(0, 1), keepdim=True).float().clamp_min(1e-4)
            smooth = (qmax.pow(0.7) / kmax.pow(1 - 0.7)).clamp(1e-4, 1e4)
            q = q * smooth.reciprocal().bfloat16()
            k = (k - k.mean(dim=(0, 1), keepdim=True)) * smooth.bfloat16()
            vec = 32 if recipe.qk_dtype == "mxfp8" else 16
            q, qsf, qs = _quantize_blockscaled_one(q, vec)
            k, ksf, ks = _quantize_blockscaled_one(k, vec)
            storage_type = cutlass.Float4E2M1FN if vec == 16 else cutlass.Float8E4M3FN
            scale = scale * qs * ks
        v, vs, channel_scales = _quantize_fp8(
            v, per_head_channel=vec != 0 and recipe.v_block_size == 1
        )
    cute_dsl_fmha_fwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        out,
        sm_scale=scale,
        lse=lse,
        scale_output=vs,
        scale_v_channels=channel_scales,
        sparse_params=sparse_params,
        timestep=timestep,
        qk_sf_vec=vec,
        q_sf=qsf,
        k_sf=ksf,
        qk_cutlass_dtype=storage_type,
    )
    return out.to(dtype)


class DenseAttentionOp(OpWrapper):
    """H3 full self-attention. VANILLA consumes HND, and fused kernels consume NHD."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(
            "packed rows with trailing padding",
            dict(backend="VANILLA", batch=1, seq=5, heads=2, dim=16),
        ),
        Cell(
            "prefix length on the fused kernel",
            dict(backend="FA4", batch=1, seq=5, heads=2, dim=128),
        ),
    )

    def __call__(self, q, k, v, *, backend, scale, mask, recipe, runner, sparse_params, timestep):
        if backend == "VANILLA":
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask[:, None, None, :] if mask is not None else None, scale=scale
            )
        if backend == "FA4":
            lengths = None if mask is None else mask.sum(dim=1).to(torch.int32)
            dtype = q.dtype
            if dtype not in (torch.float16, torch.bfloat16):
                q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
            return _fa4(q, k, v, scale, lengths).to(dtype)
        if backend == "FLASHINFER":
            return _flashinfer(q, k, v, mask, recipe, runner)
        if mask is not None:
            raise NotImplementedError(f"{backend} requires an unpadded H3 sequence")
        if backend == "CUTEDSL":
            return _cute(q, k, v, scale, recipe, sparse_params, timestep)
        if backend == "CUDNN":
            return runner.run_dense(q, k, v, is_causal=False, with_lse=False)[0]
        raise ValueError(f"Unsupported H3 attention backend {backend}")

    def reference(self, q, k, v, *, backend, scale, mask, recipe, runner, sparse_params, timestep):
        if backend != "VANILLA":
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        probs = (q.double() @ k.double().transpose(-2, -1)) * scale
        if mask is not None:
            probs = probs.masked_fill(~mask[:, None, None, :], float("-inf"))
        out = (probs.softmax(-1).nan_to_num() @ v.double()).to(q.dtype)
        return out if backend == "VANILLA" else out.transpose(1, 2)

    def is_valid(self, q, k, v, *, backend, scale, mask, recipe, runner, sparse_params, timestep):
        if backend == "FA4" and mask is not None:
            assert not ((~mask[:, :-1]) & mask[:, 1:]).any(), "FA4 requires trailing padding"


_dense_attention = DenseAttentionOp()


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
            from tensorrt_llm._torch.visual_gen.attention.backends.cudnn import CuDNNSdpaRunner

            self.runner = CuDNNSdpaRunner(
                num_heads=num_heads,
                head_dim=head_dim,
                dtype=config.torch_dtype,
                quant_attention_config=self.recipe,
            )
        elif backend == "FLASHINFER":
            from tensorrt_llm._torch.visual_gen.attention.backends.flashinfer import (
                FlashInferPrefillRunner,
            )

            if config.attention_metadata_state is None:
                raise ValueError("FLASHINFER requires shared attention metadata state.")
            self.runner = FlashInferPrefillRunner(
                num_heads=num_heads,
                head_dim=head_dim,
                dtype=config.torch_dtype,
                quant_attention_config=self.recipe,
                attention_metadata_state=config.attention_metadata_state,
            )
        elif backend == "CUTEDSL" and getattr(sparse_params, "algorithm", None) == "sol_attn":
            from tensorrt_llm._torch.visual_gen.attention.backends.cute_dsl.fmha import (
                _check_cute_runtime_available,
                _get_gpu_arch,
            )
            from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.sol_attn_backend import (
                _run_sol_attn_bthd,
            )

            self.sol = _run_sol_attn_bthd
            self.sol_params = sparse_params
            self.sparse_params = None
            self.sol_dense_layer = layer_idx in sparse_params.dense_layers
            try:
                _check_cute_runtime_available()
                _get_gpu_arch()
                self.cute_dense_available = True
            except (ImportError, RuntimeError, ValueError):
                self.cute_dense_available = False

    @staticmethod
    def support_fused_qkv():
        return False

    @torch.compiler.disable
    def _sol_dense_by_step(self, timestep):
        from tensorrt_llm._torch.attention.backends.sparse.timestep_phase import (
            graph_phase_for_timestep,
        )

        return (
            graph_phase_for_timestep(
                timestep, disabled_until_timestep=self.sol_params.disabled_until_timestep
            )
            == 0
        )

    def forward(self, q, k, v, *, key_padding_mask=None, timestep=None, **kwargs):
        if self.sol is not None:

            def dense(a, b, c):
                if key_padding_mask is None and self.cute_dense_available and a.is_cuda:
                    return _cute(a, b, c, self.scale, None, None, timestep)
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
        return _dense_attention(
            q,
            k,
            v,
            backend=self.backend,
            scale=self.scale,
            mask=key_padding_mask,
            recipe=self.recipe,
            runner=self.runner,
            sparse_params=self.sparse_params,
            timestep=timestep,
        )


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
            params = sparse.to_sparse_params(
                module_name=qualified, pretrained_config=config.pretrained_config
            )
        elif isinstance(sparse, SolAttentionConfig):
            params = sparse.to_sparse_params()
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
