# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer attention backend for visual generation models."""

import hashlib
import math
from typing import Any, ClassVar, Literal

import flashinfer
import torch

from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.visual_gen.args import QuantAttentionConfig

from ....attention.backends.fmha.flashinfer_trtllm_gen import FlashInferTrtllmGenFmha
from ....attention.backends.interface import PredefinedAttentionMask
from ....flashinfer_utils import get_env_enable_pdl
from ...cache import ForwardPlan, PagedKVTables
from .interface import AttentionBackend, AttentionTensorLayout
from .paged_rollout import forward_over_paged_rollout

_WORKSPACE_BYTES = 128 * 1024 * 1024
_BlockScaledQKMode = Literal["mxfp8", "nvfp4"]


class FlashInferAttention(AttentionBackend):
    """Dense FlashInfer attention without an LLM KV cache.

    The backend accepts the VisualGen NHD layout ``[B, S, H, D]``. The FP16/BF16
    path uses FlashInfer's single-request or batched ragged prefill kernel. Block-scaled
    MXFP8 or NVFP4 Q/K dispatches to FlashInfer's SM100/SM103 FMHA with FP8 V. The
    SM120/SM121 path supports dense NVFP4 Q/K/V, according to the validated YAML recipe.

    With a ``PagedRolloutKVCache`` plan as ``kv``, attention runs over the cache on FlashInfer's
    TRTLLM-Gen paged context kernel (SM10x).
    """

    # Per device, the zeroed workspace and multi-CTA counter buffer shared by every layer's paged
    # K/V attention. Kernels on one stream run one after another, and each kernel resets the
    # counters it used. Keep every counter allocation alive for captured graphs; geometric
    # capacities bound their total storage to less than twice the largest allocation. The
    # workspace is separate from the prefill wrappers' workspaces, which hold their plans.
    _trtllm_gen_paged_buffers: ClassVar[
        dict[torch.device, tuple[torch.Tensor, dict[int, torch.Tensor]]]
    ] = {}

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 64,
        num_kv_heads: int | None = None,
        dtype: torch.dtype | None = None,
        quant_attention_config: QuantAttentionConfig | None = None,
        attention_metadata_state: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> None:
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.dtype = dtype
        self.quant_attention_config = quant_attention_config
        self.scale = 1.0 / math.sqrt(head_dim)
        self._preferred_layout = AttentionTensorLayout.NHD
        self._metadata_state = (
            attention_metadata_state if attention_metadata_state is not None else {}
        )
        self._static_global_scale = self._metadata_state.get("_ltx23_static_global_scale") == 1.0

        self._single_prefill = flashinfer.single_prefill_with_kv_cache
        self._batch_prefill_cls = flashinfer.BatchPrefillWithRaggedKVCacheWrapper
        if quant_attention_config is not None and quant_attention_config.qk_dtype not in (
            "mxfp8",
            "nvfp4",
        ):
            raise ValueError(
                "FlashInfer quantized attention supports qk_dtype='mxfp8' or 'nvfp4', "
                f"got {quant_attention_config.qk_dtype!r}."
            )

    def _validate_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> None:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("FlashInfer attention expects 4D NHD Q/K/V tensors.")
        if q.device.type != "cuda":
            raise RuntimeError("FlashInfer attention requires CUDA tensors.")
        if q.device != k.device or q.device != v.device:
            raise ValueError("FlashInfer attention requires Q/K/V on the same device.")
        if q.dtype != k.dtype or q.dtype != v.dtype:
            raise ValueError("FlashInfer attention requires Q/K/V with the same dtype.")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("FlashInfer attention expects FP16 or BF16 Q/K/V tensors.")
        if q.shape[0] != k.shape[0] or k.shape[:2] != v.shape[:2]:
            raise ValueError("FlashInfer attention requires aligned Q/K/V batch dimensions.")
        if q.shape[2:] != (self.num_heads, self.head_dim):
            raise ValueError(
                "Invalid query shape for FlashInfer attention: "
                f"expected [..., {self.num_heads}, {self.head_dim}], got {tuple(q.shape)}."
            )
        expected_kv = (self.num_kv_heads, self.head_dim)
        if k.shape[2:] != expected_kv or v.shape[2:] != expected_kv:
            raise ValueError(
                "Invalid key/value shape for FlashInfer attention: "
                f"expected [..., {expected_kv[0]}, {expected_kv[1]}], "
                f"got K={tuple(k.shape)}, V={tuple(v.shape)}."
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("FlashInfer attention requires num_heads divisible by num_kv_heads.")
        if key_padding_mask is not None and key_padding_mask.shape != (
            q.shape[0],
            k.shape[1],
        ):
            raise ValueError(
                "key_padding_mask must have shape "
                f"{(q.shape[0], k.shape[1])}, got {tuple(key_padding_mask.shape)}."
            )

    def _run_single_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
        key_padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        custom_mask = None
        if key_padding_mask is not None:
            custom_mask = (
                key_padding_mask[0]
                .to(device=q.device, dtype=torch.bool)
                .unsqueeze(0)
                .expand(q.shape[1], -1)
                .contiguous()
            )
        output, lse = self._single_prefill(
            q[0].contiguous(),
            k[0].contiguous(),
            v[0].contiguous(),
            o_dtype=q.dtype,
            custom_mask=custom_mask,
            causal=is_causal,
            sm_scale=self.scale,
            return_lse=True,
        )
        # Single prefill reports log2 LSE; VisualGen exposes natural-log LSE.
        return output.unsqueeze(0), lse.transpose(0, 1).unsqueeze(0) * math.log(2.0)

    @torch.compiler.disable
    def _run_batch_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
        key_padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._static_global_scale:
            return self._run_capture_safe_batch_prefill(
                q, k, v, is_causal=is_causal, key_padding_mask=key_padding_mask
            )
        batch_size, query_length = q.shape[:2]
        key_value_length = k.shape[1]
        stream = torch.cuda.current_stream(q.device)
        stream_key = (q.device, stream.cuda_stream)

        workspace_cache = self._metadata_state.setdefault("flashinfer_workspace_cache", {})
        workspace = workspace_cache.get(stream_key)
        if workspace is None:
            workspace = torch.empty(
                _WORKSPACE_BYTES,
                dtype=torch.uint8,
                device=q.device,
            )
            workspace_cache[stream_key] = workspace

        plan_key = (
            *stream_key,
            batch_size,
            query_length,
            key_value_length,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            q.dtype,
            is_causal,
            key_padding_mask is not None,
        )
        plan_cache = self._metadata_state.setdefault("flashinfer_prefill_plan_cache", {})
        plan_entry = plan_cache.get(plan_key)
        if plan_entry is None:
            plan_entry = {
                "wrapper": self._batch_prefill_cls(workspace, "NHD"),
                "mask_signature": None,
                "mask_reference": None,
                "is_planned": False,
            }
            plan_cache[plan_key] = plan_entry

        mask_signature = None
        if key_padding_mask is not None:
            mask_signature = (
                id(key_padding_mask),
                key_padding_mask.data_ptr(),
                key_padding_mask._version,
            )
        if not plan_entry["is_planned"] or plan_entry["mask_signature"] != mask_signature:
            if torch.cuda.is_current_stream_capturing():
                raise ValueError(
                    "Cannot plan FlashInfer batched prefill while the CUDA stream is "
                    "capturing. Run a warmup forward pass before capture."
                )

            query_indptr = (
                torch.arange(batch_size + 1, dtype=torch.int32, device=q.device) * query_length
            )
            key_value_indptr = (
                torch.arange(batch_size + 1, dtype=torch.int32, device=q.device) * key_value_length
            )
            custom_mask = None
            if key_padding_mask is not None:
                custom_mask = (
                    key_padding_mask.to(device=q.device, dtype=torch.bool)
                    .unsqueeze(1)
                    .expand(-1, query_length, -1)
                    .reshape(-1)
                    .contiguous()
                )

            stream.synchronize()
            plan_entry["wrapper"].plan(
                query_indptr,
                key_value_indptr,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                custom_mask=custom_mask,
                causal=is_causal,
                sm_scale=self.scale,
                q_data_type=q.dtype,
                kv_data_type=k.dtype,
                o_data_type=q.dtype,
            )
            plan_entry["mask_signature"] = mask_signature
            plan_entry["mask_reference"] = key_padding_mask
            plan_entry["is_planned"] = True

        output, lse = plan_entry["wrapper"].run(
            q.contiguous().view(-1, self.num_heads, self.head_dim),
            k.contiguous().view(-1, self.num_kv_heads, self.head_dim),
            v.contiguous().view(-1, self.num_kv_heads, self.head_dim),
            return_lse=True,
        )
        output = output.view(batch_size, query_length, self.num_heads, self.head_dim)
        lse = lse.view(batch_size, query_length, self.num_heads).transpose(1, 2)
        # Batch prefill reports log2 LSE; VisualGen exposes natural-log LSE.
        return output, lse * math.log(2.0)

    def _prefill_mask_digest(self, mask: torch.Tensor | None, *, capturing: bool) -> str | None:
        if mask is None:
            return None
        version = None if torch.is_inference(mask) else mask._version
        identity = (id(mask), mask.data_ptr(), version)
        identities = self._metadata_state.setdefault("flashinfer_mask_identities", {})
        if capturing:
            if identity not in identities:
                raise RuntimeError("FlashInfer padding mask must be warmed before CUDA capture.")
            return identities[identity][1]
        digest = hashlib.sha256(mask.detach().to("cpu", torch.bool).numpy().tobytes()).hexdigest()
        # Retain the tensor so its identity cannot be recycled while a plan uses it.
        identities[identity] = (mask, digest)
        return digest

    def _run_capture_safe_batch_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
        key_padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reuse immutable mask-content plans across warmup and capture streams.

        The component's shared metadata state requires serialized forwards.
        Planning and mask readback occur only outside capture; a changed mask
        owns a new wrapper rather than mutating a plan retained by an old graph.
        """
        capturing = torch.cuda.is_current_stream_capturing()
        mask_digest = self._prefill_mask_digest(key_padding_mask, capturing=capturing)
        batch_size, query_length = q.shape[:2]
        key_value_length = k.shape[1]
        plan_key = (
            q.device,
            batch_size,
            query_length,
            key_value_length,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            q.dtype,
            is_causal,
            mask_digest,
        )
        plans = self._metadata_state.setdefault("flashinfer_immutable_prefill_plans", {})
        entry = plans.get(plan_key)
        if entry is None:
            if capturing:
                raise RuntimeError("FlashInfer batched prefill must be warmed before CUDA capture.")
            workspaces = self._metadata_state.setdefault("flashinfer_immutable_workspaces", {})
            workspace = workspaces.get(q.device)
            if workspace is None:
                workspace = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=q.device)
                workspaces[q.device] = workspace
            wrapper = self._batch_prefill_cls(workspace, "NHD")
            query_indptr = (
                torch.arange(batch_size + 1, dtype=torch.int32, device=q.device) * query_length
            )
            key_value_indptr = (
                torch.arange(batch_size + 1, dtype=torch.int32, device=q.device) * key_value_length
            )
            custom_mask = None
            if key_padding_mask is not None:
                custom_mask = (
                    key_padding_mask.to(device=q.device, dtype=torch.bool)
                    .unsqueeze(1)
                    .expand(-1, query_length, -1)
                    .reshape(-1)
                    .clone()
                )
            torch.cuda.current_stream(q.device).synchronize()
            wrapper.plan(
                query_indptr,
                key_value_indptr,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                custom_mask=custom_mask,
                causal=is_causal,
                sm_scale=self.scale,
                q_data_type=q.dtype,
                kv_data_type=k.dtype,
                o_data_type=q.dtype,
            )
            entry = (wrapper, query_indptr, key_value_indptr, custom_mask)
            plans[plan_key] = entry
        output, lse = entry[0].run(
            q.contiguous().view(-1, self.num_heads, self.head_dim),
            k.contiguous().view(-1, self.num_kv_heads, self.head_dim),
            v.contiguous().view(-1, self.num_kv_heads, self.head_dim),
            return_lse=True,
        )
        return (
            output.view(batch_size, query_length, self.num_heads, self.head_dim),
            lse.view(batch_size, query_length, self.num_heads).transpose(1, 2) * math.log(2.0),
        )

    @staticmethod
    def _quantize_qk_static_global(
        tensor: torch.Tensor, mode: _BlockScaledQKMode
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode NHD Q/K with global one and native, input-dependent block scales."""
        from flashinfer.quantization import fp4_quantize, mxfp8_quantize

        batch, sequence, heads, dimension = tensor.shape
        padded_sequence = ((sequence + 127) // 128) * 128
        data = tensor.transpose(1, 2).contiguous()
        if padded_sequence != sequence:
            data = torch.nn.functional.pad(data, (0, 0, 0, padded_sequence - sequence))
        matrix = data.reshape(batch * heads * padded_sequence, dimension)
        if mode == "mxfp8":
            encoded, factors = mxfp8_quantize(
                matrix, is_sf_swizzled_layout=True, alignment=32, backend="cute-dsl"
            )
            encoded = encoded.view(batch, heads, padded_sequence, dimension)[:, :, :sequence]
            factors = factors.view(torch.float8_e8m0fnu)
        elif mode == "nvfp4":
            # fp4_quantize takes the inverse global dequantization scale. Both
            # encode and FMHA decode use one; only native block factors vary.
            inverse_global = torch.ones(1, device=tensor.device, dtype=torch.float32)
            encoded, factors = fp4_quantize(
                matrix,
                inverse_global,
                sf_vec_size=16,
                is_sf_swizzled_layout=True,
                backend="cute-dsl",
            )
            encoded = encoded.view(batch, heads, padded_sequence, dimension // 2)[:, :, :sequence]
            factors = factors.view(torch.float8_e4m3fn)
        else:
            raise ValueError(f"Unsupported static-global Q/K mode: {mode!r}.")
        encoded = encoded.transpose(1, 2).contiguous()
        if mode == "nvfp4":
            # Copy packed bytes before viewing Float4, whose deterministic fill is unsupported.
            encoded = encoded.view(torch.float4_e2m1fn_x2)
        return encoded, factors

    @staticmethod
    def _quantize_fp8_per_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fp8_dtype = torch.float8_e4m3fn
        fp8_info = torch.finfo(fp8_dtype)
        scale = (tensor.abs().amax().float() / fp8_info.max).clamp(min=1e-12)
        quantized = (tensor / scale).clamp(min=fp8_info.min, max=fp8_info.max)
        return quantized.to(fp8_dtype).contiguous(), scale

    def _run_blockscaled_sm10x(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        qk_mode: _BlockScaledQKMode,
        is_causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.head_dim != 128:
            raise ValueError(
                "FlashInfer SM100/SM103 block-scaled attention requires head_dim=128, "
                f"got {self.head_dim}."
            )
        from flashinfer.attention.cute_dsl.fmha_blockscaled import cute_dsl_fmha_blockscaled_prefill
        from flashinfer.cute_dsl.attention.fmha.quantize import quantize_blockscaled_qk

        if self._static_global_scale:
            q_store, q_sf = self._quantize_qk_static_global(q, qk_mode)
            k_store, k_sf = self._quantize_qk_static_global(k, qk_mode)
            v_store = v.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
            q_scale = k_scale = v_scale = 1.0
        else:
            q_store, k_store, q_sf, k_sf, q_scale, k_scale = quantize_blockscaled_qk(q, k, qk_mode)
            v_store, v_scale = self._quantize_fp8_per_tensor(v)
        output = torch.empty_like(q)
        lse = torch.empty(
            (q.shape[0], q.shape[1], self.num_heads),
            dtype=torch.float32,
            device=q.device,
        )
        cute_dsl_fmha_blockscaled_prefill(
            q_store,
            k_store,
            q_sf,
            k_sf,
            v_store,
            output,
            qk_mode=qk_mode,
            is_causal=is_causal,
            sm_scale=self.scale,
            lse=lse,
            scale_q=q_scale,
            scale_k=k_scale,
            scale_v=v_scale,
            scale_o=1.0,
        )
        # CuTe DSL reports log2 LSE; VisualGen exposes natural-log LSE.
        return output, lse.transpose(1, 2).contiguous() * math.log(2.0)

    def _run_nvfp4_sm12x(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError(
                "FlashInfer SM120/SM121 NVFP4 attention supports dense self-attention only."
            )
        if self.num_heads != self.num_kv_heads or self.head_dim not in (64, 128):
            raise ValueError(
                "FlashInfer SM120/SM121 NVFP4 attention requires equal Q/K/V head counts "
                "and head_dim=64 or 128."
            )
        sequence_length = q.shape[1]
        # FlashInfer 0.6.16 pads internally but cannot mask the padded K/V tokens.
        if sequence_length % 128 != 0:
            raise ValueError(
                "FlashInfer SM120/SM121 NVFP4 attention requires sequence length "
                f"to be a multiple of 128, got {sequence_length}."
            )
        query = q.transpose(1, 2).contiguous()
        key = k.transpose(1, 2).contiguous()
        value = v.transpose(1, 2).contiguous()
        quantized_qkv = flashinfer.nvfp4_attention_sm120_quantize_qkv(
            query,
            key,
            value,
            per_block_mean=False,
        )
        output, lse = flashinfer.nvfp4_attention_sm120_fwd(
            *quantized_qkv,
            sm_scale=self.scale,
            causal=is_causal,
            per_block_mean=False,
            out_dtype=q.dtype,
        )
        output = output[:, :, :sequence_length].transpose(1, 2).contiguous()
        return output, lse[:, :, :sequence_length]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        key_padding_mask: torch.Tensor | None = None,
        kv: ForwardPlan | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Run attention.

        Args:
            q: Query tensor ``[B, S_q, H, D]``.
            k: Key tensor ``[B, S_kv, H_kv, D]``.
            v: Value tensor ``[B, S_kv, H_kv, D]``.
            attention_mask: ``CAUSAL`` or ``FULL``.
            key_padding_mask: Optional ``[B, S_kv]`` mask of the keys to attend.
            kv: A ``ForwardPlan`` from a ``PagedRolloutKVCache``. When given,
                ``k``/``v`` are the new tokens only: they are written at
                ``staging_offset`` and attention runs over everything cached before them
                plus themselves, full within a causal block and causal across causal
                blocks. The first ``kv.num_tokens`` rows are real; rows past them are
                padding, neither written to the cache nor attended, and their output rows
                are zero.

        Returns:
            Output tensor ``[B, S_q, H, D]``.
        """
        if kv is not None:
            if key_padding_mask is not None:
                raise NotImplementedError("K/V cache attention is full attention over the cache.")
            return forward_over_paged_rollout(self, self.layer_idx, q, k, v, kv, attention_mask)
        output, _ = self.forward_with_lse(
            q,
            k,
            v,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
        )
        return output

    def forward_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        key_padding_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return attention output and FP32 LSE in ``[B, H, S]`` layout."""
        self._validate_inputs(q, k, v, key_padding_mask)
        is_causal = attention_mask == PredefinedAttentionMask.CAUSAL
        if is_causal and q.shape[1] != k.shape[1]:
            raise ValueError("FlashInfer causal attention requires equal Q and K/V lengths.")
        if is_causal and key_padding_mask is not None:
            raise ValueError("FlashInfer attention does not combine causal and padding masks.")

        quant_config = self.quant_attention_config
        if quant_config is None:
            # FlashInfer's single-prefill FA2/FA3 heuristics regress on SM10x.
            if q.shape[0] == 1 and torch.cuda.get_device_capability(q.device)[0] != 10:
                return self._run_single_prefill(
                    q,
                    k,
                    v,
                    is_causal=is_causal,
                    key_padding_mask=key_padding_mask,
                )
            return self._run_batch_prefill(
                q,
                k,
                v,
                is_causal=is_causal,
                key_padding_mask=key_padding_mask,
            )
        if key_padding_mask is not None:
            raise ValueError("FlashInfer quantized attention does not support key padding masks.")

        capability = torch.cuda.get_device_capability(q.device)
        qk_dtype = quant_config.qk_dtype
        if (
            capability in ((10, 0), (10, 3))
            and qk_dtype in ("mxfp8", "nvfp4")
            and quant_config.v_dtype == "fp8"
        ):
            return self._run_blockscaled_sm10x(
                q,
                k,
                v,
                qk_mode=qk_dtype,
                is_causal=is_causal,
            )
        if capability[0] == 12 and qk_dtype == "nvfp4" and quant_config.v_dtype == "nvfp4":
            return self._run_nvfp4_sm12x(q, k, v, is_causal=is_causal)
        raise RuntimeError(
            "Unsupported FlashInfer quantized attention recipe for "
            f"SM{capability[0]}{capability[1]}: qk_dtype={qk_dtype!r}, "
            f"v_dtype={quant_config.v_dtype!r}. Use qk_dtype='mxfp8' or 'nvfp4' with "
            "v_dtype='fp8' on SM100/SM103, or qk_dtype='nvfp4' with "
            "v_dtype='nvfp4' on SM120/SM121."
        )

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        return self._preferred_layout

    @classmethod
    def support_lse(cls) -> bool:
        return True

    def support_kv_cache(self) -> bool:
        """Whether ``forward`` accepts a ``kv`` plan: on SM10x, where FlashInfer has the
        TRTLLM-Gen paged context kernel, and without quantized attention."""
        return self.quant_attention_config is None and torch.cuda.get_device_capability()[0] == 10

    def support_kv_cache_page_size(
        self, tokens_per_page: int, dtype: torch.dtype | None = None
    ) -> bool:
        """The TRTLLM-Gen paged context kernels take pages of 16, 32 or 64 tokens and bf16 or
        fp16 K/V, and have no kernels for head dims 72, 80 and 96."""
        return (
            self.support_kv_cache()
            and tokens_per_page in FlashInferTrtllmGenFmha.SUPPORTED_TOKENS_PER_BLOCK
            and self.head_dim not in FlashInferTrtllmGenFmha.UNSUPPORTED_HEAD_SIZES_CONTEXT
            and dtype in (None, torch.bfloat16, torch.float16)
        )

    @classmethod
    def _trtllm_gen_buffers(
        cls, q: torch.Tensor, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The workspace and counter buffer for paged K/V attention of ``q``
        (``[tokens, heads, head_dim]``) over ``batch_size`` sequences. They are allocated, or the
        counters grown, on first need, which must not happen during CUDA graph capture: run the
        forward eagerly once before capturing it."""
        from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes

        device = q.device
        num_tokens, num_heads, head_dim = q.shape
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        counter_bytes = get_trtllm_gen_multi_ctas_kv_counter_bytes(batch_size, num_heads, num_sms)
        buffers = cls._trtllm_gen_paged_buffers.get(device)
        counters = (
            None
            if buffers is None
            else next(
                (buf for size, buf in sorted(buffers[1].items()) if size >= counter_bytes), None
            )
        )
        if counters is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Paged K/V attention buffers first needed during CUDA graph capture; run "
                    "the forward eagerly once before capturing."
                )
            if buffers is None:
                # The kernel's scratch space: the part of its context workspace that the TRTLLM
                # attention op reserves for this kernel.
                workspace_bytes = thop.get_trtllm_gen_context_workspace_layout(
                    q.dtype, batch_size, num_tokens, num_heads, head_dim, 0, True, False
                )["trtllm_gen_workspace_size"]
                workspace = torch.zeros(workspace_bytes, dtype=torch.uint8, device=device)
                buffers = cls._trtllm_gen_paged_buffers[device] = (workspace, {})
            capacity = 1 << (counter_bytes - 1).bit_length()
            counters = torch.zeros(capacity, dtype=torch.uint8, device=device)
            buffers[1][capacity] = counters
        return buffers[0], counters

    def forward_paged_kv(
        self, q: torch.Tensor, kv_pool: torch.Tensor, tables: PagedKVTables
    ) -> torch.Tensor:
        """See ``AttentionBackend.forward_paged_kv``."""
        if not self.support_kv_cache():
            raise NotImplementedError(
                "K/V cache attention needs an SM10x GPU and does not combine with quantized "
                "attention."
            )
        batch, seq_len, num_heads, head_dim = q.shape
        page_size = kv_pool.shape[3]
        q = q.reshape(batch * seq_len, num_heads, head_dim).contiguous()
        out = torch.empty_like(q)
        workspace, counters = self._trtllm_gen_buffers(q, batch)
        # kv_pool is [num_pages, 2, H_kv, page_size, D], with K at index 0 and V at index 1 of
        # dim 1. The kernel reads it as one HND tensor, and one page id addresses both K and V.
        flashinfer.prefill.trtllm_batch_context_with_kv_cache(
            query=q,
            kv_cache=kv_pool,
            workspace_buffer=workspace,
            block_tables=tables.page_table,
            seq_lens=tables.seq_lens_kv,
            max_q_len=seq_len,
            # An upper bound fixed by the cache spec, so a captured CUDA graph stays valid.
            max_kv_len=tables.page_table.shape[1] * page_size,
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            batch_size=batch,
            cum_seq_lens_q=tables.cu_seqlens_q,
            cum_seq_lens_kv=tables.cu_seqlens_kv,
            out=out,
            kv_layout="HND",
            enable_pdl=get_env_enable_pdl(),
            uses_shared_paged_kv_idx=True,
            causal=False,
            multi_ctas_kv_counter_buffer=counters,
        )
        return out.view(batch, seq_len, num_heads, head_dim)
