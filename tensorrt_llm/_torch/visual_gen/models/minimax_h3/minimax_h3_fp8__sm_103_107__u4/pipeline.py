# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The static-FP8 H3 pipeline with packed QKV exchange on four GPUs."""

import torch

from tensorrt_llm._torch.visual_gen.attention.attention import AttentionTensorLayout
from tensorrt_llm._torch.visual_gen.attention.backends.cudnn import _a3_fmha_dense_fp8, CuDNNAttention
from tensorrt_llm._torch.visual_gen.attention.backends.parallel import UlyssesAttention
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent

from ..pipeline_minimax_h3 import MiniMaxH3Pipeline
from ..transformer_minimax_h3 import (
    MiniMaxH3Attention,
    MiniMaxH3Transformer3DModel,
)
from .block_ops import Fp8UlyssesMlpMixin, initialize_block_ops
from .output_exchange import (
    exchange_bf16_output,
    exchange_fp8_output,
    exchange_prequantized_fp8_output,
    static_fp8_output_scale,
)
from .qkv_exchange import pack_qkv, qk_norm_rope_pack_qkv
from .vae_opt import (
    configure_u4_video_vae_post_load,
    configure_u4_video_vae_pre_load,
    install_audio_vae_cuda_graph,
)

_HYBRID_CUDNN_INITIAL_STEPS = 11
_HYBRID_DIRECT_FP8_OUT_START_STEP = 20
_HYBRID_SKIP_CORR_T1_START_STEP = 25


class Fp8UlyssesAttention(MiniMaxH3Attention):
    _fp8_output_exchange_enabled: bool = True
    _bf16_output_unpack_enabled: bool = True
    _fused_qk_pack_enabled: bool = True
    _hybrid_cudnn_initial_steps: int = 11
    _hybrid_direct_fp8_out_start: int = 20
    _hybrid_t1_start: int = 25

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fp8_out_scale_cache_key = None
        self._fp8_out_scale_valid = False
        self._fp8_output_exchange_calls = 0
        self._bf16_output_exchange_calls = 0
        self._fused_qk_pack_calls = 0
        self._hybrid_gen_backend = None
        self._hybrid_gen_t1_backend = None
        self._hybrid_symm_amax = None
        self._hybrid_direct_fp8_supported = False
        self._hybrid_step_state = None
        self._hybrid_cudnn_steps = self._hybrid_cudnn_initial_steps
        self._hybrid_direct_fp8_steps = self._hybrid_direct_fp8_out_start
        self._hybrid_t1_steps = self._hybrid_t1_start
        self._hybrid_cudnn_fmha_calls = 0
        self._hybrid_gen_fmha_calls = 0
        self._hybrid_direct_fp8_out_calls = 0
        self._hybrid_t1_calls = 0

    def _validated_fp8_output_scale(self) -> torch.Tensor | None:
        if not self._fp8_output_exchange_enabled:
            return None
        if getattr(self.attn, "world_size", 1) != 4:
            return None
        inner = getattr(self.attn, "inner_backend", None)
        if getattr(inner, "preferred_layout", None) != AttentionTensorLayout.NHD:
            return None
        out_linear = self.to_out[0]
        scale = static_fp8_output_scale(out_linear)
        if scale is None:
            return None
        try:
            scale_version = scale._version
        except RuntimeError:
            scale_version = None
        key = (id(out_linear), scale.data_ptr(), scale_version)
        if key != self._fp8_out_scale_cache_key:
            self._fp8_out_scale_valid = bool(
                torch.isfinite(scale).all() and (scale > 0).all()
            )
            self._fp8_out_scale_cache_key = key
        return scale if self._fp8_out_scale_valid else None

    def _can_use_bf16_output_unpack(self) -> bool:
        if not self._bf16_output_unpack_enabled or getattr(self.attn, "world_size", 1) != 4:
            return False
        inner = getattr(self.attn, "inner_backend", None)
        return getattr(inner, "preferred_layout", None) == AttentionTensorLayout.NHD

    def _exchange_packed_output(self, send, descales, rows, *, backend=None, **kwargs):
        attn_backend = self.attn if backend is None else backend
        out_scale = self._validated_fp8_output_scale()
        if out_scale is not None:
            raw_out = attn_backend.forward_packed_fp8(
                send, fp8_descales=descales, return_pre_a2a=True, **kwargs
            )
            if (
                raw_out.ndim == 4
                and raw_out.shape == (1, 4 * rows, 14, 128)
                and raw_out.dtype == torch.bfloat16
                and raw_out.is_cuda
            ):
                self._fp8_output_exchange_calls += 1
                return exchange_fp8_output(raw_out, out_scale, attn_backend.process_group)
            self._bf16_output_exchange_calls += 1
            return attn_backend._output_a2a(raw_out, 1, 4 * rows).flatten(2)
        if self._can_use_bf16_output_unpack():
            raw_out = attn_backend.forward_packed_fp8(
                send, fp8_descales=descales, return_pre_a2a=True, **kwargs
            )
            if (
                raw_out.ndim == 4
                and raw_out.shape == (1, 4 * rows, 14, 128)
                and raw_out.dtype == torch.bfloat16
                and raw_out.is_cuda
            ):
                self._bf16_output_exchange_calls += 1
                return exchange_bf16_output(raw_out, attn_backend.process_group)
            self._bf16_output_exchange_calls += 1
            return attn_backend._output_a2a(raw_out, 1, 4 * rows).flatten(2)
        self._bf16_output_exchange_calls += 1
        return attn_backend.forward_packed_fp8(send, fp8_descales=descales, **kwargs).flatten(2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb=None,
        key_padding_mask=None,
        timestep=None,
        valid_seq_len=None,
    ) -> torch.Tensor:
        if (
            self._fused_qk_pack_enabled
            and not self.use_async_ulysses
            and self.to_gate_compress is None
            and rotary_emb is not None
            and hidden_states.ndim == 3
            and hidden_states.shape[0] == 1
            and hidden_states.shape[1] > 0
            and self.local_num_attention_heads == 56
            and self.local_num_key_value_heads == 56
            and self.head_dim == 128
            and self._use_fused_qk_norm_rope(hidden_states, rotary_emb)
        ):
            qkv = self.qkv_proj(hidden_states)
            hidden_states = self._fused_packed_forward(
                qkv,
                rotary_emb,
                key_padding_mask=key_padding_mask,
                discard_padded_queries=True,
                valid_seq_len=valid_seq_len,
                timestep=timestep,
            )
            return self.to_out[0](hidden_states)
        return super().forward(
            hidden_states,
            rotary_emb=rotary_emb,
            key_padding_mask=key_padding_mask,
            timestep=timestep,
            valid_seq_len=valid_seq_len,
        )

    def _direct_fp8_packed_exchange(self, send, descales, out_scale, rows, backend, skip_t: float, **kwargs):
        group = backend.process_group
        recv = torch.empty_like(send)
        torch.distributed.all_to_all_single(
            recv.view(torch.uint8).view(-1), send.view(torch.uint8).view(-1), group=group
        )
        qkv_full = recv.view(1, 4 * rows, 3, 14, 128)
        q, k, v = qkv_full[:, :, 0], qkv_full[:, :, 1], qkv_full[:, :, 2]
        scale_out = (descales[2:3] / out_scale.to(torch.float32)).contiguous()
        scaled_descales = torch.stack([descales[0], descales[1], scale_out[0]])
        sm_scale = 128.0**-0.5
        mask = kwargs.get("key_padding_mask")
        if mask is not None:
            valid_len = backend.inner_backend._padding_prefix_lengths(mask)[0]
            raw_fp8 = torch.zeros((1, 4 * rows, 14, 128), dtype=torch.float8_e4m3fn, device=send.device)
            if valid_len > 0:
                _a3_fmha_dense_fp8(
                    q[:, :valid_len],
                    k[:, :valid_len],
                    v[:, :valid_len],
                    scaled_descales,
                    sm_scale,
                    torch.float8_e4m3fn,
                    out=raw_fp8[:, :valid_len],
                    variant="trtllm-gen-fp16sm",
                    skip_corr_threshold=skip_t,
                    strided=True,
                )
        else:
            raw_fp8 = _a3_fmha_dense_fp8(
                q,
                k,
                v,
                scaled_descales,
                sm_scale,
                torch.float8_e4m3fn,
                variant="trtllm-gen-fp16sm",
                skip_corr_threshold=skip_t,
                strided=True,
            )
        self._fp8_output_exchange_calls += 1
        self._hybrid_direct_fp8_out_calls += 1
        return exchange_prequantized_fp8_output(raw_fp8, group)

    @torch.compiler.disable
    def _fused_packed_forward(self, qkv, rotary_emb, **kwargs):
        rows = qkv.shape[1]
        if (
            qkv.ndim != 3
            or qkv.shape != (1, rows, 3 * 56 * 128)
            or qkv.dtype != torch.bfloat16
            or not qkv.is_cuda
            or not qkv.is_contiguous()
        ):
            qkv = qkv.contiguous()
        rank = self.attn._ulysses_rank
        mask = kwargs.get("key_padding_mask")
        total = (
            4 * rows if mask is None else self.attn.inner_backend._padding_prefix_lengths(mask)[0]
        )
        valid_rows = max(0, min(rows, total - rank * rows))
        cos, sin = rotary_emb
        step = self._hybrid_step_state["step"] if self._hybrid_step_state is not None else -1
        use_gen = self._hybrid_gen_backend is not None and step >= self._hybrid_cudnn_steps
        pack_kwargs = {}
        if use_gen:
            pack_kwargs["gen_fp16sm"] = True
        if self._hybrid_symm_amax is not None:
            pack_kwargs["symm_amax"] = self._hybrid_symm_amax
        send, descales = qk_norm_rope_pack_qkv(
            qkv,
            self.norm_q.weight,
            self.norm_k.weight,
            cos,
            sin,
            float(self.norm_q.variance_epsilon),
            valid_rows,
            self.attn.process_group,
            **pack_kwargs,
        )
        self._fused_qk_pack_calls += 1
        if use_gen:
            self._hybrid_gen_fmha_calls += 1
            use_t1 = step >= self._hybrid_t1_steps
            if use_t1:
                self._hybrid_t1_calls += 1
            if self._hybrid_direct_fp8_supported and step >= self._hybrid_direct_fp8_steps:
                out_scale = self._validated_fp8_output_scale()
                if out_scale is not None:
                    return self._direct_fp8_packed_exchange(
                        send,
                        descales,
                        out_scale,
                        rows,
                        self._hybrid_gen_backend,
                        1.0 if use_t1 else 0.0,
                        **kwargs,
                    )
            backend = (
                self._hybrid_gen_t1_backend
                if use_t1 and self._hybrid_gen_t1_backend is not None
                else self._hybrid_gen_backend
            )
            return self._exchange_packed_output(send, descales, rows, backend=backend, **kwargs)
        self._hybrid_cudnn_fmha_calls += 1
        return self._exchange_packed_output(send, descales, rows, **kwargs)

    def _attn_impl(self, q, k, v, **kwargs):
        if (
            q.ndim != 3
            or q.shape[0] != 1
            or q.shape[1] == 0
            or q.shape[2] != 56 * 128
            or not all(
                t.shape == q.shape and t.dtype == torch.bfloat16 and t.stride(-1) == 1
                for t in (q, k, v)
            )
            or (
                kwargs.get("key_padding_mask") is not None
                and not kwargs.get("discard_padded_queries")
            )
        ):
            return super()._attn_impl(q, k, v, **kwargs)
        return self._packed_forward(q, k, v, **kwargs)

    @torch.compiler.disable
    def _packed_forward(self, q, k, v, **kwargs):
        rows = q.shape[1]
        rank = self.attn._ulysses_rank
        mask = kwargs.get("key_padding_mask")
        total = (
            4 * rows if mask is None else self.attn.inner_backend._padding_prefix_lengths(mask)[0]
        )
        valid_rows = max(0, min(rows, total - rank * rows))
        q, k, v = (t.view(1, rows, 56, 128) for t in (q, k, v))
        send, descales = pack_qkv(q, k, v, valid_rows, self.attn.process_group)
        return self._exchange_packed_output(send, descales, rows, **kwargs)


class Fp8UlyssesBlock(Fp8UlyssesMlpMixin):
    attention_cls = Fp8UlyssesAttention


class Fp8UlyssesTransformer(MiniMaxH3Transformer3DModel):
    block_cls = Fp8UlyssesBlock
    replicate_token_refiner = True


def _initialize_sm107_hybrid_fmha(
    transformer,
    *,
    cudnn_initial_steps: int = _HYBRID_CUDNN_INITIAL_STEPS,
    direct_fp8_out_start: int = _HYBRID_DIRECT_FP8_OUT_START_STEP,
    skip_corr_t1_start: int = _HYBRID_SKIP_CORR_T1_START_STEP,
) -> dict[str, int] | None:
    """Bind SM107 selective cuDNN (early steps) -> trtllm-gen-fp16sm (late steps) routing."""
    device = transformer.proj_out.weight.device
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (10, 7):
        return None
    attentions = [block.attn for block in transformer.transformer_blocks]
    if not attentions or not all(isinstance(a, Fp8UlyssesAttention) for a in attentions):
        return None
    stock = attentions[0].attn
    if not isinstance(stock, UlyssesAttention) or stock.world_size != 4:
        return None
    group = stock.process_group
    if any(
        not isinstance(a.attn, UlyssesAttention)
        or a.attn.process_group is not group
        or a.attn.head_dim != 128
        or a.attn.sharded_num_heads != 14
        or a.attn.sharded_num_kv_heads != 14
        or not isinstance(a.attn.inner_backend, CuDNNAttention)
        or a.attn.inner_backend.quant_dtype != "fp8"
        or a.attn.inner_backend.dtype != torch.bfloat16
        or a.use_async_ulysses
        for a in attentions
    ):
        return None
    inner = CuDNNAttention(
        num_heads=14,
        num_kv_heads=14,
        head_dim=128,
        dtype=torch.bfloat16,
        quant_attention_config=stock.inner_backend.quant_attention_config,
        fp8_dense_fmha="trtllm-gen-fp16sm",
        fp8_skip_corr_threshold=0.0,
        fp8_strided_views=True,
    )
    inner_t1 = CuDNNAttention(
        num_heads=14,
        num_kv_heads=14,
        head_dim=128,
        dtype=torch.bfloat16,
        quant_attention_config=stock.inner_backend.quant_attention_config,
        fp8_dense_fmha="trtllm-gen-fp16sm",
        fp8_skip_corr_threshold=1.0,
        fp8_strided_views=True,
    )
    gen_backend = UlyssesAttention(inner, group)
    gen_t1_backend = UlyssesAttention(inner_t1, group)
    symm_amax = None
    symm_cls = None
    try:
        from ..minimax_h3_nvfp4__sm_107__u1_u4.kernels import fused_fp8_qk

        symm_cls = getattr(fused_fp8_qk, "_SymmAmax", None)
    except (ImportError, RuntimeError, AttributeError):
        symm_cls = None
    # Vote on local readiness before entering symmetric-memory collective rendezvous.
    can_enter_rendezvous = symm_cls is not None
    if torch.distributed.is_initialized():
        ready = torch.tensor(1 if can_enter_rendezvous else 0, dtype=torch.int32, device=device)
        torch.distributed.all_reduce(ready, op=torch.distributed.ReduceOp.MIN, group=group)
        can_enter_rendezvous = int(ready.item()) == 1
    if can_enter_rendezvous and symm_cls is not None:
        try:
            symm_amax = symm_cls(group, device)
        except (ImportError, RuntimeError, AttributeError):
            symm_amax = None
        if torch.distributed.is_initialized():
            ok = torch.tensor(1 if symm_amax is not None else 0, dtype=torch.int32, device=device)
            torch.distributed.all_reduce(ok, op=torch.distributed.ReduceOp.MIN, group=group)
            if int(ok.item()) == 0:
                symm_amax = None
    direct_fp8_ok = False
    try:
        direct_fp8_ok = bool(
            torch.ops.trtllm.fmha_dense_fp8_is_supported(128, 14, torch.float8_e4m3fn, True)
        )
    except (RuntimeError, AttributeError):
        direct_fp8_ok = False
    step_state = {"step": 0}
    for attention in attentions:
        attention._hybrid_gen_backend = gen_backend
        attention._hybrid_gen_t1_backend = gen_t1_backend
        attention._hybrid_symm_amax = symm_amax
        attention._hybrid_direct_fp8_supported = direct_fp8_ok
        attention._hybrid_step_state = step_state
        attention._hybrid_cudnn_steps = int(cudnn_initial_steps)
        attention._hybrid_direct_fp8_steps = int(direct_fp8_out_start)
        attention._hybrid_t1_steps = int(skip_corr_t1_start)
    transformer.register_forward_hook(
        lambda _m, _i, _o: step_state.__setitem__("step", step_state["step"] + 1)
    )
    return step_state


class MiniMaxH3Fp8U4Pipeline(MiniMaxH3Pipeline):
    def _init_transformer(self) -> None:
        self._hybrid_step_state: dict[str, int] | None = None
        self.transformer = Fp8UlyssesTransformer(
            self.pipeline_config.model_configs[PipelineComponent.TRANSFORMER]
        )

    def post_load_weights(self) -> None:
        configure_u4_video_vae_pre_load(self.vae, self.device)
        super().post_load_weights()
        initialize_block_ops(self.transformer)
        self._hybrid_step_state = _initialize_sm107_hybrid_fmha(self.transformer)
        configure_u4_video_vae_post_load(self.vae, self.device, ulysses_size=4, enable_fp8=True)
        install_audio_vae_cuda_graph(getattr(self, "audio_vae", None), self.device)

    def forward(self, *args, **kwargs):
        if self._hybrid_step_state is not None:
            self._hybrid_step_state["step"] = 0
        try:
            return super().forward(*args, **kwargs)
        finally:
            if self._hybrid_step_state is not None:
                self._hybrid_step_state["step"] = 0
