# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 KDA production frontend for the TensorRT-LLM PyTorch executor.

The checkpoint-visible projections, short convolutions, forget/beta gates,
and gated output norm mirror the HF ``KimiDeltaAttention`` structure. The
delta-rule inner loop dispatches to the optimized Blackwell prefill,
decode, and verification kernels, with explicit FLA fallbacks.

Cache ownership
---------------
The module reads and updates three short-convolution states in the combined
``[slots, 3D, W - 1]`` bf16 pool and one delta-rule recurrent state in the
``[slots, H, V, K]`` fp32 pool. It owns the ``AttentionMetadata`` split,
cache-slot indexing, and speculative-verification state handling directly.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import torch
from fla.modules import ShortConvolution
from torch import nn

from ...attention_backend import AttentionMetadata
from ...distributed import AllReduce, AllReduceStrategy
from ..mamba.causal_conv1d import causal_conv1d_fn
from ..mamba.fuse_elementwise_ops import extract_transpose_prefill_slice
from ..mamba.layernorm_gated import RMSNorm, rms_norm_gated_token_major
from ..mamba.recurrent_state_cache import reset_recurrent_state_rows
from ..multi_stream_utils import maybe_execute_in_parallel
from ._kda_kernels import KDAKernelDispatch, fused_kda_post_conv

_KDA_INDEXED_STATE_POOL_ENABLED = os.environ.get("TLLM_KDA_ENABLE_INDEXED_STATE_POOL", "1") == "1"
# Heuristic ported from SGLang's Blackwell cutoff:
# https://github.com/sgl-project/sglang/blob/e84bbf68efb683c9e2eef4168c5198042544599d/python/sglang/srt/models/kimi_k3.py#L946-L954
# It has not been tuned for TensorRT-LLM; benchmark and retune it for TRT-LLM's
# projection kernels. Verify intentionally counts B * num_steps because those
# flattened token rows form the projection GEMMs' M dimension.
_KDA_BFA_MULTISTREAM_MAX_ROWS = 128


def _meta_safe_cast_dtype(module: nn.Module, dtype: torch.dtype) -> None:
    """Cast floating parameters while preserving meta-device construction.

    ``Module.to`` dispatches ``aten._to_copy``, which ``MetaInitMode`` rejects
    and would force the full model to fall back to eager CPU construction.
    Meta values are uninitialized, so dtype-only reallocation via
    ``empty_like`` is equivalent there; materialized tensors use ``to``.
    """

    def _cast(tensor: torch.Tensor) -> torch.Tensor:
        if not tensor.is_floating_point():
            return tensor
        if tensor.is_meta:
            return torch.empty_like(tensor, dtype=dtype)
        return tensor.to(dtype=dtype)

    module._apply(_cast)


def _kda_split_conv_sections(
    conv_state: torch.Tensor, dim: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a gathered ``[N, 3D, W - 1]`` convolution cache into Q/K/V."""
    return (
        conv_state[:, :dim].contiguous(),
        conv_state[:, dim : 2 * dim].contiguous(),
        conv_state[:, 2 * dim :].contiguous(),
    )


def _kda_expand_fla_conv_cache(conv_state: torch.Tensor) -> torch.Tensor:
    """Left-pad a production ``W - 1`` cache to FLA's full ``W`` window."""
    return torch.nn.functional.pad(conv_state, (1, 0))


class KimiKDALinearAttention(nn.Module):
    """Production Kimi K3 KDA module with direct cache-pool ownership."""

    def __init__(
        self,
        cfg,
        layer_idx: int,
        mapping=None,
        allreduce_strategy=AllReduceStrategy.AUTO,
        aux_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        lin = cfg.linear_attn_config
        self.hidden_size = cfg.hidden_size
        self.layer_idx = layer_idx
        self.head_dim = lin["head_dim"]
        self.head_k_dim = self.head_dim
        self.conv_size = lin["short_conv_kernel_size"]
        self.use_full_rank_gate = lin.get("use_full_rank_gate", True)
        self.gate_lower_bound = lin.get("gate_lower_bound", None)
        self.rms_norm_eps = cfg.rms_norm_eps
        self._use_indexed_ssm_pool = _KDA_INDEXED_STATE_POOL_ENABLED

        if mapping is not None and mapping.tp_size > 1 and not mapping.enable_attention_dp:
            self._kda_tp_size = mapping.tp_size
        else:
            self._kda_tp_size = 1
        self._kda_tp_rank = mapping.tp_rank if self._kda_tp_size > 1 else 0
        self._o_allreduce = (
            AllReduce(mapping=mapping, strategy=allreduce_strategy, dtype=torch.bfloat16)
            if self._kda_tp_size > 1
            else None
        )

        num_heads = lin["num_heads"]
        if num_heads % self._kda_tp_size != 0:
            raise ValueError(
                f"KDA num_heads {num_heads} not divisible by tp_size {self._kda_tp_size}"
            )
        self.num_heads = num_heads // self._kda_tp_size
        self.num_k_heads = self.num_heads
        projection_size = self.num_heads * self.head_dim
        self.proj_size = projection_size

        # Keep the logical projections separate for checkpoint-compatible
        # parameter names and portable fallbacks. After weight loading, the
        # Blackwell path aliases them into one fused QKVG weight buffer.
        self.q_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        self.q_conv1d = ShortConvolution(
            hidden_size=projection_size, kernel_size=self.conv_size, activation="silu"
        )
        self.k_conv1d = ShortConvolution(
            hidden_size=projection_size, kernel_size=self.conv_size, activation="silu"
        )
        self.v_conv1d = ShortConvolution(
            hidden_size=projection_size, kernel_size=self.conv_size, activation="silu"
        )

        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16))
        )
        self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)
        # dt_bias must be initialized: torch.empty heap garbage can contain
        # NaN bit patterns, which poison both the optimized and FLA gates in
        # randomly-constructed modules (parity tests) — TRTLLM-15204. This
        # matches FLA's KDA init (inverse-softplus of dt ~ LogUniform[1e-3,
        # 1e-1], fla/layers/kda.py) in its small-dt regime where
        # softplus(x) ≈ exp(x), expressed as a single uniform_ so it stays on
        # MetaInitMode's random-init allowlist (exp/expm1/clamp would raise
        # MetaInitException and force the full-CPU-init fallback). Checkpoint
        # loading overwrites the value either way.
        self.dt_bias = nn.Parameter(
            torch.empty(projection_size, dtype=torch.float32).uniform_(
                math.log(1e-3), math.log(1e-1)
            )
        )
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        if self.use_full_rank_gate:
            self.g_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        else:
            self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
            self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)
        self.o_norm = RMSNorm(self.head_dim, eps=self.rms_norm_eps)
        if not self.o_norm.weight.is_meta:
            nn.init.ones_(self.o_norm.weight)
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False)
        # Installed together by the FP8 weight loader as the fused
        # [q | k | v | g] projection and its output-section metadata.
        self.qkvg_proj: Optional[nn.Module] = None
        self.qkvg_split_sizes: Optional[List[int]] = None
        _meta_safe_cast_dtype(self, torch.bfloat16)

        # The optimized decode/verify kernels specialize the Kimi K3
        # K == V == 128 shape. Other shapes must use the portable fallback.
        kernel_shape_ok = self.head_k_dim == 128 and self.head_dim == 128
        self._dispatch = KDAKernelDispatch(
            use_optimized_decode=kernel_shape_ok,
            use_optimized_verify=kernel_shape_ok,
        )

        # Fused prefill/decode/verify projection weights, built after checkpoint
        # load. BF16 uses separate fused [q | k | v | g] and [f_a | b]
        # GEMMs; FP8 supplies qkvg through the fused projection and reuses the
        # BF16 [f_a | b] weight.
        self._qkvg_proj_weight: Optional[torch.Tensor] = None
        self._bfa_proj_weight: Optional[torch.Tensor] = None
        self._w_q_t = self._w_k_t = self._w_v_t = None
        self._A_log_f32 = self._dt_bias_f32 = self._onorm_w_f32 = None
        # Fork/join state for overlapping the small [f_a | b] -> f_b chain
        # with the wide QKVG projection during CUDA-graph execution.
        self._projection_aux_stream = aux_stream
        self._projection_fork_event = torch.cuda.Event()
        self._projection_join_event = torch.cuda.Event()
        # Output buffer for the inplace-only ``trtllm::kda_decode`` op.
        self._o_dense: Optional[torch.Tensor] = None
        self._packed_conv_weight: Optional[torch.Tensor] = None
        self._mtp_conv_weights: Optional[Tuple[torch.Tensor, ...]] = None

    def finalize_decode_weights(self) -> None:
        """Build fused projection weights and decode constants after weight load.

        1. Separate fused ``[q | k | v | g]`` and ``[f_a | b]`` projections.
           Keeping the wide qkvg output aligned avoids degrading its GEMM
           kernel selection with the small f_a and b tails. Source parameters
           are repointed to row views of the fused buffers, so prefill and
           verify paths keep using them without duplicate weight storage.
        2. Kernel-layout constants that ``_decode_via_optimized`` used to
           rebuild with ~6 device kernels per layer per decode step:
           transposed conv weights (bf16 ``[W, D]``) and fp32 copies of
           ``A_log`` / ``dt_bias`` / ``o_norm.weight``.
        """
        if self._dispatch.decode_kernel_path != "optimized" or not self.use_full_rank_gate:
            return
        if self.q_proj.weight.device.type != "cuda":
            return
        with torch.no_grad():
            qkvg_modules = (
                self.q_proj,
                self.k_proj,
                self.v_proj,
                self.g_proj,
            )
            qkvg_weight = self._merge_projection_weights(qkvg_modules)
            # Eight BF16 outputs occupy 16 bytes, so padding keeps each output row
            # aligned for vectorized f_b consumption; it is not a kernel requirement.
            bfa_weight = self._merge_projection_weights((self.f_a_proj, self.b_proj), pad_rows_to=8)
            self._build_decode_kernel_constants()
            self._bfa_proj_weight = bfa_weight
            # Publish last: both weights are required by the BF16 fast path.
            self._qkvg_proj_weight = qkvg_weight

    @staticmethod
    def _merge_projection_weights(
        modules: tuple[nn.Linear, ...], pad_rows_to: int = 1
    ) -> torch.Tensor:
        """Concatenate linear weights and repoint the modules to row views."""
        weights = [module.weight.data for module in modules]
        padding = (-sum(weight.shape[0] for weight in weights)) % pad_rows_to
        if padding:
            weights.append(weights[0].new_zeros((padding, weights[0].shape[1])))
        fused = torch.cat(weights, dim=0).contiguous()
        offset = 0
        for module in modules:
            rows = module.weight.shape[0]
            module.weight.data = fused[offset : offset + rows]
            offset += rows
        return fused

    def _build_decode_kernel_constants(self) -> None:
        """Kernel-layout constants shared by both finalize variants."""
        self._w_q_t = (
            self.q_conv1d.weight.detach().squeeze(1).transpose(0, 1).to(torch.bfloat16).contiguous()
        )
        self._w_k_t = (
            self.k_conv1d.weight.detach().squeeze(1).transpose(0, 1).to(torch.bfloat16).contiguous()
        )
        self._w_v_t = (
            self.v_conv1d.weight.detach().squeeze(1).transpose(0, 1).to(torch.bfloat16).contiguous()
        )
        self._A_log_f32 = self.A_log.detach().float().contiguous()
        self._dt_bias_f32 = self.dt_bias.detach().float().contiguous()
        self._onorm_w_f32 = self.o_norm.weight.detach().float().contiguous()
        # Build the fused-verify conv constants eagerly too, so the first
        # verify call never allocates (a capture-unsafe lazy allocation).
        self._build_mtp_conv_weights()

    def finalize_decode_weights_fp8(self) -> None:
        """FP8 counterpart of ``finalize_decode_weights()``.

        Runs AFTER ``_convert_kda_projections_to_fp8_weight_read``, so
        q/k/v/g already live in the fused FP8 ``qkvg_proj`` GEMM. Only the
        two small BF16 projections reading the same hidden — ``f_a_proj`` and
        ``b_proj`` — are fused here into one ``[f_a | b]`` weight, with the
        source parameters repointed to row views. Prefill, decode, and
        verification then share both fused projections; the kernel-layout
        constants are decode-only.
        """
        if self._dispatch.decode_kernel_path != "optimized" or not self.use_full_rank_gate:
            return
        fused_qkvg = self.qkvg_proj
        split_sizes = self.qkvg_split_sizes
        if fused_qkvg is None or split_sizes is None or len(split_sizes) != 4:
            return
        if self.f_a_proj.weight.device.type != "cuda":
            return
        with torch.no_grad():
            bfa_weight = self._merge_projection_weights((self.f_a_proj, self.b_proj), pad_rows_to=8)
            self._build_decode_kernel_constants()
            # Publish last: enables fused [f_a | b] in prefill/decode/verify.
            self._bfa_proj_weight = bfa_weight

    def forward(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        """``hidden_states``: flattened ``[num_tokens, hidden]`` (ctx tokens
        first, then one token per generation request)."""
        mamba_metadata = attn_metadata.mamba_metadata
        num_prefills = attn_metadata.num_contexts
        num_ctx_tokens = attn_metadata.num_ctx_tokens
        batch_size = attn_metadata.seq_lens.shape[0]
        state_indices = mamba_metadata.state_indices[:batch_size]
        cu_seqlens = mamba_metadata.query_start_loc_long[: num_prefills + 1]
        num_generations = batch_size - num_prefills

        layer_cache = attn_metadata.kv_cache_manager.mamba_layer_cache(self.layer_idx)
        conv_pool = layer_cache.conv  # [slots, 3D, W - 1] bf16
        ssm_pool = layer_cache.temporal  # [slots, H, V, K] fp32

        outputs: List[torch.Tensor] = []
        if num_prefills > 0:
            outputs.append(
                self.forward_prefill(
                    hidden_states[:num_ctx_tokens],
                    cu_seqlens,
                    mamba_metadata,
                    num_prefills,
                    conv_pool,
                    ssm_pool,
                    state_indices[:num_prefills],
                    layer_cache,
                )
            )
        if num_generations > 0:
            num_gen_tokens = hidden_states.shape[0] - num_ctx_tokens
            if num_gen_tokens == num_generations:
                outputs.append(
                    self.forward_decode(
                        hidden_states[num_ctx_tokens:],
                        conv_pool,
                        ssm_pool,
                        state_indices[num_prefills:],
                        mamba_metadata,
                        layer_cache,
                        ssm_state_indices=(
                            mamba_metadata.state_indices[num_prefills:batch_size]
                            if self._use_indexed_ssm_pool
                            else None
                        ),
                    )
                )
            else:
                # Speculative verification: each generation request carries
                # 1 + draft_len tokens (drafts are padded to the static max,
                # so T is uniform). Per-step states go to the manager's
                # SpeculativeState scratch buffers — never the live pools —
                # and kv_cache_manager.update_mamba_states() promotes the
                # accepted step after sampling.
                assert num_gen_tokens % num_generations == 0, (
                    f"ragged generation batch: {num_gen_tokens} tokens for {num_generations} requests"
                )
                outputs.append(
                    self.forward_verify(
                        hidden_states[num_ctx_tokens:],
                        num_gen_tokens // num_generations,
                        layer_cache,
                        conv_pool,
                        ssm_pool,
                        state_indices[num_prefills:],
                    )
                )
        out = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
        if self._o_allreduce is not None:
            # Head-sharded TP: every rank ran its head shard on the same
            # local batch; sum the row-sharded o_proj partials.
            out = self._o_allreduce(out)
        return out

    def _has_kda_replay_caches(self, layer_cache) -> bool:
        """True when the manager allocated the fused-verify replay caches."""
        return layer_cache is not None and layer_cache.has_kda_replay_caches

    def _sync_kda_replay_conv_window(self, layer_cache, slot_indices, conv_pool) -> None:
        """Seed replay conv caches from the live committed conv pool.

        The fused verify kernel keeps its own extended fp32 dim-contiguous
        conv caches; their committed window (columns ``[0, W-1)``) must hold
        the live pool's ``W - 1`` raw inputs whenever prefill or plain decode
        advances it.
        """
        if not self._has_kda_replay_caches(layer_cache):
            return
        layer_cache.commit_conv_window(slot_indices, conv_pool)

    def _project_packed_conv_input(
        self, x: torch.Tensor, x2d: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Project the block input into the ``[3D, T]`` the convolution needs.

        Returns ``(packed_conv, onorm_g)``; ``onorm_g`` is None for
        configurations without a full-rank gate.
        """
        d = self.proj_size
        if self._qkvg_proj_weight is not None:
            # Transposing the GEMM skips the repack the paths below still need.
            weight = self._qkvg_proj_weight
            packed_conv = torch.mm(weight[: 3 * d], x2d.t())
            onorm_g = torch.nn.functional.linear(x, weight[3 * d : 4 * d])
            return packed_conv, onorm_g

        onorm_g = None
        fused_qkvg = self.qkvg_proj
        if fused_qkvg is not None:
            qkvg = fused_qkvg(x)
            qkvg_split_sizes = self.qkvg_split_sizes
            if (
                self.use_full_rank_gate
                and qkvg_split_sizes is not None
                and len(qkvg_split_sizes) == 4
            ):
                onorm_g = qkvg[..., 3 * d : 4 * d]
        else:
            qkvg = torch.cat((self.q_proj(x), self.k_proj(x), self.v_proj(x)), dim=-1)

        # These have no transposed-output form, so they repack -- through the
        # fused kernel GDN and Mamba2 use here, not a strided copy.
        qkvg_2d = qkvg.squeeze(0)
        return (
            extract_transpose_prefill_slice(qkvg_2d, qkvg_2d.shape[0], 0, 3 * d),
            onorm_g,
        )

    def forward_prefill(
        self,
        x2d,
        cu_seqlens,
        mamba_metadata,
        num_prefills,
        conv_pool,
        ssm_pool,
        slot_indices,
        layer_cache=None,
    ) -> torch.Tensor:
        chunk_indices = getattr(mamba_metadata, "kda_chunk_indices", None)
        varlen_is_aligned = getattr(mamba_metadata, "kda_varlen_is_aligned", None)
        single_sequence_length = getattr(mamba_metadata, "kda_single_sequence_length", None)
        from einops import rearrange

        x = x2d.unsqueeze(0)  # [1, T, hidden]

        packed_conv, onorm_g = self._project_packed_conv_input(x, x2d)

        # Initial states: present for continuation chunks (chunked prefill)
        # and for prefix-cache hits (block reuse), where the previous
        # conv/recurrent state was onboarded into this request's slot.
        has_init = mamba_metadata.has_initial_states[:num_prefills]
        use_indexed_state = self._dispatch.can_use_indexed_prefill(
            state_pool=ssm_pool,
            state_indices=slot_indices,
            has_initial_states=has_init,
            cu_seqlens=cu_seqlens,
            num_sequences=num_prefills,
            num_tokens=x2d.shape[0],
            chunk_indices=chunk_indices,
        )
        slot_indices_long = slot_indices.long()
        recurrent_in = None
        if use_indexed_state:
            # The packed convolution clears fresh convolution rows itself.
            reset_recurrent_state_rows(ssm_pool, slot_indices, has_init)
        elif mamba_metadata.use_initial_states:
            recurrent_in = ssm_pool.index_select(0, slot_indices_long)
            recurrent_in[~has_init] = 0

        # Reuse GDN's packed variable-length causal convolution. It reads
        # and writes the live [slots, 3D, W - 1] pool directly and honors
        # has_initial_state for fresh versus continuation requests.
        assert self._packed_conv_weight is not None
        causal_conv1d_fn(
            packed_conv,
            self._packed_conv_weight,
            query_start_loc=mamba_metadata.query_start_loc[: num_prefills + 1],
            cache_indices=mamba_metadata.state_indices[:num_prefills],
            has_initial_state=has_init,
            conv_states=conv_pool,
            activation="silu",
        )

        if self._bfa_proj_weight is not None:
            bfa = torch.nn.functional.linear(x, self._bfa_proj_weight)
            f_a = bfa[..., : self.head_dim]
            beta = bfa[..., self.head_dim : self.head_dim + self.num_heads].float()
            g = self.f_b_proj(f_a)
        else:
            g = self.f_b_proj(self.f_a_proj(x))
            beta = self.b_proj(x).float()
        g = rearrange(g, "... (h d) -> ... h d", d=self.head_dim)

        q, k, v = fused_kda_post_conv(
            packed_conv,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )

        # The optimized kernel reads and writes the V-first pool directly;
        # the FLA core exchanges batch-dense recurrent states.
        lower_bound = self.gate_lower_bound
        o, final_state = self._dispatch.prefill_chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            scale=self.head_k_dim**-0.5,
            initial_state=None if use_indexed_state else recurrent_in,
            safe_gate=lower_bound is not None,
            lower_bound=lower_bound,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            state_pool=ssm_pool if use_indexed_state else None,
            state_indices=slot_indices if use_indexed_state else None,
            varlen_is_aligned=varlen_is_aligned,
            single_sequence_length=single_sequence_length,
        )

        # The packed convolution persisted the live convolution pool in place.
        if final_state is not None:
            ssm_pool.index_copy_(0, slot_indices_long, final_state.to(ssm_pool.dtype))
        else:
            assert use_indexed_state
        # Fused-verify replay caches: seed the committed conv window so the
        # first verify round convolves the correct history (pending drafts
        # are zero for a fresh request, so the tail columns are unused).
        self._sync_kda_replay_conv_window(layer_cache, slot_indices, conv_pool)

        return self._output_gate_and_proj(x, o, onorm_g)

    def forward_decode(
        self,
        x2d,
        conv_pool,
        ssm_pool,
        slot_indices,
        mamba_metadata=None,
        layer_cache=None,
        ssm_state_indices=None,
    ) -> torch.Tensor:
        """Plain T=1 decode, fast path.

        Calls ``trtllm::kda_decode`` directly with kernel-native layouts
        (nsys 07-24: the reference path spent ~70 us/layer on glue around
        the 5 us kernel — 6 separate in-projection GEMV pairs, per-step
        re-transposition of constant weights, conv-window slice/roll
        copies, per-call torch.arange defaults, and redundant dtype
        casts):

        * one wide fused qkvg GEMV on the main stream, overlapped with the
          fused [f_a | b] GEMV and f_b GEMV on the auxiliary stream for
          CUDA-graph batches up to 128 tokens;
        * live packed conv and recurrent pools updated directly by slot;
        * constant tensors (transposed conv weights, fp32 A_log/dt_bias/
          o_norm weight) reused instead of rebuilt per step.

        The direct path requires stable int32 slot indices. Static or
        unsupported layouts route to ``forward_decode_fallback``.
        """
        if self._dispatch.decode_kernel_path != "optimized":
            ssm_state_indices = None
        has_qkvg_projection = self._qkvg_proj_weight is not None or self.qkvg_proj is not None
        if (
            self._dispatch.decode_kernel_path != "optimized"
            or not has_qkvg_projection
            or self._bfa_proj_weight is None
            or mamba_metadata is None
            or ssm_pool.dtype != torch.float32
            or ssm_state_indices is None
        ):
            return self.forward_decode_fallback(
                x2d,
                conv_pool,
                ssm_pool,
                slot_indices,
                layer_cache,
                ssm_state_indices,
            )
        d = self.proj_size
        hd = self.head_dim
        H = self.num_heads
        B = x2d.shape[0]

        # kda_decode writes its [B, 1, H, hd] result into this buffer. It is
        # sized to the pool slot count on the first decode and never
        # reallocated, because captured graphs bind this pointer.
        if self._o_dense is None:
            if torch.cuda.is_current_stream_capturing():
                return self.forward_decode_fallback(
                    x2d, conv_pool, ssm_pool, slot_indices, layer_cache, ssm_state_indices
                )
            self._o_dense = torch.empty(
                max(conv_pool.shape[0], B), 1, H, hd, dtype=torch.bfloat16, device=x2d.device
            )
        else:
            assert self._o_dense.shape[0] >= B, (
                f"KDA decode output buffer holds {self._o_dense.shape[0]} rows "
                f"but the decode batch is {B}; reallocating would corrupt "
                f"previously captured CUDA graphs"
            )

        def _project_qkvg() -> torch.Tensor:
            if self._qkvg_proj_weight is not None:
                return torch.nn.functional.linear(x2d, self._qkvg_proj_weight)
            return self.qkvg_proj(x2d)

        def _project_bfa_and_fb() -> tuple[torch.Tensor, torch.Tensor]:
            bfa = torch.nn.functional.linear(x2d, self._bfa_proj_weight)
            f_a = bfa[:, :hd]
            beta = bfa[:, hd : hd + H]
            return beta, self.f_b_proj(f_a)

        projection_aux_stream = (
            self._projection_aux_stream if B <= _KDA_BFA_MULTISTREAM_MAX_ROWS else None
        )
        qkvg, (beta, g) = maybe_execute_in_parallel(
            _project_qkvg,
            _project_bfa_and_fb,
            self._projection_fork_event,
            self._projection_join_event,
            projection_aux_stream,
            disable_on_compile=True,
        )
        x_qkvg = qkvg[:, : 4 * d]

        # Section views retain the live pool's slot stride, including V2
        # manager padding. The kernel uses ssm_state_indices for both pools.
        cs_q = conv_pool[:, :d]
        cs_k = conv_pool[:, d : 2 * d]
        cs_v = conv_pool[:, 2 * d :]

        o = self._dispatch.decode_kda(
            x_q=x_qkvg[:, :d].unflatten(-1, (H, hd)).unsqueeze(0),
            x_k=x_qkvg[:, d : 2 * d].unflatten(-1, (H, hd)).unsqueeze(0),
            x_v=x_qkvg[:, 2 * d : 3 * d].unflatten(-1, (H, hd)).unsqueeze(0),
            w_q_t=self._w_q_t,
            w_k_t=self._w_k_t,
            w_v_t=self._w_v_t,
            bias_q=None,
            bias_k=None,
            bias_v=None,
            cs_q=cs_q,
            cs_k=cs_k,
            cs_v=cs_v,
            A_log=self._A_log_f32,
            g=g.unflatten(-1, (H, hd)).unsqueeze(0),
            dt_bias=self._dt_bias_f32,
            beta=beta.unsqueeze(0),
            state=ssm_pool,
            onorm_g=x_qkvg[:, 3 * d :].unflatten(-1, (H, hd)).unsqueeze(0),
            onorm_weight=self._onorm_w_f32,
            out=self._o_dense[:B],
            ssm_state_indices=ssm_state_indices,
            cu_seqlens=mamba_metadata._arange_buffer[: B + 1],
            scale=hd**-0.5,
            onorm_eps=self.o_norm.eps,
            lower_bound=self.gate_lower_bound,
            use_beta_sigmoid_in_kernel=True,
            verbose=False,
            update_conv_cache=True,
        )
        # Fused-verify replay caches (spec decoding only): keep the
        # committed conv window in sync with the plain-decode advance.
        self._sync_kda_replay_conv_window(layer_cache, slot_indices, conv_pool)

        return self.o_proj(o.view(B, d))

    def forward_decode_fallback(
        self, x2d, conv_pool, ssm_pool, slot_indices, layer_cache=None, ssm_state_indices=None
    ) -> torch.Tensor:
        """Portable or unfused decode with the production pool contract."""
        from einops import rearrange

        d = self.proj_size
        slot_indices_long = slot_indices.long()
        x = x2d.unsqueeze(1)  # [B, 1, hidden]
        cs = conv_pool.index_select(0, slot_indices_long)
        conv_q, conv_k, conv_v = _kda_split_conv_sections(cs, d)
        state = (
            ssm_pool
            if ssm_state_indices is not None
            else ssm_pool.index_select(0, slot_indices_long)
        )

        q_proj = self.q_proj(x)
        k_proj = self.k_proj(x)
        v_proj = self.v_proj(x)
        g_hidden = self.f_b_proj(self.f_a_proj(x))
        beta = self.b_proj(x).float()

        if self._dispatch.decode_kernel_path == "optimized":
            onorm_g = self.g_proj(x) if self.use_full_rank_gate else self.g_b_proj(self.g_a_proj(x))

            def _kernel_input(value: torch.Tensor) -> torch.Tensor:
                return (
                    rearrange(value, "b t (h d) -> t b h d", h=self.num_heads, d=self.head_dim)
                    .to(dtype=torch.bfloat16)
                    .contiguous()
                )

            out = self._dispatch.decode_kda(
                x_q=_kernel_input(q_proj),
                x_k=_kernel_input(k_proj),
                x_v=_kernel_input(v_proj),
                w_q_t=self.q_conv1d.weight.detach()
                .squeeze(1)
                .transpose(0, 1)
                .to(torch.bfloat16)
                .contiguous(),
                w_k_t=self.k_conv1d.weight.detach()
                .squeeze(1)
                .transpose(0, 1)
                .to(torch.bfloat16)
                .contiguous(),
                w_v_t=self.v_conv1d.weight.detach()
                .squeeze(1)
                .transpose(0, 1)
                .to(torch.bfloat16)
                .contiguous(),
                bias_q=None,
                bias_k=None,
                bias_v=None,
                cs_q=conv_q,
                cs_k=conv_k,
                cs_v=conv_v,
                A_log=self.A_log.detach().float().contiguous(),
                g=_kernel_input(g_hidden),
                dt_bias=self.dt_bias.detach().float().contiguous(),
                beta=rearrange(beta, "b t h -> t b h").to(torch.bfloat16).contiguous(),
                state=state,
                onorm_g=_kernel_input(onorm_g),
                onorm_weight=self.o_norm.weight.detach().float().contiguous(),
                out=None,
                ssm_state_indices=ssm_state_indices,
                cu_seqlens=None,
                scale=self.head_k_dim**-0.5,
                onorm_eps=self.o_norm.eps,
                lower_bound=self.gate_lower_bound,
                use_beta_sigmoid_in_kernel=True,
                verbose=False,
                update_conv_cache=False,
            )
            new_conv_q = torch.cat([conv_q[:, :, 1:], q_proj.transpose(1, 2)], dim=-1)
            new_conv_k = torch.cat([conv_k[:, :, 1:], k_proj.transpose(1, 2)], dim=-1)
            new_conv_v = torch.cat([conv_v[:, :, 1:], v_proj.transpose(1, 2)], dim=-1)
            out = self.o_proj(out.flatten(2))
        else:
            from fla.ops.kda import fused_recurrent_kda

            conv_q = _kda_expand_fla_conv_cache(conv_q)
            conv_k = _kda_expand_fla_conv_cache(conv_k)
            conv_v = _kda_expand_fla_conv_cache(conv_v)
            q, new_conv_q = self.q_conv1d(q_proj, cache=conv_q, output_final_state=True)
            k, new_conv_k = self.k_conv1d(k_proj, cache=conv_k, output_final_state=True)
            v, new_conv_v = self.v_conv1d(v_proj, cache=conv_v, output_final_state=True)
            q = rearrange(q, "... (h d) -> ... h d", d=self.head_k_dim)
            k = rearrange(k, "... (h d) -> ... h d", d=self.head_k_dim)
            v = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)
            g = rearrange(g_hidden, "... (h d) -> ... h d", d=self.head_dim)
            out, state = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                lower_bound=self.gate_lower_bound,
                state_v_first=True,
            )
            out = self._output_gate_and_proj(x, out).unsqueeze(1)
            new_conv_q = new_conv_q[:, :, 1:]
            new_conv_k = new_conv_k[:, :, 1:]
            new_conv_v = new_conv_v[:, :, 1:]

        conv_pool.index_copy_(
            0,
            slot_indices_long,
            torch.cat([new_conv_q, new_conv_k, new_conv_v], dim=1).to(conv_pool.dtype),
        )
        if ssm_state_indices is None:
            ssm_pool.index_copy_(0, slot_indices_long, state.to(ssm_pool.dtype))
        # Fused-verify replay caches: keep the committed conv window in
        # sync with the plain-decode advance. NOTE: this path is only
        # correct for requests with no pending accepted drafts
        # (prev_num_accepted_tokens == 0); with drafts pending, the live
        # pools lag by the pending prefix and only the fused verify kernel
        # can advance them. The spec workers pad drafts to the static max,
        # so drafted batches always take the verify path.
        self._sync_kda_replay_conv_window(layer_cache, slot_indices, conv_pool)

        return out.squeeze(1)

    def forward_verify(
        self, x2d, num_steps, layer_cache, conv_pool, ssm_pool, slot_indices
    ) -> torch.Tensor:
        """Speculative verification: advance each request ``num_steps``
        tokens (1 golden + ``num_steps - 1`` padded drafts).

        Two paths:

        * Fused (``trtllm::kda_mtp_decode``, when the manager allocated the
          KDA replay caches): one kernel launch replays the previous
          round's accepted drafts from the per-slot replay caches, then
          processes the new tokens, committing the recurrent state and conv
          windows **in place** after the golden token and caching the new
          drafts. ``update_mamba_states()`` afterwards only records the
          accepted count for the next round's replay.
        * Legacy (sequential per-step FLA): per-step states go to the
          manager's batch-row-indexed intermediate scratch buffers and
          ``update_mamba_states()`` promotes the accepted step's state
          after sampling.
        """
        if self._has_kda_replay_caches(layer_cache):
            assert self._dispatch.verify_kernel_path == "optimized", (
                "KDA replay caches are allocated but the fused verify "
                "kernel is unavailable; the legacy intermediate buffers "
                "were not allocated so there is no fallback"
            )
            return self.forward_verify_fused(x2d, num_steps, layer_cache, ssm_pool, slot_indices)
        return self.forward_verify_sequential(
            x2d, num_steps, layer_cache, conv_pool, ssm_pool, slot_indices
        )

    def _project_verify_inputs(
        self, x: torch.Tensor, num_rows: int
    ) -> Optional[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Optional[torch.Tensor],
        ]
    ]:
        """Project fused QKVG and [f_a | b] inputs for target verification."""
        qkvg_weight = self._qkvg_proj_weight
        fused_qkvg = self.qkvg_proj
        if qkvg_weight is None and fused_qkvg is None:
            return None

        def _project_qkvg() -> torch.Tensor:
            if qkvg_weight is not None:
                return torch.nn.functional.linear(x, qkvg_weight)
            return fused_qkvg(x)

        bfa_weight = self._bfa_proj_weight
        if bfa_weight is not None:

            def _project_bfa_and_fb() -> tuple[torch.Tensor, torch.Tensor]:
                bfa = torch.nn.functional.linear(x, bfa_weight)
                f_a = bfa[..., : self.head_dim]
                beta = bfa[..., self.head_dim : self.head_dim + self.num_heads]
                return beta, self.f_b_proj(f_a)

            projection_aux_stream = (
                self._projection_aux_stream
                if 0 < num_rows <= _KDA_BFA_MULTISTREAM_MAX_ROWS
                else None
            )
            qkvg, (beta, forget_gate) = maybe_execute_in_parallel(
                _project_qkvg,
                _project_bfa_and_fb,
                self._projection_fork_event,
                self._projection_join_event,
                projection_aux_stream,
                disable_on_compile=True,
            )
        else:
            qkvg = _project_qkvg()
            beta = self.b_proj(x)
            forget_gate = self.f_b_proj(self.f_a_proj(x))

        d = self.proj_size
        q_proj, k_proj, v_proj = (part.contiguous() for part in qkvg[..., : 3 * d].split(d, dim=-1))
        qkvg_split_sizes = self.qkvg_split_sizes
        has_onorm_gate = qkvg_weight is not None or (
            self.use_full_rank_gate and qkvg_split_sizes is not None and len(qkvg_split_sizes) == 4
        )
        onorm_g = qkvg[..., 3 * d : 4 * d].contiguous() if has_onorm_gate else None
        return q_proj, k_proj, v_proj, forget_gate, beta, onorm_g

    def forward_verify_fused(
        self, x2d, num_steps, layer_cache, ssm_pool, slot_indices
    ) -> torch.Tensor:
        """Fused multi-token verify via ``trtllm::kda_mtp_decode``.

        Token layout: the kernel indexes each request's new tokens at
        ``cu_seqlens[n] + num_accepted[n] + i``. The runtime packs the
        ``num_steps`` new tokens per request contiguously, so we pass
        ``cu_seqlens[n] = n * num_steps - num_accepted[n]`` — the shift
        lands the kernel's reads/writes exactly on the packed rows. A
        negative entry for request 0 is fine: ``bos`` is only ever used
        additively with a token offset ``>= num_accepted``.
        """
        num_generations = x2d.shape[0] // num_steps
        num_spec = num_steps - 1
        H = self.num_heads
        K = self.head_k_dim
        x = x2d.view(num_generations, num_steps, -1)  # [B, T, hidden]
        T_total = num_generations * num_steps

        projections = self._project_verify_inputs(x, T_total)
        if projections is None:
            q_proj = self.q_proj(x)
            k_proj = self.k_proj(x)
            v_proj = self.v_proj(x)
            forget_gate = self.f_b_proj(self.f_a_proj(x))
            beta_proj = self.b_proj(x)
            onorm_g = None
        else:
            q_proj, k_proj, v_proj, forget_gate, beta_proj, onorm_g = projections
        x_q = q_proj.view(1, T_total, H, K)
        x_k = k_proj.view(1, T_total, H, K)
        x_v = v_proj.view(1, T_total, H, self.head_dim)
        # Raw gate / beta: the kernel applies dt_bias, A_log, the
        # lower-bound sigmoid gate, and the beta sigmoid itself.
        g = forget_gate.view(1, T_total, H, K)
        beta = beta_proj.contiguous().view(1, T_total, H)

        w_q, w_k, w_v = self._get_mtp_conv_weights()
        lower_bound = self.gate_lower_bound

        pending = layer_cache.prev_num_accepted_tokens[
            slot_indices
        ]  # accepted drafts of the previous round, per req
        cu_seqlens = torch.arange(
            0, (num_generations + 1) * num_steps, num_steps, dtype=torch.int32, device=x2d.device
        )
        cu_seqlens[:num_generations].sub_(pending)

        out = self._dispatch.mtp_verify(
            x_q=x_q,
            x_k=x_k,
            x_v=x_v,
            w_q=w_q,
            w_k=w_k,
            w_v=w_v,
            cs_q=layer_cache.kda_conv_q,
            cs_k=layer_cache.kda_conv_k,
            cs_v=layer_cache.kda_conv_v,
            g=g,
            beta=beta,
            # .detach(): the CuTe DSL DLPack bridge rejects grad-tracking
            # tensors.
            A_log=self.A_log.detach(),
            dt_bias=self.dt_bias.detach(),
            recurrent_state=ssm_pool,
            qkg_cache=layer_cache.kda_qkg_cache,
            v_cache=layer_cache.kda_v_cache,
            beta_cache=layer_cache.kda_beta_cache,
            ssm_state_indices=slot_indices,
            cu_seqlens=cu_seqlens,
            num_spec=num_spec,
            num_accepted_tokens=pending,
            lower_bound=lower_bound,
            scale=self.head_k_dim**-0.5,
        )
        o = out.view(num_generations, num_steps, H, self.head_dim)
        return self._output_gate_and_proj(x, o, onorm_g)

    def _build_mtp_conv_weights(self) -> None:
        """Prebuild packed-prefill and fused-verify convolution weights.

        Building them lazily at first use would allocate at runtime; under
        CUDA graph capture that would bake capture-pool pointers into the
        cached tensors.
        """
        conv_weights = tuple(
            conv.weight.detach().squeeze(1)
            for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d)
        )
        self._packed_conv_weight = torch.cat(conv_weights, dim=0).to(torch.bfloat16).contiguous()
        self._mtp_conv_weights = tuple(weight.float().contiguous() for weight in conv_weights)

    def _get_mtp_conv_weights(self) -> Tuple[torch.Tensor, ...]:
        """fp32 ``[dim, W]`` conv weights for the fused verify kernel,
        prebuilt by ``_build_mtp_conv_weights()``."""
        cached = self._mtp_conv_weights
        if cached is None:
            raise RuntimeError(
                "Kimi K3 fused-verify conv weights were not prebuilt; call "
                "_build_mtp_conv_weights() (done by load_weights() and by "
                "finalize_decode_weights() / finalize_decode_weights_fp8()) "
                "after weight load and before the first verify step."
            )
        return cached

    def forward_verify_sequential(
        self, x2d, num_steps, layer_cache, conv_pool, ssm_pool, slot_indices
    ) -> torch.Tensor:
        """Sequential per-step FLA verification (legacy intermediate-buffer
        path). Live pools are read-only here; ``update_mamba_states()``
        commits the accepted step's state after sampling.
        """
        from einops import rearrange
        from fla.ops.kda import fused_recurrent_kda

        intermediate_conv = layer_cache.intermediate_conv_window
        intermediate_ssm = layer_cache.intermediate_ssm
        assert intermediate_conv is not None and intermediate_ssm is not None, (
            "speculative verification requires the cache manager's "
            "SpeculativeState (legacy intermediate-buffer path)"
        )

        d = self.proj_size
        num_generations = x2d.shape[0] // num_steps
        x = x2d.view(num_generations, num_steps, -1)  # [B, T, hidden]

        projections = self._project_verify_inputs(x, x2d.shape[0])
        if projections is None:
            q_proj_states = self.q_proj(x)
            k_proj_states = self.k_proj(x)
            v_proj_states = self.v_proj(x)
            g = self.f_b_proj(self.f_a_proj(x))
            beta = self.b_proj(x).float()
            onorm_g = None
        else:
            q_proj_states, k_proj_states, v_proj_states, g, beta, onorm_g = projections
            beta = beta.float()
        g = rearrange(g, "... (h d) -> ... h d", d=self.head_dim)

        # Gathered copies — mutated across steps, never written back to the
        # live pools.
        slot_indices_long = slot_indices.long()
        cs = conv_pool.index_select(0, slot_indices_long)
        conv_q, conv_k, conv_v = _kda_split_conv_sections(cs, d)
        conv_q = _kda_expand_fla_conv_cache(conv_q)
        conv_k = _kda_expand_fla_conv_cache(conv_k)
        conv_v = _kda_expand_fla_conv_cache(conv_v)
        state = ssm_pool.index_select(0, slot_indices_long)

        step_outputs: List[torch.Tensor] = []
        for t in range(num_steps):
            # ShortConvolution.step updates the (gathered) caches in place.
            q_t, conv_q = self.q_conv1d(
                q_proj_states[:, t : t + 1], cache=conv_q, output_final_state=True
            )
            k_t, conv_k = self.k_conv1d(
                k_proj_states[:, t : t + 1], cache=conv_k, output_final_state=True
            )
            v_t, conv_v = self.v_conv1d(
                v_proj_states[:, t : t + 1], cache=conv_v, output_final_state=True
            )

            q_t = rearrange(q_t, "... (h d) -> ... h d", d=self.head_k_dim)
            k_t = rearrange(k_t, "... (h d) -> ... h d", d=self.head_k_dim)
            v_t = rearrange(v_t, "... (h d) -> ... h d", d=self.head_dim)

            o_t, state = fused_recurrent_kda(
                q=q_t,
                k=k_t,
                v=v_t,
                g=g[:, t : t + 1],
                beta=beta[:, t : t + 1],
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                lower_bound=self.gate_lower_bound,
                state_v_first=True,
            )
            step_outputs.append(o_t)

            # Batch-row indexed ([:num_generations] prefix), matching
            # update_mamba_states()'s intermediate_state_indices.
            intermediate_conv[:num_generations, t] = torch.cat(
                [conv_q[:, :, 1:], conv_k[:, :, 1:], conv_v[:, :, 1:]], dim=1
            ).to(intermediate_conv.dtype)
            intermediate_ssm[:num_generations, t] = state.to(intermediate_ssm.dtype)

        o = torch.cat(step_outputs, dim=1)  # [B, T, H, V]
        return self._output_gate_and_proj(x, o, onorm_g)

    def _output_gate_and_proj(
        self, x: torch.Tensor, o: torch.Tensor, onorm_g: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if onorm_g is not None:
            g_out = onorm_g
        elif self.use_full_rank_gate:
            g_out = self.g_proj(x)
        else:
            g_out = self.g_b_proj(self.g_a_proj(x))
        g_out = g_out.reshape(-1, self.num_heads, self.head_dim)
        o = rms_norm_gated_token_major(
            o.reshape(-1, self.head_dim),
            g_out,
            self.o_norm.weight,
            self.o_norm.eps,
            gate_activation="sigmoid",
        )
        return self.o_proj(o.reshape(-1, self.proj_size))
