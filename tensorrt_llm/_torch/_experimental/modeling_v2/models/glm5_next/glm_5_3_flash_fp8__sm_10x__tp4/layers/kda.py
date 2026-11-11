# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KDA (Kimi delta attention) layers: this rank's heads, their recurrent state and conv
windows, and the row-split ``o_proj`` partial of every row of a step.

Under tensor parallelism every tensor that feeds a head is split by this rank's heads (16 of 64
at TP4, ``H`` heads of ``hd = 128`` channels, ``D = H * hd``): the ``q`` / ``k`` / ``v``
projections and conv taps, ``b`` (beta), the up projections ``f_b`` / ``g_b`` of the low-rank
forget and output gates, ``A_log`` and ``dt_bias``. The gates' down projections ``f_a`` /
``g_a`` and the output norm are replicated, and ``o_proj`` keeps this rank's input columns, so
the layer's output is a partial the caller reduces. The checkpoint's KDA tensors are BF16 except
the FP32 ``A_log`` and ``dt_bias``, which stay FP32.

Projections. One ``[q; k; v]`` GEMM gives the conv inputs. One ``[f_a; g_a; b]`` GEMM (rows
padded to a multiple of 8) gives both gates' low-rank inputs and the beta logits, and one batched
GEMM of its adjacent ``f_a`` / ``g_a`` columns with ``[f_b; g_b]`` gives the forget-gate and the
output-gate logits. ``[f_a; g_a; b]`` and ``o_proj`` run on ``low_m_bf16_gemm`` (its low-M
kernels for decode-sized steps, cuBLAS otherwise), the ``[q; k; v]`` GEMM on cuBLAS. In a CUDA-graph-captured
generation step of at most 128 rows the gate projections run on the aux stream beside the
``[q; k; v]`` GEMM.

Rows of a step are packed context first, and each kind of row runs one recurrence kernel:

* context rows: ``kda_prefill_conv`` (conv + SiLU from each slot's conv history, the history
  written back), then ``kda_prefill`` from each slot's recurrent state (the persistent engine
  route; slots of fresh requests are zeroed first by ``reset_recurrent_state_rows``), then the
  gated output norm ``rms_norm_gated_token_major``;
* plain decode rows (one per request): ``kda_decode``, the conv, recurrence and gated output
  norm in one launch, the conv window and state row updated in place;
* MTP verify rows (``1 + num_spec`` per request): ``kda_mtp_decode``, which replays the previous
  round's accepted drafts, runs the new rows, commits the state and the conv window after the
  golden (first) row and caches the new drafts for the next round.

Cache contract (``kv_cache_manager.mamba_layer_cache(layer)``): ``conv`` is the live
``[slots, 3D, 3]`` BF16 conv pool (``[q | k | v]`` sections), ``temporal`` the
``[slots, H, 128, 128]`` V-first recurrent pool (FP32: the GLM cache manager allocates no other). With speculative
decoding the layer cache also carries the verify kernel's replay caches: their committed conv
window is seeded from the live pool (``commit_conv_window``) after every prefill and plain
decode, and ``prev_num_accepted_tokens`` (the previous round's accepted drafts per slot, which
the runtime records after sampling) tells the verify kernel how many cached drafts to replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from ..checkpoint import Checkpoint, Cut, cols, rows, share
from ..dims import Dims
from ..ops.gemm.low_m_bf16_gemm import low_m_bf16_gemm
from ..ops.norm.rms_norm_gated_token_major import rms_norm_gated_token_major
from ..ops.ssm.kda_decode import kda_decode
from ..ops.ssm.kda_mtp_decode import kda_mtp_decode
from ..ops.ssm.kda_prefill import kda_prefill
from ..ops.ssm.kda_prefill_conv import kda_prefill_conv
from ..ops.ssm.reset_recurrent_state_rows import reset_recurrent_state_rows
from ..schedule import StepSchedule
from .streams import AuxStream

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
    from tensorrt_llm._torch.attention.backends.sparse.glm_kpool.metadata import (
        Glm5NextMamba2Metadata,
    )
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaLayerCache

#: The head size and conv width the KDA kernels implement.
_HEAD_DIM = 128
_CONV_WIDTH = 4
#: The ``[f_a; g_a; b]`` projection's rows are padded to a multiple of this.
_GATE_ROWS_ALIGN = 8
#: Captured generation steps of at most this many rows run the gate projections on the aux
#: stream beside the ``[q; k; v]`` GEMM.
_OVERLAP_MAX_ROWS = 128
#: KDA prefill takes the persistent engine route: the chunk-decomposed route, which the auto
#: router picks whenever 2 * sequences * heads fits the SMs (16 heads per rank and FP32 state
#: here), can stall one CTA of ``kernel_chain_dv2`` on its mbarrier pipeline under serving load
#: (alternating 1- and 8-request 13K-token prefills) and hang every rank at the next collective.
#: Return to the auto route once that route passes the same alternating serving load.
_PREFILL_ROUTE = "engine"


@dataclass
class KdaWeights:
    """One KDA layer at this rank's heads (``H`` heads, ``D = H * 128`` channels), in the
    layouts the operations read."""

    qkv: torch.Tensor  # [3D, hidden] bf16, rows [q; k; v]
    fa_ga_b: torch.Tensor  # [2 * 128 + H padded to 8, hidden] bf16, rows [f_a; g_a; b; zeros]
    fb_gb: torch.Tensor  # [2D, 128] bf16, rows [f_b; g_b]
    #: ``fb_gb`` as the batched GEMM's ``[2, 128, D]`` operand ``(f_b^T, g_b^T)`` (a view).
    fb_gb_t: torch.Tensor
    o_proj: torch.Tensor  # [hidden, D] bf16, this rank's input columns
    a_log: torch.Tensor  # [H] fp32
    dt_bias: torch.Tensor  # [D] fp32
    o_norm: torch.Tensor  # [128] bf16, the gated output norm's weight
    o_norm_f32: torch.Tensor  # [128] fp32, the same values for the fused decode / verify norms
    conv: torch.Tensor  # [3D, 4] bf16, the q, k and v taps stacked (prefill conv)
    conv_t: tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # q, k, v taps [4, D] bf16 (decode)
    conv_f32: tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # q, k, v taps [D, 4] fp32 (verify)
    conv_bias: torch.Tensor  # [D] bf16 zeros, the decode kernel's bias of the bias-free conv


def load_kda(ck: Checkpoint, dims: Dims, layer: int) -> KdaWeights:
    """Read layer ``layer``'s KDA tensors at this rank's heads and build the operations' layouts.

    Every ``self_attn`` tensor of the layer is read once: rows of this rank's channels for
    ``q / k / v_proj``, ``f_b / g_b_proj``, the conv taps and ``dt_bias``; rows of this rank's
    heads for ``b_proj`` and ``A_log``; this rank's input columns of ``o_proj``; ``f_a / g_a_proj``
    and ``o_norm`` whole. Projections, taps and the norm are BF16, ``A_log`` and ``dt_bias`` FP32.
    """
    assert dims.kda_head_dim == _HEAD_DIM and dims.kda_conv_size == _CONV_WIDTH, (
        f"the KDA kernels take head_dim {_HEAD_DIM} and conv width {_CONV_WIDTH}, not "
        f"{dims.kda_head_dim} and {dims.kda_conv_size}"
    )
    prefix = f"{dims.layer_prefix(layer)}self_attn."
    hidden, hd = dims.hidden, dims.kda_head_dim
    heads, d = dims.kda_heads_local, dims.kda_channels_local
    h0, h1 = share(dims.kda_heads, dims.tp_size, dims.tp_rank)
    channels = rows(h0 * hd, h1 * hd)
    device = ck.device
    bf16 = torch.bfloat16

    def read(name: str, cut: Optional[Cut] = None, dtype: torch.dtype = bf16) -> torch.Tensor:
        return ck.read(f"{prefix}{name}", cut=cut, dtype=dtype)

    qkv = torch.empty((3 * d, hidden), dtype=bf16, device=device)
    conv = torch.empty((3 * d, _CONV_WIDTH), dtype=bf16, device=device)
    for i, part in enumerate("qkv"):
        qkv[i * d : (i + 1) * d] = read(f"{part}_proj.weight", channels)
        taps = read(f"{part}_conv1d.weight", channels)
        assert taps.shape == (d, 1, _CONV_WIDTH), f"{prefix}{part}_conv1d: {tuple(taps.shape)}"
        conv[i * d : (i + 1) * d] = taps.squeeze(1)
    num_gate_rows = 2 * hd + heads
    fa_ga_b = torch.zeros(
        (-(-num_gate_rows // _GATE_ROWS_ALIGN) * _GATE_ROWS_ALIGN, hidden),
        dtype=bf16,
        device=device,
    )
    fa_ga_b[:hd] = read("f_a_proj.weight")
    fa_ga_b[hd : 2 * hd] = read("g_a_proj.weight")
    fa_ga_b[2 * hd : num_gate_rows] = read("b_proj.weight", rows(h0, h1))
    fb_gb = torch.empty((2 * d, hd), dtype=bf16, device=device)
    fb_gb[:d] = read("f_b_proj.weight", channels)
    fb_gb[d:] = read("g_b_proj.weight", channels)
    o_norm = read("o_norm.weight")
    assert o_norm.shape == (hd,), f"{prefix}o_norm: {tuple(o_norm.shape)}"
    weights = KdaWeights(
        qkv=qkv,
        fa_ga_b=fa_ga_b,
        fb_gb=fb_gb,
        fb_gb_t=fb_gb.view(2, d, hd).transpose(1, 2),
        o_proj=read("o_proj.weight", cols(h0 * hd, h1 * hd)),
        # Own allocations: the verify kernel takes every operand as 16-byte aligned unchecked, and
        # a read may be a view into the entry it was cut from.
        a_log=read("A_log", rows(h0, h1), torch.float32).clone(),
        dt_bias=read("dt_bias", channels, torch.float32).clone(),
        o_norm=o_norm,
        o_norm_f32=o_norm.float(),
        conv=conv,
        conv_t=tuple(t.t().contiguous() for t in conv.split(d)),
        conv_f32=tuple(t.float().contiguous() for t in conv.split(d)),
        conv_bias=torch.zeros(d, dtype=bf16, device=device),
    )
    assert weights.o_proj.shape == (hidden, d), f"{prefix}o_proj: {tuple(weights.o_proj.shape)}"
    assert weights.a_log.shape == (heads,) and weights.dt_bias.shape == (d,), (
        f"{prefix}A_log {tuple(weights.a_log.shape)}, dt_bias {tuple(weights.dt_bias.shape)}"
    )
    return weights


def _commit_conv_window(
    cache: MambaLayerCache, slots: torch.Tensor, conv_pool: torch.Tensor
) -> None:
    """Seed the verify kernel's replay conv caches of ``slots`` from the live conv pool, which a
    prefill or a plain decode just advanced (a layer cache without replay caches has none)."""
    if getattr(cache, "has_kda_replay_caches", False):
        cache.commit_conv_window(slots, conv_pool)


class KdaLayer:
    """One KDA layer's forward: the unreduced ``o_proj`` partial of every row of a step.

    Args:
        w: the layer's weights (``load_kda``).
        dims: the model's sizes and this rank's shares.
        layer_idx: the decoder layer's index, which names its recurrent and conv state in the
            hybrid cache.
        aux: the aux stream the gate projections of a captured generation step run on.
    """

    def __init__(self, w: KdaWeights, dims: Dims, layer_idx: int, aux: AuxStream) -> None:
        self.w = w
        self.layer_idx = layer_idx
        self.heads = dims.kda_heads_local
        self.head_dim = dims.kda_head_dim
        self.channels = dims.kda_channels_local
        self.lower_bound = dims.kda_gate_lower_bound
        self.scale = dims.kda_head_dim**-0.5
        self.norm_eps = dims.rms_eps
        self.aux = aux

    def partial(
        self, x: torch.Tensor, step: StepSchedule, attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        """This rank's ``o_proj`` partial (before the TP reduction) of every row of the step.

        Args:
            x: ``[N, hidden]`` bf16, the step's rows packed context first (``N`` is
                ``step.num_ctx_tokens + step.num_generation_tokens``).
            step: the step's schedule.
            attn_metadata: the engine-prepared metadata: its ``mamba_metadata`` (slots, packed
                context offsets, prefill schedules) and ``kv_cache_manager``.

        Returns:
            ``[N, hidden]`` bf16.
        """
        num_contexts, num_ctx_tokens = step.num_contexts, step.num_ctx_tokens
        num_rows = num_ctx_tokens + step.num_generation_tokens
        assert x.shape[0] == num_rows, f"{x.shape[0]} rows for a step of {num_rows}"
        hybrid = attn_metadata.mamba_metadata
        cache = attn_metadata.kv_cache_manager.mamba_layer_cache(self.layer_idx)
        slots = hybrid.state_indices[: num_contexts + step.num_generations]
        cores = []
        if num_contexts:
            cores.append(self._prefill(x[:num_ctx_tokens], hybrid, cache, slots[:num_contexts]))
        if step.num_generations:
            # A 16-byte-aligned copy when the verify kernel needs one, else the slots' tail.
            generation_slots = hybrid.generation_state_indices
            if generation_slots is None:
                generation_slots = slots[num_contexts:]
            generation = x[num_ctx_tokens:num_rows]
            tokens = step.gen_tokens_per_request
            if tokens == 1:
                cores.append(self._decode(generation, cache, generation_slots))
            else:
                cores.append(self._verify(generation, tokens, cache, generation_slots))
        core = cores[0] if len(cores) == 1 else torch.cat(cores)
        return low_m_bf16_gemm(core, self.w.o_proj)

    # -- projections ----------------------------------------------------------------------

    def _heads(self, columns: torch.Tensor) -> torch.Tensor:
        """``[n, D]`` columns as the kernels' ``[1, n, H, 128]`` view (the row stride kept)."""
        return columns.unflatten(-1, (self.heads, self.head_dim)).unsqueeze(0)

    def _gates(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(beta [n, H], forget [n, D], out_gate [n, D])`` logits of rows ``x [n, hidden]``:
        the ``[f_a; g_a; b]`` GEMM, then ``f_b`` / ``g_b`` as one batched GEMM over its adjacent
        ``f_a`` / ``g_a`` columns. ``beta`` is a column view of the first GEMM's output."""
        hd = self.head_dim
        fa_ga_b = low_m_bf16_gemm(x, self.w.fa_ga_b)
        beta = fa_ga_b[:, 2 * hd : 2 * hd + self.heads]
        pair = fa_ga_b[:, : 2 * hd].reshape(-1, 2, hd).transpose(0, 1)  # [2, n, hd] strided
        gates = torch.bmm(pair, self.w.fb_gb_t)  # [2, n, D]
        return beta, gates[0], gates[1]

    def _project(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """``(qkv [n, 3D], (beta, forget, out_gate))`` of generation rows ``x``. In a captured
        step of at most ``_OVERLAP_MAX_ROWS`` rows the gate projections run on the aux stream
        while the ``[q; k; v]`` GEMM runs on the current one; otherwise both run in order."""
        qkv = lambda: torch.nn.functional.linear(x, self.w.qkv)  # noqa: E731
        gates = lambda: self._gates(x)  # noqa: E731
        if x.shape[0] > _OVERLAP_MAX_ROWS:
            return qkv(), gates()
        return self.aux.fork_join(qkv, gates)

    # -- the three row kinds ----------------------------------------------------------------

    def _prefill(
        self,
        x: torch.Tensor,
        hybrid: Glm5NextMamba2Metadata,
        cache: MambaLayerCache,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        """The gated core ``[T, D]`` of the context rows ``x [T, hidden]`` of ``slots`` (int32
        ``[num_contexts]``), each sequence continuing from its slot's conv history and state."""
        w = self.w
        tokens, num_contexts = x.shape[0], slots.shape[0]
        conv_pool, state_pool = cache.conv, cache.temporal
        has_init = hybrid.has_initial_states[:num_contexts]
        qkv = torch.nn.functional.linear(x, w.qkv)
        # The recurrence reads every slot as its initial state, so fresh slots start from zero;
        # the prefill conv clears fresh conv histories itself.
        reset_recurrent_state_rows(
            recurrent_states=state_pool, state_indices=slots, has_initial_states=has_init
        )
        q, k, v = kda_prefill_conv(
            src=qkv,
            weight=w.conv,
            conv_state=conv_pool,
            starts=hybrid.query_start_loc[: num_contexts + 1],
            slots=slots,
            has_init=has_init,
            max_seqlen=hybrid.kda_max_seqlen,
        )
        beta, forget, out_gate = self._gates(x)
        o = kda_prefill(
            q=q,
            k=k,
            v=v,
            g=self._heads(forget),
            beta=beta.unsqueeze(0),
            state_pool=state_pool,
            state_indices=slots,
            cu_seqlens=hybrid.query_start_loc_long[: num_contexts + 1],
            A_log=w.a_log,
            dt_bias=w.dt_bias,
            scale=self.scale,
            lower_bound=self.lower_bound,
            mode=_PREFILL_ROUTE,
            max_seqlen=hybrid.kda_max_seqlen,
            seq_order=hybrid.kda_seq_order,
            cu_chunks=hybrid.kda_cu_chunks,
            sm_reserve=0,  # feeds the Small-BH route only
        )
        _commit_conv_window(cache, slots, conv_pool)
        core = rms_norm_gated_token_major(
            o.view(-1, self.head_dim),
            out_gate.view(tokens, self.heads, self.head_dim),
            w.o_norm,
            self.norm_eps,
            gate_activation="sigmoid",
        )
        return core.view(tokens, self.channels)

    def _decode(self, x: torch.Tensor, cache: MambaLayerCache, slots: torch.Tensor) -> torch.Tensor:
        """The gated core ``[B, D]`` of one new row per request (``x [B, hidden]``, ``slots``
        int32 ``[B]``); the requests' conv windows and FP32 state rows advance in place."""
        w = self.w
        d, batch = self.channels, x.shape[0]
        conv_pool, state_pool = cache.conv, cache.temporal
        assert state_pool.dtype == torch.float32, f"KDA state pool {state_pool.dtype}, not FP32"
        qkv, (beta, forget, out_gate) = self._project(x)
        # Allocated per call: inside a CUDA graph capture this is the graph pool's stable buffer.
        out = x.new_empty((batch, 1, self.heads, self.head_dim))
        kda_decode(
            x_q=self._heads(qkv[:, :d]),
            x_k=self._heads(qkv[:, d : 2 * d]),
            x_v=self._heads(qkv[:, 2 * d : 3 * d]),
            w_q_t=w.conv_t[0],
            w_k_t=w.conv_t[1],
            w_v_t=w.conv_t[2],
            bias_q=w.conv_bias,
            bias_k=w.conv_bias,
            bias_v=w.conv_bias,
            conv_state_q=conv_pool[:, :d],
            conv_state_k=conv_pool[:, d : 2 * d],
            conv_state_v=conv_pool[:, 2 * d :],
            a_log=w.a_log,
            g=self._heads(forget),
            dt_bias=w.dt_bias,
            beta=beta.unsqueeze(0),
            onorm_g=self._heads(out_gate),
            onorm_weight=w.o_norm_f32,
            ssm_state_indices=slots,
            state=state_pool,
            lower_bound=self.lower_bound,
            scale=self.scale,
            onorm_eps=self.norm_eps,
            output=out,
        )
        _commit_conv_window(cache, slots, conv_pool)
        return out.view(batch, d)

    def _verify(
        self, x: torch.Tensor, tokens_per_request: int, cache: MambaLayerCache, slots: torch.Tensor
    ) -> torch.Tensor:
        """The gated core ``[B * T, D]`` of ``T = tokens_per_request`` rows per request (the
        golden row, then ``T - 1`` drafts; ``x [B * T, hidden]``, ``slots`` int32 ``[B]``)."""
        if not getattr(cache, "has_kda_replay_caches", False):
            raise RuntimeError(
                f"KDA layer {self.layer_idx}: a verify step needs the KDA replay caches, which the "
                "cache manager allocates for speculative decoding the fused verify kernel supports"
            )
        w = self.w
        d = self.channels
        qkv, (beta, forget, out_gate) = self._project(x)
        out = kda_mtp_decode(
            x_q=self._heads(qkv[:, :d]),
            x_k=self._heads(qkv[:, d : 2 * d]),
            x_v=self._heads(qkv[:, 2 * d : 3 * d]),
            w_q=w.conv_f32[0],
            w_k=w.conv_f32[1],
            w_v=w.conv_f32[2],
            cs_q=cache.kda_conv_q,
            cs_k=cache.kda_conv_k,
            cs_v=cache.kda_conv_v,
            g=self._heads(forget),
            beta=beta.unsqueeze(0),
            A_log=w.a_log,
            dt_bias=w.dt_bias,
            recurrent_state=cache.temporal,
            k_cache=cache.kda_k_cache,
            g_cache=cache.kda_g_cache,
            v_cache=cache.kda_v_cache,
            beta_cache=cache.kda_beta_cache,
            ssm_state_indices=slots,
            num_accepted_tokens=cache.prev_num_accepted_tokens,
            num_spec=tokens_per_request - 1,
            lower_bound=self.lower_bound,
            scale=self.scale,
            onorm_g=self._heads(out_gate),
            onorm_weight=w.o_norm_f32,
            onorm_eps=self.norm_eps,
        )
        return out.view(x.shape[0], d)
