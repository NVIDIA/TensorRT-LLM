# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KDA linear-attention layer: Kimi K3's gated delta rule on the hybrid cache manager's state pools.

Every step runs operations of this family and target:

* prefill: the fused ``[q | k | v | g]`` projection, ``ssm/kda_prefill_conv`` (conv + SiLU), ``ssm/kda_prefill`` on
  the indexed fp32 state pool, the gated output norm (``norm/rms_norm_gated_token_major``);
* decode of one token per request: ``ssm/k3_kda_decode_attn`` (projection fused in) on the decode branch, else
  ``ssm/kda_decode`` after the fused projection;
* verify (speculative decoding), by the cache manager's mode: with the per-draft records (``kda_draft_records``),
  ``ssm/k3_kda_attn`` / ``ssm/k3_kda_verify``; with the replay caches only, ``ssm/kda_mtp_decode``; with the
  sequential-verify buffers (more drafts than ``kda_mtp_decode`` takes), FLA's ``ssm/short_conv_update`` and
  ``ssm/fused_recurrent_kda`` token by token;
* inside a breakable CUDA graph, the same core as one eager break (``graph/eager_on_graph``).

The cache manager's state layouts are read and written as they are: the conv pool ``[slots, 3 D, 3]`` bf16
(``q_k_v``), the V-first fp32 state pool, the replay caches; after every prefill or plain decode that advances the
conv pool, the replay caches' committed window follows it (``commit_conv_window``).

The replay caches and the accepted-draft record are addressed by the requests' replay slots
(``mamba_metadata.replay_state_indices``). With the per-draft records they are the state slots; with the replay caches
only, the manager keeps them in a pool of their own and the replay slots differ from the state slots.
"""

from __future__ import annotations

import math
import weakref
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.graph.eager_on_graph import (
    eager_on_graph,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.comm.k3_allreduce import (
    K3AllReduce,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.norm.rms_norm_gated_token_major import (  # noqa: E501
    rms_norm_gated_token_major,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.ssm.k3_kda_buffers import (
    K3KdaBuffers,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.ssm.k3_kda_decode_attn import (
    k3_kda_decode_attn,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.ssm.kda_decode import (
    kda_decode,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.ssm.kda_prefill import (
    kda_prefill,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.ssm.kda_prefill_conv import (
    kda_prefill_conv,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.ssm.kda_reset_state_rows import (
    kda_reset_state_rows,
)
from tensorrt_llm._torch.attention.backends import AttentionMetadata
from tensorrt_llm._torch.cute_dsl_utils import sm_reserve
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import is_in_breakable_cuda_graph
from tensorrt_llm._torch.utils import get_model_extra_attrs
from tensorrt_llm.functional import AllReduceStrategy

from . import decode_gemv as _decode_gemv
from .norm import run_in_parallel
from .ops.ssm.fused_recurrent_kda import fused_recurrent_kda
from .ops.ssm.k3_kda_attn import k3_kda_attn
from .ops.ssm.k3_kda_verify import k3_kda_verify
from .ops.ssm.kda_mtp_decode import kda_mtp_decode
from .ops.ssm.short_conv_update import short_conv_update
from .step import DecodeStep

#: Steps of up to this many projection rows overlap the [f_a | b] -> f_b chain with the wide q / k / v / g GEMM on the
#: auxiliary stream (while a CUDA graph is captured): the cutoff the built-in layer uses.
_MULTISTREAM_MAX_ROWS = 128

#: The layers a breakable CUDA graph's eager break finds by key (``_core_inplace``), weakly held.
_BREAK_LAYERS: "weakref.WeakValueDictionary[str, K3DecodeKDA]" = weakref.WeakValueDictionary()


def _core_inplace(hidden_states: torch.Tensor, layer_key: str, output: torch.Tensor) -> None:
    """The eager break of a breakable CUDA graph: the layer's core on the step's live attention metadata, into the
    graph-owned ``output``. Replays re-run it with the capture-time tensors and key, so the metadata, which changes
    every step, is read from the engine's model attributes, never passed."""
    attrs = get_model_extra_attrs()
    assert attrs is not None and attrs.get("attention_metadata") is not None, (
        "a breakable CUDA graph's KDA break needs the engine's attention metadata in the model attributes"
    )
    layer = _BREAK_LAYERS.get(layer_key)
    assert layer is not None, f"no live Kimi K3 KDA layer {layer_key}"
    layer._core(hidden_states, attrs["attention_metadata"](), output=output)


class _ConvTaps(nn.Module):
    """A short convolution's depthwise taps ``weight`` [D, 1, W], under the checkpoint's ``*_conv1d.weight`` name."""

    def __init__(self, dim: int, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, 1, width, dtype=torch.bfloat16))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))


class _GatedNormWeight(nn.Module):
    """The gated output norm's ``weight`` [head_dim] and ``eps`` (the norm is ``norm/rms_norm_gated_token_major``)."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dtype=torch.bfloat16))
        if not self.weight.is_meta:
            nn.init.ones_(self.weight)
        self.eps = eps


class K3DecodeKDA(nn.Module):
    """Kimi K3's KDA attention of one TP rank: 96 / TP heads of K = V = 128, conv width 4, a full-rank output gate.

    Parameters keep the checkpoint's names and shapes (bf16, this rank's head rows), except the six input
    projections: they are one parameter, ``k3_proj_weight`` ``[q | k | v | g | f_a | b | pad]`` (zero pad rows up to
    a multiple of 8), which the loader fills from the six checkpoint keys. ``finalize_decode_weights`` sets, once the
    weights are loaded, its ``[q | k | v | g]`` and ``[f_a | b | pad]`` row views, the conv taps in each kernel's
    layout, and fp32 copies of ``A_log``, ``dt_bias`` and the norm weight.

    * On every step ``decode_step`` classifies (the decode branch), a decode step of one token per request runs
      ``ssm/k3_kda_decode_attn``; then ``o_proj`` on the ``o_proj`` decode GEMV site where it takes the rows, and the
      TP all-reduce.
    * With the cache manager's per-token states (``KimiLinearModel.kda_draft_records``), every verify runs the
      kernels that keep them: ``ssm/k3_kda_attn`` for one request of 8 tokens (the projection fused in), else
      ``ssm/k3_kda_verify`` on the projection's rows (the ``kda_proj`` decode GEMV site where it takes them).
    * Every other step runs the generic core (``_core``) and ``o_proj``.
    """

    def __init__(
        self,
        cfg,
        layer_idx: int,
        mapping=None,
        allreduce_strategy: AllReduceStrategy = AllReduceStrategy.AUTO,
        aux_stream: Optional[torch.cuda.Stream] = None,
        model_config=None,
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
        if not (
            self.head_dim == 128
            and self.conv_size == 4
            and self.use_full_rank_gate
            and self.gate_lower_bound is not None
            and math.isfinite(self.gate_lower_bound)
            and self.gate_lower_bound < 0
        ):
            raise ValueError(
                f"Kimi K3 KDA layer {layer_idx}: the kernels take head dim 128, conv width 4, a full-rank output "
                f"gate and a finite negative gate lower bound; the config has {lin}"
            )
        declared = (getattr(model_config, "quant_config_dict", None) or {}) if model_config else {}
        quantized = [
            name
            for name in ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj")
            if getattr(declared.get(name), "quant_algo", None) is not None
        ]
        if quantized:
            raise ValueError(
                f"Kimi K3 KDA layer {layer_idx}: the kernels read bf16 projections; the checkpoint quantizes "
                f"{quantized}"
            )

        if mapping is not None and mapping.tp_size > 1 and not mapping.enable_attention_dp:
            self._kda_tp_size = mapping.tp_size
        else:
            self._kda_tp_size = 1
        self._kda_tp_rank = mapping.tp_rank if self._kda_tp_size > 1 else 0
        self._o_allreduce = (
            K3AllReduce(mapping, allreduce_strategy, torch.bfloat16)
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
        self.proj_size = proj = self.num_heads * self.head_dim

        def linear(in_features: int, out_features: int) -> nn.Linear:
            return nn.Linear(in_features, out_features, bias=False, dtype=torch.bfloat16)

        # The six input projections in one weight: q, k, v, g (this rank's heads), f_a, b (this rank's heads), then zero
        # rows up to a multiple of 8. Initialized as nn.Linear initializes each of them (all read the hidden state).
        fused_rows = 4 * proj + self.head_dim + self.num_heads
        self.k3_proj_weight = nn.Parameter(
            torch.empty(fused_rows + (-fused_rows) % 8, self.hidden_size, dtype=torch.bfloat16),
            requires_grad=False,
        )
        nn.init.kaiming_uniform_(self.k3_proj_weight, a=math.sqrt(5))
        with torch.no_grad():
            self.k3_proj_weight[fused_rows:].zero_()
        self.q_conv1d = _ConvTaps(proj, self.conv_size)
        self.k_conv1d = _ConvTaps(proj, self.conv_size)
        self.v_conv1d = _ConvTaps(proj, self.conv_size)
        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_heads, dtype=torch.bfloat16).uniform_(1, 16))
        )
        self.f_b_proj = linear(self.head_dim, proj)
        self.dt_bias = nn.Parameter(
            torch.empty(proj, dtype=torch.bfloat16).uniform_(math.log(1e-3), math.log(1e-1))
        )
        self.o_norm = _GatedNormWeight(self.head_dim, cfg.rms_norm_eps)
        self.o_proj = linear(proj, self.hidden_size)

        # SMs kept out of the prefill's persistent grids for a kernel resident for the whole step.
        self._prefill_sm_reserve = sm_reserve()
        self._projection_aux_stream = aux_stream
        self._projection_fork_event = torch.cuda.Event()
        self._projection_join_event = torch.cuda.Event()

        # Built by finalize_decode_weights once the weights are loaded.
        self._decode_weights_built = False
        # k3_proj_weight's [q | k | v | g] and [f_a | b | pad] rows, views.
        self._qkvg_proj_weight: Optional[torch.Tensor] = None
        self._bfa_proj_weight: Optional[torch.Tensor] = None
        self._w_q_t = self._w_k_t = self._w_v_t = None  # bf16 [W, D] conv taps (kda_decode)
        self._conv_weights_f32: Optional[Tuple[torch.Tensor, ...]] = (
            None  # fp32 [D, W] (verify kernels)
        )
        self._packed_conv_weight: Optional[torch.Tensor] = None  # bf16 [3 D, W] (kda_prefill_conv)
        self._A_log_f32 = self._dt_bias_f32 = self._onorm_w_f32 = None
        self._zero_bias: Optional[torch.Tensor] = None  # kda_decode's conv bias (none)
        # kda_decode writes into a buffer: one per layer, allocated at the first eager generic decode, so its address
        # stays the same across CUDA graph captures.
        self._o_dense: Optional[torch.Tensor] = None
        # The fused projection's Lamport buffers: one set per device, shared by every KDA layer.
        self.k3_buffers: Optional[K3KdaBuffers] = None
        # The decode GEMVs' state (decode_gemv.py), shared by the target's layers.
        self.decode_gemvs: Optional[_decode_gemv.K3DecodeGemvs] = None

        self._break_key = f"{layer_idx}:{id(self)}"
        _BREAK_LAYERS[self._break_key] = self

    # ------------------------------------------------------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def takes_k3_kernels(self) -> bool:
        """Whether the decode kernels can run this layer: its decode weights are built
        (``finalize_decode_weights``)."""
        return self._decode_weights_built

    def finalize_decode_weights(self) -> None:
        """The derived weights, once the checkpoint is loaded and before any CUDA graph capture: the
        ``[q | k | v | g]`` and ``[f_a | b | pad]`` row views of ``k3_proj_weight`` (no copy); the conv taps in each
        kernel's layout; fp32 ``A_log``, ``dt_bias`` and norm weight."""
        fused = self.k3_proj_weight
        assert fused.device.type == "cuda" and fused.dtype == torch.bfloat16, (
            f"Kimi K3 KDA layer {self.layer_idx}: the decode kernels read the bf16 fused projection on CUDA"
        )
        rows = 4 * self.proj_size
        with torch.no_grad():
            self._qkvg_proj_weight, self._bfa_proj_weight = fused[:rows], fused[rows:]
            self._w_q_t, self._w_k_t, self._w_v_t = (
                conv.weight.detach().squeeze(1).transpose(0, 1).to(torch.bfloat16).contiguous()
                for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d)
            )
            self._A_log_f32 = self.A_log.detach().float().contiguous()
            self._dt_bias_f32 = self.dt_bias.detach().float().contiguous()
            self._onorm_w_f32 = self.o_norm.weight.detach().float().contiguous()
            self._zero_bias = torch.zeros(self.proj_size, dtype=torch.bfloat16, device=fused.device)
        self._build_mtp_conv_weights()
        self._decode_weights_built = True

    def _build_mtp_conv_weights(self) -> None:
        """The conv taps of the prefill conv (bf16 ``[3 D, W]``) and of the verify kernels (fp32 ``[D, W]`` each),
        built eagerly so no step allocates them under capture."""
        taps = tuple(
            conv.weight.detach().squeeze(1)
            for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d)
        )
        self._packed_conv_weight = torch.cat(taps, dim=0).to(torch.bfloat16).contiguous()
        self._conv_weights_f32 = tuple(tap.float().contiguous() for tap in taps)

    # ------------------------------------------------------------------------------------------------------------------
    # The step contract
    # ------------------------------------------------------------------------------------------------------------------

    def will_run_decode_branch(self, attn_metadata: AttentionMetadata, step: DecodeStep) -> bool:
        """Whether ``forward`` runs ``step`` on the decode branch: outside a breakable CUDA graph."""
        return not is_in_breakable_cuda_graph()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        step: DecodeStep,
        reduce_output: bool = True,
        project_output: bool = True,
    ) -> torch.Tensor:
        """``forward_generic`` where ``will_run_decode_branch`` does not hold. On the decode branch: the plain decode
        on ``ssm/k3_kda_decode_attn`` on a decode step of one token per request, else the generic core; then
        ``o_proj`` on its decode GEMV site and the TP all-reduce.

        ``reduce_output=False`` returns ``o_proj``'s TP partial (no all-reduce), and, on the decode branch only,
        ``project_output=False`` the post-o_norm core ``[N, H * 128]`` (no ``o_proj``)."""
        if not self.will_run_decode_branch(attn_metadata, step):
            return self.forward_generic(hidden_states, attn_metadata, reduce_output)
        if (
            step.decode
            and step.tokens_per_request == 1
            and self.takes_k3_kernels
            and self.k3_buffers is not None
        ):
            core = self._k3_decode(hidden_states[: step.num_tokens], attn_metadata)
        else:
            core = self._core(hidden_states, attn_metadata)
        if not project_output:
            return core.reshape(-1, self.proj_size)
        return self._k3_project_output(core, reduce_output)

    def forward_generic(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        reduce_output: bool = True,
    ) -> torch.Tensor:
        """The generic core (inside a breakable CUDA graph, as one eager break), ``o_proj`` and, with
        ``reduce_output``, the TP all-reduce."""
        if is_in_breakable_cuda_graph():
            core = hidden_states.new_empty(
                (hidden_states.shape[0], self.num_heads, self.head_dim), dtype=torch.bfloat16
            )
            eager_on_graph(hidden_states, self._break_key, core, inner=_core_inplace)
        else:
            core = self._core(hidden_states, attn_metadata)
        out = F.linear(core.reshape(-1, self.proj_size), self.o_proj.weight)
        if reduce_output and self._o_allreduce is not None:
            out = self._o_allreduce(out)
        return out

    def _k3_project_output(self, core: torch.Tensor, reduce_output: bool = True) -> torch.Tensor:
        """``o_proj`` on the ``o_proj`` decode GEMV site where it takes the rows (else a GEMM), then, with
        ``reduce_output``, the TP all-reduce."""
        core2d = core.reshape(-1, self.proj_size)
        out = None
        if self.decode_gemvs is not None:
            out = self.decode_gemvs.project("o_proj", core2d, self.o_proj.weight)
        if out is None:
            out = F.linear(core2d, self.o_proj.weight)
        if reduce_output and self._o_allreduce is not None:
            out = self._o_allreduce(out)
        return out

    # ------------------------------------------------------------------------------------------------------------------
    # The generic core
    # ------------------------------------------------------------------------------------------------------------------

    def _core(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The post-o_norm core ``[N, H, 128]`` of the step's context requests, then its generation requests (plain
        decode, or verify of 1 + drafts tokens each). ``output`` (a breakable CUDA graph's buffer) receives it."""
        mamba_metadata = attn_metadata.mamba_metadata
        num_prefills = attn_metadata.num_contexts
        num_ctx_tokens = attn_metadata.num_ctx_tokens
        num_tokens = attn_metadata.num_tokens
        batch_size = attn_metadata.seq_lens.shape[0]
        state_indices = mamba_metadata.state_indices[:batch_size]
        num_decodes = batch_size - num_prefills
        layer_cache = attn_metadata.kv_cache_manager.mamba_layer_cache(self.layer_idx)
        generation_slots = getattr(mamba_metadata, "generation_state_indices", None)
        if generation_slots is None:
            generation_slots = state_indices[num_prefills:]
        # The replay caches' slots: the state slots unless the manager keeps the replay caches in a pool of their own.
        replay_indices = mamba_metadata.replay_state_indices[:batch_size]
        generation_replay_slots = getattr(mamba_metadata, "generation_replay_state_indices", None)
        if generation_replay_slots is None:
            generation_replay_slots = replay_indices[num_prefills:]

        cores = []
        if num_prefills > 0:
            core = self._prefill(
                hidden_states[:num_ctx_tokens],
                mamba_metadata,
                num_prefills,
                layer_cache,
                output[:num_ctx_tokens] if output is not None else None,
                replay_indices[:num_prefills],
            )
            cores.append(core)
        if num_decodes > 0:
            rows = num_tokens - num_ctx_tokens
            x = hidden_states[num_ctx_tokens:num_tokens]
            out = output[num_ctx_tokens:num_tokens] if output is not None else None
            if rows == num_decodes:
                core = self._decode(x, generation_slots, layer_cache, out, generation_replay_slots)
            else:
                assert rows % num_decodes == 0, (
                    f"ragged generation batch: {rows} tokens for {num_decodes} requests"
                )
                core = self._verify(
                    x,
                    rows // num_decodes,
                    layer_cache,
                    generation_slots,
                    out,
                    generation_replay_slots,
                )
            cores.append(core)
        if output is not None:
            return output
        return cores[0] if len(cores) == 1 else torch.cat(cores, dim=0)

    def _prefill(
        self,
        x2d: torch.Tensor,
        mamba_metadata,
        num_prefills: int,
        layer_cache,
        output: Optional[torch.Tensor],
        replay_slots: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The core of the step's context requests: each continues from its slot's conv window and state (a fresh
        request's rows are zeroed first) and leaves them at its last token. ``replay_slots``: the requests' rows of
        the replay caches (the state slots when None)."""
        conv_pool, ssm_pool = layer_cache.conv, layer_cache.temporal
        slots = mamba_metadata.state_indices[:num_prefills]
        has_init = mamba_metadata.has_initial_states[:num_prefills]
        if x2d.shape[0] == 0:
            kda_reset_state_rows(ssm_pool, slots, has_init, conv_pool)
            self._commit_conv_window(layer_cache, slots, conv_pool, replay_slots)
            return self._store_core(x2d.new_empty((0, self.num_heads, self.head_dim)), output)
        d = self.proj_size
        x = x2d.unsqueeze(0)  # [1, T, hidden]
        cu_seqlens = mamba_metadata.query_start_loc_long[: num_prefills + 1]
        qkvg = F.linear(x, self._qkvg_proj_weight)  # [1, T, 4 D]
        onorm_g = qkvg[..., 3 * d : 4 * d]
        packed = qkvg.squeeze(0)
        self._check_prefill_layout(ssm_pool, slots, has_init, cu_seqlens, num_prefills)
        # kda_prefill reads every selected state row; kda_prefill_conv zeroes the fresh conv windows itself.
        kda_reset_state_rows(ssm_pool, slots, has_init)
        q, k, v = kda_prefill_conv(
            packed,
            self._packed_conv_weight,
            conv_pool,
            mamba_metadata.query_start_loc[: num_prefills + 1],
            slots,
            has_init,
            getattr(mamba_metadata, "kda_max_seqlen", packed.shape[0]),
        )
        beta, g = self._project_gate_inputs(x)
        o = kda_prefill(
            q,
            k,
            v,
            g.unflatten(-1, (self.num_heads, self.head_dim)),
            beta,
            ssm_pool,
            slots,
            cu_seqlens,
            self._A_log_f32,
            self._dt_bias_f32,
            self.head_k_dim**-0.5,
            self.gate_lower_bound,
            "auto",
            getattr(mamba_metadata, "kda_max_seqlen", 0),
            getattr(mamba_metadata, "kda_seq_order", None),
            getattr(mamba_metadata, "kda_cu_chunks", None),
            self._prefill_sm_reserve,
        )
        self._commit_conv_window(layer_cache, slots, conv_pool, replay_slots)
        return self._store_core(self._output_gate(o, onorm_g), output)

    @staticmethod
    def _check_prefill_layout(ssm_pool, slots, has_init, cu_seqlens, num_prefills: int) -> None:
        """The indexed prefill's host-side layout contract (state pool rows, int32 slots, bool flags, int64
        offsets), checked before any pool is written."""
        _, heads, v_dim, k_dim = ssm_pool.shape
        ok = (
            cu_seqlens.dtype == torch.int64
            and cu_seqlens.shape == (num_prefills + 1,)
            and cu_seqlens.is_contiguous()
            and ssm_pool.dtype in (torch.bfloat16, torch.float32)
            and ssm_pool.stride()[1:] == (v_dim * k_dim, k_dim, 1)
            and (v_dim, k_dim) == (128, 128)
            and ssm_pool.stride(0) >= heads * v_dim * k_dim
            and ssm_pool.stride(0) * ssm_pool.element_size() % 16 == 0
            and ssm_pool.data_ptr() % 16 == 0
            and slots.ndim == 1
            and slots.shape[0] == num_prefills
            and slots.dtype == torch.int32
            and slots.is_contiguous()
            and has_init.ndim == 1
            and has_init.shape[0] == num_prefills
            and has_init.dtype == torch.bool
            and has_init.is_contiguous()
            and ssm_pool.device == slots.device == has_init.device
        )
        if not ok:
            raise ValueError("Unsupported FlashInfer KDA state layout or prepared metadata")

    def _decode(
        self,
        x2d: torch.Tensor,
        slots: torch.Tensor,
        layer_cache,
        output: Optional[torch.Tensor],
        replay_slots: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``ssm/kda_decode`` of one token of each of the step's generation requests after the fused projection
        (the ``[f_a | b] -> f_b`` chain on the auxiliary stream); each slot's conv window and state advance in place.
        ``replay_slots``: the requests' rows of the replay caches (``slots`` when None)."""
        conv_pool, ssm_pool = layer_cache.conv, layer_cache.temporal
        d, hd, heads = self.proj_size, self.head_dim, self.num_heads
        rows = x2d.shape[0]
        if output is not None:
            kda_out = output.view(rows, 1, heads, hd)
        else:
            if self._o_dense is None and not torch.cuda.is_current_stream_capturing():
                self._o_dense = torch.empty(
                    max(conv_pool.shape[0], rows),
                    1,
                    heads,
                    hd,
                    dtype=torch.bfloat16,
                    device=x2d.device,
                )
            if self._o_dense is None:
                # A capture before any eager decode of this layer: a buffer of the graph's own.
                kda_out = x2d.new_empty((rows, 1, heads, hd))
            else:
                assert self._o_dense.shape[0] >= rows, (
                    f"KDA decode output buffer holds {self._o_dense.shape[0]} rows but the decode batch is {rows}; "
                    "reallocating would corrupt previously captured CUDA graphs"
                )
                kda_out = self._o_dense[:rows]
        aux_stream = self._projection_aux_stream if rows <= _MULTISTREAM_MAX_ROWS else None
        qkvg, (beta, g) = run_in_parallel(
            lambda: F.linear(x2d, self._qkvg_proj_weight),
            lambda: self._project_gate_inputs(x2d),
            self._projection_fork_event,
            self._projection_join_event,
            aux_stream,
        )

        def heads_of(t: torch.Tensor) -> torch.Tensor:
            return t.unflatten(-1, (heads, hd)).unsqueeze(0)

        kda_decode(
            heads_of(qkvg[:, :d]),
            heads_of(qkvg[:, d : 2 * d]),
            heads_of(qkvg[:, 2 * d : 3 * d]),
            self._w_q_t,
            self._w_k_t,
            self._w_v_t,
            self._zero_bias,
            self._zero_bias,
            self._zero_bias,
            conv_pool[:, :d],
            conv_pool[:, d : 2 * d],
            conv_pool[:, 2 * d :],
            self._A_log_f32,
            heads_of(g),
            self._dt_bias_f32,
            beta.unsqueeze(0),
            heads_of(qkvg[:, 3 * d : 4 * d]),
            self._onorm_w_f32,
            slots,
            ssm_pool,
            True,
            True,
            True,
            True,
            float(self.gate_lower_bound),
            float(hd**-0.5),
            float(self.o_norm.eps),
            kda_out,
        )
        self._commit_conv_window(layer_cache, slots, conv_pool, replay_slots)
        return kda_out.view(rows, heads, hd)

    # ------------------------------------------------------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------------------------------------------------------

    def _verify(
        self,
        x2d: torch.Tensor,
        num_steps: int,
        layer_cache,
        slots: torch.Tensor,
        output: Optional[torch.Tensor],
        replay_slots: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The core of N requests of ``num_steps`` verify tokens each, by the cache manager's speculative state:
        the per-draft records (``kda_state_tok``) go with the K3 kernels; the replay caches alone with
        ``ssm/kda_mtp_decode``; the sequential-verify buffers with the token-by-token FLA verify.

        ``replay_slots`` are the requests' rows of the replay caches (``slots`` when None). The K3 kernels address the
        replay caches by the state slot, so with the per-draft records the manager keeps them in the state slot
        (``KDAReplayState.replay_shares_state_slot``)."""
        replay = bool(getattr(layer_cache, "has_kda_replay_caches", False))
        if replay and layer_cache.kda_state_tok is not None:
            assert self.takes_k3_kernels and self.k3_buffers is not None, (
                f"Kimi K3 KDA layer {self.layer_idx}: the cache manager keeps the per-token verify states, which "
                "only the decode kernels write, and this layer has not built its decode weights or has no buffers"
            )
            assert replay_slots is None or replay_slots.data_ptr() == slots.data_ptr(), (
                f"Kimi K3 KDA layer {self.layer_idx}: the decode kernels address the replay caches by the state slot"
            )
            core = self._k3_verify(x2d, num_steps, layer_cache, layer_cache.temporal, slots)
            return self._store_core(core, output)
        if replay:
            return self._verify_replay(x2d, num_steps, layer_cache, slots, output, replay_slots)
        return self._verify_sequential(x2d, num_steps, layer_cache, slots, output)

    def _project_verify(self, x: torch.Tensor, num_rows: int):
        """The fused ``[q | k | v | g]`` rows and ``(beta, f_b gate)`` of ``x`` [N, T, hidden], the gate chain on
        the auxiliary stream for up to ``_MULTISTREAM_MAX_ROWS`` rows."""
        aux_stream = self._projection_aux_stream if 0 < num_rows <= _MULTISTREAM_MAX_ROWS else None
        return run_in_parallel(
            lambda: F.linear(x, self._qkvg_proj_weight),
            lambda: self._project_gate_inputs(x),
            self._projection_fork_event,
            self._projection_join_event,
            aux_stream,
        )

    def _verify_replay(
        self, x2d, num_steps, layer_cache, slots, output, replay_slots=None
    ) -> torch.Tensor:
        """``ssm/kda_mtp_decode``: each request replays its accepted drafts of the last round from the replay caches,
        then runs its new tokens; the state and conv window commit after the golden token, the drafts are cached.
        The state pool is addressed by ``slots``, the replay caches and the accepted counts by ``replay_slots``
        (``slots`` when None)."""
        num_requests = x2d.shape[0] // num_steps
        heads, hd, d = self.num_heads, self.head_dim, self.proj_size
        rows = num_requests * num_steps
        x = x2d.view(num_requests, num_steps, -1)
        qkvg, (beta, g) = self._project_verify(x, rows)
        q, k, v = qkvg[..., : 3 * d].split(d, dim=-1)
        w_q, w_k, w_v = self._conv_weights_f32
        out = kda_mtp_decode(
            q.view(1, rows, heads, hd),
            k.view(1, rows, heads, hd),
            v.view(1, rows, heads, hd),
            w_q,
            w_k,
            w_v,
            layer_cache.kda_conv_q,
            layer_cache.kda_conv_k,
            layer_cache.kda_conv_v,
            g.view(1, rows, heads, hd),
            beta.view(1, rows, heads),
            self.A_log.detach(),
            self.dt_bias.detach(),
            layer_cache.temporal,
            layer_cache.kda_k_cache,
            layer_cache.kda_g_cache,
            layer_cache.kda_v_cache,
            layer_cache.kda_beta_cache,
            slots,
            None,
            num_steps - 1,
            layer_cache.prev_num_accepted_tokens,
            self.gate_lower_bound,
            self.head_k_dim**-0.5,
            True,
            qkvg[..., 3 * d : 4 * d].view(1, rows, heads, hd),
            self._onorm_w_f32,
            self.o_norm.eps,
            True,
            replay_state_indices=replay_slots,
        )
        return self._store_core(out, output)

    def _verify_sequential(self, x2d, num_steps, layer_cache, slots, output) -> torch.Tensor:
        """The token-by-token FLA verify (more drafts than ``ssm/kda_mtp_decode`` takes): the live pools are read
        only; each step's conv window and state go to the manager's sequential-verify buffers (row n for request
        n), which the manager promotes after sampling."""
        conv_steps = layer_cache.intermediate_conv_window
        state_steps = layer_cache.intermediate_ssm
        assert conv_steps is not None and state_steps is not None, (
            "speculative verification requires the cache manager's sequential-verify buffers"
        )
        num_requests = x2d.shape[0] // num_steps
        heads, hd, d = self.num_heads, self.head_dim, self.proj_size
        x = x2d.view(num_requests, num_steps, -1)
        qkvg, (beta, g) = self._project_verify(x, x2d.shape[0])
        q_all, k_all, v_all = (t.contiguous() for t in qkvg[..., : 3 * d].split(d, dim=-1))
        onorm_g = qkvg[..., 3 * d : 4 * d].contiguous()
        beta = beta.float()
        g = g.unflatten(-1, (heads, hd))
        slots_long = slots.long()
        windows = layer_cache.conv.index_select(0, slots_long)
        conv_q, conv_k, conv_v = (
            F.pad(windows[:, i * d : (i + 1) * d].contiguous(), (1, 0)) for i in range(3)
        )
        state = layer_cache.temporal.index_select(0, slots_long).float()
        taps = [conv.weight.squeeze(1) for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d)]
        outs = []
        for t in range(num_steps):
            q_t, conv_q = short_conv_update(q_all[:, t : t + 1], conv_q, taps[0], "silu")
            k_t, conv_k = short_conv_update(k_all[:, t : t + 1], conv_k, taps[1], "silu")
            v_t, conv_v = short_conv_update(v_all[:, t : t + 1], conv_v, taps[2], "silu")
            o_t, state = fused_recurrent_kda(
                q_t.unflatten(-1, (heads, hd)),
                k_t.unflatten(-1, (heads, hd)),
                v_t.unflatten(-1, (heads, hd)),
                g[:, t : t + 1],
                beta[:, t : t + 1],
                self.A_log,
                self.dt_bias,
                state,
                self.gate_lower_bound,
            )
            outs.append(o_t)
            conv_steps[:num_requests, t] = torch.cat(
                [conv_q[:, :, 1:], conv_k[:, :, 1:], conv_v[:, :, 1:]], dim=1
            ).to(conv_steps.dtype)
            state_steps[:num_requests, t] = state.to(state_steps.dtype)
        core = self._output_gate(torch.cat(outs, dim=1), onorm_g)
        return self._store_core(core, output)

    # ------------------------------------------------------------------------------------------------------------------
    # The decode kernels
    # ------------------------------------------------------------------------------------------------------------------

    def _k3_decode(self, x: torch.Tensor, attn_metadata: AttentionMetadata) -> torch.Tensor:
        """``ssm/k3_kda_decode_attn``: the core output ``[R, H, 128]`` of one token of each of the step's R requests;
        each slot's conv window and state advance in place."""
        mamba_metadata = attn_metadata.mamba_metadata
        slots = getattr(mamba_metadata, "generation_state_indices", None)
        if slots is None:
            slots = mamba_metadata.state_indices[: x.shape[0]]
        # The replay caches' slots, which differ from the state slots when the manager keeps them in a pool of their
        # own.
        replay_slots = getattr(mamba_metadata, "generation_replay_state_indices", None)
        if replay_slots is None:
            replay_slots = mamba_metadata.replay_state_indices[: x.shape[0]]
        layer_cache = attn_metadata.kv_cache_manager.mamba_layer_cache(self.layer_idx)
        w_q, w_k, w_v = self._conv_weights_f32
        core = k3_kda_decode_attn(
            x.contiguous(),
            self.k3_proj_weight,
            self.f_b_proj.weight,
            w_q,
            w_k,
            w_v,
            self._A_log_f32,
            self._dt_bias_f32,
            self._onorm_w_f32,
            layer_cache.conv,
            layer_cache.temporal,
            slots,
            self.k3_buffers,
            float(self.gate_lower_bound),
            self.head_k_dim**-0.5,
            float(self.o_norm.eps),
        )
        self._commit_conv_window(layer_cache, slots, layer_cache.conv, replay_slots)
        return core

    def _k3_verify(self, x, num_steps, layer_cache, ssm_pool, slot_indices) -> torch.Tensor:
        """The core output ``[N T, H, 128]`` of N requests of T verify tokens: ``ssm/k3_kda_attn`` for one request of
        8 tokens, else ``ssm/k3_kda_verify`` on the projection's rows (the ``kda_proj`` decode GEMV site where it
        takes them). Each slot's state after the golden token, its drafts' states and its conv window are written in
        place."""
        num_requests = x.shape[0] // num_steps
        slots = slot_indices[:num_requests]
        w_q, w_k, w_v = self._conv_weights_f32
        constants = (float(self.gate_lower_bound), self.head_k_dim**-0.5, float(self.o_norm.eps))
        if num_requests == 1 and num_steps == 8:
            out = k3_kda_attn(
                x.contiguous(),
                self.k3_proj_weight,
                self.f_b_proj.weight,
                w_q,
                w_k,
                w_v,
                self._A_log_f32,
                self._dt_bias_f32,
                self._onorm_w_f32,
                layer_cache.kda_conv_q,
                layer_cache.kda_conv_k,
                layer_cache.kda_conv_v,
                ssm_pool,
                layer_cache.kda_state_tok,
                slots,
                layer_cache.prev_num_accepted_tokens,
                self.k3_buffers,
                num_steps - 1,
                *constants,
            )
        else:
            rows = None
            if self.decode_gemvs is not None:
                rows = self.decode_gemvs.project("kda_proj", x, self.k3_proj_weight)
            if rows is None:
                rows = F.linear(x, self.k3_proj_weight)
            out = k3_kda_verify(
                rows,
                self.f_b_proj.weight,
                w_q,
                w_k,
                w_v,
                self._A_log_f32,
                self._dt_bias_f32,
                self._onorm_w_f32,
                layer_cache.kda_conv_q,
                layer_cache.kda_conv_k,
                layer_cache.kda_conv_v,
                ssm_pool,
                layer_cache.kda_state_tok,
                slots,
                layer_cache.prev_num_accepted_tokens,
                num_steps - 1,
                *constants,
            )
        return out.view(-1, self.num_heads, self.head_dim)

    # ------------------------------------------------------------------------------------------------------------------
    # Shared pieces
    # ------------------------------------------------------------------------------------------------------------------

    def _project_gate_inputs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(beta logits, f_b gate)`` from the fused ``[f_a | b]`` GEMM and the ``f_b`` GEMM."""
        hd = self.head_dim
        bfa = F.linear(x, self._bfa_proj_weight)
        return bfa[..., hd : hd + self.num_heads], F.linear(bfa[..., :hd], self.f_b_proj.weight)

    def _output_gate(self, o: torch.Tensor, onorm_g: torch.Tensor) -> torch.Tensor:
        """``rmsnorm(o) * weight * sigmoid(onorm_g)`` per head row (``norm/rms_norm_gated_token_major``)."""
        return rms_norm_gated_token_major(
            o.reshape(-1, self.head_dim),
            onorm_g.reshape(-1, self.num_heads, self.head_dim),
            self.o_norm.weight,
            self.o_norm.eps,
            None,
            "sigmoid",
        )

    def _store_core(self, core: torch.Tensor, output: Optional[torch.Tensor]) -> torch.Tensor:
        """``core`` as ``[N, H, 128]``, copied into ``output`` (a breakable CUDA graph's buffer) when given."""
        core = core.reshape(-1, self.num_heads, self.head_dim)
        if output is None:
            return core
        output.copy_(core)
        return output

    @staticmethod
    def _commit_conv_window(
        layer_cache,
        slots: torch.Tensor,
        conv_pool: torch.Tensor,
        replay_slots: Optional[torch.Tensor] = None,
    ) -> None:
        """Keep the replay caches' committed conv window (speculative decoding) equal to the live pool's after a
        prefill or plain decode advanced it; without it the next verify convolves a stale history, silently. Pool row
        ``slots[n]`` goes to replay row ``replay_slots[n]`` (``slots[n]`` when None)."""
        if bool(getattr(layer_cache, "has_kda_replay_caches", False)):
            layer_cache.commit_conv_window(slots, conv_pool, replay_slots)
