# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash block-FP8 on sm_100 / sm_103, tp4: model construction and forward.

The checkpoint is the composite ``Glm5NextForConditionalGeneration`` release, served text
only (``disable_mm_encoder``). The engine builds ``ModelingV2Glm53FlashFp8Sm10xTp4``, a
``DecoderModelForCausalLM`` shell around ``Glm53FlashFp8Sm10xTp4Core``; one-model MTP-3
drafts with the checkpoint's MTP layer through the engine's speculative worker.

Each step runs one of two targets:

- ``PrefillTarget`` for steps with context rows, sequence parallel: rank ``r`` owns rows
  ``[r * n, (r + 1) * n)`` of the packed step (``n = ceil(T / 4)``) and keeps the mHC
  streams of those rows only, so every boundary (post-mapping, pre-mapping, RMSNorm) runs
  over a quarter of the rows. Each sublayer is all-gather, the sublayer over all rows,
  reduce-scatter: the same bytes as the all-reduce it replaces. KDA, the MoE and the dense
  MLPs produce head- or intermediate-split partials as under TP. The MLA layers attend
  their context rows sequence parallel (all 64 heads, the owned rows only, zero for the
  others, so the reduce-scatter leaves each owned row's full value); generation rows of a
  mixed step stay head-parallel.
- ``DecodeTarget`` for generation-only steps (the steps CUDA graphs capture): the decoder
  over every row under TP, with the TP all-reduce of a step of at most four rows fused into
  the mHC boundary after each sublayer.

Every layer's computation is this target's: ``layers/`` holds the KDA, sparse MLA, FFN and
mHC layers on catalog and target operations, ``weights.py`` reads the checkpoint, and
``schedule.py`` derives the step's row schedules from the engine's prepared metadata.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import (
    disable_v2_by_default,
    step_contract_enabled,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.allgather import allgather
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.allreduce import (
    allreduce,
    allreduce_group_args,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.reducescatter import reducescatter
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp8_quantize_1x128 import (
    fp8_quantize_1x128,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.distributed.ops import get_allreduce_workspace
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.speculative import get_spec_worker, should_use_separate_draft_kv_cache
from tensorrt_llm._utils import mpi_disabled
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceStrategy
from tensorrt_llm.logger import logger

from . import weights as _weights
from .dims import Dims
from .layers.ffn import DenseMlp, MoeLayer
from .layers.kda import KdaLayer
from .layers.mhc import Mhc
from .layers.mla import MlaLayer, SparseAttentionScratch
from .ops.comm.mhc_ar_boundary import WORLD as _FUSED_BOUNDARY_WORLD
from .ops.comm.mhc_ar_boundary import mhc_ar_boundary
from .ops.gemm.low_m_bf16_gemm import low_m_bf16_gemm
from .ops.moe.decode_moe import MAX_ROWS as _DECODE_MOE_ROWS
from .ops.moe.prefill_moe import PrefillMoeWorkspace
from .schedule import DENSE_MLP, LINEAR_ATTENTION, StepSchedule

# The deployment this target is written for; route() selects it on the same facts, and
# the constructor states them again for a direct construction.
_SMS = ((10, 0), (10, 3))
_TP = 4

#: The TP all-reduce of more rows than this runs on NCCL; fewer on the one-shot kernel,
#: whose workspace the fused boundary shares.
_ONESHOT_MAX_ROWS = 512
#: Head products of at most this many rows run on ``low_m_bf16_gemm``.
_LOW_M_ROWS = 32
_AR_NONE = int(AllReduceFusionOp.NONE)

#: The prepared hybrid-cache metadata fields the KDA layers read (``StepSchedule.build`` reads
#: the GLM ones).
_HYBRID_FIELDS = (
    "state_indices",
    "generation_state_indices",
    "has_initial_states",
    "query_start_loc",
    "query_start_loc_long",
    "kda_max_seqlen",
    "kda_seq_order",
    "kda_cu_chunks",
)
#: The KV cache manager members the layers use: the MLA layers' latent and indexer pools and
#: the KDA layers' state.
_CACHE_MANAGER_MEMBERS = (
    "tokens_per_block",
    "get_latent_state_buffer",
    "get_index_state_buffer",
    "mamba_layer_cache",
)
#: A KDA layer cache's fused-verify replay members, read with MTP.
_KDA_REPLAY_FIELDS = (
    "kda_conv_q",
    "kda_conv_k",
    "kda_conv_v",
    "kda_k_cache",
    "kda_g_cache",
    "kda_v_cache",
    "kda_beta_cache",
    "prev_num_accepted_tokens",
    "commit_conv_window",
)


class _Comm:
    """This rank's TP collectives: the all-reduce (one-shot up to ``_ONESHOT_MAX_ROWS`` rows,
    NCCL above), and the all-gather / reduce-scatter of sequence-parallel prefill."""

    def __init__(self, mapping) -> None:
        self.tp_size = mapping.tp_size
        self.tp_rank = mapping.tp_rank
        self.group = list(mapping.tp_group)
        self.process_group = mapping.tp_group_pg.boxed() if mpi_disabled() else None
        self.small = allreduce_group_args(mapping, int(AllReduceStrategy.ONESHOT))
        self.large = allreduce_group_args(mapping, int(AllReduceStrategy.NCCL))
        # The one-shot all-reduce's Lamport workspace, which the fused boundary also rotates.
        self.workspace = get_allreduce_workspace(mapping)

    def allreduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return x
        args = self.small if x.shape[0] <= _ONESHOT_MAX_ROWS else self.large
        return allreduce(
            x.contiguous(),
            None,
            None,
            None,
            None,
            op=_AR_NONE,
            eps=0.0,
            trigger_completion_at_end=False,
            **args,
        )

    def gather(self, local: torch.Tensor, num_rows: int) -> torch.Tensor:
        """Every rank's rows ``local`` stacked in rank order, cut to ``num_rows``."""
        return allgather(local, None, self.group, self.process_group)[:num_rows]

    def scatter(self, partial: torch.Tensor, rows: int) -> torch.Tensor:
        """This rank's ``rows`` rows of the sum of every rank's ``partial``."""
        pad = rows * self.tp_size - partial.shape[0]
        if pad:
            partial = torch.nn.functional.pad(partial, (0, 0, 0, pad))
        return reducescatter(partial, None, self.group, self.process_group)

    def gather_vocab(self, local: torch.Tensor) -> torch.Tensor:
        """``[rows, vocab / tp]`` shards -> ``[rows, vocab]``, rank-major along the vocabulary."""
        if self.tp_size == 1:
            return local
        rows = local.shape[0]
        stacked = allgather(local.contiguous(), None, self.group, self.process_group)
        return stacked.view(self.tp_size, rows, -1).transpose(0, 1).reshape(rows, -1)


def _expand_streams(x: torch.Tensor, mult: int) -> torch.Tensor:
    """``[rows, hidden]`` -> ``[rows, mult, hidden]`` exact copies: the streams diverge at the
    first mixer. Contiguous, because post-mapping writes each stream separately."""
    return x.unsqueeze(-2).expand(-1, mult, -1).contiguous()


def _collapse_streams(streams: torch.Tensor, norm_weight: torch.Tensor, eps: float) -> torch.Tensor:
    """The decoder's output: the streams' unweighted mean (GLM-5.3-Flash's stack ends in a plain
    mean, not a learned head), then the final RMSNorm."""
    return flashinfer_rmsnorm(streams.mean(dim=-2), norm_weight, eps)


class _Layer:
    """One decoder layer's parts: the attention (KDA or MLA), the FFN, the two mHC sites and
    the two RMSNorm weights."""

    def __init__(self, index, kind, attention, mlp, hc_attn, hc_ffn, input_norm, post_norm):
        self.index = index
        self.kind = kind
        self.attention = attention
        self.mlp = mlp
        self.hc_attn = hc_attn
        self.hc_ffn = hc_ffn
        self.input_norm = input_norm
        self.post_norm = post_norm

    @property
    def is_kda(self) -> bool:
        return self.kind == LINEAR_ATTENTION


class _Mtp:
    """The MTP layer: front end, sparse MLA, MoE, head norm; plain residuals, no mHC."""

    def __init__(self, w: _weights.MtpLayerWeights, attention: MlaLayer, mlp: MoeLayer) -> None:
        self.w = w
        self.attention = attention
        self.mlp = mlp


class PrefillTarget:
    """Steps with context rows, sequence parallel over the step's rows (see the module
    docstring)."""

    def __init__(self, core: "Glm53FlashFp8Sm10xTp4Core") -> None:
        self.core = core
        self.comm = core.comm
        self.moe_workspace = PrefillMoeWorkspace()

    def forward(self, step: StepSchedule, input_ids: torch.Tensor, attn_metadata) -> torch.Tensor:
        core = self.core
        comm = self.comm
        dims = core.dims
        eps = dims.rms_eps
        embeds = torch.nn.functional.embedding(input_ids, core.w.embed)
        num_tokens = embeds.shape[0]
        rows = -(-num_tokens // comm.tp_size)
        first_row = comm.tp_rank * rows
        own = embeds[first_row : first_row + rows]
        if own.shape[0] < rows:
            own = torch.nn.functional.pad(own, (0, 0, 0, rows - own.shape[0]))

        layers = core.layers
        residual = _expand_streams(own, dims.hc_mult)
        post, comb, x = layers[0].hc_attn.pre_mapping(residual)
        x = flashinfer_rmsnorm(x, layers[0].input_norm, eps)
        for i, layer in enumerate(layers):
            full = comm.gather(x, num_tokens)
            if layer.is_kda:
                partial = layer.attention.partial(full, step, attn_metadata)
            else:
                partial = self._mla_partial(layer.attention, full, step, rows)
            del full
            residual, post, comb, x = layer.hc_ffn.fused_hc(
                comm.scatter(partial, rows), residual, post, comb, layer.post_norm, eps
            )
            del partial
            mlp_out = comm.scatter(self._ffn_partial(layer.mlp, x, num_tokens), rows)
            if i + 1 < len(layers):
                nxt = layers[i + 1]
                residual, post, comb, x = nxt.hc_attn.fused_hc(
                    mlp_out, residual, post, comb, nxt.input_norm, eps
                )
            else:
                residual = layer.hc_ffn.post_mapping(mlp_out, residual, post, comb)
            del mlp_out
        return comm.gather(_collapse_streams(residual, core.w.final_norm, eps), num_tokens)

    def mtp_forward(
        self, step: StepSchedule, input_ids: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """The MTP layer over a step with context rows, sequence parallel like the decoder:
        its row-parallel eh_proj reduce-scatters to the owned rows, the attention and the FFN
        are all-gather, sublayer, reduce-scatter, and the head-normalized output is gathered
        back to every row."""
        core = self.core
        comm = self.comm
        mtp = core.mtp
        eps = core.dims.rms_eps
        num_tokens = hidden_states.shape[0]
        rows = -(-num_tokens // comm.tp_size)
        own_cols = _mtp_own_columns(core, input_ids, hidden_states)
        residual = comm.scatter(torch.nn.functional.linear(own_cols, mtp.w.eh_proj), rows)
        full = comm.gather(flashinfer_rmsnorm(residual, mtp.w.input_norm, eps), num_tokens)
        partial = self._mla_partial(mtp.attention, full, step, rows)
        del full
        residual = residual + comm.scatter(partial, rows)
        x = flashinfer_rmsnorm(residual, mtp.w.post_norm, eps)
        residual = residual + comm.scatter(self._ffn_partial(mtp.mlp, x, num_tokens), rows)
        return comm.gather(flashinfer_rmsnorm(residual, mtp.w.head_norm, eps), num_tokens)

    def _ffn_partial(self, mlp, x: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """The FFN's intermediate-split output for every row, from this rank's rows ``x``.

        Every FFN GEMM input is the 1 x 128 block-FP8 quantization of the row, a per-row
        function: each rank quantizes its own rows (and runs the router on them), and the
        gather moves FP8 rows, their scales and the router logits instead of BF16 rows every
        rank would quantize again.
        """
        comm = self.comm
        x_fp8, x_scale = fp8_quantize_1x128(x)
        x_fp8 = comm.gather(x_fp8.view(torch.uint8), num_tokens).view(torch.float8_e4m3fn)
        # Gathered row-major, then returned to the K-block-major layout the GEMMs read.
        x_scale = comm.gather(x_scale.t().contiguous(), num_tokens).t().contiguous()
        if isinstance(mlp, MoeLayer):
            logits = comm.gather(mlp.router_logits(x), num_tokens)
            return mlp.prefill_partial(x_fp8, x_scale, logits, self.moe_workspace)
        return mlp.partial_from_fp8(x_fp8, x_scale)

    def _mla_partial(
        self, attention: MlaLayer, x: torch.Tensor, step: StepSchedule, rows: int
    ) -> torch.Tensor:
        """This rank's contribution to the MLA output of every row: the full value of its own
        context rows, zero for the other context rows, and the head-shard partial of the
        generation rows."""
        out = x.new_zeros(x.shape)
        num_ctx = step.num_ctx_tokens
        if step.num_contexts:
            lo = min(self.comm.tp_rank * rows, num_ctx)
            hi = min(lo + rows, num_ctx)
            attention.context_rows(x[:num_ctx], step, lo, hi, out)
        if step.num_generations:
            out[num_ctx:] = attention.generation_partial(x[num_ctx:], step)
        return out


class DecodeTarget:
    """Generation-only steps under TP: every row is a generation (or verify) row. A step of at
    most ``_DECODE_MOE_ROWS`` rows ends each sublayer unreduced and fuses the TP all-reduce
    into the mHC boundary after it."""

    def __init__(self, core: "Glm53FlashFp8Sm10xTp4Core") -> None:
        self.core = core
        self.comm = core.comm

    def forward(self, step: StepSchedule, input_ids: torch.Tensor, attn_metadata) -> torch.Tensor:
        core = self.core
        comm = self.comm
        eps = core.dims.rms_eps
        layers = core.layers
        x = torch.nn.functional.embedding(input_ids, core.w.embed)
        fused = x.shape[0] <= _DECODE_MOE_ROWS and comm.tp_size == _FUSED_BOUNDARY_WORLD
        residual = _expand_streams(x, core.dims.hc_mult)
        post, comb, x = layers[0].hc_attn.pre_mapping(residual)
        x = flashinfer_rmsnorm(x, layers[0].input_norm, eps)
        for i, layer in enumerate(layers):
            if layer.is_kda:
                partial = layer.attention.partial(x, step, attn_metadata)
            else:
                partial = layer.attention.generation_partial(x, step)
            if fused:
                residual, post, comb, x = self._fused_boundary(
                    partial, residual, post, comb, layer.hc_ffn, layer.post_norm
                )
            else:
                residual, post, comb, x = layer.hc_ffn.fused_hc(
                    comm.allreduce(partial), residual, post, comb, layer.post_norm, eps
                )
            del partial
            last = i + 1 == len(layers)
            if fused and not last and isinstance(layer.mlp, MoeLayer):
                nxt = layers[i + 1]
                residual, post, comb, x = self._fused_boundary(
                    layer.mlp.partial(x), residual, post, comb, nxt.hc_attn, nxt.input_norm
                )
                continue
            mlp_out = comm.allreduce(layer.mlp.partial(x))
            if not last:
                nxt = layers[i + 1]
                residual, post, comb, x = nxt.hc_attn.fused_hc(
                    mlp_out, residual, post, comb, nxt.input_norm, eps
                )
            else:
                residual = layer.hc_ffn.post_mapping(mlp_out, residual, post, comb)
            del mlp_out
        return _collapse_streams(residual, core.w.final_norm, eps)

    def mtp_forward(
        self, step: StepSchedule, input_ids: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """The MTP layer over generation rows under TP: eh_proj's row-parallel partial, then
        attention and FFN with plain residuals, each reduced once."""
        core = self.core
        comm = self.comm
        mtp = core.mtp
        eps = core.dims.rms_eps
        own_cols = _mtp_own_columns(core, input_ids, hidden_states)
        residual = comm.allreduce(_linear(own_cols, mtp.w.eh_proj))
        x = flashinfer_rmsnorm(residual, mtp.w.input_norm, eps)
        residual = residual + comm.allreduce(mtp.attention.generation_partial(x, step))
        x = flashinfer_rmsnorm(residual, mtp.w.post_norm, eps)
        residual = residual + comm.allreduce(mtp.mlp.partial(x))
        return flashinfer_rmsnorm(residual, mtp.w.head_norm, eps)

    def _fused_boundary(self, partial, residual, post, comb, hc: Mhc, norm_weight):
        """What ``hc.fused_hc(allreduce(partial), residual, post, comb, norm_weight, eps)``
        returns, with the TP all-reduce of ``partial`` fused in."""
        rows, mult = residual.shape[0], hc.mult
        residual_out, post_out, comb_out, layer_input = mhc_ar_boundary(
            partial,
            self.comm.workspace,
            self.comm.tp_rank,
            residual.reshape(rows, mult, -1),
            post.reshape(rows, mult),
            comb.reshape(rows, mult, mult),
            hc.fn,
            hc.scale,
            hc.base,
            norm_weight,
            hc.sinkhorn_iters,
            hc.norm_eps,
            hc.eps,
            hc.sinkhorn_eps,
            hc.post_mult_value,
            self.core.dims.rms_eps,
        )
        return residual_out.view_as(residual), post_out.view(rows, mult), comb_out, layer_input


def _linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x @ weight.T`` for a BF16 weight: the low-M kernel for decode-sized steps."""
    rows = x.reshape(-1, x.shape[-1])
    if rows.shape[0] <= _LOW_M_ROWS:
        return low_m_bf16_gemm(rows.contiguous(), weight).view(*x.shape[:-1], -1)
    return torch.nn.functional.linear(x, weight)


def _mtp_own_columns(core, input_ids: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    """This rank's input columns of the MTP layer's row-parallel eh_proj:
    ``cat(enorm(embed(input_ids)), hnorm(hidden_states))[:, rank's quarter]``."""
    mtp = core.mtp
    eps = core.dims.rms_eps
    e = flashinfer_rmsnorm(torch.nn.functional.embedding(input_ids, core.w.embed), mtp.w.enorm, eps)
    h = flashinfer_rmsnorm(hidden_states, mtp.w.hnorm, eps)
    joined = torch.cat([e, h], dim=-1)
    tp, rank = core.comm.tp_size, core.comm.tp_rank
    return torch.chunk(joined, tp, dim=-1)[rank].contiguous()


class Glm53FlashFp8Sm10xTp4Core(DecoderModel):
    """The text decoder: this rank's weights, the layers built on them, and the per-step choice
    between the two targets."""

    def __init__(self, model_config: ModelConfig, text_config: PretrainedConfig) -> None:
        super().__init__(model_config)
        spec = model_config.spec_config
        mtp = spec is not None and spec.spec_dec_mode.is_mtp_one_model()
        self.text_config = text_config
        self.dims = Dims.from_config(
            text_config,
            model_config.mapping,
            mtp,
            model_config.moe_max_num_tokens or model_config.max_num_tokens,
        )
        self.w: Optional[_weights.ModelWeights] = None
        self.layers: list[_Layer] = []
        self.mtp: Optional[_Mtp] = None
        self.comm: Optional[_Comm] = None
        self._prefill: Optional[PrefillTarget] = None
        self._decode: Optional[DecodeTarget] = None
        # Read once: TRTLLM_MODELING_V2_VALIDATE asks the first forward to check the metadata
        # surface this target reads (``_check_step_contract``).
        self._contract_pending = step_contract_enabled()

    def load(self, weights: Any, lm_head_weight: torch.Tensor) -> None:
        device = torch.device("cuda", torch.cuda.current_device())
        self.w = _weights.load_model_weights(weights, self.dims, lm_head_weight, device)

    @torch.no_grad()
    def post_load_weights(self) -> None:
        """Build the layers and the two targets once the weights are real. Reached more than
        once (the shell and the loader's module walk), so it builds once. Streams and the
        collectives' workspaces are made here, not at construction: the engine constructs the
        model on the meta device, and a stream made inside a graph capture is not capturable."""
        if self._decode is not None:
            return
        assert self.w is not None, "post_load_weights before load_weights"
        dims = self.dims
        mapping = self.model_config.mapping
        self.comm = _Comm(mapping)
        aux_stream = torch.cuda.Stream()
        events = (torch.cuda.Event(), torch.cuda.Event())
        scratch = SparseAttentionScratch(dims)
        self.layers = []
        for i, w in enumerate(self.w.layers):
            if w.attention_kind == LINEAR_ATTENTION:
                attention = KdaLayer(w.attention, dims, i, aux_stream, events)
            else:
                attention = MlaLayer(w.attention, dims, i, aux_stream, events, scratch)
            mlp = DenseMlp(w.mlp, dims, i) if w.mlp_kind == DENSE_MLP else MoeLayer(w.mlp, dims, i)
            self.layers.append(
                _Layer(
                    i,
                    w.attention_kind,
                    attention,
                    mlp,
                    Mhc(w.hc_attn, dims),
                    Mhc(w.hc_ffn, dims),
                    w.input_norm,
                    w.post_norm,
                )
            )
        if self.w.mtp is not None:
            m = self.w.mtp
            layer = dims.mtp_layer
            self.mtp = _Mtp(
                m,
                MlaLayer(m.attention, dims, layer, aux_stream, events, scratch),
                MoeLayer(m.mlp, dims, layer),
            )
        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)

    def select_target(self, attn_metadata) -> "PrefillTarget | DecodeTarget":
        """A step with any context row is the prefill target's (mixed steps included); a
        generation-only step is the decode target's. One host int read, so a captured decode
        graph sees a constant."""
        if self._decode is None:
            self.post_load_weights()
        return self._decode if attn_metadata.num_contexts == 0 else self._prefill

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert inputs_embeds is None, "this text-only target embeds the input ids itself"
        target = self.select_target(attn_metadata)
        if self._contract_pending:
            self._check_step_contract(attn_metadata)
        return target.forward(StepSchedule.build(attn_metadata), input_ids, attn_metadata)

    def _check_step_contract(self, attn_metadata) -> None:
        """Opt-in first-forward fail-fast (``TRTLLM_MODELING_V2_VALIDATE``): every field of the
        engine-prepared metadata and every KV cache manager member the layers read must exist,
        so an upstream rename fails here rather than mid-forward. They are fixed at engine
        construction, so one check per instance is sound."""
        try:
            StepSchedule.build(attn_metadata)
            for name in ("seq_lens_cuda", "padded_num_tokens"):
                getattr(attn_metadata, name)
            for name in _HYBRID_FIELDS:
                getattr(attn_metadata.mamba_metadata, name)
            manager = attn_metadata.kv_cache_manager
            for name in _CACHE_MANAGER_MEMBERS:
                getattr(manager, name)
            cache = manager.mamba_layer_cache(next(x.index for x in self.layers if x.is_kda))
            for name in ("conv", "temporal") + (_KDA_REPLAY_FIELDS if self.dims.mtp else ()):
                getattr(cache, name)
        except AttributeError as exc:
            raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
        self._contract_pending = False

    def mtp_forward(self, input_ids, hidden_states, attn_metadata) -> torch.Tensor:
        """The MTP layer on the metadata of this draft call: the worker rewrites the metadata
        in place between draft steps, so the target is chosen per call."""
        step = StepSchedule.build(attn_metadata)
        return self.select_target(attn_metadata).mtp_forward(step, input_ids, hidden_states)


class _VocabParallelHead:
    """The shell's vocabulary-parallel LM head on this target's GEMMs: decode-sized products on
    ``low_m_bf16_gemm``, the rest on cuBLAS, this rank's vocabulary shard or, with
    ``gather_output``, every rank's shards rank-major. The speculative worker and the logits
    processor call it as they call the shell's head, whose TP attributes it carries."""

    def __init__(self, lm_head: nn.Module, comm: _Comm) -> None:
        self.lm_head = lm_head
        self.comm = comm
        self.tp_mode = lm_head.tp_mode
        self.gather_output = lm_head.gather_output
        self.padding_size = getattr(lm_head, "padding_size", 0)

    @property
    def weight(self) -> torch.Tensor:
        return self.lm_head.weight

    def __call__(
        self,
        hidden_states: torch.Tensor,
        gather_output: Optional[bool] = None,
        is_spec_decoding_head: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        local = _linear(hidden_states, self.lm_head.weight)
        gather = self.gather_output if gather_output is None else gather_output
        return self.comm.gather_vocab(local) if gather else local


class MTPLayer:
    """The checkpoint's MTP layer as the speculative worker drives it: ``__call__`` runs the
    layer (whose output is already ``shared_head.norm``-ed, unlike DeepSeek's) and
    ``shared_head`` projects it, without another norm, to this rank's vocabulary shard of
    the draft logits; the worker's greedy draft sampler gathers the argmax across ranks."""

    def __init__(self, core: Glm53FlashFp8Sm10xTp4Core) -> None:
        self.core = core

    def __call__(
        self,
        embed_tokens: Any = None,
        all_rank_num_tokens: Any = None,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attn_metadata: Any = None,
        spec_metadata: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del embed_tokens, all_rank_num_tokens, position_ids, kwargs  # NoPE; own embedding
        hidden_states = self.core.mtp_forward(input_ids, hidden_states, attn_metadata)
        if spec_metadata is not None:
            spec_metadata.maybe_capture_hidden_states(0, hidden_states, None)
        return hidden_states

    def shared_head(
        self, hidden_states, lm_head, attn_metadata, return_context_logits: bool = False
    ) -> torch.Tensor:
        if not return_context_logits:
            last = torch.cumsum(attn_metadata.seq_lens_cuda, dim=0, dtype=torch.long) - 1
            hidden_states = hidden_states[last]
        return _linear(hidden_states, lm_head.weight)


class DraftModel:
    """The container the speculative worker reaches the drafter through. ``embed_tokens``
    resolves through the core on every read: the container is built before the weights are."""

    def __init__(self, core: Glm53FlashFp8Sm10xTp4Core, lm_head) -> None:
        self.core = core
        self.mtp_layers = [MTPLayer(core)]
        self.lm_head = lm_head

    @property
    def embed_tokens(self) -> torch.Tensor:
        return self.core.w.embed


def _text_config(pretrained_config: PretrainedConfig) -> PretrainedConfig:
    return getattr(pretrained_config, "text_config", None) or pretrained_config


def _normalize_top_config(config: PretrainedConfig, text: PretrainedConfig) -> None:
    """Expose on the composite config the text fields the engine reads off it (capacity
    planning, the speculative metadata and the LM head), keeping values already there."""
    for name in ("num_hidden_layers", "hidden_size", "vocab_size"):
        if getattr(config, name, None) is None:
            setattr(config, name, int(getattr(text, name)))
    if getattr(config, "torch_dtype", None) is None:
        config.torch_dtype = getattr(text, "torch_dtype", None) or torch.bfloat16
    if getattr(config, "num_nextn_predict_layers", None) is None:
        config.num_nextn_predict_layers = int(getattr(text, "num_nextn_predict_layers", 0) or 0)


@disable_v2_by_default
class ModelingV2Glm53FlashFp8Sm10xTp4(
    DecoderModelForCausalLM[Glm53FlashFp8Sm10xTp4Core, PretrainedConfig]
):
    """GLM-5.3-Flash block-FP8, sm_100 / sm_103, tp4, text only."""

    def __init__(self, model_config: ModelConfig, *args, **kwargs) -> None:
        mapping = model_config.mapping
        assert torch.cuda.get_device_capability() in _SMS, (
            f"this target runs on sm100 / sm103, not {torch.cuda.get_device_capability()}"
        )
        assert (
            mapping.world_size == _TP
            and mapping.tp_size == _TP
            and mapping.moe_tp_size == _TP
            and not mapping.enable_attention_dp
            and mapping.pp_size == 1
            and mapping.cp_size == 1
        ), f"this target is tp4 with the experts split by intermediate, not {mapping}"
        assert model_config.disable_mm_encoder, "this target serves the text model only"
        config = model_config.pretrained_config
        text = _text_config(config)
        assert not bool(getattr(text, "tie_word_embeddings", False)), (
            "the checkpoint's lm_head is its own weight, not the embedding"
        )
        _normalize_top_config(config, text)
        super().__init__(
            Glm53FlashFp8Sm10xTp4Core(model_config, text),
            config=model_config,
            hidden_size=int(text.hidden_size),
            vocab_size=int(text.vocab_size),
        )
        self.text_config = text
        # device='cpu' keeps the constant real when the engine constructs on the meta device.
        self._mm_token_ids = torch.tensor(
            [int(config.image_token_id), int(config.video_token_id)],
            dtype=torch.int32,
            device="cpu",
        )
        self.spec_config = model_config.spec_config
        self.draft_model = None
        self.spec_worker = None
        self._head: Optional[_VocabParallelHead] = None
        if self.model.dims.mtp:
            self.draft_model = DraftModel(self.model, self.lm_head)
            self.spec_worker = get_spec_worker(
                self.spec_config,
                model_config,
                mapping,
                use_separate_draft_kv_cache=should_use_separate_draft_kv_cache(self.spec_config),
            )
            self.spec_worker.set_draft_model(self.draft_model)
            self.spec_worker.set_target_model(self)
        logger.info(f"modeling_v2: {type(self).__name__} built")

    # -- engine contracts ----------------------------------------------------------------

    @property
    def mamba_metadata_cls(self) -> type:
        """The hybrid-cache metadata the engine prepares for this model: GLM block tables, KDA
        state indices and the context schedules ``schedule.StepSchedule`` reads."""
        from tensorrt_llm._torch.attention.backends.sparse.glm_kpool.metadata import (
            Glm5NextMamba2Metadata,
        )

        return Glm5NextMamba2Metadata

    @property
    def multimodal_token_ids(self) -> torch.Tensor:
        """The image and video placeholder ids, which this text-only target never embeds."""
        return self._mm_token_ids

    def infer_max_seq_len(self) -> int:
        """The text config's context limit; the model has no RoPE (NoPE MLA, KDA)."""
        return int(self.text_config.max_position_embeddings)

    def validate_request(self, request: Any) -> None:
        """Reject media: the vision tower is not built."""
        params = getattr(request, "multimodal_params", None) or getattr(
            getattr(request, "py_request", None), "multimodal_params", None
        )
        if params is not None and getattr(params, "multimodal_data", None):
            raise ValueError("this GLM-5.3-Flash deployment serves text only (disable_mm_encoder)")

    def set_guided_decoder(self, guided_decoder) -> bool:
        if self.spec_worker is not None:
            return self.spec_worker.set_guided_decoder(guided_decoder)
        return False

    # -- weights -------------------------------------------------------------------------

    def load_weights(self, weights: Any, *args: Any, **kwargs: Any) -> None:
        assert self.lm_head.weight.dtype == torch.bfloat16, (
            f"the checkpoint's lm_head is BF16, the shell built {self.lm_head.weight.dtype}"
        )
        self.model.load(weights, self.lm_head.weight)

    def post_load_weights(self) -> None:
        super().post_load_weights()
        self.model.post_load_weights()
        if self._head is None:
            self._head = _VocabParallelHead(self.lm_head, self.model.comm)

    # -- forward -------------------------------------------------------------------------

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        spec_metadata=None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ):
        """One engine step. ``resource_manager`` stays inside ``**kwargs``, and ``position_ids``
        reaches the worker in the engine's ``[1, T]`` shape (the worker squeezes it itself).

        A step padded for a prefill CUDA graph or attention-DP peers (``padded_num_tokens``,
        neither of which this deployment runs) is cut back to its ``num_tokens`` rows before
        the decoder, as the built-in model cuts its output."""
        multimodal = kwargs.get("multimodal_params")
        assert not multimodal, "this target serves text only"
        if attn_metadata.padded_num_tokens is not None:
            num_tokens = attn_metadata.num_tokens
            input_ids = input_ids[:num_tokens]
            if position_ids is not None:
                position_ids = position_ids[..., :num_tokens]
        hidden = self.model(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        head = self._head
        if self.spec_worker is None:
            return self.logits_processor.forward(hidden, head, attn_metadata, return_context_logits)
        logits = self.spec_worker.target_logits(
            hidden[spec_metadata.gather_ids],
            head,
            self.logits_processor,
            attn_metadata,
            spec_metadata,
            self.draft_model,
        )
        return self.spec_worker(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden,
            logits=logits,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=self.draft_model,
            resource_manager=kwargs.get("resource_manager"),
        )
