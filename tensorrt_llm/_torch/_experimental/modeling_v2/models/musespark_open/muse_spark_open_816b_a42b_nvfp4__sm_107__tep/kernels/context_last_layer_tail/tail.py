# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``context_last_layer_tail``: the last decoder layer of a sequence-parallel context step past
its K/V append, on the rows the LM head reads only.

A lever (``levers.LEVERS``): lossless-numeric, on only by its knob until its accuracy gate holds,
served on disaggregated sm107 TEP4 context servers, on top of ``context_sequence_parallel``.

Ownership. On a context step the output of the last decoder layer feeds nothing but the final
norm, and the final norm feeds nothing but the LM head (the DFlash drafter reads the capture
buffer, filled by the capture layers before the last one, and takes only the shape of the final
hidden states). With the one-engine DFlash worker the LM head reads the rows of
``spec_metadata.gather_ids``: the last row of every context request and every generation row
(``needed_rows``). The last layer's K and V, by contrast, are read by every later step of every
request, so its attention-input gather, its QKV projection, its prep and its K/V append still run
on every row of the step.

What runs instead. ``ContextSequenceParallel.forward`` hands the last layer to
``LastLayerTail.forward``, which runs the gather and the attention module as the decoder layer
does, with two changes:

* An eager break (``eager_on_graph``) right before the attention module arms the layer's
  ``TailPureContextFmha``, which sits in front of the library that serves the layer's pure
  context steps. An armed call runs ``thop.trtllm_gen_context_preprocess`` on every row (the
  K/V append, with the parameters the thop op's context phase and ``fmk_pure_context_fmha``
  use), then flashinfer's trtllm-gen decode FMHA on one query row per context request: the
  request's last row, over all of its keys, with that row's BF16 query, the generation step's
  operands. Its output rows go to their places in the layer's attention output; the other rows
  are not written. An unarmed call (any other caller of the layer) runs the library behind it.
* The attention module's output projection is an eager break (``LastLayerTail._tail``) that
  reads the step's metadata and runs, on the needed rows only, padded to a power of two with
  copies of the last one: the o_proj GEMM, one all-gather of its partial sums and of the needed
  rows' FP32 residual (each row from the rank that owns it), the sum of the partials in rank
  order, the post-attention join, the MoE (its replicated arm, the same as a decode step's), the
  post-FFN join and the final norm. It writes those rows into the step's output, which is zero
  elsewhere, and disarms the library.

Being eager, both breaks see the step's own lengths on every replay of a breakable prefill CUDA
graph; the graph segments around them keep the shapes they were captured with.

Depth cap. The decode FMHA reads every key of a request with one query row per request, so its
cost grows with the request's depth where the context FMHA's grows with the rows; past a few
hundred thousand keys the tail is slower than the layer it replaces. A step whose deepest context
request (its cached prefix plus this chunk) reaches ``max_depth`` keys runs the layer unchanged:
the first break leaves the library unarmed (the library behind it runs the FMHA on every row)
and the second runs the decoder layer's own post-attention path on every row (the output
projection and its reduce-scatter, the joins, the sequence-parallel MoE, the final norm and the
all-gather of the step's rows), with the same modules, arguments and shapes as the layer's
captured call. The decision is made per step in the first break, from the step's attention
metadata.

Numerics: not bitwise. Every K/V byte of every layer and every capture row are the built-in
path's (the layers before the last are unchanged, and the last layer's append is the thop op's
preprocessing). The LM-head rows differ in rounding: the decode FMHA reads the BF16 query where
the context FMHA reads it in E4M3, the o_proj and MoE GEMMs run another tile on a few rows, and
the partial sums are added in rank order where the reduce-scatter adds them in its ring order.

Refusals. Without one-engine DFlash speculative metadata the step runs the layer as before:
without it the LM head can read every row (context logits). With it, the runner hands the LM
head the ``gather_ids`` rows only, so context logits and prompt logprobs never read another
row. A last layer that is a DFlash capture layer also runs as before.
"""

from __future__ import annotations

import contextlib
import itertools
from typing import Iterator, List, Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch.attention.backends.fmha import flashinfer_trtllm_gen as _fi_gen
from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
from tensorrt_llm._torch.attention.backends.fmha.interface import FmhaPhase
from tensorrt_llm._torch.attention.backends.fmha.phased import FmhaParams, PhasedFmha
from tensorrt_llm._torch.attention.backends.fmha.utils import (
    get_attention_chunk_size,
    get_bmm1_scale,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionInputType
from tensorrt_llm._torch.distributed import allgather, reducescatter
from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE, get_env_enable_pdl
from tensorrt_llm._torch.models.modeling_musespark_open import (
    MuseSparkOpenGatedResidual,
    _pad_sequence_parallel_tokens,
)
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import eager_on_graph
from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata
from tensorrt_llm._torch.utils import get_model_extra_attrs
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.functional import AllReduceParams, AttentionMaskType
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantMode

#: The lever this module installs.
LEVER = "context_last_layer_tail"
#: The tensor-parallel size of the sequence-parallel path it runs on.
TP_SIZE = 4
#: Bytes of the decode FMHA's workspace (its multi-CTA partial results; at most
#: ``_DECODE_SM_SCALE_MAX`` x 128 rows x 136 B per SM, about 21 MB at 148 SMs).
DECODE_WORKSPACE_BYTES = 128 << 20
#: The page sizes flashinfer's trtllm-gen decode FMHA takes.
PAGE_SIZES = (16, 32, 64)


def needed_rows(context_lens: Sequence[int], num_ctx_tokens: int, num_tokens: int) -> List[int]:
    """The rows of a step the LM head reads: each context request's last row, then every
    generation row.

    Args:
        context_lens: the step's context requests' row counts, in batch order.
        num_ctx_tokens: the rows of the context requests (``sum(context_lens)``).
        num_tokens: all the step's rows; rows past ``num_ctx_tokens`` are generation rows.
    """
    rows = [end - 1 for end in itertools.accumulate(int(n) for n in context_lens)]
    return rows + list(range(int(num_ctx_tokens), int(num_tokens)))


def padded_count(count: int) -> int:
    """The row count a tail call runs: the next power of two at or above ``count`` (at least 1).
    Few distinct shapes keep the GEMM and MoE tactics the warm-up chose."""
    return 1 << max(0, int(count) - 1).bit_length()


def owner_plan(
    rows: Sequence[int], shard: int, rank: int
) -> Tuple[List[int], List[int], List[int]]:
    """Where each needed row's residual comes from.

    Rank ``r`` holds rows ``[r * shard, (r + 1) * shard)`` of the step's FP32 residual.

    Returns:
        ``(slots, sources, flat)``: the slots of ``rows`` this rank owns and their rows in its
        shard, and for every slot the row of the all-gathered ``[tp * len(rows), H]`` buffer
        that holds it (``owner * len(rows) + slot``).
    """
    count = len(rows)
    slots, sources, flat = [], [], []
    for slot, row in enumerate(rows):
        owner = int(row) // shard
        flat.append(owner * count + slot)
        if owner == rank:
            slots.append(slot)
            sources.append(int(row) - rank * shard)
    return slots, sources, flat


def _host_index(values: Sequence[int], device: torch.device) -> torch.Tensor:
    """``values`` as a device int64 tensor, copied from pinned memory in stream order."""
    host = torch.tensor(list(values), dtype=torch.int64, pin_memory=prefer_pinned())
    return host.to(device, non_blocking=True)


def deepest_context_depth(attn_metadata) -> int:
    """The keys of the step's deepest context request after this chunk: its cached prefix plus
    the chunk's rows (0 for a step without context requests).

    Reads the host lengths the attention op reads (``kv_lens_runtime``), else the cached counts
    plus ``seq_lens``. Context requests come first in a step's batch.
    """
    count = int(getattr(attn_metadata, "num_contexts", 0) or 0)
    if count <= 0:
        return 0
    kv_lens = getattr(attn_metadata, "kv_lens_runtime", None)
    if kv_lens is not None and len(kv_lens) >= count:
        return int(max(kv_lens[:count].tolist()))
    params = getattr(attn_metadata, "kv_cache_params", None)
    cached = getattr(params, "num_cached_tokens_per_seq", None)
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    if cached is None or seq_lens is None:
        return 0
    lens = seq_lens[:count].tolist()
    return int(max(int(c) + int(n) for c, n in zip(cached[:count], lens)))


@contextlib.contextmanager
def _sp_step_active(step) -> Iterator[None]:
    """``step.active`` set while the block runs, as the sequence-parallel loop sets it around
    the layers (its joins and MoE read it); ``step`` None is a no-op."""
    if step is None:
        yield
        return
    previous = step.active
    step.active = True
    try:
        yield
    finally:
        step.active = previous


def takes_pure_context(attn_metadata) -> bool:
    """Whether a step is a pure context step: context requests, no generation request, beam
    width 1. Both counts are FMHA cache key fields, so a cached selection never crosses this
    test."""
    return (
        getattr(attn_metadata, "num_contexts", 0) > 0
        and getattr(attn_metadata, "num_generations", 0) == 0
        and getattr(attn_metadata, "beam_width", 1) == 1
    )


def _preprocess_workspace_bytes(dtype, batch, num_tokens, num_heads, head_dim, rope_dim) -> int:
    """Bytes of the preprocess's context workspace layout (no trtllm-gen FMHA workspace)."""
    layout = thop.get_trtllm_gen_context_workspace_layout(
        dtype, batch, num_tokens, num_heads, head_dim, rope_dim, True, True, True
    )
    return int(layout["total_size"])


class TailState:
    """Shared by the layer's library and the tail: whether the next pure context call of the
    last layer's attention runs only the needed rows' FMHA."""

    def __init__(self) -> None:
        self.armed = False
        #: Calls of the library: needed rows only, declined to the library behind, unarmed.
        self.pruned_calls = 0
        self.declined_calls = 0
        self.passed_calls = 0
        #: Tail calls, and the needed rows they ran.
        self.tail_calls = 0
        self.tail_rows = 0
        #: Steps at or past the depth cap, which ran the layer unchanged.
        self.depth_declined_steps = 0


class TailPureContextFmha(PhasedFmha):
    """The last layer's pure context steps: K/V append on every row, FMHA on the needed rows.

    Sits in front of the library that serves the layer's pure context steps
    (``fmk_pure_context_fmha``'s, or the thop fallback) and is offered on the same steps, so the
    manager's cache key is unchanged. An unarmed call, and an armed call ``_decline`` refuses,
    runs that library with the manager's arguments.
    """

    supports_workspace_reclamation = True

    def __init__(self, attn, behind, fallback: FallbackFmha, state: TailState) -> None:
        PhasedFmha.__init__(self, attn)
        self._behind = behind
        self._fallback = fallback
        self._state = state
        self._call: Optional[Tuple] = None
        self._sm_count: Optional[int] = None
        self._kv_scale: Optional[float] = None
        self._workspace: Optional[torch.Tensor] = None
        self._counter: Optional[torch.Tensor] = None
        self._enable_pdl = get_env_enable_pdl()

    def _is_supported(self, q, k, v, metadata, forward_args, *, phase=None) -> bool:
        if phase != FmhaPhase.CONTEXT or not takes_pure_context(metadata):
            return False
        if forward_args.attention_input_type not in (
            AttentionInputType.mixed,
            AttentionInputType.context_only,
        ):
            return False
        if forward_args.output is None or forward_args.output_sf is not None:
            return False
        return self._fallback.is_supported(q, k, v, metadata, forward_args)

    def forward(self, q, k, v, metadata, forward_args) -> None:
        if not self._state.armed:
            self._state.passed_calls += 1
            self._behind.forward(q, k, v, metadata, forward_args)
            return
        self._call = (q, k, v, metadata, forward_args)
        try:
            super().forward(q, k, v, metadata, forward_args)
        finally:
            self._call = None

    def run_generation(self, params: FmhaParams) -> None:
        raise RuntimeError("TailPureContextFmha serves pure context steps only")

    def run_context(self, params: FmhaParams) -> None:
        self._grow_workspace(params)
        reason = self._decline(params)
        if reason is not None:
            self._state.declined_calls += 1
            logger.info_once(
                f"[musespark_open] {LEVER}: a pure context step's last-layer FMHA ran on every "
                f"row (layer {self.attn.layer_idx}): {reason}",
                key=f"musespark_open.{LEVER}.decline.{reason}",
            )
            self._behind.forward(*self._call)
            return
        self._run_needed_rows(params)
        self._state.pruned_calls += 1
        logger.info_once(
            f"[musespark_open] {LEVER}: the last layer's FMHA ran on the context requests' last "
            f"rows ({params.batch_size} of {params.num_tokens} rows, layer {self.attn.layer_idx})",
            key=f"musespark_open.{LEVER}.ran",
        )

    @staticmethod
    def _grow_workspace(params: FmhaParams) -> None:
        """Grow the attention workspace to the preprocess's layout in place, as the thop op
        grows it for its own call (the eager reclaimer can leave it empty at a step's start)."""
        qkv, workspace, attn = params.qkv_input, params.workspace, params.attn
        if qkv is None or workspace is None or torch.cuda.is_current_stream_capturing():
            return
        need = _preprocess_workspace_bytes(
            qkv.dtype,
            params.batch_size,
            params.num_tokens,
            attn.num_heads,
            attn.head_dim,
            attn.rope_params.dim,
        )
        if workspace.numel() * workspace.element_size() < need:
            workspace.resize_(-(-need // workspace.element_size()))

    def _decline(self, params: FmhaParams) -> Optional[str]:
        if torch.cuda.is_current_stream_capturing():
            return "a stream capture is in progress"
        attn, fwd, meta = params.attn, params.fwd, params.meta
        if params.is_cross or fwd.cross_kv is not None:
            return "cross attention"
        if AttentionMaskType(fwd.mask_type) != AttentionMaskType.causal:
            return "a mask other than causal"
        if fwd.attention_sinks is not None:
            return "attention sinks"
        if fwd.softmax_stats_tensor is not None:
            return "softmax statistics requested"
        if fwd.mrope_rotary_cos_sin is not None:
            return "M-RoPE"
        if get_attention_chunk_size(attn) != 0:
            return "chunked attention"
        if 0 < params.cyclic_attention_window_size < params.max_past_kv_length:
            return "a sliding window shorter than the context"
        if not meta.use_paged_context_fmha:
            return "no paged context FMHA"
        if params.tokens_per_block not in PAGE_SIZES:
            return f"a page size of {params.tokens_per_block}"
        qkv = params.qkv_input
        if qkv is None or qkv.dtype != torch.bfloat16 or qkv.dim() != 2 or qkv.stride(-1) != 1:
            return "no BF16 fused QKV input"
        out = params.output
        if out is None or out.dtype != torch.bfloat16 or not out.is_contiguous():
            return "an output that is not contiguous BF16"
        q_lens = meta.prompt_lens_cpu_runtime[: params.batch_size].tolist()
        kv_lens = meta.kv_lens_runtime[: params.batch_size].tolist()
        if (
            len(q_lens) != params.batch_size
            or sum(q_lens) != params.num_tokens
            or any(q <= 0 or k < q for q, k in zip(q_lens, kv_lens))
        ):
            return "host lengths that disagree with the call"
        need = _preprocess_workspace_bytes(
            qkv.dtype,
            params.batch_size,
            params.num_tokens,
            attn.num_heads,
            attn.head_dim,
            attn.rope_params.dim,
        )
        if params.workspace.numel() * params.workspace.element_size() < need:
            return "a workspace smaller than the preprocess needs"
        return None

    def _decode_buffers(self, device: torch.device, meta) -> Tuple[torch.Tensor, torch.Tensor]:
        """The decode FMHA's workspace and its multi-CTA counters (zeroed once; the kernel
        leaves them zero), allocated at the first armed call (eager, outside any capture)."""
        attn = self.attn
        if self._sm_count is None:
            self._sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        need = _fi_gen._get_multi_ctas_kv_counter_size(
            attn.num_heads,
            meta.max_num_sequences or meta.max_num_requests,
            self._sm_count * _fi_gen._DECODE_SM_SCALE_MAX,
        )
        counter = self._counter
        if counter is None or counter.numel() < need:
            self._counter = torch.zeros(need, dtype=torch.uint8, device=device)
        if self._workspace is None:
            self._workspace = torch.empty(DECODE_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        return self._workspace, self._counter

    def _kv_dequant_scale(self, fwd) -> float:
        """The cache's K/V dequantization scale, read once (a host read at the first call)."""
        if self._kv_scale is None:
            scale = fwd.kv_scale_quant_orig
            self._kv_scale = 1.0 if scale is None else float(scale.reshape(-1)[0].item())
        return self._kv_scale

    def _run_needed_rows(self, params: FmhaParams) -> None:
        attn, meta, fwd = params.attn, params.meta, params.fwd
        rope = attn.rope_params
        batch, num_tokens = params.batch_size, params.num_tokens
        heads, head_dim = attn.num_heads, attn.head_dim
        qkv = params.qkv_input
        device = qkv.device
        workspace, counter = self._decode_buffers(device, meta)
        # The K/V append of every row: the thop op's context preprocessing, as
        # ``fmk_pure_context_fmha`` runs it (its E4M3 query buffer is not read here).
        (_, kv_pool, block_tables, *_) = thop.trtllm_gen_context_preprocess(
            qkv_input=qkv,
            workspace=params.workspace,
            sequence_lengths=params.sequence_lengths,
            context_lengths=params.context_lengths,
            kv_cache_block_offsets=meta.kv_cache_block_offsets,
            host_kv_cache_pool_pointers=meta.host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=meta.host_kv_cache_pool_mapping,
            kv_scale_orig_quant=fwd.kv_scale_orig_quant,
            kv_scale_quant_orig=fwd.kv_scale_quant_orig,
            attention_output_orig_quant=fwd.out_scale,
            rotary_inv_freq=attn.rotary_inv_freq,
            rotary_cos_sin=attn.rotary_cos_sin,
            mrope_rotary_cos_sin=None,
            layer_idx=attn.local_layer_idx,
            num_heads=heads,
            num_kv_heads=attn.num_kv_heads,
            head_size=head_dim,
            tokens_per_block=params.tokens_per_block,
            mask_type=fwd.mask_type,
            kv_cache_quant_mode=attn.quant_mode,
            max_attention_window_size=params.max_attention_window_size,
            cyclic_attention_window_size=params.cyclic_attention_window_size,
            num_tokens=num_tokens,
            batch_size=batch,
            input_seq_length=params.input_seq_length,
            max_past_kv_length=params.max_past_kv_length,
            rotary_embedding_dim=rope.dim,
            rotary_embedding_base=rope.theta,
            rotary_embedding_scale_type=int(rope.scale_type),
            rotary_embedding_scale=rope.scale,
            rotary_embedding_max_positions=rope.max_positions,
            position_embedding_type=attn.position_embedding_type,
            bmm1_scale=get_bmm1_scale(attn),
            bmm2_scale=1.0,
            attention_chunk_size=0,
            fp8_context_fmha=True,
            paged_context_fmha=True,
            is_mla_enable=False,
            multi_processor_count=self._sm_count,
            total_num_blocks=params.total_num_blocks,
            kv_factor=params.kv_factor,
            need_build_kv_cache_metadata=True,
            cross_kv=None,
            cross_attention=False,
            skip_fmha_workspace=True,
        )
        q_lens = meta.prompt_lens_cpu_runtime[:batch].tolist()
        max_kv = int(max(meta.kv_lens_runtime[:batch].tolist()))
        last = _host_index(needed_rows(q_lens, num_tokens, num_tokens), device)
        query = (
            qkv.index_select(0, last)[:, : heads * head_dim]
            .reshape(batch, heads, head_dim)
            .contiguous()
        )
        out = torch.empty(batch, heads, head_dim, dtype=torch.bfloat16, device=device)
        kv_scale = self._kv_dequant_scale(fwd)
        workspace_bytes = workspace.numel() * workspace.element_size()
        _fi_gen._trtllm_gen_decode(
            _fi_gen._decode_sm_scale(
                batch, attn.num_kv_heads, head_dim, self._sm_count, workspace_bytes, -1
            ),
            query=query,
            block_tables=block_tables[:batch],
            seq_lens=params.sequence_lengths[:batch],
            out=out,
            kv_cache=(kv_pool, kv_pool),
            workspace_buffer=workspace,
            max_seq_len=max_kv,
            bmm1_scale=get_bmm1_scale(attn) * kv_scale,
            bmm2_scale=kv_scale,
            window_left=-1,
            sinks=None,
            kv_layout=_fi_gen.FlashInferTrtllmGenFmha.DEFAULT_KV_LAYOUT,
            enable_pdl=self._enable_pdl,
            backend="trtllm-gen",
            q_len_per_req=1,
            max_q_len=None,
            cum_seq_lens_q=None,
            kv_cache_sf=None,
            uses_shared_paged_kv_idx=_fi_gen.FlashInferTrtllmGenFmha.USE_SHARED_PAGED_KV_IDX,
            bmm1_scale_log2=None,
            multi_ctas_kv_counter_buffer=counter,
        )
        params.output.view(num_tokens, heads, head_dim).index_copy_(0, last, out)


class LastLayerTail:
    """The last decoder layer of a sequence-parallel context step (module docstring).

    Args:
        max_depth: the depth cap: a step whose deepest context request reaches this many keys
            runs the layer unchanged; None runs the tail on every step.
        step: the sequence-parallel loop's ``SequenceParallelStep``, set active while a capped
            step runs the layer's post-attention path (None leaves it alone).
    """

    def __init__(
        self,
        model,
        library: TailPureContextFmha,
        state: TailState,
        max_depth: Optional[int] = None,
        step=None,
    ) -> None:
        self.model = model
        self.layer = model.layers[-1]
        self.mapping = model._sp_mapping
        self.library = library
        self.state = state
        self.max_depth = None if max_depth is None else int(max_depth)
        self.step = step
        #: Whether the step in flight is at or past the depth cap (set by ``_arm``).
        self._unchanged = False
        self._arm_op = eager_on_graph(self._arm)
        self._tail_op = eager_on_graph(self._tail)

    @staticmethod
    def _step_metadata():
        return get_model_extra_attrs()["attention_metadata"]()

    def _arm(self) -> None:
        """Arm the library for this step, unless the step is at or past the depth cap."""
        self._unchanged = False
        if self.max_depth is not None:
            depth = deepest_context_depth(self._step_metadata())
            if depth >= self.max_depth:
                self._unchanged = True
                self.state.depth_declined_steps += 1
                logger.info_once(
                    f"[musespark_open] {LEVER}: a step whose deepest context request reaches "
                    f"{depth} keys (cap {self.max_depth}) ran the last layer unchanged "
                    f"(layer {self.layer.layer_idx})",
                    key=f"musespark_open.{LEVER}.depth_cap",
                )
                return
        self.state.armed = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_input: torch.Tensor,
        attn_metadata,
        position_ids: Optional[torch.Tensor],
        rows: int,
        kwargs: dict,
    ) -> Optional[torch.Tensor]:
        """The step's final hidden states (``[rows, H]``, zero but on ``needed_rows``), or None
        when this step must run the last layer as the decoder layer does (refusals above).

        ``hidden_states`` is this rank's FP32 residual shard and ``attn_input`` its BF16
        attention input, both as the previous layer returns them.
        """
        layer = self.layer
        spec_metadata = kwargs.get("spec_metadata")
        if not isinstance(spec_metadata, DFlashSpecMetadata) or spec_metadata.is_layer_capture(
            layer.layer_idx
        ):
            return None
        gather = layer._sp_attn_input_gather
        if gather is None:
            full_input = allgather(attn_input, self.mapping, dim=0)[:rows]
        else:
            full_input = gather(attn_input, rows)
        hidden_size = hidden_states.shape[1]
        output = torch.zeros(
            (rows, hidden_size), dtype=self.model.dtype, device=hidden_states.device
        )
        self._arm_op()
        try:
            layer.self_attn(
                position_ids=position_ids,
                hidden_states=full_input,
                attn_metadata=attn_metadata,
                output_projection=lambda attention: self._tail_op(attention, hidden_states, output),
                **dict(kwargs, all_reduce_params=AllReduceParams(enable_allreduce=False)),
            )
        finally:
            self.state.armed = False
        return output

    def _tail(
        self, attention: torch.Tensor, hidden_states: torch.Tensor, output: torch.Tensor
    ) -> None:
        """o_proj, joins, MoE and final norm of the step's needed rows (every row on a step at or
        past the depth cap), written into ``output``."""
        try:
            if self._unchanged:
                self._run_layer(attention, hidden_states, output)
            else:
                self._run_tail(attention, hidden_states, output)
        finally:
            self.state.armed = False
            self._unchanged = False

    def _run_layer(
        self, attention: torch.Tensor, hidden_states: torch.Tensor, output: torch.Tensor
    ) -> None:
        """The decoder layer's own sequence-parallel path past its attention, on every row:
        ``MuseSparkOpenDecoderLayer.forward`` with ``sp_num_tokens`` and no next-layer norm,
        then the sequence-parallel forward's final norm and all-gather of the step's rows."""
        layer = self.layer
        attn = layer.self_attn
        rows = output.shape[0]
        metadata = self._step_metadata()
        with _sp_step_active(self.step):
            partial = attn.o_proj(
                attention,
                all_reduce_params=AllReduceParams(enable_allreduce=False),
                lora_params=None,
                layer_idx=attn.layer_idx,
            )
            attn_out = reducescatter(
                _pad_sequence_parallel_tokens(partial, layer.mapping.tp_size), layer.mapping, dim=0
            )
            hidden, ffn_input = layer.add_attn._add_after_rmsnorm_and_norm(
                hidden_states, attn_out, layer.post_attn_norm.variance_epsilon, layer.pre_ffn_norm
            )
            ffn_out = layer.feed_forward(ffn_input, metadata, sp_num_tokens=rows)
            hidden = layer.add_ffn.add_after_rmsnorm(
                hidden,
                ffn_out,
                layer.post_ffn_norm.variance_epsilon,
                layer.post_ffn_norm.gain(),
            )
        final = self.model.norm(hidden.to(self.model.dtype))
        output.copy_(allgather(final, self.mapping, dim=0)[:rows])

    def _run_tail(
        self, attention: torch.Tensor, hidden_states: torch.Tensor, output: torch.Tensor
    ) -> None:
        layer, mapping = self.layer, self.mapping
        metadata = self._step_metadata()
        num_contexts = int(metadata.num_contexts)
        rows = needed_rows(
            metadata.seq_lens[:num_contexts].tolist(),
            metadata.num_ctx_tokens,
            metadata.num_tokens,
        )
        if not rows:
            return
        count = padded_count(len(rows))
        padded = rows + [rows[-1]] * (count - len(rows))
        device = hidden_states.device
        index = _host_index(padded, device)
        slots, sources, flat = owner_plan(padded, hidden_states.shape[0], mapping.tp_rank)
        residual = torch.zeros(
            (count, hidden_states.shape[1]), dtype=hidden_states.dtype, device=device
        )
        if slots:
            residual.index_copy_(
                0,
                _host_index(slots, device),
                hidden_states.index_select(0, _host_index(sources, device)),
            )
        partial = layer.self_attn.o_proj(
            attention.index_select(0, index),
            all_reduce_params=AllReduceParams(enable_allreduce=False),
        )
        partials, residuals = allgather([partial, residual], mapping, dim=0)
        partials = partials.view(mapping.tp_size, count, -1)
        summed = partials[0].float()
        for rank in range(1, mapping.tp_size):
            summed = summed + partials[rank].float()
        attn_out = summed.to(partial.dtype)
        residual = residuals.index_select(0, _host_index(flat, device))
        hidden, ffn_input = MuseSparkOpenGatedResidual._add_after_rmsnorm_and_norm(
            layer.add_attn,
            residual,
            attn_out,
            layer.post_attn_norm.variance_epsilon,
            layer.pre_ffn_norm,
        )
        ffn_out = layer.feed_forward(ffn_input, metadata)
        hidden = MuseSparkOpenGatedResidual.add_after_rmsnorm(
            layer.add_ffn,
            hidden,
            ffn_out,
            layer.post_ffn_norm.variance_epsilon,
            layer.post_ffn_norm.gain(),
        )
        final = self.model.norm(hidden.to(self.model.dtype))
        output.index_copy_(0, index[: len(rows)], final[: len(rows)])
        self.state.tail_calls += 1
        self.state.tail_rows += len(rows)


def refusal(model) -> Optional[str]:
    """Why ``model``'s last layer cannot run the tail, or None. Changes nothing."""
    mapping = getattr(model, "_sp_mapping", None)
    if mapping is None or mapping.tp_size != TP_SIZE:
        return f"it needs the TP{TP_SIZE} sequence-parallel path"
    if not IS_FLASHINFER_AVAILABLE:
        return "flashinfer is not installed"
    missing = _fi_gen.FlashInferTrtllmGenFmha._missing_fused_nanobind_ops()
    if missing:
        return f"the binding lacks {', '.join(missing)}"
    attention = model.layers[-1].self_attn
    if attention.is_sliding:
        return "the last layer is a sliding-window layer"
    attn = attention.attn
    manager = getattr(attn, "_fmha_manager", None)
    if manager is None:
        return "the last layer's attention has no FMHA manager"
    if not any(isinstance(fmha, FallbackFmha) for fmha in manager.fmha_libs):
        return "no thop fallback among the last layer's FMHA libraries"
    if getattr(attn, "is_mla_enable", False):
        return "MLA"
    if not QuantMode(attn.quant_mode).has_fp8_kv_cache():
        return "a KV cache that is not FP8"
    if attn.num_heads % attn.num_kv_heads:
        return f"{attn.num_heads} query heads over {attn.num_kv_heads} KV heads"
    return None


def _pure_context_index(libs) -> int:
    """The index of the library serving a pure context step: ``fmk_pure_context_fmha``'s when
    installed (it sits right before the thop fallback), else the thop fallback."""
    from ..fmk_context_fmha.attention import FmkPureContextFmha

    for index, fmha in enumerate(libs):
        if isinstance(fmha, (FmkPureContextFmha, FallbackFmha)):
            return index
    raise ValueError("no library serves the last layer's pure context steps")


def install(model, max_depth: Optional[int] = None, step=None) -> LastLayerTail:
    """Put a ``TailPureContextFmha`` in front of the library serving the last layer's pure
    context steps and return the tail (``max_depth`` and ``step`` as ``LastLayerTail`` takes
    them). Run after ``fmk_pure_context_fmha``'s install and before any CUDA graph is captured.

    Raises:
        ValueError: when ``refusal`` names a reason.
    """
    reason = refusal(model)
    if reason is not None:
        raise ValueError(f"[musespark_open] {LEVER}: {reason}")
    attn = model.layers[-1].self_attn.attn
    libs = attn._fmha_manager.fmha_libs
    existing = next((fmha for fmha in libs if isinstance(fmha, TailPureContextFmha)), None)
    if existing is None:
        index = _pure_context_index(libs)
        fallback = next(fmha for fmha in libs if isinstance(fmha, FallbackFmha))
        existing = TailPureContextFmha(attn, libs[index], fallback, TailState())
        libs.insert(index, existing)
        attn._fmha_manager._cache.clear()
    cap = (
        "on every step"
        if max_depth is None
        else f"on steps whose deepest context request is below {int(max_depth)} keys"
    )
    logger.info(
        f"[musespark_open] {LEVER}: layer {model.layers[-1].layer_idx} of sequence-parallel "
        f"context steps runs its FMHA, o_proj, MoE and final norm on the LM-head rows only "
        f"(in front of {type(existing._behind).__name__}), {cap}"
    )
    return LastLayerTail(model, existing, existing._state, max_depth=max_depth, step=step)


__all__ = [
    "LEVER",
    "LastLayerTail",
    "TailPureContextFmha",
    "TailState",
    "deepest_context_depth",
    "install",
    "needed_rows",
    "owner_plan",
    "padded_count",
    "refusal",
    "takes_pure_context",
]
