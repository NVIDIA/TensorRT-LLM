# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast path ``context_sequence_parallel``: a context step's per-row work on a quarter of its rows.

What it replaces. ``PrefillTarget`` runs every step with context rows (a prefill
chunk, possibly with generation rows beside it) through the built-in text-model
forward, in which each rank of the TP4 group computes every row of the work that
is per row: the embedding norm, the two residual joins of every layer with their
norms, the MoE router (an FP32 projection behind an FP32 cast of its input), the
pool and unpool projections (replicated BF16 GEMMs) and the final norm. Only the
attention heads and the experts are divided between the ranks.

What runs instead. A step with context rows that ``ContextSequenceParallel.takes``
(below) runs the built-in decoder layers' sequence-parallel arm (``sp_num_tokens``). Rank ``r`` owns
rows ``[r * shard, (r + 1) * shard)`` of the step, the last shard padded with zero
rows, keeps the FP32 residual stream of those rows only and runs the per-row work on
them (the vocabulary-parallel token embedding still embeds every row). Each layer
all-gathers the attention input in front of the QKV projection and reduce-scatters the
output projection's partial sums in place of their all-reduce; its MoE quantizes the
pooled rows of the shard to NVFP4 and all-gathers them with the router logits in front
of the experts (``moe.ContextSequenceParallelMoE``, the built-in arm's expert input
bit for bit) and reduce-scatters the experts' partial sums in place of their
all-reduce. Attention and experts still see every row of the step, so the generation
rows of a mixed step are served as the context rows are. After the last layer the
final norm runs on the shard and the step's rows are all-gathered. On sm103
(``SHARD_PARTS_BY_SM``) each rank also routes its own rows and gathers their top-8 ids
and weights instead of the router logits (``moe.shard_routing``), and the attention-side
residual join writes the router's FP32 input next to the pre-FFN norm output
(``router_input``); both are bitwise. There the attention input of steps of at least
``ce_gather.MIN_ROWS`` rows is gathered by the copy engines and the QKV projection runs
in row blocks behind per-peer flags (``ce_gather``); also bitwise. On sm107 the router
logits and the attention input are all-gathered.

DFlash taps the residual stream after its capture layers, and that stream is
sharded here. For the forward the step's capture buffer
(``DFlashSpecMetadata.captured_hidden_states``) is swapped for a shard-height
buffer, which the layers' own capture calls fill; after the last layer it is
all-gathered together with the final hidden states and copied into the step
buffer's rows, so the drafter reads what it reads after the replicated forward.

Numerics: not bitwise against the replicated forward. The per-row kernels compute
the same values per row. The reductions run in another order (a reduce-scatter of
the output projection's and the experts' BF16 partials instead of their
all-reduce), and the router, pool and unpool GEMMs run on a quarter of the rows,
for which the GEMM library may choose another tactic and accumulation order.

Which steps. Every all-reduce of the replicated forward becomes an all-gather and a
reduce-scatter, and the eager forward's host side (the launches of 62 layers) grows by
those collectives' calls, while the work moved to a quarter of the rows grows with the
rows. So the path takes:

* a step captured into the breakable prefill CUDA graph (``is_in_breakable_cuda_graph``
  at capture) of at least ``MIN_ROWS`` rows. Replays carry no host cost; a step is padded
  up to the next captured bucket, so with buckets ``..., 768, 1024, ...`` every step of
  769 rows or more replays a sequence-parallel graph, generation rows beside the
  context rows included;
* an eager step of at least the device's eager threshold (``EAGER_MIN_ROWS_BY_SM``,
  ``EAGER_MIN_ROWS`` on sm103), where the GPU work saved exceeds the host time added.

Eager steps below that threshold (including the breakable graph's eager warmups of
its buckets, and every step when no prefill graph is configured), and every decode step,
run what they ran before.

The MNNVL all-reduce grows its workspace when a call needs more than it holds, and
a CUDA graph captured before the growth keeps addressing the replaced buffers. The
warmup step of ``max_num_tokens`` rows grows it before any graph is captured when it
runs replicated; here that step runs sequence-parallel, without the all-reduce, so
``ContextSequenceParallel.reserve`` grows every MNNVL workspace of the model to the
size that step asks for, on the first prefill step: after the KV cache is sized (as
the replicated warmup's growth) and before any CUDA graph is captured.
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, Iterator, Optional, Tuple

import torch

from tensorrt_llm._torch.distributed import allgather
from tensorrt_llm._torch.models.modeling_musespark_open import (
    MuseSparkOpenModel,
    _pad_sequence_parallel_tokens,
)
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import is_in_breakable_cuda_graph
from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata

#: The fewest rows a step captured into the breakable prefill CUDA graph needs to run
#: sequence-parallel: below it the two collectives' fixed cost per all-reduce replaced
#: outweighs the work moved to a quarter of the rows. Above every layer's fused
#: finalize all-reduce rows (``moe.install`` checks), which reduce their partial sums
#: themselves.
MIN_ROWS = 1024

#: The fewest rows an eager step needs to run sequence-parallel. The eager forward is
#: bound by its host launches up to about this many rows sequence-parallel (the
#: collectives add their own host launches to it), so between the graph's largest bucket
#: and this the replicated forward is faster; above it the sequence-parallel GPU time is
#: the shorter. Measured on sm103 with its Grace host.
EAGER_MIN_ROWS = 4608

#: ``EAGER_MIN_ROWS`` per SM version (``dispatch.sm_of``). On sm107 with its Vera host the
#: eager sequence-parallel forward is already the faster one at 3584 rows, the smallest
#: eager step above the default prefill graph buckets (3072 rows), so every eager step of
#: the largest bucket's rows or more takes the path there. Any other SM takes sm103's.
EAGER_MIN_ROWS_BY_SM = {103: EAGER_MIN_ROWS, 107: 3072}

#: The ranks the rows are split over: one tray's TP4 group, experts parallel over it.
TP_SIZE = 4


def refusal(model: MuseSparkOpenModel) -> Optional[str]:
    """Why this model's context steps cannot run sequence-parallel, or None. Changes nothing.

    The experts' partial sums are reduce-scattered, which needs every rank to hold
    a partial sum over its own experts (expert parallelism over the TP group) and
    no MoE communication of its own (no attention DP).
    """
    mapping = getattr(model, "_sp_mapping", None)
    if mapping is None:
        return "the model has no parallel mapping"
    if not (
        mapping.tp_size == TP_SIZE
        and mapping.moe_ep_size == TP_SIZE
        and mapping.moe_tp_size == 1
        and mapping.cp_size == 1
        and mapping.pp_size == 1
        and not mapping.enable_attention_dp
    ):
        return (
            f"it needs TP{TP_SIZE} with the experts parallel over the TP group, got tp "
            f"{mapping.tp_size}, moe_ep {mapping.moe_ep_size}, moe_tp {mapping.moe_tp_size}, cp "
            f"{mapping.cp_size}, pp {mapping.pp_size}, attention_dp {mapping.enable_attention_dp}"
        )
    if getattr(model, "dtype", None) != torch.bfloat16:
        return f"it needs a BF16 model, got {getattr(model, 'dtype', None)}"
    with_comm = [
        layer.layer_idx
        for layer in model.layers
        if getattr(layer.feed_forward.experts, "comm", None) is not None
    ]
    if with_comm:
        return f"the experts of layers {with_comm} run their own communication"
    if getattr(model, "_prefill_sp_enabled", False):
        return "the built-in prefill sequence parallelism is switched on as well"
    return None


#: The SM versions ``install`` wires, each with its own set of the path's parts
#: (``SHARD_PARTS_BY_SM``).
SHARD_PARTS_BY_SM = {
    # sm103: the MoE re-class with routing on the shard, the router's FP32 input from the
    # attention-side join, and the copy-engine gather of the attention input.
    103: ("moe", "shard_routing", "router_input", "ce_gather"),
    # sm107: the MoE re-class only; the router logits are gathered and every row of the
    # step is routed, as the built-in arm does, and the attention input is all-gathered.
    107: ("moe",),
}


def install(model: MuseSparkOpenModel, sm: Optional[int]) -> "ContextSequenceParallel":
    """The path's forward for ``model``, with every layer's MoE re-classed (``moe.install``).

    ``sm`` is the SM version the model's composition was built for
    (``dispatch.sm_of``), required: the parts differ per SM, so none is assumed. It
    picks the eager threshold (``EAGER_MIN_ROWS_BY_SM``) and the parts installed
    beside the MoE re-class (``SHARD_PARTS_BY_SM``).

    The MNNVL all-reduce workspaces are grown on the first prefill step
    (``ContextSequenceParallel.reserve``).

    Raises:
        ValueError: when ``refusal`` names a reason, a layer's MoE is not the built-in one,
            a layer's fused finalize all-reduce serves ``MIN_ROWS`` rows, or ``sm`` is
            neither 103 nor 107 (None included).
    """
    from . import moe as _moe

    parts = SHARD_PARTS_BY_SM.get(sm)
    if parts is None:
        got = "no SM version" if sm is None else f"sm{sm}"
        raise ValueError(
            f"[musespark_open] context_sequence_parallel is written for sm103 and sm107, got {got}"
        )
    path = ContextSequenceParallel(model, sm)
    _moe.install(model, min_rows=MIN_ROWS, shard_routing="shard_routing" in parts)
    if "router_input" in parts:
        from . import router_input as _router_input

        _router_input.install(model, path.step)
    if "ce_gather" in parts:
        from . import ce_gather as _ce_gather

        path.ce_gather = _ce_gather.install(model)
    return path


def _reserve_all_reduce_workspaces(model: MuseSparkOpenModel) -> int:
    """Grow every MNNVL all-reduce workspace of ``model`` for a ``max_num_tokens`` step; returns how many.

    The size is the one the replicated forward's all-reduce asks for on a step of
    ``max_num_tokens`` rows, so no step the path declines grows a workspace after the
    CUDA graphs are captured. Collective over each workspace's group: every rank of the
    TP group runs the same steps.
    """
    from tensorrt_llm._torch.distributed.ops import (
        MNNVLAllReduce,
        get_or_scale_allreduce_mnnvl_workspace,
    )

    config = model.model_config
    rows = config.max_num_tokens
    hidden = config.pretrained_config.hidden_size
    sized = []
    for module in model.modules():
        if not isinstance(module, MNNVLAllReduce):
            continue
        key = (module.mapping, module.dtype)
        if key in sized:
            continue
        sized.append(key)
        get_or_scale_allreduce_mnnvl_workspace(
            module.mapping,
            module.dtype,
            buffer_size_bytes=MNNVLAllReduce.get_required_workspace_size(
                rows, hidden, module.mapping.tp_size, module.dtype
            ),
        )
    return len(sized)


@contextlib.contextmanager
def _shard_capture(
    spec_metadata: Optional[DFlashSpecMetadata], shard: int
) -> Iterator[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
    """``(shard buffer, step buffer)``: the DFlash capture buffer swapped for a shard-height one.

    ``(None, None)`` when the step captures nothing. The step buffer is put back on
    exit, whatever happens inside.
    """
    step = None if spec_metadata is None else spec_metadata.captured_hidden_states
    if step is None:
        yield None, None
        return
    local = step.new_empty((shard, step.shape[1]))
    spec_metadata.captured_hidden_states = local
    try:
        yield local, step
    finally:
        spec_metadata.captured_hidden_states = step


class ContextSequenceParallel:
    """The sequence-parallel forward of a context step, and which steps take it.

    Raises:
        ValueError: at construction, when ``refusal`` names a reason (every rank of
            the TP group reads the same facts, so every rank raises alike).
    """

    def __init__(self, model: MuseSparkOpenModel, sm: Optional[int] = None) -> None:
        from .router_input import SequenceParallelStep

        reason = refusal(model)
        if reason is not None:
            raise ValueError(f"[musespark_open] context_sequence_parallel: {reason}")
        self.mapping = model._sp_mapping
        self.reserved = False
        # The copy-engine gather of the attention input (``ce_gather.install``); None when
        # a layer's QKV projection cannot run in row blocks.
        self.ce_gather = None
        # Set while ``forward`` runs the layers: the attention-side joins then also write
        # the router's FP32 input (``router_input``).
        self.step = SequenceParallelStep()
        #: The fewest rows of an eager step that runs sequence-parallel on this SM.
        self.eager_min_rows = EAGER_MIN_ROWS_BY_SM.get(sm, EAGER_MIN_ROWS)

    def reserve(self, model: MuseSparkOpenModel) -> None:
        """Grow ``model``'s MNNVL all-reduce workspaces once, on the first prefill step.

        Raises:
            RuntimeError: when that first step is captured into a CUDA graph, where the
                workspaces cannot be grown.
        """
        if self.reserved:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "[musespark_open] context_sequence_parallel: the first prefill step is "
                "captured into a CUDA graph; the MNNVL all-reduce workspaces must be grown "
                "on an eager prefill step before any capture"
            )
        _reserve_all_reduce_workspaces(model)
        if self.ce_gather is not None:
            self.ce_gather.build()
            self.ce_gather.start_watch()
        self.reserved = True

    def takes(self, attn_metadata: Any, rows: int, kwargs: Dict[str, Any]) -> bool:
        """Whether a step of ``rows`` rows with these layer keywords runs sequence-parallel.

        Host-side facts only, the same on every rank of the TP group: the rows, whether
        the step is being captured into the breakable prefill CUDA graph, and the keywords.
        """
        spec_metadata = kwargs.get("spec_metadata")
        return (
            rows >= MIN_ROWS
            and (rows >= self.eager_min_rows or is_in_breakable_cuda_graph())
            and attn_metadata.num_contexts > 0
            and not torch.compiler.is_compiling()
            and kwargs.get("lora_params") is None
            and kwargs.get("all_reduce_params") is None
            and (spec_metadata is None or isinstance(spec_metadata, DFlashSpecMetadata))
        )

    def forward(
        self,
        core: MuseSparkOpenModel,
        attn_metadata: Any,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """The step's final hidden states, every row, as the built-in forward returns them."""
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        mapping = self.mapping
        # The token embedding is vocabulary-parallel: each rank looks up the ids its
        # shard owns and the rows are all-reduced, so every rank embeds every row.
        if inputs_embeds is None:
            inputs_embeds = core.embed_tokens(input_ids)
        rows = int(inputs_embeds.shape[0])
        shard = -(-rows // mapping.tp_size)
        mine = _pad_sequence_parallel_tokens(inputs_embeds, mapping.tp_size).narrow(
            0, mapping.tp_rank * shard, shard
        )
        embeds = core.embed_norm(mine.to(core.dtype))
        if core.m_emb != 1.0:
            embeds = embeds * core.m_emb
        # The FP32 residual stream of this rank's rows, entered once.
        hidden_states = embeds.to(torch.float32)
        ce_gather = (
            self.ce_gather if self.ce_gather is not None and self.ce_gather.takes(rows) else None
        )
        with _shard_capture(kwargs.get("spec_metadata"), shard) as (capture, step_capture):
            layers = core.layers
            precomputed_attn_input = None
            self.step.active = True
            if ce_gather is not None:
                ce_gather.begin()
            try:
                for index, decoder_layer in enumerate(layers):
                    next_norm = (
                        layers[index + 1].pre_qkv_proj_norm if index + 1 < len(layers) else None
                    )
                    result = decoder_layer(
                        position_ids=position_ids,
                        hidden_states=hidden_states,
                        attn_metadata=attn_metadata,
                        precomputed_attn_input=precomputed_attn_input,
                        next_layer_norm=next_norm,
                        sp_num_tokens=rows,
                        **kwargs,
                    )
                    if next_norm is None:
                        hidden_states = result
                    else:
                        hidden_states, precomputed_attn_input = result
            finally:
                self.step.active = False
                # Also counted when a layer raised: the watch then sees the forward end once
                # its queued work does, rather than a forward that never ends.
                if ce_gather is not None:
                    ce_gather.end()
        output = core.norm(hidden_states.to(core.dtype))
        if capture is None:
            return allgather(output, mapping, dim=0)[:rows]
        output, capture = allgather([output, capture], mapping, dim=0)
        step_capture[:rows].copy_(capture[:rows])
        return output[:rows]
