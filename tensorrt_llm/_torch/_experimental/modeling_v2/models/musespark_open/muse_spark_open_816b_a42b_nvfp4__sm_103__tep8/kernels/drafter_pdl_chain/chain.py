# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Programmatic-dependent-launch chain of the DFlash 2 drafter and its draft head (``drafter_pdl_chain``).

The drafter's GEMMs (``drafter_fixed_gemms``' fixed split-K tactics, flashinfer's
``SplitKDenseGemmKernel``) load their weight before their own
``griddepcontrol.wait`` and read their activation only after it. A GEMM
can therefore stream its weight while its predecessor runs -- but only if
the predecessor lets it launch, i.e. executes ``griddepcontrol.launch_dependents``
early. In the drafter step (``draft.windowed_block_decode``, ``drafter_block_decode``)
four of the predecessors do not: the fused block convolution in front of
``qkv_proj`` and ``gate_up_proj`` is a plain launch, the window-attention
merge in front of ``o_proj`` triggers at its exit, the built-in
``silu_and_mul`` in front of ``down_proj`` is a plain launch, and in front of
the draft LM head two plain torch kernels (the clamp of the slot ids and the
row gather) follow the last fused hop. Each launch below changes how one of
these is launched or removes it; none changes an arithmetic operation:

* ``pdl_conv`` -- the prepare-side block convolution of every layer
  (``ops/spec/dflash2_block_conv.py``) is launched with programmatic
  dependent launch, triggers its dependents at entry and waits for its
  producers before its first load.
* ``pdl_merge`` -- the window-attention merge (``ops/spec/dflash_window_attention.py``)
  triggers its dependents before its existing wait.
* ``pdl_silu`` -- the MLP's ``silu(gate) * up`` runs as a programmatic-dependent-launch
  copy of the built-in Triton kernel (``ops/spec/drafter_silu_and_mul.py``)
  that triggers at entry and waits before its first load; the drafter's
  ``GatedMLP`` modules are re-classed in place (``EntryTriggerGatedMLP``).
* ``pdl_head`` -- the draft head's slot ids, on a one-request step whose K slots
  are one contiguous row range, are handed to the built-in forward as an
  object whose ``clamp`` returns that range as a ``slice``: the built-in
  ``hidden_states_out[gen_gather_ids]`` is then the ``narrow`` view of the
  same rows (no clamp and no gather launch), and the draft LM-head GEMM
  follows the last fused hop directly.
* ``selector_fork`` -- the candidate selector's ``hidden_projection`` (a cuBLAS GEMM
  that reads only the block rows) is issued on a captured side stream forked
  right before the draft LM-head GEMM and joined where the projection is
  called, after the key all-reduce, instead of on the main stream after it.

Scope: block decodes whose successors are known. ``pdl_conv`` /
``pdl_merge`` / ``pdl_silu`` act on the block decode's 8 rows (one request, K + 1,
K = 7), where ``drafter_fixed_gemms`` has a fixed tactic for ``qkv_proj`` / ``gate_up_proj`` /
``o_proj`` / ``down_proj`` / the first convolution's ``kernel_projection``. ``pdl_conv``
also acts on every other block width of ``ENTRY_ROW_CANDIDATES`` (2..4 requests: 16, 24
and 32 rows) at which the drafter's table lists a fixed tactic for all four of those
successors (``entry_rows``); a width without one runs the base launches. ``pdl_silu``
and ``pdl_merge`` act on the widths their device's table lists among those
(``SILU_ENTRY_ROWS_BY_SM`` / ``MERGE_ENTRY_ROWS_BY_SM``), each measured in the in-model
drafter chain on that GPU. On GB300 all three triggers together are the fastest chain at
1..4 requests. On Rubin an entry-released ``o_proj`` competes with the multi-request merge
it follows (the chain is slower with the merge trigger at 2..4 requests), and the silu
trigger on top of the conv trigger gains nothing at two requests and loses at three.
``pdl_head`` acts on a one-request step whose K slots are one contiguous row
range (any K); its successor is the low-M dispatcher's runner for the LM
head (cuBLAS through ``torch.mm``, which has no programmatic attribute, or
flashinfer's split-K / direct kernels, which wait before reading the
activation). ``selector_fork`` forks only for the one-request draft rows
(``HEAD_ROWS`` = K = 7 rows). Every other call -- more requests, other row
counts, prefill -- runs the base path.

All five launches are installed together, and only with ``TRTLLM_ENABLE_PDL=1``.

PDL memory order (the record the tests assert in the generated PTX):

* block convolution (``pdl_conv``): ``launch_dependents`` and ``griddepcontrol.wait``
  precede every global load; it reads the normed rows (the previous hop's
  output, or layer 0's RMSNorm output), the coefficient GEMM's output and the
  base kernel, all after the wait. Its dependent (the split-K GEMM) reads the
  convolution's output after its own wait and loads only its weight before.
* merge (``pdl_merge``): ``launch_dependents`` precedes ``griddepcontrol.wait``,
  which precedes every global load (the split's partials and the lengths).
  Its dependent (``o_proj``'s split-K GEMM) reads the merged attention after
  its own wait.
* ``silu_and_mul`` copy (``pdl_silu``): as the convolution; its dependent
  (``down_proj``'s split-K GEMM) reads its output after its wait.
* draft LM head (``pdl_head``): its predecessor becomes the last ``drafter_hops`` hop,
  which triggers before its output stores; every runner of the low-M
  dispatcher reads the hop's rows only after its wait (or, cuBLAS, behind a
  full dependency).

Transitivity: every kernel between a producer and a reader that reads it
after its own wait waits before it completes (the convolution and the silu
copy wait; the merge, the split, the GEMMs and the hops wait as well), so a
completed predecessor implies completed producers, as ``boundary_early_release``
records for its entry triggers. Early-resident dependents cannot starve their predecessor: a
dependent launches only after every CTA of its predecessor has executed the
trigger (or exited), i.e. is resident.

Installed from the shell's drafter install after ``drafter_hops.install``;
idempotent. With ``TRTLLM_ENABLE_PDL`` off nothing is installed (logged).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.linear import Linear, UnquantizedLinearMethod
from tensorrt_llm.logger import logger

from ...ops.gemm.fixed_low_m_gemm import device_capability
from ...ops.spec.drafter_silu_and_mul import drafter_silu_and_mul

#: The drafter block rows of one request, K + 1 = 8: the width ``pdl_conv`` / ``pdl_merge`` /
#: ``pdl_silu`` require a fixed tactic at, and the ``pdl_head`` / ``selector_fork`` step.
ROWS = 8

#: The block widths the conv / merge / silu entry triggers may act on: 1..4 requests at
#: K = 7, at most the fixed GEMM's ``MAX_M`` of 32 rows. A width is taken when the
#: drafter's tactic table covers every successor at it (``entry_rows``).
ENTRY_ROW_CANDIDATES = (8, 16, 24, 32)

#: The block widths of the merge trigger per compute capability (``trigger_rows``), among
#: those of ``entry_rows``. sm103: one to four requests. sm107: the one-request block only.
MERGE_ENTRY_ROWS_BY_SM: dict[tuple[int, int], frozenset] = {
    (10, 3): frozenset(ENTRY_ROW_CANDIDATES),
    (10, 7): frozenset({ROWS}),
}

#: The block widths of the silu trigger per compute capability (``trigger_rows``), among
#: those of ``entry_rows``. sm103: one to four requests. sm107: one and four requests.
SILU_ENTRY_ROWS_BY_SM: dict[tuple[int, int], frozenset] = {
    (10, 3): frozenset(ENTRY_ROW_CANDIDATES),
    (10, 7): frozenset({ROWS, 4 * ROWS}),
}


def trigger_rows(table: dict, capability: Optional[tuple] = None) -> frozenset:
    """The block widths ``table`` (``MERGE_ENTRY_ROWS_BY_SM`` / ``SILU_ENTRY_ROWS_BY_SM``)
    lists for ``capability`` (the drafter GEMMs' device, ``fixed_low_m_gemm.device_capability``,
    when None).

    Raises:
        ValueError: on a device the table does not list.
    """
    key = tuple(device_capability() if capability is None else capability)
    rows = table.get(key)
    if rows is None:
        raise ValueError(
            f"drafter_pdl_chain: no trigger widths for SM {key} (supported: {sorted(table)})"
        )
    return rows


#: The draft head's rows for that request: its K = 7 draft slots. ``selector_fork``
#: forks the selector projection only for this many rows.
HEAD_ROWS = ROWS - 1


# --------------------------------------------------------------------------
# Successor checks
# --------------------------------------------------------------------------
def _fixed_tactic_reason(linear: nn.Module, rows: int, where: str) -> Optional[str]:
    """Why ``linear`` is not on the ``drafter_fixed_gemms`` split-K tactic at ``rows`` rows, or None.

    The tactic is the one the installed method runs: its table is the drafter's TP size's.
    """
    from ...spec_worker import FixedTacticLinearMethod

    if not isinstance(linear, Linear):
        return f"{where} is not a Linear"
    method = linear.quant_method
    if type(method) is not FixedTacticLinearMethod:
        return f"{where} is not on the drafter_fixed_gemms GEMM tactic (select drafter_fixed_gemms)"
    if not method.enable_pdl:
        return f"{where}'s fixed GEMM is launched without programmatic dependent launch"
    out_features, in_features = linear.weight.shape
    if method.tactic_for(rows, in_features, out_features) is None:
        return f"{where} has no fixed tactic at {rows} rows"
    return None


def _conv_reason(draft_model: nn.Module) -> Optional[str]:
    from ..drafter_block_decode.block_decode import FusedDFlash2BlockConv

    layers = draft_model.model.layers
    for index, layer in enumerate(layers):
        for kind, successor in (
            ("attention_convs", layer.self_attn.qkv_proj),
            ("mlp_convs", layer.mlp.gate_up_proj),
        ):
            conv = getattr(draft_model, kind)[index]
            if type(conv) is not FusedDFlash2BlockConv:
                return f"{kind}.{index} is not the fused convolution (select drafter_block_conv)"
            reason = _fixed_tactic_reason(successor, ROWS, f"the GEMM after {kind}.{index}")
            if reason is not None:
                return reason
    return None


def _merge_reason(draft_model: nn.Module) -> Optional[str]:
    from ..drafter_block_decode import block_decode as _draft

    bound = draft_model.__dict__.get("dflash_forward")
    if getattr(bound, "__func__", None) is not _draft.windowed_block_decode:
        return "the drafter does not run the windowed block decode (select drafter_block_decode)"
    if not getattr(draft_model, "_block_decode_enable_pdl", False):
        return "the windowed attention is launched without programmatic dependent launch"
    for index, layer in enumerate(draft_model.model.layers):
        reason = _fixed_tactic_reason(layer.self_attn.o_proj, ROWS, f"layer {index} o_proj")
        if reason is not None:
            return reason
    return None


def entry_rows(draft_model: nn.Module) -> frozenset:
    """The widths of ``ENTRY_ROW_CANDIDATES`` at which every successor of the conv / merge /
    silu triggers (``qkv_proj``, ``gate_up_proj``, ``o_proj``, ``down_proj`` of every layer)
    has a fixed tactic. A function of the drafter's shapes and TP table: rank-uniform."""
    rows = []
    for width in ENTRY_ROW_CANDIDATES:
        if all(
            _fixed_tactic_reason(linear, width, "") is None
            for layer in draft_model.model.layers
            for linear in (
                layer.self_attn.qkv_proj,
                layer.mlp.gate_up_proj,
                layer.self_attn.o_proj,
                layer.mlp.down_proj,
            )
        ):
            rows.append(width)
    return frozenset(rows)


class EntryTriggerGatedMLP(GatedMLP):
    """The built-in ``GatedMLP`` whose ``silu(gate) * up`` releases ``down_proj`` at entry on the
    widths of ``_pdl_entry_rows`` (``entry_rows``; 8 rows when unset).

    Never constructed: ``install`` re-classes the drafter's ``GatedMLP``
    modules in place. Every other call is the built-in method's.
    """

    def _apply_activation(self, x, *, has_lora: bool = False):
        if (
            not has_lora
            and isinstance(x, torch.Tensor)
            and x.dim() == 2
            and x.shape[0] in self.__dict__.get("_pdl_entry_rows", (ROWS,))
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
        ):
            return drafter_silu_and_mul(x, True)
        return super()._apply_activation(x, has_lora=has_lora)


def _mlp_reason(draft_model: nn.Module) -> Optional[str]:
    for index, layer in enumerate(draft_model.model.layers):
        mlp = layer.mlp
        where = f"layer {index} mlp"
        if type(mlp) not in (GatedMLP, EntryTriggerGatedMLP):
            return f"{where} is {type(mlp).__name__}, not the built-in GatedMLP"
        if mlp.activation is not F.silu or not mlp._is_plain_swiglu() or mlp.swiglu_limit:
            return f"{where} is not a plain SwiGLU"
        if mlp.split_gate_up:
            return f"{where} splits its gate/up projection"
        down = mlp.down_proj
        if getattr(down, "has_fp8_qdq", False) or getattr(down, "has_w4a8_nvfp4_fp8", False):
            return f"{where} quantizes the activation for down_proj"
        if mlp._can_fuse_gate_up_swiglu() or mlp._can_fuse_swiglu_fp8_quant():
            return f"{where} fuses its activation into a GEMM"
        reason = _fixed_tactic_reason(down, ROWS, f"{where} down_proj")
        if reason is not None:
            return reason
    return None


def _head_reason(llm: nn.Module) -> Optional[str]:
    from tensorrt_llm._torch.modules import low_m_gemm as _low_m_gemm

    from ..draft_head import head as _draft_head

    worker = getattr(llm, "spec_worker", None)
    draft_model = getattr(llm, "draft_model", None)
    if not isinstance(worker, _draft_head.ShardedDraftHeadDFlashWorker):
        return "the DFlash worker does not run the sharded draft head (select draft_head)"
    lm_head = getattr(draft_model, "lm_head", None)
    if not isinstance(lm_head, Linear) or type(lm_head.quant_method) is not UnquantizedLinearMethod:
        return "the draft LM head is not an unquantized Linear"
    if getattr(lm_head, "use_custom_cublas_mm", False) or getattr(
        lm_head, "use_cute_dsl_bf16_gemm", False
    ):
        return "the draft LM head runs a GEMM path pdl_head was not checked against"
    if not _low_m_gemm.LOW_M_GEMM_ACTIVE:
        return "the low-M GEMM dispatcher is off (TRTLLM_LOW_M_GEMM_BACKEND)"
    if not _draft_head.gathers_column_shards(lm_head):
        return "the draft LM head is not a gathered bf16 column-parallel head"
    return None


# --------------------------------------------------------------------------
# pdl_head: the draft rows as a view
# --------------------------------------------------------------------------
class DraftRowRange:
    """The draft head's slot ids for a step whose K slots are rows ``[start, stop)``.

    What the draft head's ``_draft_slot_ids`` returns under ``pdl_head`` instead of the
    cached id tensor. The built-in step calls only ``clamp(max=rows - 1)`` on
    it and then indexes the drafter output with the result; when the bound
    covers every slot (the ids are unchanged by the clamp) that result is
    ``slice(start, stop)``, so the index is the view of exactly the rows the
    gather would have copied. Otherwise ``clamp`` is the tensor's own.
    Anything else is an ``AttributeError``: the step does not run the code of
    the built-in ``DFlashWorker._forward_impl`` this was written against
    (``gen_gather_ids.clamp(max=...)``, then ``hidden_states_out[gen_gather_ids]``).
    """

    __slots__ = ("ids", "start", "stop")

    def __init__(self, ids: torch.Tensor, start: int, stop: int) -> None:
        self.ids = ids
        self.start = start
        self.stop = stop

    def clamp(self, min=None, max=None):  # noqa: A002 - torch's keyword names
        if (max is None or self.stop - 1 <= max) and (min is None or self.start >= min):
            return slice(self.start, self.stop)
        return self.ids.clamp(min=min, max=max)


def draft_row_range(num_gens: int, block_size: int, num_draft_tokens: int, ids: torch.Tensor):
    """``(start, stop)`` when ``ids`` are one contiguous row range of a one-request step, else None.

    Reads ``ids`` back to the host (a device sync): called only when the draft head
    first caches the ids, outside any capture. The expected values are the
    built-in ``dflash_draft_slot_ids`` evaluated on the host.
    """
    from tensorrt_llm._torch.speculative.dflash import dflash_draft_slot_ids

    if num_gens != 1 or torch.cuda.is_current_stream_capturing():
        return None
    expected = dflash_draft_slot_ids(num_gens, block_size, num_draft_tokens, False, device="cpu")
    values = expected.tolist()
    if not values or values != list(range(values[0], values[0] + len(values))):
        return None
    if ids.dtype != expected.dtype or ids.tolist() != values:
        return None
    return values[0], values[0] + len(values)


# --------------------------------------------------------------------------
# selector_fork: the selector projection on a side stream
# --------------------------------------------------------------------------
class SelectorFork:
    """The draft head's side stream for the selector's ``hidden_projection``.

    ``launch`` (from the draft head's logits processor, right before the LM-head GEMM)
    issues the projection of the block rows on the side stream after the main
    stream's work so far -- only for the one-request draft rows
    (``HEAD_ROWS`` contiguous rows); any other call forks nothing and the
    projection runs inline where it is called. ``take`` (the projection
    module's own call, after the key all-reduce) returns it after making the
    main stream wait for the side stream -- a full dependency in front of the
    merge. A pending projection that is not taken is joined by the next
    ``launch`` (``settle``).
    """

    def __init__(self, projection: nn.Linear) -> None:
        from ..context_kv_fork.fork import dedicated_stream

        self.projection = projection
        self.stream = dedicated_stream()
        self.launches = 0
        self.taken = 0
        self._pending = None

    def settle(self) -> None:
        from ..context_kv_fork.fork import current_capture_id

        pending, self._pending = self._pending, None
        if pending is not None and pending[3] == current_capture_id():
            torch.cuda.current_stream().wait_stream(self.stream)

    def launch(self, hidden_states: torch.Tensor) -> None:
        from ..context_kv_fork.fork import current_capture_id

        self.settle()
        weight = self.projection.weight
        if not (
            isinstance(hidden_states, torch.Tensor)
            and hidden_states.dim() == 2
            and hidden_states.shape[0] == HEAD_ROWS
            and hidden_states.dtype == weight.dtype
            and hidden_states.shape[-1] == weight.shape[1]
            and hidden_states.is_contiguous()
        ):
            return
        main = torch.cuda.current_stream()
        if main == self.stream:
            raise RuntimeError(
                "[musespark_open] selector_fork: the draft head runs on the selector's side stream"
            )
        self.stream.wait_stream(main)
        with torch.cuda.stream(self.stream):
            gate = nn.Linear.forward(self.projection, hidden_states)
        self._pending = (hidden_states, gate, main, current_capture_id())
        self.launches += 1

    def take(self, input: torch.Tensor) -> Optional[torch.Tensor]:
        from ..context_kv_fork.fork import current_capture_id

        pending = self._pending
        if pending is None:
            return None
        source, gate, main, capture = pending
        if (
            torch.cuda.current_stream() != main
            or current_capture_id() != capture
            or input.dtype != source.dtype
            or input.data_ptr() != source.data_ptr()
            or input.numel() != source.numel()
            or input.shape[-1] != source.shape[-1]
            or not input.is_contiguous()
        ):
            self.settle()
            return None
        self._pending = None
        main.wait_stream(self.stream)
        self.taken += 1
        return gate.view(*input.shape[:-1], gate.shape[-1])


class ForkedSelectorProjection(nn.Linear):
    """The selector's ``hidden_projection`` returning the side-stream result ``selector_fork`` issued for it.

    Never constructed: ``install`` re-classes the module in place. A call
    whose input is not the rows the fork projected computes as the built-in.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        fork = self.__dict__.get("_selector_fork")
        if fork is not None:
            gate = fork.take(input)
            if gate is not None:
                return gate
        return super().forward(input)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def unsupported_reason(llm: nn.Module) -> Optional[str]:
    """Why the five PDL chain launches cannot be installed on ``llm``, or None."""
    from ..drafter_block_decode import block_decode as _draft

    draft_model = getattr(llm, "draft_model", None)
    if draft_model is None:
        return "no drafter"
    bound = draft_model.__dict__.get("dflash_forward")
    if getattr(bound, "__func__", None) is not _draft.windowed_block_decode:
        return "the drafter does not run the windowed block decode"
    capability = device_capability()
    if capability not in MERGE_ENTRY_ROWS_BY_SM or capability not in SILU_ENTRY_ROWS_BY_SM:
        return f"no merge / silu trigger widths for SM {capability}"
    reason = _conv_reason(draft_model)
    if reason is not None:
        return f"pdl_conv: {reason}"
    reason = _merge_reason(draft_model)
    if reason is not None:
        return f"pdl_merge: {reason}"
    reason = _mlp_reason(draft_model)
    if reason is not None:
        return f"pdl_silu: {reason}"
    reason = _head_reason(llm)
    if reason is not None:
        return f"pdl_head,selector_fork: {reason}"
    selector = getattr(draft_model, "candidate_selector", None)
    projection = getattr(selector, "hidden_projection", None)
    if type(projection) not in (nn.Linear, ForkedSelectorProjection):
        return "selector_fork: the selector's hidden_projection is not a plain nn.Linear"
    processor = getattr(draft_model, "logits_processor", None)
    from ..draft_head import head as _draft_head

    if type(processor) is not _draft_head.ShardDraftLogitsProcessor:
        return "selector_fork: the drafter's logits processor is not the draft head's"
    return None


def installed(llm: nn.Module) -> frozenset:
    """The PDL chain launches in place on ``llm``."""
    draft_model = getattr(llm, "draft_model", None)
    worker = getattr(llm, "spec_worker", None)
    found = set()
    if draft_model is None:
        return frozenset()
    convs = list(getattr(draft_model, "attention_convs", None) or ()) + list(
        getattr(draft_model, "mlp_convs", None) or ()
    )
    if convs and all(conv.__dict__.get("_pdl_conv") for conv in convs):
        found.add("pdl_conv")
    if draft_model.__dict__.get("_pdl_merge"):
        found.add("pdl_merge")
    layers = getattr(getattr(draft_model, "model", None), "layers", None) or ()
    if layers and all(type(layer.mlp) is EntryTriggerGatedMLP for layer in layers):
        found.add("pdl_silu")
    if worker is not None and worker.__dict__.get("_pdl_head"):
        found.add("pdl_head")
    selector = getattr(draft_model, "candidate_selector", None)
    projection = getattr(selector, "hidden_projection", None)
    if projection is not None and projection.__dict__.get("_selector_fork") is not None:
        found.add("selector_fork")
    return frozenset(found)


def _widths(widths) -> str:
    return ",".join(str(w) for w in sorted(widths)) or "none"


def install(llm: nn.Module, mapping=None) -> frozenset:
    """Install the five PDL chain launches; return the names this call installed.

    The refusal is agreed over ``mapping``'s TP group (``tp_agreement``, a
    collective when the group has more than one rank: every rank calls this at
    the same point), so a rank whose checks fail cannot raise alone while its
    peers go on to the next collective. Programmatic dependent launch is a
    precondition of the composition (``dispatch.PRECONDITIONS``).

    Raises:
        ValueError: when a launch cannot run here on some rank
            (``unsupported_reason``), on every rank alike.
    """
    from ... import dispatch as _tp_agreement

    reason = unsupported_reason(llm)
    reason = _tp_agreement.agreed_refusal(
        mapping, None if reason is None else f"[musespark_open] {reason}"
    )
    if reason is not None:
        raise ValueError(reason)
    before = installed(llm)
    draft_model = llm.draft_model
    widths = entry_rows(draft_model)
    for conv in list(draft_model.attention_convs) + list(draft_model.mlp_convs):
        conv._pdl_conv = True
        conv._pdl_entry_rows = widths
    merge_rows = widths & trigger_rows(MERGE_ENTRY_ROWS_BY_SM)
    silu_rows = widths & trigger_rows(SILU_ENTRY_ROWS_BY_SM)
    draft_model._pdl_merge = True
    draft_model._pdl_merge_rows = merge_rows
    for layer in draft_model.model.layers:
        if type(layer.mlp) is GatedMLP:
            layer.mlp.__class__ = EntryTriggerGatedMLP
        layer.mlp._pdl_entry_rows = silu_rows
    llm.spec_worker._pdl_head = True
    projection = draft_model.candidate_selector.hidden_projection
    if type(projection) is nn.Linear:
        projection.__class__ = ForkedSelectorProjection
    if projection.__dict__.get("_selector_fork") is None:
        projection._selector_fork = SelectorFork(projection)
    draft_model.logits_processor._selector_fork = projection._selector_fork
    now = installed(llm)
    changed = now - before
    if changed:
        logger.info(
            f"[musespark_open] {','.join(sorted(changed))}: drafter PDL chain installed "
            f"(in place: {','.join(sorted(now))}; block rows with the conv trigger "
            f"{_widths(widths)}, the silu trigger {_widths(silu_rows)}, "
            f"the merge trigger {_widths(merge_rows)})"
        )
    return changed
