# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``swa_block``: a sliding-window layer's decode attention block in fewer launches.

With ``preattn_prep`` and ``swa_paged_decode`` a sliding-window decode step runs, between the QKV
projection and the output projection: ``preattn_prep``, the split and combine
kernels of ``paged_decode_attention`` and the output norm-gate -- four
dependent launches, each waiting for the previous grid and only then starting
its own chain of dependent loads. ``ops/attention/swa_block`` is that block
with the same arithmetic and fewer edges:

* Pre-wait loads: the split kernel reads the sequence length, its block-table entries
  and the cached K/V before ``griddepcontrol.wait``, and ``preattn_prep``
  triggers its dependent at entry, so the split grid lands and loads the cache
  while prep runs. Prep is launched only after the step is known to fit
  ``swa_block``, so its entry trigger releases nothing but the split; a step
  that does not fit gets prep with its usual trigger (dependents at its exit).
* Fused prep (``SwaBlockPlan.fused_prep``, on top of the pre-wait loads): the split programs run ``preattn_prep``'s
  statements on their own query rows and the step's keys, reading the raw
  projection; the prep launch is gone on these layers. The output gate is read
  where the projection put it (a strided view), which is the same values prep
  would have copied.
* Fused gate (``SwaBlockPlan.fused_gate``, on top of the pre-wait loads): the
  combine kernel applies the output norm-gate to the head it has merged and lets
  the output projection launch after its own wait, as the PDL norm-gate copy of
  ``boundary_early_release`` does.

* Key blocks per split program (the ``swa_blocks_per_program`` lever, installed by
  ``install_multi``): off, every step runs the one-block split grid; on (``AUTO``),
  each step of 2..4 sequences picks its geometry from its sequence count
  (``AUTO_SPLIT_GEOMETRY``: blocks and the split kernel's register cap), and a
  one-sequence step keeps the one-block grid. The partials and every output bit
  are unchanged (``ops/attention/swa_block``); only the program count and the
  programs resident per SM change.

Full-attention layers, prefill steps and any step ``paged_decode_attention``
would not take run exactly as without these features. Every value is the one
the four-kernel chain computes.

Memory order (what the split reads before its wait, and why nothing in flight
writes it): ``seq_lens``, the block table (``kv_cache_block_offsets``, a view)
and the positions are inputs written before the graph launch and not inside
the step before the target's layers; the cached K/V of keys before the step
(``j < past``) were written by earlier graphs, which complete before this one
starts, and this layer's only cache writer in the step is the split grid's
``(0, h, 0)`` program, which writes positions ``>= past`` after its wait. The
grids that can run concurrently with the split's pre-wait loads are the QKV
GEMM and (with the pre-wait loads) prep: neither writes any of those buffers. ``preattn_prep``
itself still reads nothing before its wait.

Constraint on any other change. The pre-wait loads really do run while the
split's predecessors are still executing, so the argument above is what keeps
the values right. A kernel that writes ``seq_lens`` / ``kv_lens_cuda_runtime``,
the block table, the positions, the KV scale or a sliding-window layer's cache
pages *inside* the captured target forward -- for example a between-graph
commit moved into the graph, or spec-loop glue that rewrites the KV lengths or
positions before the target's layers -- must have completed before any split
grid starts: it must reach layer 0's QKV GEMM through a full dependency (a
plain launch or a stream join), not only through programmatic edges, and must
not run on a stream that joins the main chain after the attention. (A chain of
programmatic edges is safe only while every kernel on it waits before it
triggers, as the QKV GEMM does; a kernel that triggers at entry, such as prep
launched with its entry trigger, passes its predecessor's progress on unfinished.) Whoever adds
such a writer re-runs the captured-graph edge census of
``test_modeling_v2_musespark_open_schedule.py`` and the pre-wait
window and late-predecessor tests of ``attention/test_modeling_v2_swa_block.py``.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple, Union

import torch

from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenAttention
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.logger import logger

from ...attention import MuseSparkOpenTepAttention, MuseSparkOpenTepDecodeAttention
from ...dispatch import LAUNCH_PDL
from ...ops.attention.paged_decode_attention import window_splits
from ...ops.attention.preattn_prep import HEAD_DIM, preattn_prep
from ...ops.attention.swa_block import (
    MAX_BLOCKS_PER_PROGRAM,
    MAX_ROWS,
    MULTI_MAX_REQUESTS,
    SPLIT_REGISTER_CAP,
    max_blocks,
    swa_block_attention,
    swa_block_multi_attention,
)

#: The geometry setting that picks blocks per split program by the step's sequence count.
AUTO = "auto"
#: ``AUTO``: (key blocks per split program, split register cap or ``None``) by the step's
#: sequence count, the fastest geometry of a tep4 rank on sm107. The split grid's time is set
#: by its rounds of resident programs first: the one-block kernel fits 3 programs per SM, so
#: ``n`` sequences need ``n`` rounds; 2 blocks per program keep 3 per SM (2 sequences in one
#: round); 3 or 4 blocks per program fit 3 per SM only under ``SPLIT_REGISTER_CAP`` (3
#: sequences at 3 blocks, 4 at 4, in one round); one sequence is a single round already.
AUTO_SPLIT_GEOMETRY = {
    1: (1, None),
    2: (2, None),
    3: (3, SPLIT_REGISTER_CAP),
    4: (4, SPLIT_REGISTER_CAP),
}


def check_setting(setting: Union[int, str]) -> Union[int, str]:
    """``setting`` when it is ``AUTO`` or a fixed ``1`` .. ``MAX_BLOCKS_PER_PROGRAM``.

    Raises:
        ValueError: on any other value.
    """
    if setting == AUTO or (type(setting) is int and 1 <= setting <= MAX_BLOCKS_PER_PROGRAM):
        return setting
    raise ValueError(
        f"[musespark_open] swa_block: blocks per split program must be 1..{MAX_BLOCKS_PER_PROGRAM} "
        f"or {AUTO!r}, got {setting!r}"
    )


def split_geometry(setting: Union[int, str], requests: int) -> Tuple[int, Optional[int]]:
    """(key blocks per split program, split register cap or ``None``) of a step of
    ``requests`` sequences under ``setting``; a fixed ``setting`` has no cap."""
    if setting == AUTO:
        return AUTO_SPLIT_GEOMETRY.get(requests, (1, None))
    return setting, None


def describe_geometry(bpp: int, max_registers: Optional[int]) -> str:
    """``bpp`` with its register cap, as the install lines print it."""
    return f"{bpp}" if max_registers is None else f"{bpp} (<= {max_registers} registers)"


def describe_blocks_per_program(setting: Union[int, str]) -> str:
    """The install lines' wording of ``setting``."""
    if setting == AUTO:
        table = ", ".join(
            f"{n}: {describe_geometry(*geometry)}"
            for n, geometry in sorted(AUTO_SPLIT_GEOMETRY.items())
        )
        return f"key blocks per split program by sequences (blocks {{{table}}})"
    if setting == 1:
        return "1 key block per split program"
    return f"{setting} key blocks per split program"


@dataclasses.dataclass(frozen=True)
class SwaBlockPlan:
    """What the sliding-window layers' decode attention block runs under ``swa_block``.

    Attributes:
        fused_prep: the split kernel runs the prep.
        fused_gate: the combine kernel runs the output norm-gate.
    """

    fused_prep: bool
    fused_gate: bool


class _SwaBlock:
    """The ``swa_block`` decode path of a sliding-window layer's attention.

    A mixin over the ``preattn_prep`` + ``swa_paged_decode`` attention class (and,
    with ``boundary_early_release``, its norm-gate class); never used alone. A call carrying ``swa_block`` (only
    ``DecodeTarget`` passes it) on a sliding-window layer runs the block
    through ``ops/attention/swa_block``; every other call is the base class's.
    The per-call attributes are set by ``forward`` and cleared when it returns.

    With ``_swa_multi`` (set by the ``wide_swa_block`` fast path) a decode step of
    2..``MULTI_MAX_REQUESTS`` generation sequences of equal row count runs the block too,
    through ``swa_block_multi_attention``: each sequence's rows, cache bytes and output are
    the one-sequence block's.
    """

    _swa_multi: bool = False
    _swa_blocks_per_program: Union[int, str] = 1
    _swa_block: Optional[SwaBlockPlan]
    _swa_raw: bool
    _swa_gate: Optional[torch.Tensor]
    _swa_gated: bool
    _swa_position_ids: Optional[torch.Tensor]

    def forward(
        self,
        position_ids,
        hidden_states,
        attn_metadata,
        *args,
        swa_block: Optional[SwaBlockPlan] = None,
        **kwargs,
    ):
        self._swa_block = swa_block if self.is_sliding else None
        self._swa_raw = False
        self._swa_gate = None
        self._swa_gated = False
        self._swa_position_ids = position_ids
        try:
            return super().forward(position_ids, hidden_states, attn_metadata, *args, **kwargs)
        finally:
            self._swa_block = None
            self._swa_raw = False
            self._swa_gate = None
            self._swa_gated = False
            self._swa_position_ids = None

    def preprocess_qkv(self, qkv: torch.Tensor, position_ids: Optional[torch.Tensor]):
        plan = self._swa_block
        if (
            plan is None
            or self._preattn_prep is None
            or not self._fits_preattn_prep(qkv, position_ids)
        ):
            q, k, v, gate = super().preprocess_qkv(qkv, position_ids)
            self._swa_gate = gate
            return q, k, v, gate
        # Hand on the raw projection, and the gate where the projection put it.
        # ``_attn_impl`` launches the prep once it knows which kernel follows:
        # with the fused prep the split kernel runs it; otherwise ``preattn_prep``
        # runs with its entry trigger right before the split; a step that does
        # not fit ``swa_block`` gets ``preattn_prep`` as it is launched without
        # this class (dependents at its exit) and then the paged decode / built-in
        # attention. Nothing launches between here and ``_attn_impl``.
        self._swa_raw = True
        self._swa_gate = self._raw_gate(qkv)
        self._swa_position_ids = position_ids
        return qkv, None, None, self._swa_gate

    def _raw_gate(self, qkv: torch.Tensor) -> torch.Tensor:
        """The output gate ``[rows, H, 64]`` as a view of the raw projection (prep copies it)."""
        rows, heads = qkv.shape[0], self.num_heads
        if self._qkv_layout_inverse is not None:
            start = (heads + 2 * self.num_key_value_heads) * HEAD_DIM
            return qkv[:, start:].view(rows, heads, HEAD_DIM)
        return qkv[:, : 2 * heads * HEAD_DIM].view(rows, heads, 2 * HEAD_DIM)[..., HEAD_DIM:]

    def _prep_now(self, qkv: torch.Tensor, trigger_at_entry: bool = False) -> torch.Tensor:
        """``preattn_prep`` of the raw projection ``preprocess_qkv`` handed on; returns the fused operand.

        The launch the base class's ``preprocess_qkv`` makes (the op ``DecodeTarget``
        handed this call), with the entry trigger only when ``trigger_at_entry``:
        ``_attn_impl`` asks for it only when ``swa_block``'s split is the next
        launch. The output gate becomes prep's copy, as in the base class.
        """
        rope = not self.skip_rope
        op = self._preattn_prep if self._preattn_prep is not None else preattn_prep
        fused, gate = op(
            qkv,
            self.rotary_emb.rotary_cos_sin if rope else None,
            self._swa_position_ids if rope else None,
            self.num_heads,
            self.num_key_value_heads,
            self.q_norm.variance_epsilon,
            self.scale_query_by,
            self._qkv_layout_inverse is not None,
            LAUNCH_PDL,
            **({"trigger_at_entry": True} if trigger_at_entry else {}),
        )
        self._swa_raw = False
        self._swa_gate = gate
        return fused

    def _attn_impl(
        self,
        q,
        k,
        v,
        attn_metadata,
        attention_mask,
        mrope_rotary_cos_sin,
        mrope_position_deltas,
        attention_window_size,
        attention_mask_data,
        output=None,
        output_sf=None,
        attention_sinks=None,
        relative_attention_bias=None,
        relative_attention_max_distance=0,
        has_lora=False,
    ):
        plan = self._swa_block
        decode = self._decode_attention
        tail = (
            attn_metadata,
            attention_mask,
            mrope_rotary_cos_sin,
            mrope_position_deltas,
            attention_window_size,
            attention_mask_data,
        )
        tail_kwargs = dict(
            output=output,
            output_sf=output_sf,
            attention_sinks=attention_sinks,
            relative_attention_bias=relative_attention_bias,
            relative_attention_max_distance=relative_attention_max_distance,
            has_lora=has_lora,
        )
        if plan is None or decode is None or not decode.sliding_window:
            if self._swa_raw:
                q = self._prep_now(q)
            return super()._attn_impl(q, k, v, *tail, **tail_kwargs)
        raw = self._swa_raw
        fused_width = (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim
        requests = self._step_requests(attn_metadata)
        decline = self._paged_decode_decline(
            q.narrow(1, 0, fused_width) if raw and q.dim() == 2 else q,
            k,
            v,
            attn_metadata,
            attention_mask,
            mrope_rotary_cos_sin,
            mrope_position_deltas,
            attention_window_size,
            output,
            output_sf,
            attention_sinks,
            relative_attention_bias,
            has_lora,
            requests=requests,
        )
        if decline is None and window_splits(attention_window_size) < max_blocks(
            attention_window_size, attn_metadata.num_tokens // requests
        ):
            decline = "fewer splits than key blocks"
        if decline is not None:
            logger.warning_once(
                f"[musespark_open] swa_block: a decode step did not fit swa_block ({decline}); "
                "running the paged decode / built-in attention",
                key=f"musespark_open.swa_block.decline_{decline}",
            )
            if raw:
                q = self._prep_now(q)
            return super()._attn_impl(q, k, v, *tail, **tail_kwargs)
        if raw and not plan.fused_prep:
            # Pre-wait loads only: swa_block's split is the next launch, so prep may let it
            # launch at prep's entry.
            q = self._prep_now(q, trigger_at_entry=True)
            raw = False
        gate = self._swa_gate if plan.fused_gate and self._fits_fused_gate(q) else None
        logger.info_once(
            f"[musespark_open] swa_block: sliding-window decode attention block runs swa_block "
            f"(fused_prep={raw}, fused_gate={gate is not None}, q_fp8={decode.q_fp8})",
            key="musespark_open.swa_block.installed",
        )
        bpp, max_registers = split_geometry(self._swa_blocks_per_program, requests)
        logger.info_once(
            f"[musespark_open] swa_block: decode steps of {requests} sequence(s) run "
            f"{describe_geometry(bpp, max_registers)} key block(s) per split program",
            key=f"musespark_open.swa_block.blocks_per_program_{requests}_{bpp}_{max_registers}",
        )
        out = self._swa_block_call(
            q,
            attn_metadata,
            attention_window_size,
            decode.q_fp8,
            raw,
            gate,
            requests,
            bpp,
            max_registers,
        )
        self._swa_gated = gate is not None
        return out, None

    def _fits_fused_gate(self, q: torch.Tensor) -> bool:
        """Exactly when the built-in ``apply_output_gate`` would launch its Triton kernel."""
        gate = self._swa_gate
        rows = q.shape[0]
        return (
            gate is not None
            and not torch.compiler.is_compiling()
            and gate.is_cuda
            and gate.dtype == torch.bfloat16
            and gate.ndim in (2, 3)
            and gate.stride(-1) == 1
            and gate.shape[0] == rows
            and gate.numel() == rows * self.num_heads * self.head_dim
            and (gate.ndim == 2 or gate.shape[-1] == self.head_dim)
            and 0 < rows <= 128
        )

    def _step_requests(self, md) -> int:
        """The sequences of this step the block takes in one call: 1, or under ``_swa_multi`` the
        2..``MULTI_MAX_REQUESTS`` generation sequences of a step whose host ``seq_lens`` are all
        ``rows / sequences`` (at most ``MAX_ROWS``). Host-side reads only."""
        if not self._swa_multi:
            return 1
        n = md.num_seqs
        rows = md.num_tokens
        seq_lens = getattr(md, "seq_lens", None)
        if (
            not 2 <= n <= MULTI_MAX_REQUESTS
            or md.num_generations != n
            or rows % n
            or rows // n > MAX_ROWS
            or seq_lens is None
            or seq_lens.is_cuda
            or seq_lens.numel() < n
            or bool((seq_lens[:n] != rows // n).any())
        ):
            return 1
        return n

    def _swa_block_call(
        self,
        q,
        md,
        window: int,
        q_fp8: bool,
        raw: bool,
        gate,
        requests: int = 1,
        bpp: int = 1,
        max_registers: Optional[int] = None,
    ):
        attn = self.attn
        rows = md.num_tokens
        local_layer_idx = attn.get_local_layer_idx(md)
        kv_pool, block_tables, _ = thop.build_trtllm_gen_kv_cache_metadata(
            md.host_kv_cache_pool_pointers,
            md.host_kv_cache_pool_mapping,
            md.kv_cache_block_offsets,
            local_layer_idx,
            self.num_key_value_heads,
            md.tokens_per_block,
            self.head_dim,
            2,
            self._total_num_blocks_fmha()._get_total_num_blocks(md),
            attn.quant_mode,
            0,
            requests,
            torch.bfloat16,
        )
        rope = raw and not self.skip_rope
        multi = requests > 1
        op = swa_block_multi_attention if multi else swa_block_attention
        return op(
            q[:rows],
            kv_pool,
            block_tables if multi else block_tables[0],
            md.kv_lens_cuda_runtime[:requests],
            attn.kv_scale_orig_quant,
            self.num_heads,
            self.num_key_value_heads,
            window,
            window_splits(window),
            q_fp8,
            fused_prep=raw,
            cos_sin=self.rotary_emb.rotary_cos_sin if rope else None,
            position_ids=self._swa_position_ids if rope else None,
            prep_eps=self.q_norm.variance_epsilon,
            q_scale=self.scale_query_by,
            qkv_permuted=self._qkv_layout_inverse is not None,
            gate=gate,
            gate_eps=self.o_head_norm.variance_epsilon,
            enable_pdl=LAUNCH_PDL,
            blocks_per_program=bpp,
            split_max_registers=max_registers,
        )

    def apply_output_gate(self, attention_output, gate):
        if self._swa_gated:
            # The combine kernel applied it.
            self._swa_gated = False
            return attention_output
        if self._swa_gate is not None:
            # The gate of this call: prep's copy once ``_attn_impl`` ran ``preattn_prep``
            # (what the base class hands the norm-gate), else the one ``preprocess_qkv`` returned.
            gate = self._swa_gate
        return super().apply_output_gate(attention_output, gate)

    @classmethod
    def _base(cls) -> type:
        return cls.__mro__[2]

    @classmethod
    def check(cls, attention: MuseSparkOpenAttention) -> None:
        """The base class's check; changes nothing.

        Raises:
            TypeError, ValueError: as the base class's ``check``.
        """
        cls._base().check(attention)

    @classmethod
    def adopt(cls, attention: MuseSparkOpenAttention):
        """Re-class one built-in attention module as the base class does, then as this class."""
        cls._base().adopt(attention)
        attention.__class__ = cls
        attention._swa_block = None
        attention._swa_raw = False
        attention._swa_gate = None
        attention._swa_gated = False
        attention._swa_position_ids = None
        attention._swa_blocks_per_program = 1
        return attention


_CLASSES: dict = {}


def attention_class(base: Optional[type]) -> type:
    """The class carrying the ``swa_block`` path over ``base``.

    ``base`` is the attention class ``preattn_prep`` + ``swa_paged_decode``
    (+ ``boundary_early_release``) pick.

    Raises:
        ValueError: if ``base`` does not carry both ``preattn_prep`` and the paged decode attention.
    """
    if (
        base is None
        or not issubclass(base, MuseSparkOpenTepAttention)
        or not issubclass(base, MuseSparkOpenTepDecodeAttention)
    ):
        raise ValueError(
            "[musespark_open] swa_block replaces preattn_prep and the paged sliding-window decode "
            "attention; it needs both (select preattn_prep and swa_paged_decode), got the "
            f"attention class {getattr(base, '__name__', None)}"
        )
    if base not in _CLASSES:
        _CLASSES[base] = type(
            f"{base.__name__}SwaBlock", (_SwaBlock, base), {"__module__": __name__}
        )
    return _CLASSES[base]


def log_installed(layers, plan: SwaBlockPlan) -> None:
    sliding = sum(1 for layer in layers if layer.self_attn.is_sliding)
    logger.info(
        f"[musespark_open] swa_block on the decode path of {sliding} sliding-window layers "
        f"(pre-wait loads; fused prep {plan.fused_prep}; fused norm-gate {plan.fused_gate}"
        + ("; prep triggers at entry" if not plan.fused_prep else "")
        + ")"
    )


def install_multi(model, blocks_per_program: Union[int, str] = 1) -> int:
    """``wide_swa_block``: let every sliding-window layer's ``swa_block`` take 2..4-request steps.

    Called after the attention re-class. Sets ``_swa_multi`` on each sliding-window layer
    whose attention carries ``swa_block``, and its key blocks per split program to
    ``blocks_per_program`` (``AUTO`` under the ``swa_blocks_per_program`` lever, else the
    one-block grid); returns how many.

    Raises:
        ValueError: if a sliding-window layer does not carry ``swa_block``, or for a
            ``blocks_per_program`` ``check_setting`` refuses.
    """
    setting = check_setting(blocks_per_program)
    count = 0
    for layer in model.layers:
        attention = layer.self_attn
        if not attention.is_sliding:
            continue
        if not isinstance(attention, _SwaBlock):
            raise ValueError(
                "[musespark_open] wide_swa_block extends swa_block; layer "
                f"{layer.layer_idx} carries {type(attention).__name__}"
            )
        attention._swa_multi = True
        attention._swa_blocks_per_program = setting
        count += 1
    logger.info(
        f"[musespark_open] wide_swa_block: swa_block on decode steps of 2..{MULTI_MAX_REQUESTS} "
        f"requests of up to {MAX_ROWS} rows each on {count} sliding-window layers "
        f"({describe_blocks_per_program(setting)})"
    )
    return count


__all__ = ["AUTO", "SwaBlockPlan", "attention_class", "install_multi", "log_installed"]
