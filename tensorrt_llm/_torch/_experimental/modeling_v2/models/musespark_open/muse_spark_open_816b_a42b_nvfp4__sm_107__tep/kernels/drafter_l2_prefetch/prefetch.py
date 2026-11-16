# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The drafter's GEMM weights asked of the L2 ahead of their GEMMs (``tep8_drafter_wide_gemms__PREFETCH``).

Every drafter GEMM streams its weight from HBM cold: the drafter's 5 layers read
~280 MB of BF16 weights per TP8 rank per step (two convolution coefficient
projections of 33.5 MB, qkv 21 MB, o_proj 16.8 MB, gate_up 117 MB, down 58.7 MB),
in 25 short launches that reach 3-8 TB/s each (4.4-14.2 us), while the kernels
between them (QK norm + RoPE, the window attention, the hops and their
convolutions and norms) leave HBM mostly idle. Here a side stream asks the L2 for
the next GEMMs' weights while those kernels run (``ops/gemm/dense_gemm.l2_prefetch``:
``cp.async.bulk.prefetch.L2``, one per 32 KiB, a few single-warp programs), so
the GEMMs find them in L2. Per layer ``i`` of a block decode
(``drafter_block_decode.windowed_block_decode``), by ``LEVEL``:

* 1, fork ``attn`` after ``qkv_proj``'s launch: ``o_proj`` and the MLP convolution's
  coefficient projection (50 MB), behind the QK norm + RoPE and the window attention;
* 2, also fork ``mlp`` after the window attention's launch: ``gate_up_proj`` and
  ``down_proj`` (176 MB), behind ``o_proj``, its hop and the MLP convolution;
* 3, also fork ``next`` after the MLP's launches: layer ``i + 1``'s attention
  convolution coefficient projection and ``qkv_proj`` (55 MB), behind the MLP's
  hop (the last layer forks none).

Graph structure: a fork makes the side stream wait on the current stream (an event
recorded right after the last launch), so the prefetch kernel starts once that
kernel completed and runs beside the next ones; nothing waits for it until the end
of the forward, where the current stream waits on the side stream once (a captured
graph must rejoin every fork). A prefetch reads no value and writes nothing anyone
reads (``l2_prefetch``'s sink is never stored), so no value of the step depends on it:
the drafter's bits are the same with any LEVEL. Its programs are single warps that
exit after issuing their requests, so they hold no SM a hop's clusters wait for.

Calls of fewer than ``MIN_ROWS`` rows fork nothing. ``LEVEL`` 0 (the default)
installs nothing.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn

from tensorrt_llm.logger import logger

#: ``level`` / ``min_rows`` (``configure``; ``tep8_drafter_wide_gemms__PREFETCH`` / ``__PREFETCH_MIN_ROWS``).
LEVEL = 0
MIN_ROWS = 1
#: ``weights`` (``configure``; ``tep8_drafter_wide_gemms__PREFETCH_WEIGHTS``): the weights the forks of ``LEVEL``
#: ask for, by name (the L2 may not hold a whole layer's MLP: gate_up alone is 117 MB per TP8 rank).
WEIGHT_NAMES = ("o_proj", "conv_mlp", "gate_up", "down", "conv_attn", "qkv")
#: The default: without the MLP's 176 MB (Rubin's L2 is 126 MiB; W10 prefetch chain: all six +10..+22 us/step,
#: these four -9..-11).
DEFAULT_WEIGHTS = ("o_proj", "conv_mlp", "conv_attn", "qkv")
WEIGHTS = DEFAULT_WEIGHTS
#: The forks of each level, in order. ``first`` (level 4): layer 0's attention convolution coefficients + qkv
#: asked for at the end of the context-K/V side chain (``context_kv_fork``), which runs beside the verify
#: sampler, so the drafter's first GEMMs find them in L2 too.
STAGES = ("attn", "mlp", "next", "first")
#: int32 sink entries (one per 32 KiB request lane): 2 GiB of weight per prefetch call at most.
SINK_ENTRIES = 1 << 16
#: The drafter attribute the block decode reads.
ATTRIBUTE = "_drafter_l2_prefetch"


def configure(*, level: int = 0, min_rows: int = 1, weights: tuple = DEFAULT_WEIGHTS) -> None:
    """Set ``LEVEL`` (0..3) and ``MIN_ROWS`` (1..1024). ``lever_settings.apply`` sets them from the
    model's plan before any install.

    Raises:
        ValueError: for a value outside its band; nothing is set then.
    """
    global LEVEL, MIN_ROWS, WEIGHTS
    level, rows = int(level), int(min_rows)
    names = tuple(weights)
    if not names or set(names) - set(WEIGHT_NAMES):
        raise ValueError(
            f"[musespark_open] drafter_l2_prefetch: weights must be a non-empty subset of {list(WEIGHT_NAMES)} "
            f"(tep8_drafter_wide_gemms__PREFETCH_WEIGHTS), got {list(names)}"
        )
    if not 0 <= level <= len(STAGES):
        raise ValueError(
            f"[musespark_open] drafter_l2_prefetch: level must be in 0..{len(STAGES)} "
            f"(tep8_drafter_wide_gemms__PREFETCH), got {level}"
        )
    if not 1 <= rows <= 1024:
        raise ValueError(
            f"[musespark_open] drafter_l2_prefetch: min_rows must be in 1..1024 "
            f"(tep8_drafter_wide_gemms__PREFETCH_MIN_ROWS), got {rows}"
        )
    LEVEL, MIN_ROWS, WEIGHTS = level, rows, names


def stage_weights(draft_model: nn.Module, stage: str, layer_idx: int, names=None) -> list:
    """The weights fork ``stage`` of layer ``layer_idx`` asks for (empty past the last layer), those of
    ``names`` only when given (``WEIGHT_NAMES``)."""
    layers = draft_model.model.layers
    if stage == "attn":
        named = [
            ("o_proj", lambda: layers[layer_idx].self_attn.o_proj.weight),
            ("conv_mlp", lambda: draft_model.mlp_convs[layer_idx].kernel_projection.weight),
        ]
    elif stage == "mlp":
        named = [
            ("gate_up", lambda: layers[layer_idx].mlp.gate_up_proj.weight),
            ("down", lambda: layers[layer_idx].mlp.down_proj.weight),
        ]
    elif stage == "first":
        named = [
            ("conv_attn", lambda: draft_model.attention_convs[0].kernel_projection.weight),
            ("qkv", lambda: layers[0].self_attn.qkv_proj.weight),
        ]
    elif stage == "next":
        if layer_idx + 1 >= len(layers):
            return []
        named = [
            (
                "conv_attn",
                lambda: draft_model.attention_convs[layer_idx + 1].kernel_projection.weight,
            ),
            ("qkv", lambda: layers[layer_idx + 1].self_attn.qkv_proj.weight),
        ]
    else:
        raise ValueError(f"[musespark_open] drafter_l2_prefetch: unknown stage {stage!r}")
    return [get() for name, get in named if names is None or name in names]


def unsupported_reason(draft_model: Optional[nn.Module]) -> Optional[str]:
    """Why the drafter's weights cannot be prefetched as above, or None."""
    from ..drafter_block_decode.block_decode import windowed_block_decode

    if draft_model is None:
        return "no drafter"
    bound = draft_model.__dict__.get("dflash_forward")
    if getattr(bound, "__func__", None) is not windowed_block_decode:
        return "the drafter does not run drafter_block_decode's block decode"
    if not getattr(draft_model, "has_block_conv", False):
        return "the drafter has no DFlash 2 block convolutions"
    try:
        for layer_idx in range(len(draft_model.model.layers)):
            for stage in STAGES:
                for weight in stage_weights(draft_model, stage, layer_idx):
                    if not (weight.is_cuda and weight.is_contiguous()):
                        return (
                            f"a {stage} weight of layer {layer_idx} is not a contiguous CUDA tensor"
                        )
                    if weight.data_ptr() % 16 or (weight.numel() * weight.element_size()) % 16:
                        return (
                            f"a {stage} weight of layer {layer_idx} is not 16-byte aligned / sized"
                        )
    except (AttributeError, IndexError) as error:
        return f"the drafter's modules are not the DFlash 2 layout ({error})"
    return None


class DrafterPrefetch:
    """The side stream, the sink and the fork schedule of one drafter."""

    def __init__(self, draft_model: nn.Module, level: int, min_rows: int, weights=None) -> None:
        self.draft_model = draft_model
        self.stages = frozenset(STAGES[:level])
        #: The weight names asked for (all when None).
        self.weights = None if weights is None else tuple(weights)
        self.min_rows = min_rows
        device = draft_model.model.layers[0].self_attn.qkv_proj.weight.device
        self.stream = torch.cuda.Stream(device=device)
        self.sink = torch.zeros(SINK_ENTRIES, dtype=torch.int32, device=device)
        #: Whether this forward forked (and so must join).
        self.forked = False
        #: Forks issued (a test counts them).
        self.calls = 0

    def takes(self, rows: int) -> bool:
        return rows >= self.min_rows

    def fork(self, stage: str, layer_idx: int, rows: int) -> None:
        """Ask the L2 for fork ``stage``'s weights of layer ``layer_idx`` from the side stream."""
        if stage not in self.stages or not self.takes(rows):
            return
        weights = stage_weights(self.draft_model, stage, layer_idx, self.weights)
        if not weights:
            return
        from ...ops.gemm.dense_gemm import l2_prefetch

        current = torch.cuda.current_stream()
        self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            for weight in weights:
                l2_prefetch(weight.detach(), self.sink)
        self.forked = True
        self.calls += 1

    def first(self, rows: int) -> None:
        """Level 4: ask the L2 for layer 0's front weights on the CURRENT stream (the context-K/V side chain's,
        which the main stream joins before the drafter forward)."""
        if "first" not in self.stages or not self.takes(rows):
            return
        from ...ops.gemm.dense_gemm import l2_prefetch

        for weight in stage_weights(self.draft_model, "first", 0, self.weights):
            l2_prefetch(weight.detach(), self.sink)
        self.calls += 1

    def join(self) -> None:
        """The forward's end: the current stream waits on the side stream once (graph rejoin)."""
        if self.forked:
            torch.cuda.current_stream().wait_stream(self.stream)
            self.forked = False


def install(draft_model: Optional[nn.Module]) -> bool:
    """Bind a ``DrafterPrefetch`` of ``LEVEL`` / ``MIN_ROWS`` to the drafter; nothing at LEVEL 0.

    Call after ``drafter_block_decode``'s install (and after any install that relays
    the drafter's weights in place: the forks read the parameters at each call).
    Returns whether this call changed the drafter. Idempotent.

    Raises:
        ValueError: when LEVEL is set and the drafter cannot take it (``unsupported_reason``).
    """
    if LEVEL == 0:
        return False
    reason = unsupported_reason(draft_model)
    if reason is not None:
        raise ValueError(
            "[musespark_open] tep8_drafter_wide_gemms__PREFETCH cannot run here: " + reason
        )
    if isinstance(draft_model.__dict__.get(ATTRIBUTE), DrafterPrefetch):
        return False
    setattr(draft_model, ATTRIBUTE, DrafterPrefetch(draft_model, LEVEL, MIN_ROWS, WEIGHTS))
    per_layer = sum(
        w.numel() * w.element_size()
        for stage in STAGES[:LEVEL]
        for w in stage_weights(draft_model, stage, 0, WEIGHTS)
    )
    logger.info(
        f"[musespark_open] drafter_l2_prefetch: drafter GEMM weights asked of the L2 ahead of use "
        f"(level {LEVEL}: {', '.join(STAGES[:LEVEL])} forks of {','.join(WEIGHTS)}, "
        f"{per_layer / 2**20:.0f} MiB per layer, "
        f"block decodes of {MIN_ROWS}+ rows)"
    )
    return True


def forks_of(level: int, layers: int) -> Sequence[tuple]:
    """``(stage, layer)`` of every fork a forward of ``layers`` layers issues at ``level``, in order
    (``next`` of the last layer issues nothing; ``first`` is the side chain's, not the forward's)."""
    stages = STAGES[:level]
    return [
        (stage, layer)
        for layer in range(layers)
        for stage in stages
        if stage != "first" and not (stage == "next" and layer + 1 >= layers)
    ]
