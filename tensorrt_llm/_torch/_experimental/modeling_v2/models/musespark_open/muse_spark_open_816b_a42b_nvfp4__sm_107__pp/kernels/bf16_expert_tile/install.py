# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A fixed FlashInfer tactic for the BF16 expert calls of a stage's context steps.

Layers 0 and 61 keep the checkpoint's BF16 experts on the built-in FlashInfer TRTLLM-Gen BF16
leaf (``prefill_cutedsl_experts`` keeps it for them). That leaf asks FlashInfer's own
``AutoTuner`` for a ``flashinfer::trtllm_bf16_moe`` tactic, and a context server never tunes
FlashInfer for this op, so every call takes the op's untuned fallback: of the token tiles it
offers at 256 rows per expert (64 and 128) the launcher takes the first, 64.

Every BF16 expert layer of the stage runs its calls of ``MIN_ROWS..MAX_ROWS`` rows (the context
steps of at least 4096 tokens) at the fixed tactic ``PINNED_TACTIC`` -- token tile 128, config 0
-- on every stage alike, with no tuning and no per-rank timing. Every other call, and every
other FlashInfer op, keeps the chooser it had. The pin acts in the forward's thread only, while
the wrapped expert forward runs, so a CUDA graph captured around the call records the pinned
kernels. If the pinned tactic is not among the op's valid tactics for a call's shape, that call
keeps the op's own choice (logged once).

Numerics: bitwise. Every FlashInfer BF16 MoE tactic at these shapes reduces each output
element over the same K order in FP32 (BF16 in, BF16 weights, FP32 accumulate), so the pinned
calls' rows equal the fallback's bit for bit.
"""

from __future__ import annotations

import threading
import types
from typing import Any, List, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

from ... import dispatch as _dispatch
from ..._tep import module as _tep_module

#: The FlashInfer op whose tactic is pinned.
OP = "flashinfer::trtllm_bf16_moe"
#: The pinned ``[tile_N, config]`` of the FlashInfer launcher.
PINNED_TACTIC = (128, 0)
#: The rows of the calls the pin acts on: the context steps of 4096..8192 tokens.
MIN_ROWS = 4096
MAX_ROWS = 8192
#: The experts of every MoE layer (the leaf must hold all of them on a stage).
EXPERTS = 256


class _Pin(threading.local):
    """The tactic pinned in this thread while a wrapped expert forward runs, else None."""

    tactic: Optional[Tuple[int, int]] = None


_PIN = _Pin()
_CHOOSER_LOCK = threading.Lock()
#: (layer, rows bucket, capturing) of the pinned calls already logged.
_LOGGED: set = set()


def _tactic_key(tactic: Any) -> Optional[Tuple[int, ...]]:
    try:
        return tuple(int(v) for v in tactic)
    except TypeError:
        return None


def _install_chooser() -> None:
    """Route the FlashInfer ``AutoTuner`` singleton's ``choose_one`` through the thread's pin.

    Idempotent. Outside a pinned forward (every other op, every other thread, every unwrapped
    call) the call is the tuner's own ``choose_one`` unchanged.
    """
    from flashinfer.autotuner import AutoTuner

    with _CHOOSER_LOCK:
        tuner = AutoTuner.get()
        if getattr(tuner, "_musespark_open_pp_tile_pin", False):
            return
        own = tuner.choose_one

        def choose_one(custom_op, runners, tuning_config, inputs, **kwargs):
            pinned = _PIN.tactic
            if pinned is None or custom_op != OP or not runners:
                return own(custom_op, runners, tuning_config, inputs, **kwargs)
            try:
                valid = {_tactic_key(t) for t in runners[0].get_valid_tactics(inputs, None)}
            except Exception as error:  # noqa: BLE001 - an unknown shape keeps the op's own choice
                valid = set()
                reason = repr(error)
            else:
                reason = "not among the op's valid tactics"
            if tuple(pinned) not in valid:
                key = ("invalid", tuple(tuple(getattr(t, "shape", ())) for t in inputs[:6]))
                if key not in _LOGGED:
                    _LOGGED.add(key)
                    logger.warning(
                        f"{_dispatch.TAG} bf16_expert_tile: tactic {list(pinned)} for {OP} "
                        f"{reason} at this shape; the op's own choice is kept for it"
                    )
                return own(custom_op, runners, tuning_config, inputs, **kwargs)
            return runners[0], list(pinned)

        tuner.choose_one = choose_one
        tuner._musespark_open_pp_tile_pin = True


def refusal(moe) -> Optional[str]:
    """Why this MoE layer's experts are not the BF16 leaf the pin acts on, or None."""
    ring_buddy = _tep_module("kernels.replicated_experts.ring_buddy")
    if getattr(moe, "experts", None) is None:
        return "the layer has no routed experts"
    if getattr(moe.experts, "comm", None) is not None:
        return "the layer has an external MoE communication strategy"
    if ring_buddy.expert_format(moe) != "bf16":
        return "the experts are not the FlashInfer TRTLLM-Gen BF16 leaf"
    backend = ring_buddy._backend(moe)
    if (
        getattr(backend, "num_slots", None) != EXPERTS
        or getattr(backend, "expert_size_per_partition", None) != EXPERTS
        or getattr(backend, "slot_start", None) != 0
    ):
        return f"the leaf does not hold all {EXPERTS} experts from slot 0"
    return None


def _wrap(moe, layer_idx: int, tactic: Tuple[int, int]) -> None:
    experts = moe.experts
    prior = experts.forward

    def forward(self, x, router_logits, *args, **kwargs):
        if (
            not isinstance(x, torch.Tensor)
            or x.dim() != 2
            or x.dtype != torch.bfloat16
            or not MIN_ROWS <= x.shape[0] <= MAX_ROWS
            or torch.compiler.is_compiling()
        ):
            return prior(x, router_logits, *args, **kwargs)
        rows = int(x.shape[0])
        capturing = torch.cuda.is_current_stream_capturing()
        key = (layer_idx, rows, capturing)
        if key not in _LOGGED and len(_LOGGED) < 64:
            _LOGGED.add(key)
            logger.info(
                f"{_dispatch.TAG} bf16_expert_tile: layer {layer_idx} call of {rows} rows runs "
                f"{OP} at tactic {list(tactic)} (capturing {capturing})"
            )
        outer = _PIN.tactic
        _PIN.tactic = tactic
        try:
            result = prior(x, router_logits, *args, **kwargs)
        finally:
            _PIN.tactic = outer
        return result

    experts.forward = types.MethodType(forward, experts)


def install(text_model, mapping) -> List[int]:
    """Pin the BF16 expert layers of the stage; returns the layer indices.

    Collective over the stages (``agreed_install``). A stage without a BF16 expert layer installs
    nothing (stages 1 and 2 of the 62-layer split).
    """
    tactic = PINNED_TACTIC

    def build():
        layers, skipped = [], {}
        for layer in _dispatch.owned_layers(text_model):
            reason = refusal(layer.feed_forward)
            if reason is None:
                layers.append(layer)
            else:
                skipped[layer.layer_idx] = reason
        _install_chooser()
        return layers, skipped

    layers, skipped = _dispatch.agreed_install(mapping, build, (ValueError, ImportError))
    for layer in layers:
        _wrap(layer.feed_forward, int(layer.layer_idx), tactic)
    installed = [int(layer.layer_idx) for layer in layers]
    logger.info(
        f"{_dispatch.TAG} bf16_expert_tile: {OP} pinned to tactic {list(tactic)} for the "
        f"{MIN_ROWS}..{MAX_ROWS}-row expert calls of layers {installed} on stage "
        f"{text_model._pp_stage}; {len(skipped)} other layers keep their expert call"
    )
    return installed
