# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast path ``expert_tactic_pin``: the NVFP4 expert calls of 512..1023 rows on one pinned tactic.

Where the call is: above the stacked paths' row limits, every NVFP4 MoE layer's experts run the
built-in ``trtllm::fp4_block_scale_moe_runner`` (SwiGLU, top-8 of 256 experts, the rank's own
64-expert window at tp 4). That op takes its ``[tile_n, config]`` from the AutoTuner, which
profiles one tactic per power-of-two row bucket (``FP4BlockScaleMoERunner``'s
``last_positive_power_of_2``), separately on every rank (``INDEPENDENT``) and on a dummy routing.
So every call of 512..1023 rows, the decode steps of 64..127 requests at 8 rows each, runs the
tactic one rank's warmup profiled at 512 rows, and the ranks of one server can draw different
tiles: a 128-token tile whose FC2 runs slower on served routings than the 64-token tiles, while
the tensor-parallel join after the experts waits for the slowest rank.

``install`` wraps every NVFP4 MoE layer's ``experts.forward`` (the 64-expert window) with
``ExpertTacticPin.ensure``, a host-side check before the call it delegates to. Outside the
AutoTuner's tuning mode, and only when the profiling cache gained entries since the last check,
it sets this rank's cache entries of the expert op at bucket ``BUCKET`` for the 64-expert window
(``unique_id`` ``(8, intermediate, 64, 0)``) to the pinned tactic, the way
``hot_replicas._owner_window_tactics`` rewrites cache entries. The entries are rewritten after the
tuning pass, never seeded before it: a hit at the tuning call's bucket returns before the op's
other buckets are profiled. The captured CUDA graphs then carry the pinned kernels, on every rank
alike. Every other bucket, window and op keeps the AutoTuner's choice, and a cache file the
tuning pass saves keeps the autotuned values. The rewrite is in the process's cache, so an
``autotune()`` context with ``TLLM_AUTOTUNER_CACHE_PATH`` that exits after it saves the pinned
entries to that file, which a later process loads with or without the pin. When the tuned cache
holds the expert op but no entry the pin serves (no 512-row call was tuned), the pin warns once
that it is idle.

Values. Only the kernel pair inside the op changes; the routing, the permuted rows, the
finalize and every buffer are the op's own. The pinned tactic is valid for every row count of
the bucket (``get_valid_configs``), or the entry is left to the AutoTuner. The config index is
an index into the TRTLLM-gen runner's list of (FC1, FC2) kernel pairs for one tile, built from
the kernels the running GPU loads, so the default is for sm107 and this kernel library
(``VALIDATED_CAPABILITIES``); ``install`` refuses another device capability.
"""

from __future__ import annotations

import functools
import types
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import torch

from tensorrt_llm.logger import logger

#: The autotuned op whose entries the pin rewrites.
EXPERT_OP = "trtllm::fp4_block_scale_moe_runner"
#: The op's hidden-states input (its tuned dimension 0 is the row count).
HIDDEN_STATES = 2
#: The pinned bucket: every call of BUCKET..2*BUCKET-1 rows looks it up.
BUCKET = 512
#: The rank's own expert window at tp 4, the routed top-k and the SwiGLU act type the pin serves.
WINDOW = 64
TOP_K = 8
ACT_SWIGLU = 0
#: The device capabilities the default tactic is measured on.
VALIDATED_CAPABILITIES = frozenset({(10, 7)})
#: The pinned ``[tile_n, config]`` by default: a 64-token tile whose FC1 is
#: ``t128x64x512_s4_et128x32_m256x64x128`` and FC2 ``t128x64x512_s4_et128x64_m256x64x128``
#: (sm107a cubins), a pair the AutoTuner itself picks on some ranks.
DEFAULT_TILE_N = 64
DEFAULT_CONFIG = 26

TAG = "[musespark_open] expert_tactic_pin"

#: ``(top_k, hidden, intermediate, window, rows) -> the valid [tile_n, config] pairs``.
ValidFn = Callable[[int, int, int, int, int], FrozenSet[Tuple[int, int]]]


@functools.lru_cache(maxsize=None)
def _runner_valid(
    top_k: int, hidden: int, intermediate: int, window: int, rows: int
) -> FrozenSet[Tuple[int, int]]:
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner

    runner = FP4BlockScaleMoERunner.runner_dict.get((ACT_SWIGLU,))
    if runner is None:
        runner = torch.classes.trtllm.FP4BlockScaleMoERunner(ACT_SWIGLU)
        FP4BlockScaleMoERunner.runner_dict[(ACT_SWIGLU,)] = runner
    configs = runner.get_valid_configs(top_k, hidden, intermediate, window, rows)
    return frozenset(tuple(int(v) for v in c) for c in configs)


def parse_unique_id(unique_id: str) -> Optional[Tuple[int, int, int, int]]:
    """``FP4BlockScaleMoERunner.unique_id()`` as the cache key holds it (its ``str``):
    ``(top_k, intermediate, local experts, act type)``, or None for another format."""
    fields = unique_id.strip().strip("()").split(",")
    if len(fields) != 4:
        return None
    try:
        return tuple(int(v) for v in fields)  # type: ignore[return-value]
    except ValueError:
        return None


def pinned_keys(cache: Dict[Tuple, Tuple]) -> List[Tuple]:
    """The cache keys the pin rewrites: the expert op, the 64-expert SwiGLU top-8 window,
    hidden states at the ``BUCKET`` row count."""
    keys = []
    for key in cache:
        if len(key) != 4 or key[0] != EXPERT_OP:
            continue
        ident = parse_unique_id(key[2])
        if ident is None:
            continue
        top_k, _, window, act_type = ident
        if top_k != TOP_K or window != WINDOW or act_type != ACT_SWIGLU:
            continue
        shapes = key[3]
        if len(shapes) <= HIDDEN_STATES or len(shapes[HIDDEN_STATES]) != 2:
            continue
        if shapes[HIDDEN_STATES][0] != BUCKET:
            continue
        keys.append(key)
    return keys


def valid_for_bucket(
    tactic: Tuple[int, int], top_k: int, hidden: int, intermediate: int, valid: ValidFn
) -> bool:
    """Whether ``tactic`` is a valid config of the window's call at every row count of the bucket."""
    return all(
        tuple(tactic) in valid(top_k, hidden, intermediate, WINDOW, rows)
        for rows in range(BUCKET, 2 * BUCKET)
    )


def rewrite(
    cache: Dict[Tuple, Tuple], tactic: Tuple[int, int], valid: ValidFn
) -> Tuple[int, int, List[Tuple]]:
    """Set every pinned key's tactic to ``tactic`` where it is valid for the whole bucket.

    Returns ``(rewritten, matched, invalid keys)``; an entry already holding ``tactic``
    counts as rewritten. The runner index and recorded time of each entry are kept.
    """
    rewritten = 0
    invalid = []
    keys = pinned_keys(cache)
    for key in keys:
        top_k, intermediate, _, _ = parse_unique_id(key[2])
        hidden = 2 * key[3][HIDDEN_STATES][1]
        if not valid_for_bucket(tactic, top_k, hidden, intermediate, valid):
            invalid.append(key)
            continue
        entry = cache[key]
        cache[key] = (entry[0], [int(tactic[0]), int(tactic[1])], *entry[2:])
        rewritten += 1
    return rewritten, len(keys), invalid


class ExpertTacticPin:
    """The pin of one process: its tactic and when it last checked the AutoTuner's cache."""

    def __init__(self, tactic: Sequence[int], valid: Optional[ValidFn] = None):
        self.tactic = (int(tactic[0]), int(tactic[1]))
        self.valid = valid or _runner_valid
        self._seen = -1
        self._reported = False
        self._warned_idle = False

    def ensure(self) -> None:
        """Rewrite the pinned entries if the cache gained entries since the last check; a no-op
        under the AutoTuner's tuning mode (the tuning pass profiles every bucket first)."""
        from tensorrt_llm._torch.autotuner import AutoTuner

        tuner = AutoTuner.get()
        if tuner.is_tuning_mode:
            return
        cache = tuner.profiling_cache.cache
        size = len(cache)
        if size == self._seen:
            return
        self._seen = size
        rewritten, matched, invalid = rewrite(cache, self.tactic, self.valid)
        if invalid:
            logger.warning(
                f"{TAG}: {list(self.tactic)} is not a valid config at every {BUCKET}.."
                f"{2 * BUCKET - 1}-row call of {len(invalid)} cache entries; they keep the "
                "AutoTuner's tactic"
            )
        if (
            not matched
            and not self._warned_idle
            and any(len(key) == 4 and key[0] == EXPERT_OP for key in cache)
        ):
            self._warned_idle = True
            logger.warning(
                f"{TAG}: the tuned cache has no {BUCKET}-row expert-call entry of the "
                f"{WINDOW}-expert window; the pin changes no call"
            )
        if matched and not self._reported:
            self._reported = True
            logger.info(
                f"{TAG}: {rewritten} of {matched} {BUCKET}-row expert-call entries of the "
                f"{WINDOW}-expert window run {list(self.tactic)}"
            )


def _wrap(experts, pin: ExpertTacticPin) -> None:
    previous_forward = experts.forward

    def forward(self, *args, **kwargs):
        pin.ensure()
        return previous_forward(*args, **kwargs)

    experts.forward = types.MethodType(forward, experts)


def nvfp4_layers(text_model) -> List[int]:
    """The layers whose experts are the native NVFP4 leaf over a ``WINDOW``-expert window."""
    from ..replicated_experts import ring_buddy as _ring_buddy

    layers = []
    for index, decoder_layer in enumerate(text_model.layers):
        moe = getattr(decoder_layer, "feed_forward", None)
        if moe is None or getattr(moe, "experts", None) is None:
            continue
        if _ring_buddy.expert_format(moe) != "nvfp4":
            continue
        backend = _ring_buddy._backend(moe)
        if getattr(backend, "expert_size_per_partition", None) != WINDOW:
            continue
        layers.append(index)
    return layers


def install(
    text_model,
    tactic: Iterable[int],
    capability: Optional[Tuple[int, int]] = None,
    layers: Optional[Sequence[int]] = None,
) -> ExpertTacticPin:
    """Wrap the NVFP4 MoE layers' expert forwards with the pin; returns it.

    ``capability`` defaults to the current device's. ``layers`` defaults to ``nvfp4_layers``.

    Raises:
        ValueError: on a device capability the tactic is not measured on, or when no layer
            has the NVFP4 64-expert window.
    """
    tactic = tuple(int(v) for v in tactic)
    if capability is None:
        capability = tuple(torch.cuda.get_device_capability())
    if tuple(capability) not in VALIDATED_CAPABILITIES:
        raise ValueError(
            f"{TAG}: the tactic index is measured on {sorted(VALIDATED_CAPABILITIES)}, "
            f"not {tuple(capability)}"
        )
    if layers is None:
        layers = nvfp4_layers(text_model)
    if not layers:
        raise ValueError(f"{TAG}: no layer has the NVFP4 {WINDOW}-expert window")
    pin = ExpertTacticPin(tactic)
    for index in layers:
        _wrap(text_model.layers[index].feed_forward.experts, pin)
    text_model._expert_tactic_pin = pin
    logger.info(
        f"{TAG}: {list(tactic)} pinned for the {BUCKET}..{2 * BUCKET - 1}-row expert calls of "
        f"{len(layers)} NVFP4 MoE layers (applied after the AutoTuner's tuning pass)"
    )
    return pin
