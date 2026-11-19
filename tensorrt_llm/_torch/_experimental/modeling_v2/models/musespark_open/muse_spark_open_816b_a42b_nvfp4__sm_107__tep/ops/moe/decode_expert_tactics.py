# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed TRTLLM-gen tactics for the NVFP4 expert call of tep4 decode steps (sm107).

``trtllm::fp4_block_scale_moe_runner`` takes its ``[tile_n, config]`` tactic from the
autotuner, which profiles one tactic per power-of-two row bucket at warmup, on a dummy
routing, separately on every rank: every row count of 256 to 511 runs the tactic
profiled at 256 rows, where it settles on 64-token tiles.

This table pins the tactic of the stacked expert call (``ring_buddy._nvfp4_call``) per
``(expert window, rows)`` for listed decode row counts of tep4 (8 rows per request at the
DFlash K = 7 verify width), one table per GPU:

* sm107: the 32-token tile, for 256 to 384 rows, and from ``TILE32_MIN_ROWS`` rows
  (``MUSESPARK_OPEN_DECODE_TILE32_MIN_ROWS``, a multiple of 8 from 128 to 256; unset: 256).
  The autotuner's 128-row profile, which serves 128 to 255 rows, is a timing pick on a
  random routing and can settle on the 64-token tile, whose FC1 runs slow on served
  routings. The pinned tile computes every expert row bitwise as the autotuned tactics
  whose FC2 has the same MMA K; tactics whose FC2 MMA K differs can round a rare row
  differently in its last place (the autotuner picks from both).

With ``batch_half_window_experts`` (``pin_half_batch_steps``) sm107 also pins the 96-expert
half window's 40..64-row decode steps (five to eight requests) to the same tile: the
autotuner profiles that window per rank, and a rank that settles on another tile there runs
every layer's experts slower than the others, which the finalize then waits for.

Row counts and windows not listed, and tuning passes, keep the autotuner. A GPU without a
table is not supported. With the switch below 256, the first call that reaches a listed row
count logs, once per process and call shape, at how many of the window's listed row counts
the pinned tactic is a valid config (``fixed_tactic`` leaves the others to the autotuner).

The lever ``l061_tile32_pin``: the owner-only 64-expert window of the NVFP4 layers
``hot_replicas`` leaves on the built-in path (0 and 61, experts quantized at load) at 256 to
384 rows on sm107, ``owner_tactic``. There the autotuner's 256-row profile settles on 64-token
tiles, whose FC1 runs slow on served routings; the lever pins a 32-token tile
(``L061_TILE32_DEFAULT``). Every expert row is
bitwise the autotuned tactics' whose FC2 has the same MMA K; the 64-token tactic's FC2 MMA K
can differ from the pinned tactic's, and then a rare row can round differently in its last
place.

The config index is an index into the TRTLLM-gen MoE runner's list of (FC1, FC2) kernel
pairs for one tile N, built from the kernels the running GPU loads, so an entry is only
meaningful for the GPU and kernel library it was measured on; ``fixed_tactic`` also
requires the entry to be a valid config of the call's shape (``get_valid_configs``) and
returns None otherwise.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

import torch

#: The 32-token tile (FC1 ``t128x32x512u2_s5 … m256x32x128``, FC2
#: ``t128x32x512_s5 … m256x32x128``), pinned for the decode steps of ``TILE32_MIN_ROWS`` to
#: 384 rows (32 to 48 requests by default) of the 68-expert ``hot_replicas`` window.
#: Its FC2's MMA K is 128, as is that of the tile-64 tactics the autotuner picks there
#: most, so every expert row is bitwise theirs.
_TILE32_DECODE = (32, 236)

#: The first decode row count the sm107 pin covers when the switch is unset.
_TILE32_DEFAULT_MIN_ROWS = 256
#: The switch: the first decode row count the sm107 pin covers, a multiple of 8 from 128
#: (16 requests at K = 7) to 256.
TILE32_MIN_ROWS_ENV = "MUSESPARK_OPEN_DECODE_TILE32_MIN_ROWS"


def _tile32_min_rows() -> int:
    value = os.environ.get(TILE32_MIN_ROWS_ENV, "").strip()
    if not value:
        return _TILE32_DEFAULT_MIN_ROWS
    rows = int(value) if value.isdigit() else -1
    if rows % 8 or not 128 <= rows <= _TILE32_DEFAULT_MIN_ROWS:
        raise ValueError(
            f"{TILE32_MIN_ROWS_ENV}={value!r}: expected a multiple of 8 from 128 to "
            f"{_TILE32_DEFAULT_MIN_ROWS}"
        )
    return rows


TILE32_MIN_ROWS = _tile32_min_rows()

#: ``(local_num_experts, rows) -> (tile_n, config)`` on sm107. Window 68 = the 64 own
#: experts plus 4 hot copies (``hot_replicas``, 33..512 rows).
SM107_TACTICS: Dict[Tuple[int, int], Tuple[int, int]] = {
    (68, rows): _TILE32_DECODE for rows in range(TILE32_MIN_ROWS, 385, 8)
}

_TABLES = {(10, 7): SM107_TACTICS}

#: ``(local_num_experts, rows) -> (tile_n, config)`` on sm107 for the 40..64-row decode steps
#: of the 96-expert half window (``batch_half_window_experts``): the tile of
#: ``SM107_TACTICS``. ``fixed_tactic`` keeps the autotuner where it is not a valid config.
SM107_HALF_BATCH_TACTICS: Dict[Tuple[int, int], Tuple[int, int]] = {
    (96, rows): _TILE32_DECODE for rows in range(40, 65, 8)
}
#: The half-window batch-step tables ``pin_half_batch_steps`` serves, per device capability.
_HALF_BATCH_TABLES = {(10, 7): SM107_HALF_BATCH_TACTICS}
#: The capabilities whose half-window batch-step table is served; empty by default.
_PINNED_HALF_BATCH: Set[Tuple[int, int]] = set()


def pin_half_batch_steps() -> None:
    """Serve the half-window batch-step tables too (``batch_half_window_experts``)."""
    if not _PINNED_HALF_BATCH:
        from tensorrt_llm.logger import logger

        logger.info(
            "[musespark_open] decode expert tactics: the 96-expert half window's "
            f"{min(r for _, r in SM107_HALF_BATCH_TACTICS)}.."
            f"{max(r for _, r in SM107_HALF_BATCH_TACTICS)}-row decode steps run tactic "
            f"{list(_TILE32_DECODE)} on sm107"
        )
    _PINNED_HALF_BATCH.update(_HALF_BATCH_TABLES)


#: The owner window the pin serves and the decode rows it pins (32 to 48 requests at K = 7).
L061_WINDOW = 64
L061_ROWS = frozenset(range(256, 385, 8))
#: The pinned tactic by default: the hot path's 32-token tile (``_TILE32_DECODE``).
L061_TILE32_DEFAULT = _TILE32_DECODE


def owner_tactic(
    device: torch.device,
    pinned: Optional[Tuple[int, int]],
    act_type: int,
    top_k: int,
    hidden: int,
    intermediate: int,
    window: int,
    rows: int,
) -> Optional[List[int]]:
    """The pinned ``[tile_n, config]`` of an owner-window expert call of a layer the hot path
    leaves, or None to leave it to the autotuner: sm107 only, the SwiGLU expert, the
    ``L061_WINDOW`` window at ``L061_ROWS`` rows, and a valid config of the call's shape."""
    if pinned is None or act_type != 0 or window != L061_WINDOW or rows not in L061_ROWS:
        return None
    if tuple(torch.cuda.get_device_capability(device)) != (10, 7):
        return None
    if tuple(pinned) not in _valid(act_type, top_k, hidden, intermediate, window, rows):
        return None
    return list(pinned)


def table_for(capability: Tuple[int, int]) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """The table of a device capability: sm107; any other GPU is not supported."""
    table = _TABLES.get(tuple(capability))
    if table is None:
        raise NotImplementedError(
            f"decode expert tactics: no table for compute capability {tuple(capability)} "
            "(measured for sm107 only)"
        )
    return table


@lru_cache(maxsize=None)
def _valid(act_type: int, top_k: int, hidden: int, intermediate: int, window: int, rows: int):
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner

    runner = FP4BlockScaleMoERunner.runner_dict.get((act_type,))
    if runner is None:
        runner = torch.classes.trtllm.FP4BlockScaleMoERunner(act_type)
        FP4BlockScaleMoERunner.runner_dict[(act_type,)] = runner
    configs = runner.get_valid_configs(top_k, hidden, intermediate, window, rows)
    return frozenset(tuple(int(v) for v in c) for c in configs)


_PINNED_ROWS_REPORTED: Set[Tuple[int, int, int, int, int]] = set()


def _report_pinned_rows(
    table: Dict[Tuple[int, int], Tuple[int, int]],
    act_type: int,
    top_k: int,
    hidden: int,
    intermediate: int,
    window: int,
) -> None:
    """Log once per process and call shape at how many of ``window``'s listed row counts the
    pinned tactic is a valid config; the others keep the autotuner (``fixed_tactic``)."""
    key = (act_type, top_k, hidden, intermediate, window)
    if key in _PINNED_ROWS_REPORTED:
        return
    _PINNED_ROWS_REPORTED.add(key)
    from tensorrt_llm.logger import logger

    listed = sorted(rows for (w, rows) in table if w == window)
    invalid = [
        rows
        for rows in listed
        if table[(window, rows)] not in _valid(act_type, top_k, hidden, intermediate, window, rows)
    ]
    message = (
        f"[musespark_open] decode expert tactics: {TILE32_MIN_ROWS_ENV}={TILE32_MIN_ROWS} pins "
        f"{list(table[(window, listed[0])])} for the {window}-expert window's "
        f"{listed[0]}..{listed[-1]}-row calls: a valid config at {len(listed) - len(invalid)} "
        f"of {len(listed)} row counts"
    )
    if invalid:
        logger.warning(f"{message}; the autotuner keeps {invalid}")
    else:
        logger.info(message)


def fixed_tactic(
    device: torch.device,
    act_type: int,
    top_k: int,
    hidden: int,
    intermediate: int,
    window: int,
    rows: int,
) -> Optional[List[int]]:
    """The pinned ``[tile_n, config]`` of an expert call, or None to leave it to the autotuner.

    None when the ``(window, rows)`` is not listed, the call is not the SwiGLU expert
    (``act_type`` 0), or the listed tactic is not a valid config of the call's shape; a
    device without a table raises (``table_for``).
    """
    if act_type != 0:
        return None
    capability = tuple(torch.cuda.get_device_capability(device))
    table = table_for(capability)
    tactic = table.get((window, rows))
    if tactic is not None:
        if TILE32_MIN_ROWS < _TILE32_DEFAULT_MIN_ROWS and table is SM107_TACTICS:
            _report_pinned_rows(table, act_type, top_k, hidden, intermediate, window)
    elif capability in _PINNED_HALF_BATCH:
        tactic = _HALF_BATCH_TABLES[capability].get((window, rows))
    if tactic is None:
        return None
    if tactic not in _valid(act_type, top_k, hidden, intermediate, window, rows):
        return None
    return list(tactic)


def run_fixed(
    tactic: List[int],
    inputs: List[Optional[torch.Tensor]],
    num_experts: int,
    top_k: int,
    intermediate: int,
    window: int,
    routing_method_type: int,
    act_type: int,
    tune_max_num_tokens: int,
) -> List[torch.Tensor]:
    """The unfinalized pre-routed expert call on ``tactic``, without the autotuner.

    ``inputs`` are ``FP4BlockScaleMoEInputs``'s 18 tensors in order (``routing_logits``
    None, ``topk_weights`` / ``topk_ids`` the caller's). Returns what
    ``trtllm::fp4_block_scale_moe_runner`` returns for ``do_finalize=False``.
    """
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner

    runner = FP4BlockScaleMoERunner(
        num_experts,
        top_k,
        None,
        None,
        intermediate,
        0,
        window,
        None,
        routing_method_type,
        False,
        act_type,
        tune_max_num_tokens=tune_max_num_tokens,
    )
    return runner(inputs, tactic=tactic)
