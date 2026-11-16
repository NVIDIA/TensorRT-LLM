# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every cell a catalog entry is certified over: its own, and its targets'.

An entry's own `CELLS` are the configurations that belong to the op whoever
calls it -- tile edges, the decode and prefill row counts, the dtypes its
kernel takes. The configurations a particular target runs belong to that
target, and live in the entry's test, keyed by the target's directory under
`models/`:

    TARGET_CELLS = {"deepseek_v3/r1_0528_nvfp4__sm_10x__default": (Cell(...), ...)}
    CELLS = certified_cells(op, TARGET_CELLS)

The test drives the union, and a spec several targets declare runs once.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell, OpWrapper


def certified_cells(
    op: OpWrapper | type[OpWrapper], target_cells: Mapping[str, Sequence[Cell]]
) -> tuple[Cell, ...]:
    """`op`'s own cells, then every target's, each distinct spec once."""
    entry = op if isinstance(op, type) else type(op)
    declared = [*entry.CELLS]
    for cells in target_cells.values():
        declared.extend(cells)
    seen, cells = set(), []
    for cell in declared:
        key = repr(sorted(cell.spec.items()))
        if key not in seen:
            seen.add(key)
            cells.append(cell)
    return tuple(cells)
