# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every cell a catalog entry is certified over: its own, and its targets'.

An entry's own `CELLS` are the configurations that belong to the op whoever
calls it -- tile edges, the decode and prefill row counts, the dtypes its
kernel takes. The configurations a particular target runs belong to that
target, in a `cells.py` beside its `modeling.py`:

    CELLS = {FlashinferSiluAndMul: (Cell(why=..., spec=...), ...)}

A target's cells change no forward, so a change to them reruns the entry tests
they feed and nothing of the target's. An entry's test drives the union, and a
spec several targets declare runs once.
"""

from __future__ import annotations

import functools
import importlib
from pathlib import Path

import tensorrt_llm._torch._experimental.modeling_v2 as _modeling_v2
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell, OpWrapper

_ROOT = Path(_modeling_v2.__file__).resolve().parent


@functools.cache
def target_cells() -> dict[str, dict[type[OpWrapper], tuple[Cell, ...]]]:
    """Each `cells.py` under `models/`, keyed by its directory relative to `models/`."""
    found = {}
    for path in sorted((_ROOT / "models").rglob("cells.py")):
        module = ".".join(path.relative_to(_ROOT).with_suffix("").parts)
        cells = importlib.import_module(f"{_modeling_v2.__name__}.{module}").CELLS
        found[path.parent.relative_to(_ROOT / "models").as_posix()] = cells
    return found


def certified_cells(op: OpWrapper | type[OpWrapper]) -> tuple[Cell, ...]:
    """`op`'s own cells, then every target's, each distinct spec once."""
    entry = op if isinstance(op, type) else type(op)
    declared = [*entry.CELLS]
    for by_entry in target_cells().values():
        declared.extend(by_entry.get(entry, ()))
    seen, cells = set(), []
    for cell in declared:
        key = repr(sorted(cell.spec.items()))
        if key not in seen:
            seen.add(key)
            cells.append(cell)
    return tuple(cells)
