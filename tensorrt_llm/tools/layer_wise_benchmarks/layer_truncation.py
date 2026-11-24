# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a model with only its first N decoder layers, for the layer-wise benchmark engine path.

Config: ``num_hidden_layers`` becomes N on the top-level config and on nested text / LLM configs,
and any list or tuple attribute whose length equals the original layer count (a per-layer
schedule such as ``layer_types``) keeps its first N entries. Index lists (``full_attn_layers``)
are left as is; models filter them against ``num_hidden_layers``.

Checkpoint and per-layer quantization keys ``[...model.]layers.{i}....``: kept for ``i < N``;
layers stored past the original count (MTP / next-n modules at ``L..L+k-1``) move to
``N..N+k-1``; the rest are dropped.

A model that asserts its own layer count compares ``original_num_hidden_layers(config)``.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, MutableMapping, Optional, Tuple

ORIGINAL_NUM_LAYERS_ATTR = "_lwb_original_num_hidden_layers"
_LAYER_KEY = re.compile(r"^((?:.*\bmodel\.)?layers\.)(\d+)(\..*)$")
_NESTED = ("text_config", "llm_config", "language_config")


def original_num_hidden_layers(config: Any) -> Optional[int]:
    """``num_hidden_layers`` before truncation, for checks that pin a model's layer count."""
    return getattr(config, ORIGINAL_NUM_LAYERS_ATTR, getattr(config, "num_hidden_layers", None))


def _truncate_one(config: Any, n: int) -> bool:
    total = getattr(config, "num_hidden_layers", None)
    if not isinstance(total, int) or total <= n:
        return False
    for name, value in list(vars(config).items()):
        if isinstance(value, (list, tuple)) and len(value) == total and name != "architectures":
            setattr(config, name, type(value)(value[:n]))
    config.num_hidden_layers = n
    setattr(config, ORIGINAL_NUM_LAYERS_ATTR, total)
    return True


def truncate_config(config: Any, n: int) -> bool:
    """Keep the first ``n`` decoder layers of ``config`` and its nested configs, in place.

    Returns True if any of them had more than ``n``.
    """
    configs = [c for c in (config, *(getattr(config, a, None) for a in _NESTED)) if c is not None]
    return any([_truncate_one(c, n) for c in configs])


def checkpoint_num_layers(checkpoint_dir: Optional[str]) -> Optional[int]:
    """The decoder layer count the checkpoint's own ``config.json`` states."""
    if not checkpoint_dir:
        return None
    path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        cfg = json.load(f)
    for nested in (cfg, *(cfg.get(a) or {} for a in _NESTED)):
        if isinstance(nested.get("num_hidden_layers"), int):
            return nested["num_hidden_layers"]
    return None


def filter_layer_keys(entries: MutableMapping[str, Any], n: int, total: int) -> Tuple[int, int]:
    """Hide the decoder layers ``n..total-1`` of ``entries`` in place; returns (dropped, moved)."""
    if total <= n:
        return 0, 0
    dropped = moved = 0
    for key in list(entries.keys()):
        m = _LAYER_KEY.match(key)
        if m is None:
            continue
        i = int(m.group(2))
        if i < n:
            continue
        value = entries.pop(key)
        if i >= total:
            entries[f"{m.group(1)}{i - total + n}{m.group(3)}"] = value
            moved += 1
        else:
            dropped += 1
    return dropped, moved
