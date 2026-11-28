# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Builders of the K3.1 Rubin targets' collective resources: a runtime helper, not an OpWrapper.

A target calls them once in ``post_load_weights``, outside CUDA graph capture, at the same point
on every rank of the group (each one is collective). It holds the results in its runtime
resources and passes them to the entries as arguments:

* :func:`rubin_all_reduce`: the engine ``AllReduce`` of a TP group (``tp_allreduce`` of
  ``ops/comm/allreduce_module.py``), which ``allreduce_module`` takes;
* :func:`rubin_latent_ar_workspace`: the routed latent all-reduce's own MNNVL Lamport workspace,
  which ``rubin_latent_ar_norm_row`` takes (built in ``ops/comm/rubin_latent_ar_norm.py``).

Every group comes from the mapping the caller passes: on helix8 that is the mapping whose TP
group is the 8 CP ranks (``Mapping.repurpose_helix_cp_to_tp``), never the helix mapping itself,
whose TP group is one rank.
"""

from __future__ import annotations

from typing import Any

import torch

from tensorrt_llm.mapping import Mapping

from .allreduce_module import tp_allreduce
from .rubin_latent_ar_norm import check_group, rubin_latent_ar_workspace


def rubin_all_reduce(mapping: Mapping, strategy: Any, group_size: int):
    """The engine ``AllReduce`` of ``mapping``'s TP group (``group_size`` ranks), in BF16."""
    check_group(mapping, group_size)
    return tp_allreduce(mapping, strategy, torch.bfloat16)


__all__ = ["rubin_all_reduce", "rubin_latent_ar_workspace"]
