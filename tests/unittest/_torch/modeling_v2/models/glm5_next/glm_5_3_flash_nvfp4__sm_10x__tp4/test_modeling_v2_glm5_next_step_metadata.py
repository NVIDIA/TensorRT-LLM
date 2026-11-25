# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.3-Flash target's step metadata keeps its slot tables current on every step.

The sparse layers' slot tables are per-step state the metadata builds in ``prepare()``: a row
takes a new page every ``tokens_per_block`` tokens. The runtime prepares a step through more
than one hook of the Mamba metadata (a decode step whose batch is the previous step's takes
``prepare_steady_gen_step``), and a hook that does not reach the tables leaves the kernels
reading a new page through the previous step's table. These read the classes only: no GPU or
engine.
"""

from __future__ import annotations

import pytest

from tensorrt_llm._torch.modeling_v2.models.glm5_next.glm_5_3_flash_nvfp4__sm_10x__tp4 import (  # noqa: E501
    modeling as M,
)

pytestmark = pytest.mark.cpu_only

Metadata = M._SlotTablesMetadata


def test_a_steady_decode_step_runs_the_whole_prepare():
    prepared = []
    metadata = object.__new__(Metadata)
    metadata.prepare = prepared.append
    attn_metadata = object()

    metadata.prepare_steady_gen_step(attn_metadata)

    assert prepared == [attn_metadata]


def test_every_step_hook_of_the_runtime_reaches_the_slot_tables():
    """A step hook the runtime's metadata gains has to be taken here, or it skips the tables."""
    runtime_hooks = {
        name for base in Metadata.__mro__[1:] for name in vars(base) if name.startswith("prepare")
    }
    assert {"prepare", "prepare_steady_gen_step"} <= runtime_hooks
    assert runtime_hooks <= set(vars(Metadata)), (
        f"step hooks that do not build the slot tables: {sorted(runtime_hooks - set(vars(Metadata)))}"
    )
