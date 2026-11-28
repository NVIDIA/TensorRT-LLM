# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Rubin DSA metadata's generation placeholder rows under forced acceptance (CPU).

``prepare_generation_placeholder_rows`` uploads, every step, one flag per generation row in batch order: whether
the cache manager marked the row's request as a placeholder-draft request (its first generation step after a
context step or a disaggregated handoff, which keeps its own accepted-token count). A step without such requests
clears the previous step's flags.
"""

import importlib
import importlib.util
from types import SimpleNamespace

import pytest
import torch

# The family's DSA modules import its kernels, which need the CuTe DSL.
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("cutlass") is None, reason="this environment lacks cutlass"
)
cpu = pytest.mark.cpu_only


def _metadata_module():
    return importlib.import_module(
        "tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.rubin_dsa_metadata"
    )


def _md(manager, request_ids, num_contexts):
    return SimpleNamespace(
        kv_cache_manager=manager,
        request_ids=request_ids,
        num_contexts=num_contexts,
        num_seqs=len(request_ids),
        max_num_requests=8,
        kv_lens_cuda=torch.zeros(1),
        generation_placeholder_rows=None,
    )


@cpu
def test_the_metadata_uploads_the_generation_rows_flags_every_step():
    """One flag per generation row in batch order; a step without placeholder rows clears the previous step's."""
    upload = _metadata_module().prepare_generation_placeholder_rows
    manager = SimpleNamespace(first_step_pending={}, placeholder_requests=frozenset({5}))
    md = _md(manager, [9, 5, 6, 1 << 62], num_contexts=1)
    upload(md)
    assert md.generation_placeholder_rows[:3].tolist() == [True, False, False]
    manager.placeholder_requests = frozenset()
    upload(md)
    assert md.generation_placeholder_rows[:3].tolist() == [False, False, False]
