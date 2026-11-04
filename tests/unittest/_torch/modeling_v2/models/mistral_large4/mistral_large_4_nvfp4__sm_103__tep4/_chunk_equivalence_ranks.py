# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The rank side of `test_modeling_v2_chunk_equivalence.py`.

`rank_probe` runs on every rank of the launcher's MPI world (four ranks at tep4, eight over two
nodes of four at tep8), through the session the engines use, while no engine is up (the session
runs one task at a time). It reports the rank's device free memory and the context steps the
target ran since the last probe, and installs or removes an observe-only record of those steps:
`PrefillTarget.forward` and the module's `allgather` wrapped so each context step appends what
it received -- host values the step already holds -- and how many selection gathers it made.
Both wrappers call the original unchanged.

Nothing here touches a device at import: the module is imported on every rank, and a CUDA call
there would open a context on the wrong device.
"""

from __future__ import annotations

import gc
import importlib
import socket
from types import SimpleNamespace

import torch

from tensorrt_llm._utils import global_mpi_rank

TARGET = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4."
    "mistral_large_4_nvfp4__sm_103__tep4.modeling"
)

# This rank's context-step records, and the originals they wrap while the record is installed.
_RECORD = SimpleNamespace(steps=[], originals=None)


class _CountedGather:
    """The target's selection gather, counted: the same call, the same result."""

    def __init__(self, gather):
        self.gather = gather
        self.calls = 0

    def __call__(self, input, sizes, group, process_group=None):
        self.calls += 1
        return self.gather(input, sizes, group, process_group)

    def __getattr__(self, name):
        return getattr(self.gather, name)


def _install(modeling) -> None:
    if _RECORD.originals is not None:
        return
    forward, gather = modeling.PrefillTarget.forward, modeling.allgather
    counted = _CountedGather(gather)

    def recorded(self, attn_metadata, *args, **kwargs):
        md = attn_metadata
        ctx = int(md.num_contexts)
        q_split = md.indexer_prefill_q_split
        if q_split is not None:
            q_split = (int(q_split[0]), int(q_split[1]), [int(s) for s in q_split[2]])
        counted.calls = 0
        out = forward(self, attn_metadata, *args, **kwargs)
        _RECORD.steps.append(
            dict(
                rank=self.core.tp_rank,
                contexts=ctx,
                cached=[int(c) for c in md.kv_cache_params.num_cached_tokens_per_seq[:ctx]],
                new=[int(n) for n in md.seq_lens[:ctx].tolist()],
                tc=int(md.num_ctx_tokens),
                gen=int(md.num_tokens) - int(md.num_ctx_tokens),
                skip_ctx=bool(md.skip_indexer_for_ctx_reqs),
                cached_kv=bool(md.enable_context_mla_with_cached_kv),
                q_split=q_split,
                chunks=[
                    (c.token_start, c.token_end, c.k_token_start, c.k_token_end)
                    for c in (md.indexer_prefill_chunks or [])
                ],
                gathers=counted.calls,
            )
        )
        return out

    _RECORD.originals = (forward, gather)
    modeling.PrefillTarget.forward = recorded
    modeling.allgather = counted


def _remove(modeling) -> None:
    if _RECORD.originals is not None:
        modeling.PrefillTarget.forward, modeling.allgather = _RECORD.originals
        _RECORD.originals = None


def rank_probe(action: str) -> dict:
    """This rank's device free memory after a garbage collection and a cache release, what of
    the device this process's torch allocator still holds, and the context steps recorded since
    the last probe. `action` is "install" (start recording), "collect" (keep recording) or
    "remove" (restore the target's own methods)."""
    rank = global_mpi_rank()
    # The engine's device for this rank: its local rank among the node's devices.
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    gc.collect()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    modeling = importlib.import_module(TARGET)
    steps, _RECORD.steps = _RECORD.steps, []
    if action == "install":
        _install(modeling)
    elif action == "remove":
        _remove(modeling)
    return dict(
        rank=rank,
        host=socket.gethostname(),
        device=device,
        free=free,
        total=total,
        allocated=torch.cuda.memory_allocated(device),
        reserved=torch.cuda.memory_reserved(device),
        steps=steps,
    )
