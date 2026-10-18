# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for ``draft_path_walk_tail`` (fast path ``path_walk_tail``).

Gated on its cells with ``torch.equal`` against its reference (``draft_path_walk``'s
reference, then the built-in KV-length rewind and next-token lines), against
``draft_path_walk`` itself for the draft tokens, and against the built-in
worker methods (``DFlashWorker._apply_kv_rewind_after_draft``,
``SpecWorkerBase._prepare_next_new_tokens``) for the tail, eagerly and
replayed from a CUDA graph.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec.draft_path_walk_tail import (  # noqa: E501
    DraftPathWalkTail,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec.draft_path_walk_tail import (  # noqa: E501
    draft_path_walk_tail as op,
)
from tensorrt_llm._torch.speculative.dflash import DFlashWorker

__extra_import_path__ = ["../../../../..", ".", "../../kernels/draft_head"]
from _draft_head_cases import draft_path_walk, seed_of  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401
from test_modeling_v2_draft_path_walk import _walk_inputs  # noqa: E402

assert torch.cuda.is_available(), "the path-walk tail requires a CUDA device"


def _tail_inputs(spec, seed):
    requests, positions = spec["requests"], spec["positions"]
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    rows = requests + 3 if spec.get("permuted") else requests
    accepted_full = torch.randint(0, 202048, (rows, positions + 2), generator=g, device="cuda")
    accepted = accepted_full.int()[:, : positions + 1]  # a strided view when permuted
    if not spec.get("permuted"):
        accepted = accepted.contiguous()
    num_accepted = torch.randint(1, positions + 2, (requests,), generator=g, device="cuda").int()
    batch_indices = torch.arange(rows, device="cuda", dtype=torch.int32)
    if spec.get("permuted"):
        batch_indices = torch.randperm(rows, generator=g, device="cuda").int()
    kv_lens = rewind = None
    if spec.get("rewind"):
        low = 0 if spec.get("clamp") else 64
        kv_lens = torch.randint(low, low + 8, (requests + 4,), generator=g, device="cuda").int()
        rewind = (1 - num_accepted).contiguous()
        if spec.get("clamp"):
            rewind = torch.randint(-3, 10, (requests,), generator=g, device="cuda").int()
    return accepted, num_accepted, batch_indices, kv_lens, rewind


def _builtin_tail(tokens, accepted, num_accepted, batch_indices, kv_lens, rewind):
    """The built-in worker methods on a generation-only step."""
    worker = DFlashWorker.__new__(DFlashWorker)
    # The DFlash context arena's windowing state, as DFlashWorker.__init__ sets it.
    worker._ctx_base = None
    worker._ctx_base_host = None
    worker._ctx_keep = 0
    worker.windowed_ctx_arena = False
    worker._saved_ctx_base = None
    worker._saved_ctx_base_host = None
    worker._kv_rewind_amount = rewind
    worker._kv_rewind_nc, worker._kv_rewind_bs = 0, tokens.shape[0]
    worker._kv_rewind_pending = True
    if kv_lens is not None:
        DFlashWorker._apply_kv_rewind_after_draft(
            worker, SimpleNamespace(kv_lens_cuda=kv_lens), SimpleNamespace(is_cuda_graph=False)
        )
    return DFlashWorker._prepare_next_new_tokens(
        worker, accepted, tokens, batch_indices, tokens.shape[0], num_accepted
    )


@pytest.mark.parametrize(
    "cell", DraftPathWalkTail.CELLS, ids=[c.why[:48] for c in DraftPathWalkTail.CELLS]
)
def test_path_walk_tail_cells(cell, certifying):  # noqa: F811
    certifying(op)
    seed = seed_of(cell.why)
    edges, unary, ids = _walk_inputs(cell.spec, seed)
    accepted, num_accepted, batch_indices, kv_lens, rewind = _tail_inputs(cell.spec, seed)
    kv_ref = None if kv_lens is None else kv_lens.clone()
    kv_builtin = None if kv_lens is None else kv_lens.clone()
    tokens, next_tokens = op(
        edges, unary, ids, accepted, num_accepted, batch_indices, kv_lens, rewind
    )
    ref_tokens, ref_next = op.reference(
        edges, unary, ids, accepted, num_accepted, batch_indices, kv_ref, rewind
    )
    assert torch.equal(tokens, ref_tokens) and torch.equal(next_tokens, ref_next)
    assert tokens.dtype == next_tokens.dtype == torch.int32
    assert torch.equal(tokens, draft_path_walk(edges, unary, ids))
    builtin_next = _builtin_tail(tokens, accepted, num_accepted, batch_indices, kv_builtin, rewind)
    assert builtin_next.dtype == next_tokens.dtype and torch.equal(builtin_next, next_tokens)
    if kv_lens is not None:
        assert torch.equal(kv_lens, kv_ref) and torch.equal(kv_lens, kv_builtin)
        assert (kv_lens[tokens.shape[0] :] == kv_ref[tokens.shape[0] :]).all()


def test_cuda_graph_replay_matches_the_reference():
    spec = dict(requests=1, positions=7, scores="random", rewind=True)
    edges, unary, ids = _walk_inputs(spec, 99)
    accepted, num_accepted, batch_indices, kv_lens, rewind = _tail_inputs(spec, 99)
    op(edges, unary, ids, accepted, num_accepted, batch_indices, kv_lens.clone(), rewind)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        tokens, next_tokens = op(
            edges, unary, ids, accepted, num_accepted, batch_indices, kv_lens, rewind
        )
    torch.cuda.current_stream().wait_stream(stream)
    for step in range(200):
        e, u, i = _walk_inputs(spec, 1000 + step)
        a, n, _, kv, _ = _tail_inputs(spec, 1000 + step)
        edges.copy_(e), unary.copy_(u), ids.copy_(i), accepted.copy_(a), num_accepted.copy_(n)
        rewind.copy_(1 - n)
        kv_lens.copy_(kv)
        kv_ref = kv.clone()
        graph.replay()
        ref_tokens, ref_next = op.reference(
            edges, unary, ids, accepted, num_accepted, batch_indices, kv_ref, rewind
        )
        torch.cuda.synchronize()
        assert torch.equal(tokens, ref_tokens) and torch.equal(next_tokens, ref_next)
        assert torch.equal(kv_lens, kv_ref)


def test_guard_refuses_what_the_kernel_cannot_take():
    spec = dict(requests=1, positions=7, scores="random", rewind=True)
    edges, unary, ids = _walk_inputs(spec, 5)
    accepted, num_accepted, batch_indices, kv_lens, rewind = _tail_inputs(spec, 5)
    with validating(op):
        with pytest.raises(ValueError):  # int64 accepted tokens
            op(edges, unary, ids, accepted.long(), num_accepted, batch_indices, kv_lens, rewind)
        with pytest.raises(ValueError):  # a rewind without KV lengths
            op(edges, unary, ids, accepted, num_accepted, batch_indices, None, rewind)
