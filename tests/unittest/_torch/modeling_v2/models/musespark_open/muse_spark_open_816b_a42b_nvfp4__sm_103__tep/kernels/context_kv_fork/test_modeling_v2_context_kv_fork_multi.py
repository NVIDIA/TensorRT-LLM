# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for ``context_kv_fork`` with several generation requests per step, on the published drafter.

Two and three requests (verify width M = 16 / 24 rows) in distinct slots that
are not the identity mapping, a different accepted count per request every
step: the forked prepare against the built-in one, then the drafter's block
decode, bitwise -- eagerly with a busy main stream, and under CUDA-graph
replay whose inputs change between replays (captured rows, counts, and the
batch-to-slot mapping).

Needs ``MUSESPARK_REFERENCE_DRAFTER`` (the published drafter checkpoint).
"""

import os
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.context_kv_fork import (  # noqa: E501
    fork as context_kv_fork,
)

__extra_import_path__ = ["."]
from test_modeling_v2_context_kv_fork import (  # noqa: E402
    MAX_CTX,
    TAPS,
    K,
    _assert_same,
    _busy_main_stream,
    _decode,
    _worker,
    drafter,  # noqa: E402,F401
)

_DRAFTER_DIR = os.environ.get("MUSESPARK_REFERENCE_DRAFTER")
NUM_SLOTS = 5


class _Spec:
    def __init__(self, hidden, gens):
        self.hidden_size = hidden
        self.num_capture_layers = len(TAPS)
        self.layers_to_capture = list(TAPS)
        self.runtime_draft_len = K
        self.captured_hidden_states = torch.zeros(
            gens * (K + 1) + 8, hidden * len(TAPS), dtype=torch.bfloat16, device="cuda"
        )

    def get_hidden_states(self, num_tokens):
        return self.captured_hidden_states[
            :num_tokens, : self.hidden_size * self.num_capture_layers
        ]


def _attn(gens):
    return SimpleNamespace(
        num_contexts=0,
        num_seqs=gens,
        num_tokens=gens * (K + 1),
        num_ctx_tokens=0,
        _seq_lens_cuda=torch.zeros(gens, dtype=torch.int32, device="cuda"),
        _seq_lens=torch.zeros(gens, dtype=torch.int32),
    )


def _mworker(model, starts, slots, fork=False):
    worker = _worker(model, 0)
    L, nkv, hd = model._num_attn_layers, model._num_kv_heads, model._head_dim
    g = torch.Generator(device="cuda").manual_seed(77)
    shape = (NUM_SLOTS, L, MAX_CTX + K + 1, nkv, hd)
    worker._ctx_k_buf = torch.randn(shape, generator=g, device="cuda").to(torch.bfloat16)
    worker._ctx_v_buf = torch.randn(shape, generator=g, device="cuda").to(torch.bfloat16)
    lengths = [11 * (s + 1) for s in range(NUM_SLOTS)]
    for slot, start in zip(slots, starts):
        lengths[slot] = start
    worker._ctx_len = torch.tensor(lengths, dtype=torch.int64, device="cuda")
    worker._batch_to_slot = torch.tensor(list(slots), dtype=torch.long, device="cuda")
    if fork:
        context_kv_fork._reclass(worker)
        worker._kv_fork = context_kv_fork.ContextKvFork(worker, model, "head")
    return worker


def _copy(dst, src):
    for name in ("_ctx_k_buf", "_ctx_v_buf", "_ctx_len", "_batch_to_slot"):
        getattr(dst, name).copy_(getattr(src, name))


class _MInputs:
    def __init__(self, hidden, gens):
        self.gens = gens
        self.spec = _Spec(hidden, gens)
        self.attn = _attn(gens)
        self.accepted = torch.zeros(gens, K + 1, dtype=torch.int32, device="cuda")
        self.num_accepted = torch.ones(gens, dtype=torch.int32, device="cuda")
        self.hidden = torch.zeros(gens * (K + 1), hidden, dtype=torch.bfloat16, device="cuda")

    def load(self, step, vocab):
        g = torch.Generator(device="cuda").manual_seed(5000 + step)
        rows = self.spec.captured_hidden_states
        rows.copy_((2.0 * torch.randn(rows.shape, generator=g, device="cuda")).to(rows.dtype))
        self.accepted.copy_(
            torch.randint(0, vocab - 1, self.accepted.shape, generator=g, device="cuda")
        )
        counts = [1 + (step * (r + 3) + r) % (K + 1) for r in range(self.gens)]
        self.num_accepted.copy_(torch.tensor(counts, dtype=torch.int32))

    def prepare(self, worker, model):
        return worker.prepare_1st_drafter_inputs(
            input_ids=None,
            position_ids=None,
            hidden_states=self.hidden,
            accepted_tokens=self.accepted,
            num_accepted_tokens=self.num_accepted,
            attn_metadata=self.attn,
            spec_metadata=self.spec,
            draft_model=model,
            total_target_tokens=self.gens * (K + 1),
        )


def _check(forked, base, got, ref, what):
    for key in ("noise_embedding", "query_positions", "num_ctx_per_req", "first_prev_tokens",
                "ctx_cache_batch_idx"):  # fmt: skip
        _assert_same(got[key], ref[key], f"{what} {key}")
    _assert_same(forked._ctx_len, base._ctx_len, f"{what} ctx_len")
    _assert_same(forked._ctx_k_buf, base._ctx_k_buf, f"{what} arena K")
    _assert_same(forked._ctx_v_buf, base._ctx_v_buf, f"{what} arena V")


@pytest.mark.skipif(not _DRAFTER_DIR, reason="needs MUSESPARK_REFERENCE_DRAFTER")
@pytest.mark.parametrize("slots", [(3, 1), (2, 4, 0)], ids=["gens2", "gens3"])
@torch.inference_mode()
def test_multi_request_eager_bitwise(drafter, slots):  # noqa: F811
    gens = len(slots)
    starts = [1_000 + 37 * i for i in range(gens)]
    base = _mworker(drafter, starts, slots)
    forked = _mworker(drafter, starts, slots, fork=True)
    vocab = drafter.draft_model_full.model.embed_tokens.num_embeddings
    inp = _MInputs(drafter.config.hidden_size, gens)
    for step in range(2 * (K + 1)):
        inp.load(step, vocab)
        ref = inp.prepare(base, drafter)
        prepared = _mworker(drafter, starts, slots)
        _copy(prepared, base)
        ref_out = _decode(drafter, ref, base)
        fork = forked._kv_fork
        assert fork.arm(inp.spec, inp.attn) is None
        fork.fire("head")
        assert fork.launched, "a multi-request decode step must fork"
        _busy_main_stream()
        got = inp.prepare(forked, drafter)
        _check(forked, prepared, got, ref, f"gens {gens} step {step}")
        got_out = _decode(drafter, got, forked)
        _assert_same(got_out, ref_out, f"gens {gens} step {step} drafter output")
    print(f"context_kv_fork multi-request eager gens={gens}: {2 * (K + 1)} steps bitwise")


@pytest.mark.skipif(not _DRAFTER_DIR, reason="needs MUSESPARK_REFERENCE_DRAFTER")
@torch.inference_mode()
def test_multi_request_graph_replay_with_changing_slot_map(drafter):  # noqa: F811
    slots = (3, 1)
    gens = len(slots)
    starts = [2_000, 2_345]
    base = _mworker(drafter, starts, slots)
    forked = _mworker(drafter, starts, slots, fork=True)
    snapshot = _mworker(drafter, starts, slots)
    _copy(snapshot, forked)
    vocab = drafter.draft_model_full.model.embed_tokens.num_embeddings
    inp = _MInputs(drafter.config.hidden_size, gens)
    fork = forked._kv_fork
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for step in range(2):
            inp.load(step, vocab)
            fork.arm(inp.spec, inp.attn)
            fork.fire("head")
            _decode(drafter, inp.prepare(forked, drafter), forked)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    fork.arm(inp.spec, inp.attn)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fork.fire("head")
        static_inputs = inp.prepare(forked, drafter)
        static_out = _decode(drafter, static_inputs, forked)
    _copy(forked, snapshot)
    _copy(base, snapshot)
    maps = [(3, 1), (3, 1), (1, 3), (0, 4), (4, 2), (3, 1)]
    for step in range(3 * (K + 1)):
        mapping = torch.tensor(maps[step % len(maps)], dtype=torch.long, device="cuda")
        base._batch_to_slot.copy_(mapping)
        forked._batch_to_slot.copy_(mapping)
        inp.load(300 + step, vocab)
        ref = inp.prepare(base, drafter)
        ref_out = _decode(drafter, ref, base)
        graph.replay()
        torch.cuda.synchronize()
        _check(forked, base, static_inputs, ref, f"replay {step} map {maps[step % len(maps)]}")
        _assert_same(static_out, ref_out, f"replay {step} drafter output")
    print(
        f"context_kv_fork multi-request graph replay: {3 * (K + 1)} replays bitwise, slot map changing"
    )
