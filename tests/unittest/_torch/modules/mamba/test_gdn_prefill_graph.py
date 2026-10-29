# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The context GDN core's CUDA graphs under a breakable prefill graph match its eager break."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.mamba import gdn_mixer
from tensorrt_llm._torch.modules.mamba.gdn_mixer import Qwen3NextGatedDeltaNet
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    enable_breakable_cuda_graph,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import InforwardSnapshotEntry

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not gdn_mixer._use_flashinfer_gdn_prefill(),
    reason="needs the FlashInfer GDN prefill",
)

NUM_TOKENS, NUM_SLOTS, STAGE_ROWS = 256, 8, 4
QKV_DIM, D_CONV = 2048, 4


def _make_layer() -> Qwen3NextGatedDeltaNet:
    layer = Qwen3NextGatedDeltaNet.__new__(Qwen3NextGatedDeltaNet)
    torch.nn.Module.__init__(layer)
    layer.layer_idx = 0
    layer.prefill_graphable = True
    layer.model_config = SimpleNamespace(extra_attrs={})
    layer.attn_tp_size = 1
    layer.num_k_heads = layer.num_k_heads_per_tp = 4
    layer.num_v_heads = layer.num_v_heads_per_tp = 8
    layer.head_k_dim = layer.head_v_dim = 128
    layer.activation = "silu"
    layer.conv1d = SimpleNamespace(
        weight=torch.randn(QKV_DIM, D_CONV, device="cuda", dtype=torch.bfloat16) * 0.1,
        bias=None,
    )
    layer.A_log = torch.randn(8, device="cuda") * 0.1
    layer.dt_bias = torch.randn(8, device="cuda") * 0.1
    return layer


class _Step:
    """One context-only step: the metadata the GDN core reads, over persistent buffers."""

    def __init__(self):
        self.conv = torch.zeros(NUM_SLOTS, QKV_DIM, D_CONV - 1, device="cuda", dtype=torch.bfloat16)
        self.ssm = torch.zeros(NUM_SLOTS, 8, 128, 128, device="cuda", dtype=torch.bfloat16)
        self.stage_ssm = torch.zeros(STAGE_ROWS, 8, 128, 128, device="cuda", dtype=torch.bfloat16)
        self.stage_conv = torch.zeros(
            STAGE_ROWS, QKV_DIM, D_CONV - 1, device="cuda", dtype=torch.bfloat16
        )
        self.slots = {}
        self.manager = SimpleNamespace(
            mamba_layer_cache=lambda layer_idx: SimpleNamespace(conv=self.conv, temporal=self.ssm),
            inforward_snapshot_stage_views=lambda layer_idx: (self.stage_ssm, self.stage_conv),
            conv_state_shape=[QKV_DIM, D_CONV - 1],
            inforward_snapshot_rows=STAGE_ROWS,
            inforward_snapshot_max_tail=4,
            state_index=lambda rid: self.slots[rid],
        )
        self.mamba = Mamba2Metadata(max_batch_size=4, chunk_size=128)
        self.attn = SimpleNamespace(
            kv_cache_manager=self.manager,
            mamba_metadata=self.mamba,
            # Size the pre-Blackwell prefill state workspace.
            max_num_sequences=4,
            max_num_requests=4,
            max_num_tokens=NUM_TOKENS,
        )

    def set(self, requests):
        """requests: (live slot, chunk length, has initial state, tokens after P or 0)."""
        n = len(requests)
        lengths = [length for _, length, _, _ in requests]
        self.slots = {rid: slot for rid, (slot, _, _, _) in enumerate(requests)}
        self.attn.num_contexts = n
        self.attn.seq_lens = torch.tensor(lengths, dtype=torch.int32)
        self.attn.num_ctx_tokens = self.attn.num_tokens = sum(lengths)
        self.attn.request_ids = list(range(n))
        mamba = self.mamba
        cu = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0, dtype=torch.int32)
        mamba.cu_seqlens[: n + 1].copy_(cu)
        mamba.query_start_loc = mamba.cu_seqlens[: n + 1]
        mamba._cu_seqlens_long[: n + 1].copy_(mamba.query_start_loc)
        mamba.query_start_loc_long = mamba._cu_seqlens_long[: n + 1]
        mamba.state_indices[:n].copy_(torch.tensor([slot for slot, _, _, _ in requests]))
        mamba.has_initial_states[:n].copy_(torch.tensor([init for _, _, init, _ in requests]))
        mamba.inforward_split = None
        plan = {
            rid: InforwardSnapshotEntry(row, length - after, length - after, after)
            for row, (rid, (_, length, _, after)) in enumerate(
                (rid, r) for rid, r in enumerate(requests) if r[3]
            )
        }
        if plan:
            mamba._prepare_inforward_split(self.attn, self.manager, plan, n, n)

    def states(self):
        return [t.clone() for t in (self.conv, self.ssm, self.stage_ssm, self.stage_conv)]

    def load(self, states):
        for t, saved in zip((self.conv, self.ssm, self.stage_ssm, self.stage_conv), states):
            t.copy_(saved)


@pytest.fixture
def harness(monkeypatch):
    torch.manual_seed(0)
    layer = _make_layer()
    step = _Step()
    monkeypatch.setattr(gdn_mixer, "_extract_gdn_extra_attrs", lambda idx: (step.attn, layer, None))
    # The prefill graph body runs on the token rows the BCG segment produced.
    projected_in = torch.zeros(NUM_TOKENS, QKV_DIM + 1024, device="cuda", dtype=torch.bfloat16)
    gates_in = torch.zeros(NUM_TOKENS, 16, device="cuda", dtype=torch.bfloat16)
    result = torch.zeros(1, NUM_TOKENS, 8, 128, device="cuda", dtype=torch.bfloat16)

    def body():
        projected, gates = projected_in * 1, gates_in * 1
        out = projected.new_empty(1, NUM_TOKENS, 8, 128)
        gdn_mixer.maybe_bcg_gdn_custom_op_inplace(
            projected[:, :QKV_DIM], gates[:, 8:], gates[:, :8], "0", out
        )
        result.copy_(out)

    step.set([(1, 100, False, 0)])
    graph = BreakableCUDAGraph()
    with enable_breakable_cuda_graph():
        with BreakableCUDAGraphCapture(graph, stream=torch.cuda.Stream()):
            body()
    torch.cuda.synchronize()
    pool = torch.cuda.graph_pool_handle()

    def run(requests, seed, capture=False):
        """Replay the bucket graph on a fresh step; returns (output rows, states)."""
        step.set(requests)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        projected_in.copy_(torch.randn(projected_in.shape, generator=gen, device="cuda") * 0.1)
        gates_in.copy_(torch.randn(gates_in.shape, generator=gen, device="cuda") * 0.1)
        # Garbage past the real tokens, as an attention output's unwritten rows.
        projected_in[step.attn.num_tokens :] = float("nan")
        gates_in[step.attn.num_tokens :] = float("nan")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), enable_breakable_cuda_graph():
            graph.replay(break_graph_pool=pool if capture else None)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        return result[:, : step.attn.num_tokens].clone(), step.states()

    def eager(requests, seed):
        """The eager break on the same step (the reference)."""
        step.set(requests)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        projected = torch.randn(projected_in.shape, generator=gen, device="cuda") * 0.1
        gates = torch.randn(gates_in.shape, generator=gen, device="cuda") * 0.1
        projected, gates = projected.to(torch.bfloat16), gates.to(torch.bfloat16)
        out = torch.zeros(1, NUM_TOKENS, 8, 128, device="cuda", dtype=torch.bfloat16)
        gdn_mixer.gdn_custom_op_inplace(
            projected[:, :QKV_DIM], gates[:, 8:], gates[:, :8], "0", out
        )
        torch.cuda.synchronize()
        return out[:, : step.attn.num_tokens].clone(), step.states()

    def check(requests, seed, initial):
        step.load(initial)
        expected_out, expected_states = eager(requests, seed)
        step.load(initial)
        out, states = run(requests, seed)
        torch.testing.assert_close(out, expected_out, rtol=1.6e-2, atol=1e-3)
        for actual, expected in zip(states, expected_states):
            torch.testing.assert_close(actual, expected, rtol=1.6e-2, atol=1e-3)

    return SimpleNamespace(graph=graph, step=step, run=run, check=check)


def _initial(step):
    gen = torch.Generator(device="cuda").manual_seed(7)
    return [
        (torch.randn(t.shape, generator=gen, device="cuda") * 0.1).to(t.dtype)
        for t in step.states()
    ]


# (live slot, chunk length, has initial state, tokens after P or 0)
_FRESH = [(1, 100, False, 0)]
_CACHED_PREFIX = [(3, 177, True, 0)]
_TWO_SPLIT_ONE = [(5, 64, True, 0), (2, 130, True, 2)]


@pytest.mark.parametrize(
    "warm, cases",
    [
        # Fresh prompts of other lengths than the warmup's, padded to the bucket.
        (_FRESH, [[(6, 37, False, 0)], [(0, 256, False, 0)]]),
        # Prefix reuse: the core starts from the live slot's state.
        (_CACHED_PREFIX, [[(4, 9, True, 0)], [(3, 200, True, 0)]]),
        # In-forward snapshot split: other lengths and slots under one key.
        (
            _TWO_SPLIT_ONE,
            [[(7, 21, False, 0), (1, 90, True, 2)], [(0, 100, True, 0), (4, 6, True, 2)]],
        ),
    ],
)
def test_prefill_graph_matches_eager(harness, warm, cases):
    initial = _initial(harness.step)
    harness.step.load(initial)
    harness.run(warm, seed=1, capture=True)
    assert harness.graph.num_break_graphs == 1
    for seed, requests in enumerate(cases, start=2):
        harness.check(requests, seed, initial)
    # Each case replayed the warmup's graph.
    assert harness.graph.num_break_graphs == 1


def test_prefill_graph_does_not_replay_against_moved_metadata(harness):
    initial = _initial(harness.step)
    harness.step.load(initial)
    harness.run(_TWO_SPLIT_ONE, seed=1, capture=True)
    # The metadata is rebuilt: its buffers move, and the old ones hold another batch.
    old = harness.step.mamba
    harness.step.mamba = harness.step.attn.mamba_metadata = Mamba2Metadata(4, 128)
    harness.step.set([(5, 64, True, 0), (2, 130, True, 2)])
    old.cu_seqlens.fill_(1)
    old.state_indices.fill_(0)
    old._inforward_split_buf.zero_()
    harness.check([(6, 50, True, 0), (1, 70, False, 2)], seed=2, initial=initial)
    assert harness.graph.num_break_graphs == 1


def test_prefill_graph_key_only_for_context_only_flashinfer_batches():
    layer = _make_layer()
    attn = SimpleNamespace(num_contexts=1, seq_lens=torch.tensor([10, 1]))
    assert layer.prefill_graph_key(attn) is None
    layer.prefill_graphable = False
    attn = SimpleNamespace(num_contexts=1, seq_lens=torch.tensor([10]))
    assert layer.prefill_graph_key(attn) is None
