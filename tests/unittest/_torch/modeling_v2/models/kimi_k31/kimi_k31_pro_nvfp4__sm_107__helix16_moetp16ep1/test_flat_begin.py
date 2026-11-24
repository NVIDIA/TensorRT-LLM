# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The flat chain's folded step begin (``ops.embedding.begin_fold``) against the unfolded ops.

The unfolded begin is ``layers.embedding`` then ``FlatChainRuntime.begin_step``, whose
``KDAReplayState.checkpoint_indices`` writes the checkpoint slots and accepted counts. Both
begins are captured in CUDA graphs from the same static inputs and replayed over a sequence of
steps: b1 and b4, q8 and plain q1, CUDA-graph padding rows (1-3 live requests in the b4 graph,
dummy state slot and padded token ids, including ids outside the table), a released slot reused
by a new request, accepted counts at 0 and at 7, and slot strides whose int64 products truncate
in the int32 cast. Every tensor the begin writes, the embedding rows included, must be bitwise
equal to the unfolded one and to an eager reference.
"""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEV = "cuda"
TOKENS = 8
CAPACITY = 12
DUMMY_SLOT = CAPACITY - 1
VOCAB = 1000
HIDDEN = 7168
COUNT_STRIDE = 4

_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1"


def _replay(units):
    from importlib import import_module

    state = object.__new__(
        import_module(
            f"{_PKG.rpartition('.')[0]}.ops.ssm.rubin_ull_kda_dsa_cache_manager"
        ).KDAReplayState
    )
    state.checkpoints = True
    state.checkpoint_slot_units, state.checkpoint_state_units, state.checkpoint_header_units = units
    state.num_speculative_tokens = TOKENS - 1
    # The manager's count is the first int32 of each slot's record: a strided view.
    record = torch.full((CAPACITY, COUNT_STRIDE), -3, dtype=torch.int32, device=DEV)
    state.prev_num_accepted_tokens = record[:, 0]
    return state


def _chain(replay, physical, fold, rows):
    from importlib import import_module

    chain = object.__new__(import_module(f"{_PKG}.modeling").FlatChainRuntime)
    manager = SimpleNamespace(uses_kda_checkpoints=True, _kda_replay=replay)
    chain.rows, chain.tokens, chain.kda = rows, TOKENS, {}
    chain._manager = manager
    chain.state_indices = lambda metadata: physical
    chain.begin_fold = fold
    chain._begin_fold_logged = False
    chain.mapping = SimpleNamespace(tp_rank=0)
    chain._slots = chain._accepted = None
    return chain, SimpleNamespace(num_tokens=None, num_contexts=0, kv_cache_manager=manager)


def _begin(chain, metadata, ids, table):
    """What ``FlatModel.__call__`` runs: the folded begin, else the gather and ``begin_step``."""
    from importlib import import_module

    glue = import_module(f"{_PKG}.modeling")
    rows = chain.begin_step_embedding(metadata, ids, table)
    if rows is None:
        rows = glue.embedding(ids, glue.Bf16Weight(table))
        chain.begin_step(metadata)
    return rows, chain._slots, chain._accepted


def _reference(replay, physical, ids, table):
    from importlib import import_module

    glue = import_module(f"{_PKG}.modeling")
    slots, accepted = replay.checkpoint_indices(physical)
    return glue.embedding(ids, glue.Bf16Weight(table)), slots, accepted


# (live state slots, previous accepted counts of those slots, token ids or None for random)
_STEPS = [
    ([3], [0], None),
    ([3], [7], None),
    ([3, 5], [2, 0], None),
    ([3, 5, 0], [7, 7, 1], None),
    ([3, 5, 0, 9], [0, 3, 7, 6], None),
    # Request on slot 5 left; a new one reuses slot 5 with a fresh count.
    ([3, 0, 9, 5], [4, 7, 0, 0], None),
    # Ids at and outside the table edges (copied as zero rows by both begins).
    ([9, 5], [1, 7], [0, VOCAB - 1, VOCAB, -1, VOCAB + 7, 1, 2, 3]),
    ([10], [7], None),
]


def _fill(physical_buf, ids_buf, replay, step, live_rows_per_req, gen):
    slots, counts, ids = step
    batch = physical_buf.shape[0]
    padded = slots + [DUMMY_SLOT] * (batch - len(slots))
    physical_buf.copy_(torch.tensor(padded, dtype=torch.int32))
    replay.prev_num_accepted_tokens.fill_(-3)
    replay.prev_num_accepted_tokens[DUMMY_SLOT] = 0
    for slot, count in zip(slots, counts):
        replay.prev_num_accepted_tokens[slot] = count
    live = len(slots) * live_rows_per_req
    values = torch.randint(0, VOCAB, (ids_buf.shape[0],), generator=gen).to(ids_buf.dtype)
    if ids is not None:
        pattern = torch.tensor(ids, dtype=ids_buf.dtype)
        values[: min(live, len(ids))] = pattern[: min(live, len(ids))]
    # Padding rows carry the dummy request's ids (here 0) and an out-of-table id.
    values[live:] = 0
    if live < values.shape[0]:
        values[-1] = VOCAB + 3
    ids_buf.copy_(values)


@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("width", [TOKENS, 1])
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize(
    "units",
    [(3 * TOKENS + 2, 3, 1), ((1 << 28) + 5, (1 << 26) + 1, 7)],
    ids=["dense", "truncating"],
)
def test_folded_begin_matches_the_unfolded_ops_under_graph_replay(units, batch, width, ids_dtype):
    gen = torch.Generator().manual_seed(batch * 31 + width)
    table = torch.randn((VOCAB, HIDDEN), generator=gen).to(torch.bfloat16).to(DEV)
    rows = batch * width
    physical = torch.zeros(batch, dtype=torch.int32, device=DEV)
    ids = torch.zeros(rows, dtype=ids_dtype, device=DEV)
    replay = _replay(units)

    captured = {}
    for fold in (True, False):
        chain, metadata = _chain(replay, physical, fold, (rows,))
        metadata.num_tokens = rows
        _fill(physical, ids, replay, _STEPS[0], width, gen)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _begin(chain, metadata, ids, table)  # warm-up (compiles the kernel)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = _begin(chain, metadata, ids, table)
        captured[fold] = (graph, outputs)
    for step in _STEPS:
        if len(step[0]) > batch:
            continue
        _fill(physical, ids, replay, step, width, gen)
        reference = _reference(replay, physical, ids, table)
        for fold in (True, False):
            for out in captured[fold][1]:
                out.fill_(float("nan") if out.is_floating_point() else -1)
            captured[fold][0].replay()
        torch.cuda.synchronize()
        names = ("embedding", "slots", "accepted")
        for name, folded, unfolded, ref in zip(
            names, captured[True][1], captured[False][1], reference
        ):
            assert folded.dtype == unfolded.dtype == ref.dtype, name
            assert folded.shape == unfolded.shape == ref.shape, name
            # Bitwise: compare the bytes, so NaN or signed zeros cannot pass by value.
            assert torch.equal(_bits(folded), _bits(unfolded)), f"{name} differs at step {step}"
            assert torch.equal(_bits(folded), _bits(ref)), f"{name} differs from eager at {step}"


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t


def test_fold_switch_and_inputs_it_does_not_take():
    from importlib import import_module

    flat_begin = import_module(f"{_PKG.rpartition('.')[0]}.kernels.embedding.rubin_ull_flat_begin")
    table = torch.zeros((VOCAB, HIDDEN), dtype=torch.bfloat16, device=DEV)
    replay = _replay((3 * TOKENS + 2, 3, 1))
    physical = torch.zeros(1, dtype=torch.int32, device=DEV)
    chain, metadata = _chain(replay, physical, False, (TOKENS,))
    metadata.num_tokens = TOKENS
    ids = torch.zeros(TOKENS, dtype=torch.int32, device=DEV)
    assert chain.begin_step_embedding(metadata, ids, table) is None
    assert chain._slots is None, "the off switch must leave the begin to begin_step"
    chain.begin_fold = True
    assert chain.begin_step_embedding(metadata, ids[None], table) is None
    assert chain.begin_step_embedding(metadata, ids, table.t().contiguous().t()) is None
    assert chain._slots is None
    assert chain.begin_step_embedding(metadata, ids, table) is not None
    assert chain._slots is not None
    assert not hasattr(flat_begin, "begin_fold_enabled"), "the fold has no kill switch"


def test_a_mixed_step_is_refused_before_any_begin():
    from importlib import import_module

    flat_model = import_module(f"{_PKG}.modeling")
    model = object.__new__(flat_model.FlatModel)
    calls = []
    model.backend = SimpleNamespace(
        begin_step_embedding=lambda *a: calls.append("fold"),
        begin_step=lambda *a: calls.append("begin"),
    )
    metadata = SimpleNamespace(num_contexts=1, num_tokens=TOKENS + 5, is_ragged_verify=False)
    ids = torch.zeros(TOKENS + 5, dtype=torch.int32, device=DEV)
    with pytest.raises(ValueError, match="does not serve context requests"):
        model(metadata, input_ids=ids)
    assert not calls

    # The chain refuses a context row count itself as well, folded or not.
    replay = _replay((3 * TOKENS + 2, 3, 1))
    for fold in (True, False):
        chain, chain_metadata = _chain(
            replay, torch.zeros(2, dtype=torch.int32, device=DEV), fold, (TOKENS,)
        )
        chain_metadata.num_tokens, chain_metadata.num_contexts = TOKENS, 1
        table = torch.zeros((VOCAB, HIDDEN), dtype=torch.bfloat16, device=DEV)
        with pytest.raises(ValueError, match="generation-only"):
            _begin(chain, chain_metadata, ids[:TOKENS], table)
        assert chain._slots is None
