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
"""``layout_mapping`` against a byte oracle: each rank's page holds, at every byte of every tensor,
the id of the global element there; after every source rank of an instance sends what
``map_buffers`` gives it, every byte a destination rank holds of every buffer both transfer has
arrived exactly once, holding its own global element, and only from source ranks it pairs with;
a pairing whose source ranks do not hold everything a destination holds is refused."""

from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest
from utils.page_layout_oracle import assert_layouts_reshard

from tensorrt_llm._torch.pyexecutor.kv_cache.layout_mapping import (
    BufferRuns,
    check_compatible,
    fan,
    map_buffers,
    merge_runs,
    sent_bytes,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.page_layout import (
    Axis,
    BufferDesc,
    LayerGroupDesc,
    PageLayout,
    Progression,
    RankTopology,
    Tensor,
    head_major,
    held,
    replicated,
    sectioned,
    token_major,
)

pytestmark = pytest.mark.cpu_only

HEAD = 16  # bytes of one head of one page
TOKENS = 4


@dataclass(frozen=True)
class Instance:
    """An instance: its parallel sizes and the global layers of each pipeline stage."""

    tp: int = 1
    cp: int = 1
    attention_dp: bool = False
    stages: tuple[tuple[int, ...], ...] = ((0, 1),)

    def ranks(self, dp: int) -> list[RankTopology]:
        """The ranks serving the requests of attention data-parallel rank ``dp``."""
        tps = [dp] if self.attention_dp else range(self.tp)
        return [
            RankTopology(
                tp_size=self.tp,
                tp_rank=t,
                pp_size=len(self.stages),
                pp_rank=p,
                cp_size=self.cp,
                cp_rank=c,
                dp_size=self.tp if self.attention_dp else 1,
                dp_rank=t if self.attention_dp else 0,
                attention_dp=self.attention_dp,
            )
            for p in range(len(self.stages))
            for t in tps
            for c in range(self.cp)
        ]


# A buffer spec: (role, topology -> tensors, kind of its layer group).
Spec = tuple[str, Callable[[RankTopology], tuple], str]


def _tp(topology: RankTopology) -> int:
    return 1 if topology.attention_dp else topology.tp_size


def kv(total: int, layout: str = "hnd") -> Callable[[RankTopology], tuple]:
    def tensors(topology):
        heads = -(-total // _tp(topology))
        if layout == "nhd":
            return token_major(TOKENS * heads * HEAD, TOKENS, heads, total)
        return head_major(heads * HEAD, heads, total)

    return tensors


def state_heads(total: int) -> Callable[[RankTopology], tuple]:
    def tensors(topology):
        ranks = _tp(topology) * topology.cp_size
        return head_major(total // ranks * HEAD, total // ranks, total, ("tp", "cp"))

    return tensors


def conv(sections: tuple[int, ...]) -> Callable[[RankTopology], tuple]:
    def tensors(topology):
        ranks = _tp(topology) * topology.cp_size
        return sectioned([s // ranks for s in sections], ranks, ("tp", "cp"))

    return tensors


def whole(size: int) -> Callable[[RankTopology], tuple]:
    return lambda topology: replicated(size)


def layout(topology: RankTopology, layers: tuple[int, ...], specs: list[Spec]) -> PageLayout:
    """One rank's page: each kind's layers in a pool group of their own, buffers back to back."""
    kinds = sorted({kind for _, _, kind in specs})
    buffers, widths = [], {}
    for g, kind in enumerate(kinds):
        offset = 0
        for layer in layers:
            for role, make, k in specs:
                if k != kind:
                    continue
                tensors = make(topology)
                size = max(
                    t.offset + 1 + sum((len(held(a, topology)) - 1) * a.stride for a in t.axes)
                    for t in tensors
                )
                buffers.append(BufferDesc(layer, role, g, 0, offset, size, tensors))
                offset += size
        widths[g] = (offset,)
    groups = tuple(
        LayerGroupDesc(kind, None, 0, g, layers) for g, kind in enumerate(kinds) if widths[g][0]
    )
    return PageLayout(TOKENS, widths, groups, tuple(buffers), topology)


def check(src: Instance, dst: Instance, specs: list[Spec], *, src_dp: int = 0) -> None:
    """Every destination rank of every data-parallel group gets each byte it holds exactly once,
    holding the right element, and only from source ranks of its fan."""
    sources = [layout(t, src.stages[t.pp_rank], specs) for t in src.ranks(src_dp)]
    destinations = [
        layout(t, dst.stages[t.pp_rank], specs)
        for dst_dp in range(dst.tp if dst.attention_dp else 1)
        for t in dst.ranks(dst_dp)
    ]
    assert_layouts_reshard(sources, destinations)


TPS = (1, 2, 4, 8)


def fan_holds(held_by: Callable[[int, int], set], src_n: int, dst_n: int) -> bool:
    """Whether every destination rank's indices lie on the source ranks of its fan, given
    ``held_by(rank, count)``: the indices rank ``rank`` of ``count`` holds."""
    return all(
        held_by(d, dst_n) <= set().union(*(held_by(s, src_n) for s in fan(src_n, dst_n, d)))
        for d in range(dst_n)
    )


def contiguous_heads(total: int) -> Callable[[int, int], set]:
    return lambda rank, count: set(
        range(rank * total // count, rank * total // count + -(-total // count))
    )


@pytest.mark.parametrize("heads", [1, 2, 4, 8, 6])
@pytest.mark.parametrize("kv_layout", ["hnd", "nhd"])
@pytest.mark.parametrize("src_tp,dst_tp", list(itertools.product(TPS, TPS)))
def test_attention_heads_reshard_over_tensor_ranks(heads, kv_layout, src_tp, dst_tp):
    """Heads reshard wherever each destination's fan holds its heads; a split that leaves a
    destination's head outside its fan (6 heads over 8 ranks to 4) is refused."""
    specs = [("key", kv(heads, kv_layout), "attention"), ("index", whole(24), "attention")]
    if fan_holds(contiguous_heads(heads), src_tp, dst_tp):
        check(Instance(tp=src_tp), Instance(tp=dst_tp), specs)
    else:
        with pytest.raises(ValueError, match="no source rank it pairs with"):
            check(Instance(tp=src_tp), Instance(tp=dst_tp), specs)


def test_some_uneven_head_splits_are_refused():
    """The refusal above is reached: 6 heads over 8 ranks leave head 1 of TP4 rank 0 on rank 2."""
    assert not fan_holds(contiguous_heads(6), 8, 4)


@pytest.mark.parametrize("src_adp,dst_adp", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("src_tp,dst_tp", [(2, 4), (4, 2), (4, 4), (1, 4)])
def test_attention_data_parallel_holds_whole_heads(src_adp, dst_adp, src_tp, dst_tp):
    specs = [("key", kv(8), "attention"), ("index", whole(24), "attention")]
    for src_dp in range(src_tp if src_adp else 1):
        check(
            Instance(tp=src_tp, attention_dp=src_adp),
            Instance(tp=dst_tp, attention_dp=dst_adp),
            specs,
            src_dp=src_dp,
        )


@pytest.mark.parametrize("src_tp,dst_tp", list(itertools.product((1, 2, 4), (1, 2, 4))))
@pytest.mark.parametrize("dst_cp", [1, 2])
def test_recurrent_state_splits_over_tensor_times_context_ranks(src_tp, dst_tp, dst_cp):
    specs = [
        ("ssm", state_heads(8), "state"),
        ("conv", conv((16, 8, 8)), "state"),
        ("ple", whole(12), "state"),
        ("key", kv(4), "attention"),
    ]
    check(Instance(tp=src_tp), Instance(tp=dst_tp, cp=dst_cp), specs)


@pytest.mark.parametrize(
    "src_stages,dst_stages",
    [
        (((0, 1, 2, 3),), ((0, 1), (2, 3))),
        (((0, 1), (2, 3)), ((0,), (1, 2), (3,))),
        (((0, 1, 2), (3,)), ((0, 1, 2, 3),)),
    ],
)
@pytest.mark.parametrize("src_tp,dst_tp", [(1, 2), (2, 1), (2, 2)])
def test_pipeline_stages_pair_by_layer(src_stages, dst_stages, src_tp, dst_tp):
    specs = [("key", kv(4), "attention"), ("ssm", state_heads(4), "state")]
    check(Instance(tp=src_tp, stages=src_stages), Instance(tp=dst_tp, stages=dst_stages), specs)


def test_a_two_dimensional_split_reshards_both_axes():
    """Heads over tensor ranks and tokens over context ranks, which no single split expresses."""

    def tensors(topology):
        heads = 4 // topology.tp_size
        tokens = 8 // topology.cp_size
        return (
            Tensor(
                (
                    Axis("token", 8, heads * HEAD, ("cp",), tokens),
                    Axis("head", 4, HEAD, ("tp",), heads),
                    Axis("byte", HEAD, 1),
                )
            ),
        )

    specs = [("key", tensors, "attention")]
    for src, dst in [((1, 1), (2, 2)), ((2, 2), (1, 1)), ((2, 1), (1, 2)), ((4, 2), (2, 4))]:
        check(Instance(tp=src[0], cp=src[1]), Instance(tp=dst[0], cp=dst[1]), specs)


def test_a_cyclic_split_hands_each_rank_every_nth_index():
    """A cyclic split reshards to or from a single rank; between other rank counts a destination's
    rows lie outside its fan, which is refused."""

    def tensors(topology):
        return (Tensor((Axis("row", 12, HEAD, ("tp",), cyclic=True), Axis("byte", HEAD, 1))),)

    def rows(rank, count):
        return set(range(rank, 12, count))

    specs = [("rows", tensors, "attention")]
    for src_tp, dst_tp in [(1, 3), (3, 1), (2, 2), (2, 4), (4, 2), (3, 2)]:
        if fan_holds(rows, src_tp, dst_tp):
            check(Instance(tp=src_tp), Instance(tp=dst_tp), specs)
        else:
            with pytest.raises(ValueError, match="no source rank it pairs with"):
                check(Instance(tp=src_tp), Instance(tp=dst_tp), specs)
    assert not fan_holds(rows, 2, 4)


def test_runs_merge_where_both_sides_continue():
    src = np.array([0, 8, 32, 40], dtype=np.int64)
    dst = np.array([100, 108, 200, 240], dtype=np.int64)
    sizes = np.array([8, 8, 8, 8], dtype=np.int64)
    merged = merge_runs(src, dst, sizes)
    assert [list(a) for a in merged] == [[0, 32, 40], [100, 200, 240], [16, 8, 8]]


def test_a_whole_page_moves_as_one_run_when_layouts_match():
    specs = [("key", kv(4), "attention"), ("value", kv(4), "attention")]
    topology = Instance(tp=2).ranks(0)[1]
    page = layout(topology, (0, 1), specs)
    maps = map_buffers(page, page)
    src = np.concatenate([m.src_offsets for m in maps])
    sizes = np.concatenate([m.sizes for m in maps])
    assert len(src) == len(maps) and int(sizes.sum()) == page.pool_slot_bytes[0][0]


def test_progressions_intersect_by_their_common_residues():
    a, b = Progression(1, 4, 10), Progression(3, 6, 10)  # 1, 5, 9, ...; 3, 9, 15, ...
    assert a.intersect(b) == Progression(9, 12, 3)  # 9, 21, 33
    assert Progression(0, 2, 5).intersect(Progression(1, 2, 5)) is None
    assert Progression(4, 1, 2).intersect(Progression(0, 1, 4)) is None


def test_mismatched_tensors_are_refused():
    topology = Instance().ranks(0)[0]
    a = layout(topology, (0,), [("key", kv(4), "attention")])
    b = layout(topology, (0,), [("key", kv(8), "attention")])
    with pytest.raises(ValueError, match="extent"):
        map_buffers(a, b)


def test_layouts_of_one_model_correspond_across_parallelism():
    """Ranks of different TP, attention DP and pipeline stages describe the same global tensors."""
    specs = [
        ("key", kv(8), "attention"),
        ("index", whole(24), "attention"),
        ("ssm", state_heads(8), "state"),
        ("conv", conv((16, 8, 8)), "state"),
    ]
    pages = [
        layout(t, layers, specs)
        for instance, layers in [
            (Instance(tp=1), (0, 1)),
            (Instance(tp=4), (0, 1)),
            (Instance(tp=4, attention_dp=True), (0,)),
            (Instance(tp=2, cp=2), (1, 2)),
        ]
        for t in instance.ranks(0)[:1]
    ]
    for a, b in itertools.product(pages, pages):
        check_compatible(a, b)


def test_layouts_that_do_not_correspond_are_refused():
    topology = Instance().ranks(0)[0]
    base = layout(topology, (0,), [("key", kv(4), "attention"), ("ple", whole(12), "state")])
    with pytest.raises(ValueError, match=r"\(0, 'key'\) tensor '': axes \{'head': 4"):
        check_compatible(
            base, layout(topology, (0,), [("key", kv(8), "attention"), ("ple", whole(12), "state")])
        )
    with pytest.raises(ValueError, match=r"\(0, 'ple'\) is transferred by one rank only"):
        check_compatible(base, layout(topology, (0,), [("key", kv(4), "attention")]))
    untransferred = dataclasses.replace(
        base,
        buffers=tuple(dataclasses.replace(b, transferred=b.role != "ple") for b in base.buffers),
    )
    with pytest.raises(ValueError, match="transferred by one rank only"):
        check_compatible(base, untransferred)
    with pytest.raises(ValueError, match="pages of 4 tokens against 8"):
        check_compatible(base, dataclasses.replace(base, tokens_per_page=8))
    # Layers only one rank holds are not compared: they simply do not move.
    check_compatible(base, layout(topology, (1,), [("key", kv(8), "attention")]))


def test_the_oracle_refuses_runs_into_bytes_no_transferred_tensor_holds(monkeypatch):
    """Every transferred byte arriving once is not enough: a run into a local-only buffer, or one
    past the slot, fails the oracle too."""
    import utils.page_layout_oracle as oracle

    topology = Instance().ranks(0)[0]
    page = layout(topology, (0,), [("key", kv(4), "attention")])
    width = page.pool_slot_bytes[0][0]
    scratch = BufferDesc(0, "scratch", 0, 0, width, 8, transferred=False)
    page = dataclasses.replace(
        page, pool_slot_bytes={0: (width + 8,)}, buffers=(*page.buffers, scratch)
    )
    oracle.assert_layouts_reshard([page], [page])
    sent = oracle.map_buffers

    def also_writing(at: int):
        def runs(src, dst, **kwargs):
            out = sent(src, dst, **kwargs)
            first = out[0]
            extra = BufferRuns(
                first.src, first.dst, np.array([0]), np.array([at]), np.array([4], dtype=np.int64)
            )
            return [*out, extra]

        return runs

    monkeypatch.setattr(oracle, "map_buffers", also_writing(width))
    with pytest.raises(AssertionError, match="written, but no transferred tensor holds it"):
        oracle.assert_layouts_reshard([page], [page])
    monkeypatch.setattr(oracle, "map_buffers", also_writing(width + 6))
    with pytest.raises(AssertionError):
        oracle.assert_layouts_reshard([page], [page])


def test_the_oracle_gives_layer_groups_sharing_a_pool_group_slots_of_their_own():
    """Two layer groups drawing from one pool group, as windowed and full layers of equal slot
    sizes do, lay their buffers at the same offsets of slots of their own; the oracle keeps them
    apart rather than counting both writes in one slot."""
    topology = Instance().ranks(0)[0]
    size = 4 * HEAD
    buffers = tuple(BufferDesc(layer, "key", 0, 0, 0, size, kv(4)(topology)) for layer in (0, 1))
    groups = (
        LayerGroupDesc("attention", None, 0, 0, (0,)),
        LayerGroupDesc("attention", 64, 0, 0, (1,)),
    )
    page = PageLayout(TOKENS, {0: (size,)}, groups, buffers, topology)
    assert_layouts_reshard([page], [page])


def _with_group(page: PageLayout, **changes) -> PageLayout:
    """``page`` with its first layer group's fields changed."""
    first, *rest = page.layer_groups
    return dataclasses.replace(page, layer_groups=(dataclasses.replace(first, **changes), *rest))


def test_layouts_whose_bytes_line_up_but_mean_other_things_are_refused():
    """Bytes of the same extents hold other values under another element type, quantization,
    sub-page count or layer group kind; such peers are refused, not paired."""
    topology = Instance().ranks(0)[0]
    base = layout(topology, (0,), [("key", kv(4), "attention")])
    typed = layout(topology, (0,), [("ssm", state_heads(4), "state")])
    with pytest.raises(ValueError, match="KV elements of type BF16 against HALF"):
        check_compatible(
            dataclasses.replace(base, dtype="BF16"), dataclasses.replace(base, dtype="HALF")
        )
    with pytest.raises(ValueError, match="KV quantization fp8 against None"):
        check_compatible(dataclasses.replace(base, quant_mode="fp8"), base)
    bf16, fp16 = (
        dataclasses.replace(
            typed,
            buffers=tuple(
                dataclasses.replace(
                    b, tensors=tuple(dataclasses.replace(t, dtype=dtype) for t in b.tensors)
                )
                for b in typed.buffers
            ),
        )
        for dtype in ("bfloat16", "float16")
    )
    with pytest.raises(ValueError, match="'ssm'\\) tensor '': elements of type bfloat16 against"):
        check_compatible(bf16, fp16)
    check_compatible(bf16, bf16)
    with pytest.raises(ValueError, match="a layer group of attention against one of state"):
        check_compatible(base, _with_group(base, kind="state"))
    expanded = dataclasses.replace(
        base, buffers=tuple(dataclasses.replace(b, expansion=2) for b in base.buffers)
    )
    with pytest.raises(ValueError, match="'key'\\): 2 sub-pages against 1"):
        check_compatible(expanded, base)


def test_layers_of_one_name_but_other_model_layers_are_refused():
    """Peers pair buffers by global layer id. A side naming its layers otherwise, as a stage of a
    mixed-precision model naming model layer ``m`` as ``m`` where the other stages name it
    ``2 m``, would move one model layer's KV into another's pages; such peers are refused."""
    topology = Instance().ranks(0)[0]
    base = layout(topology, (2,), [("key", kv(4), "attention")])
    check_compatible(base, _with_group(base, model_layers=(2,)))
    with pytest.raises(ValueError, match="'key'\\): of model layer 1 against 2"):
        check_compatible(_with_group(base, model_layers=(1,)), base)


def test_model_layers_named_by_other_layers_are_refused():
    """Peers sharing no layer id can still hold the same model layers: a stage naming model layers
    4 and 5 as 8 and 10 against one naming them 4 and 5 would pair nothing and leave both
    unwritten, so it is refused."""
    topology = Instance().ranks(0)[0]
    plain = layout(topology, (4, 5), [("key", kv(4), "attention")])
    doubled = _with_group(
        layout(topology, (8, 10), [("key", kv(4), "attention")]), model_layers=(4, 5)
    )
    with pytest.raises(
        ValueError, match=r"model layer 4's key buffers in layers \[8\] against \[4\]"
    ):
        check_compatible(doubled, plain)
    with pytest.raises(
        ValueError, match=r"model layer 5's key buffers in layers \[5\] against \[10\]"
    ):
        check_compatible(plain, doubled)
    check_compatible(doubled, doubled)


def test_a_role_one_side_alone_transfers_pairs_only_where_that_side_rebuilds_it():
    """A CSA2 context-only worker keeps no sliding cache past its split: its boundary layer 20
    holds only the global cache, layer 61, where the generation worker holds both, 60 and 61. A
    generation worker that rebuilds the sliding one per request pairs the global cache; one that
    does not would leave it unwritten, and is refused."""
    topology = Instance().ranks(0)[0]
    both = layout(topology, (60, 61), [("swa", kv(4), "attention"), ("global", kv(4), "attention")])
    full = _with_group(
        dataclasses.replace(
            both,
            buffers=tuple(
                b for b in both.buffers if (b.layer, b.role) in {(60, "swa"), (61, "global")}
            ),
        ),
        model_layers=(20, 20),
    )
    context_only = _with_group(
        layout(topology, (61,), [("global", kv(4), "attention")]), model_layers=(20,)
    )
    with pytest.raises(
        ValueError, match="model layer 20's swa buffers are transferred by one rank"
    ):
        check_compatible(context_only, full)
    rebuilding = dataclasses.replace(full, request_rebuilt_roles=frozenset({"swa"}))
    check_compatible(context_only, rebuilding)
    check_compatible(rebuilding, context_only)
    assert {(r.dst.layer, r.dst.role) for r in map_buffers(context_only, rebuilding)} == {
        (61, "global")
    }
    assert_layouts_reshard([context_only], [rebuilding])
    with pytest.raises(AssertionError, match="no source sends it"):
        assert_layouts_reshard([context_only], [full])


def test_a_role_rebuilt_per_request_may_share_a_global_layer_with_paired_buffers():
    """A rebuilt role need not lie in a global layer of its own: kept on layer 60 next to a cache
    both sides transfer, it pairs where the side holding it rebuilds it per request, whichever
    side sends, and is refused otherwise."""
    topology = Instance().ranks(0)[0]
    full = layout(topology, (60,), [("swa", kv(4), "attention"), ("global", kv(4), "attention")])
    context_only = layout(topology, (60,), [("global", kv(4), "attention")])
    for pair in ((context_only, full), (full, context_only)):
        with pytest.raises(ValueError, match=r"buffer \(60, 'swa'\) is transferred by one rank"):
            check_compatible(*pair)
    rebuilding = dataclasses.replace(full, request_rebuilt_roles=frozenset({"swa"}))
    check_compatible(context_only, rebuilding)
    check_compatible(rebuilding, context_only)
    assert {(r.dst.layer, r.dst.role) for r in map_buffers(context_only, rebuilding)} == {
        (60, "global")
    }
    assert_layouts_reshard([context_only], [rebuilding])
    with pytest.raises(AssertionError, match="no source sends it"):
        assert_layouts_reshard([context_only], [full])


def test_layer_groups_keeping_other_windows_correspond():
    """Retention differs legitimately between the sides: a context worker with block reuse keeps a
    reuse-backoff token more (GPT-OSS with Eagle3 keeps 129 tokens of a 128-token window), a
    generation worker drafting tokens keeps them too (DeepSeek-V4 DSpark keeps W + 5), and a window
    reaching ``max_seq_len`` on one side is unbounded there. The transfer moves the blocks both
    sides hold, so none of these is refused."""
    topology = Instance().ranks(0)[0]
    base = layout(topology, (0,), [("key", kv(4), "attention")])
    pairs = [
        (_with_group(base, kept_window=129), _with_group(base, kept_window=128)),
        (_with_group(base, kept_window=64), _with_group(base, kept_window=69)),
        (_with_group(base, kept_window=4096), base),
        (_with_group(base, kept_window=64, num_sink_blocks=1), _with_group(base, kept_window=64)),
    ]
    for one, other in pairs:
        check_compatible(one, other)
        check_compatible(other, one)


def test_state_pages_of_other_token_counts_correspond():
    """Recurrent state holds no tokens, so only attention pages must hold as many."""
    topology = Instance().ranks(0)[0]
    state = layout(topology, (0,), [("ssm", state_heads(4), "state")])
    check_compatible(state, dataclasses.replace(state, tokens_per_page=8))


def test_sent_bytes_count_what_one_source_sends_into_each_layer_group():
    """Per page of each of the destination's layer groups (attention 0, state 1 here), what one
    source rank sends it."""
    specs = [
        ("key", kv(4), "attention"),
        ("ssm", state_heads(4), "state"),
        ("ple", whole(12), "state"),
    ]
    dst = layout(Instance(tp=2).ranks(0)[0], (0, 1), specs)
    src = layout(Instance().ranks(0)[0], (1, 2), specs)
    # Layer 1 only: 2 of its 4 K heads, its SSM share (2 of 4 heads) and its replicated PLE.
    assert sent_bytes(src, dst) == {0: 2 * HEAD, 1: 2 * HEAD + 12}
    # Two tensor ranks fan into one: each sends its heads of both layers, one of them the PLE.
    whole_dst = layout(Instance().ranks(0)[0], (0, 1), specs)
    shares = [sent_bytes(layout(t, (0, 1), specs), whole_dst) for t in Instance(tp=2).ranks(0)]
    assert sorted(shares, key=lambda s: -s[1]) == [
        {0: 4 * HEAD, 1: 4 * HEAD + 24},
        {0: 4 * HEAD, 1: 4 * HEAD},
    ]
