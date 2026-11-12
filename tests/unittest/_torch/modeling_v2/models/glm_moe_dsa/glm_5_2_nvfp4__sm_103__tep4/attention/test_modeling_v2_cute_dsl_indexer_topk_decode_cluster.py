# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test of the cluster-split decode top-k (a long row split across a cluster of CTAs).

Every cluster size is launched directly (independent of the row-count policy) and compared
with the exact reference (fp32 bit order, lower column first) and with the one-CTA kernel, on
rows below, at and above the split threshold, with distinct, tied and fp16-rounded scores.
"""

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel is built for sm_103", allow_module_level=True)

import cuda.bindings.driver as cuda  # noqa: E402

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.kernels.attention import (  # noqa: E402, E501
    self_attn_indexer_topk_decode as kernel,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention.cute_dsl_indexer_topk_decode import (  # noqa: E402, E501
    build,
    select_rows,
    uses_cluster,
)

TOP_K = 2048
WIDTH = 212992
#: Written past each row's causal end. A kernel that read past the end would
#: select these before any real score.
POISON = 1.0e9
SIZES = (2, 4, 8, 16)


def _stream() -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def _logits(kind: str, ends: list[int], seed: int, pad: int = 256) -> torch.Tensor:
    """`[rows, WIDTH]` scores on a row-padded arena; POISON from each causal end on."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = len(ends)
    arena = POISON + torch.arange(WIDTH + pad, dtype=torch.float32, device="cuda").repeat(rows, 1)
    for r, end in enumerate(ends):
        if kind == "distinct":
            vals = torch.randperm(end, generator=g, device="cuda").float()
        elif kind == "randn":
            vals = torch.randn(end, generator=g, device="cuda")
        elif kind == "outliers":
            # The served indexer's shape: the k-th score negative, a few hundred positive
            # outliers; one key window over the candidates would span the sign boundary.
            vals = torch.randn(end, generator=g, device="cuda") * 8 - 60
            hot = torch.randint(0, end, (300,), generator=g, device="cuda")
            vals[hot] = torch.rand(300, generator=g, device="cuda") * 90
        elif kind == "fp16":
            # Many exact ties near the k-th value.
            vals = torch.randn(end, generator=g, device="cuda").half().float()
        else:
            # `levels<n>`: n values, so the k-th value is tied hundreds to thousands of times.
            levels = int(kind[len("levels") :])
            vals = torch.randint(0, levels, (end,), generator=g, device="cuda").float()
        arena[r, :end] = vals
    return arena[:, :WIDTH]


def _ends(lens: list[int], next_n: int) -> list[int]:
    return [
        max(0, min(WIDTH, lens[r // next_n] - next_n + r % next_n + 1))
        for r in range(len(lens) * next_n)
    ]


def _compare(out: torch.Tensor, ref: torch.Tensor, what: str) -> None:
    got, want = out.sort(dim=1).values, ref.sort(dim=1).values
    bad = (got != want).any(dim=1)
    assert not bool(bad.any()), (
        f"{what}: {int(bad.sum())} of {out.shape[0]} rows select a different key set; "
        f"first at row {int(bad.nonzero()[0])}"
    )


CASES = [
    # (next_n, seq_lens, kind): rows at / around the split threshold and top_k, odd lengths,
    # the longest contexts, a row-padded view (every case), MTP1 / MTP3 / MTP5 widths.
    (1, [kernel._CLUSTER_SPLIT_MIN], "distinct"),
    (2, [kernel._CLUSTER_SPLIT_MIN + 2, 2049], "distinct"),
    (6, [20001, 1000], "randn"),
    (6, [163840, 131071], "randn"),
    (6, [163840, 65536], "outliers"),
    (1, [WIDTH], "distinct"),
    (4, [70000, 2050, 17000], "fp16"),
    (2, [200001, 99999], "fp16"),
    # Tie groups of ~3000 at the cut: the candidate refine, column digits past 32768.
    (1, [200000], "levels64"),
    # ~28000 tied at the cut (> the candidate capacity): every step overflows.
    (2, [200000, 40000], "levels7"),
    # Short rows with ~5000 tied at the cut: rank 0's step 0 overflows, the whole one-CTA
    # schedule reruns (down to its tied-column scan).
    (2, [kernel._CLUSTER_SPLIT_MIN, 9000], "levels3"),
]


@pytest.mark.parametrize("case", CASES, ids=[f"n{c[0]}-{c[1][0]}-{c[2]}" for c in CASES])
def test_every_cluster_size_selects_the_exact_set(case) -> None:
    next_n, lens, kind = case
    ends = _ends(lens, next_n)
    logits = _logits(kind, ends, seed=sum(lens) + next_n)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    end_t = torch.tensor(ends, device="cuda")
    ref = select_rows(logits, torch.zeros_like(end_t), end_t, TOP_K)
    assert bool((ref >= 0).any()) and max(ends) > TOP_K, "degenerate reference"
    one = kernel.compile_indexer_topk_decode(
        device_id=torch.cuda.current_device(), max_num_columns=WIDTH, next_n=next_n
    )
    out = torch.full((len(ends), TOP_K), -7, dtype=torch.int32, device="cuda")
    one._raw(logits, seq_lens, out, _stream())
    _compare(out, ref, "one-CTA kernel")
    for size in SIZES:
        raw = kernel.compile_indexer_topk_decode_cluster(torch.cuda.current_device(), next_n, size)
        out.fill_(-7)
        raw(logits, seq_lens, out, _stream())
        _compare(out, ref, f"cluster {size}")


@pytest.mark.parametrize("size", SIZES)
def test_graph_replay_matches_eager(size) -> None:
    """Replays of a captured call select the eager call's keys, also after new scores."""
    next_n, lens = 6, [163840, 60000]
    ends = _ends(lens, next_n)
    logits = _logits("randn", ends, seed=size)
    new_logits = _logits("randn", ends, seed=100 + size)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    raw = kernel.compile_indexer_topk_decode_cluster(torch.cuda.current_device(), next_n, size)
    out = torch.full((len(ends), TOP_K), -7, dtype=torch.int32, device="cuda")
    eager = []
    for x in (logits.clone(), new_logits):
        raw(x, seq_lens, out, _stream())
        eager.append(out.clone())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        raw(logits, seq_lens, out, _stream())
    first = logits.clone()
    for scores, want in zip((first, new_logits), eager):
        logits.copy_(scores)
        out.fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        _compare(out, want, "replay")
    assert not torch.equal(eager[0].sort(dim=1).values, eager[1].sort(dim=1).values)


def test_handle_policy() -> None:
    """Wide engines carry the cluster launchers, chosen by row count; narrow ones do not."""
    dev = torch.cuda.current_device()
    wide = build(dev, max_columns=WIDTH, top_k=TOP_K, next_n=6)
    assert [wide.launcher_for(rows)[0] for rows in (6, 12, 18, 24, 48, 192)] == [16, 8, 4, 4, 1, 1]
    assert uses_cluster(wide, 24) and not uses_cluster(wide, 48)
    narrow = build(dev, max_columns=kernel.CLUSTER_MIN_COLUMNS, top_k=TOP_K, next_n=6)
    assert not uses_cluster(narrow, 6)


@pytest.mark.parametrize("size", (8, 16))
def test_pdl_launch_selects_the_exact_set(size) -> None:
    """The PDL launch (TRTLLM_GLM52_DEC_TOPK_CLUSTER_PDL=1) behind a producer of the logits."""
    next_n, lens = 6, [163840, 2600]
    ends = _ends(lens, next_n)
    want = _logits("outliers", ends, seed=size)
    logits = torch.zeros_like(want)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    end_t = torch.tensor(ends, device="cuda")
    ref = select_rows(want, torch.zeros_like(end_t), end_t, TOP_K)
    raw = kernel.compile_indexer_topk_decode_cluster(
        torch.cuda.current_device(), next_n, size, pdl=True
    )
    out = torch.full((len(ends), TOP_K), -7, dtype=torch.int32, device="cuda")
    logits.copy_(want)
    raw(logits, seq_lens, out, _stream())
    _compare(out, ref, f"cluster {size} with PDL")
