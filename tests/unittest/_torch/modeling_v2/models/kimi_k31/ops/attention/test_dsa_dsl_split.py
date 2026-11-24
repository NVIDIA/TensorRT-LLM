# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How the Kimi K3.1 DSA metadata splits a verify step's next_n for the CuTe DSL paged MQA logits.

``_pick_dsl_expand`` reshapes ``[B, next_n]`` into ``[B * factor, atom]`` with ``atom`` a native kernel tile;
``_pick_dsl_split`` adds the in-kernel uneven split for a next_n no native atom > 1 divides (5 or 7 on the general
target's FP8 atoms), and ``_atom_ctx_offsets`` gives each split atom its causal context shortening.
"""

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.dsa_indexer_plan import (
    _atom_ctx_offsets,
    _pick_dsl_expand,
    _pick_dsl_split,
)

pytestmark = pytest.mark.cpu_only

#: The general target's FP8 atoms and the SM 10.3 perf targets' native tiles (dsa_metadata.prepare_for_spec_decode).
GENERAL_ATOMS = (1, 2, 3, 4)
PERF_ATOMS = (1, 2, 3, 4, 5, 6, 7, 8)
NUM_SMS = 148


@pytest.mark.parametrize(
    "next_n, batch_size, max_ctx, expected",
    [
        (4, 1, 4096, (4, 1)),  # 64 tasks < 148 SMs: one wave either way, the larger factor
        (4, 64, 4096, (1, 4)),  # 1024 tasks at factor 1 already take 7 waves: the fewest waves
        (2, 1, 4096, (2, 1)),  # wave tie: the larger factor
        (6, 0, 0, (2, 3)),  # workload unknown: the largest atom dividing next_n
        (5, 0, 0, (5, 1)),  # workload unknown, no atom > 1 divides 5
    ],
)
def test_expand_pick(next_n, batch_size, max_ctx, expected) -> None:
    assert _pick_dsl_expand(next_n, NUM_SMS, batch_size, max_ctx, GENERAL_ATOMS) == expected


@pytest.mark.parametrize("atoms", [GENERAL_ATOMS, PERF_ATOMS])
@pytest.mark.parametrize("next_n", range(1, 9))
@pytest.mark.parametrize(
    "batch_size, max_ctx", [(0, 0), (1, 300), (8, 4096), (64, 4096), (256, 100_000)]
)
def test_split_covers_next_n(atoms, next_n, batch_size, max_ctx) -> None:
    """Either a caller-side reshape onto one native atom or an in-kernel split covers exactly next_n tokens."""
    factor, atom, num_atoms = _pick_dsl_split(next_n, NUM_SMS, batch_size, max_ctx, atoms)
    assert atom in atoms
    if num_atoms == 1:
        assert factor * atom == next_n
    else:
        # In-kernel split: no caller-side reshape, atoms of ceil(next_n / num_atoms) tokens at most.
        assert factor == 1
        assert num_atoms == -(-next_n // max(atoms))
        assert atom == -(-next_n // num_atoms) and atom <= max(atoms)
        assert not any(next_n % a == 0 for a in atoms if a > 1)


@pytest.mark.parametrize("next_n", range(1, 9))
def test_perf_atoms_never_split(next_n) -> None:
    """Every next_n up to 8 is a native tile of the perf targets: no split, the whole step in one atom."""
    assert _pick_dsl_split(next_n, NUM_SMS, 0, 0, PERF_ATOMS) == (1, next_n, 1)
    assert _pick_dsl_split(next_n, NUM_SMS, 512, 100_000, PERF_ATOMS) == (1, next_n, 1)


@pytest.mark.parametrize("next_n", [5, 7])
def test_general_split_choice(next_n) -> None:
    """The general target's next_n 5 / 7: the split when the workload is unknown or atom 1 needs >= 16 waves that
    the split shortens, atom 1 (next_n KV re-reads) below that."""
    split = (1, -(-next_n // 2), 2)
    assert _pick_dsl_split(next_n, NUM_SMS, 0, 0, GENERAL_ATOMS) == split
    # B=1, 4K context: next_n * 16 tiles = 80 / 112 tasks, one atom-1 wave.
    assert _pick_dsl_split(next_n, NUM_SMS, 1, 4096, GENERAL_ATOMS) == (next_n, 1, 1)
    # B=64, 32K context: 64 * next_n * 128 tiles is 277 / 388 atom-1 waves, the split needs 111.
    assert _pick_dsl_split(next_n, NUM_SMS, 64, 32768, GENERAL_ATOMS) == split


@pytest.mark.parametrize(
    "next_n, num_atoms, expected",
    [
        (5, 2, [2, 0]),  # atoms of 3 and 2 tokens
        (7, 2, [3, 0]),  # atoms of 4 and 3 tokens
        (7, 3, [4, 2, 0]),  # atoms of 3, 2 and 2 tokens
        (8, 4, [6, 4, 2, 0]),  # even split
        (3, 1, [0]),
    ],
)
def test_atom_ctx_offsets(next_n, num_atoms, expected) -> None:
    assert _atom_ctx_offsets(next_n, num_atoms) == expected


@pytest.mark.parametrize("next_n", range(1, 17))
def test_atom_ctx_offsets_match_the_kernel_split(next_n) -> None:
    """Atom i sees its context shortened by the tokens of the atoms after it; the first next_n % num_atoms atoms take
    one extra token (the kernel's split)."""
    for num_atoms in range(1, next_n + 1):
        base, rem = divmod(next_n, num_atoms)
        sizes = [base + (1 if i < rem else 0) for i in range(num_atoms)]
        after = [sum(sizes[i + 1 :]) for i in range(num_atoms)]
        assert _atom_ctx_offsets(next_n, num_atoms) == after
