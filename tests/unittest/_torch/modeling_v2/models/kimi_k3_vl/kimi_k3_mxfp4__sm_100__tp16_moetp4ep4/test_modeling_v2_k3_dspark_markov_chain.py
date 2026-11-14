# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the target's spec/k3_dspark_markov_chain entry (sm_100 / sm_103).

Each cell checks the op against its ``reference`` (the eager chain) and bit for bit against the speculative runtime's
``vocab_parallel.markov_chain_logits``, the chain the built-in DSpark drafter runs. The vocabulary-shard form is
checked bit for bit against the runtime's on one GPU, with an exchange that returns this rank's buffer unchanged.
"""

import importlib

import pytest
import torch

assert torch.cuda.is_available(), "k3_dspark_markov_chain requires a CUDA device"

if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
    pytest.skip("k3_dspark_markov_chain is certified on sm_100 and sm_103", allow_module_level=True)

_op = importlib.import_module(
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl."
    "kimi_k3_mxfp4__sm_100__tp16_moetp4ep4.ops.spec.k3_dspark_markov_chain"
)
k3_dspark_markov_chain = _op.k3_dspark_markov_chain

__extra_import_path__ = ["../../.."]
from _validating import validating  # noqa: E402

from tensorrt_llm._torch.speculative import vocab_parallel  # noqa: E402


def _inputs(spec: dict, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    b, k, vocab, rank, dtype = (spec[n] for n in ("batch", "block", "vocab", "rank", "dtype"))
    base = torch.randn(b, k, vocab, generator=gen, device="cuda") * 4.0
    w1 = (torch.randn(vocab, rank, generator=gen, device="cuda") * 0.5).to(dtype)
    w2 = (torch.randn(vocab, rank, generator=gen, device="cuda") * 0.5).to(dtype)
    first = torch.randint(0, vocab, (b,), generator=gen, device="cuda")
    if dtype == torch.float32:
        # Anchors outside the vocabulary take a zero bias.
        first[0], first[-1] = -1, vocab
    return base, first, w1, w2


@pytest.mark.parametrize(
    "cell", k3_dspark_markov_chain.CELLS, ids=[c.why for c in k3_dspark_markov_chain.CELLS]
)
def test_k3_dspark_markov_chain(cell) -> None:
    """The op against its reference (same tokens; the logits within fp32 rounding of the bias add) and bit for bit
    against the speculative runtime's chain."""
    base, first, w1, w2 = _inputs(cell.spec, seed=len(cell.why))
    with validating(k3_dspark_markov_chain):
        got = k3_dspark_markov_chain(base, first, w1, w2)
    want = k3_dspark_markov_chain.reference(base, first, w1, w2)
    assert torch.equal(got.argmax(-1), want.argmax(-1))
    torch.testing.assert_close(got, want, rtol=0, atol=1e-5)
    builtin = vocab_parallel.markov_chain_logits(base, first, w1, w2)
    assert torch.equal(got, builtin), "differs from the speculative runtime's markov_chain_logits"


class _OwnBuffer:
    """A stand-in for one rank's TP all-reduce: its buffer unchanged."""

    def __call__(self, buffer: torch.Tensor) -> torch.Tensor:
        return buffer


@pytest.mark.parametrize("tp_size, tp_rank", [(2, 1), (16, 5)])
def test_k3_dspark_markov_chain_shard_matches_the_runtime(tp_size, tp_rank) -> None:
    """The vocabulary-shard form (``tp_exchange``) bit for bit against the runtime's chain on the same exchange."""
    spec = dict(batch=4, block=7, vocab=163840, rank=256, dtype=torch.bfloat16)
    base, first, w1, w2 = _inputs(spec, seed=tp_size)
    shard = spec["vocab"] // tp_size
    local = base[..., tp_rank * shard : (tp_rank + 1) * shard].contiguous()
    exchange = vocab_parallel._PairExchange(_OwnBuffer(), tp_size, tp_rank)
    with validating(k3_dspark_markov_chain):
        got = k3_dspark_markov_chain(local, first, w1, w2, exchange)
    builtin = vocab_parallel.markov_chain_logits(local, first, w1, w2, exchange)
    assert torch.equal(got, builtin)


def test_k3_dspark_markov_chain_rejects_out_of_contract() -> None:
    """The guard refuses bf16 logits and a vocabulary the head does not have."""
    spec = dict(batch=2, block=3, vocab=5000, rank=4, dtype=torch.float32)
    base, first, w1, w2 = _inputs(spec, seed=1)
    for args in ((base.bfloat16(), first, w1, w2), (base[..., :-1].contiguous(), first, w1, w2)):
        with validating(k3_dspark_markov_chain), pytest.raises(AssertionError):
            k3_dspark_markov_chain(*args)


def _num_compiled(kernel) -> int:
    """Specializations of a ``triton.jit`` kernel compiled in this process; a launch that raises the count compiled a
    new variant. Skips the test when this Triton does not expose the per-device caches."""
    caches = getattr(kernel, "device_caches", None)
    if caches is None:
        pytest.skip("this Triton has no JITFunction.device_caches")
    return sum(len(cache[0]) for cache in caches.values())


@pytest.mark.parametrize(
    "tp_size, tp_rank, expected",
    [
        # A single shard's pair buffer is 2 * splits wide, so its row stride follows the row count.
        (1, 0, {(0, 16, 32), (0, 11, 22)}),
        # Rank 1 of 2 (identity all-reduce): the row stride is the ZERO_WIDTH.
        (2, 1, {(64, 16, 64), (64, 11, 64)}),
    ],
    ids=["single_shard", "tp2_rank1"],
)
def test_new_row_counts_reuse_the_compiled_kernels(tp_size, tp_rank, expected) -> None:
    """A 65536-column shard takes 16 splits for 9-17 rows and 11 for 18-25: one ZERO_WIDTH / BLOCK_SLOTS bucket in
    which the pair offsets, the slot count and the pair-buffer row stride are and are not multiples of 16. After the
    first row count, the other row counts compile no new variant of the chain's kernels."""
    kern = importlib.import_module(
        "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl."
        "kimi_k3_mxfp4__sm_100__tp16_moetp4ep4.kernels.k3_dspark_markov_chain.k3_dspark_markov_chain_kernel"
    )
    n_cols, rank, block = 32 * 2048, 8, 2
    exchange = kern._PairExchange(None if tp_size == 1 else (lambda pairs: pairs), tp_size, tp_rank)
    col_base = tp_rank * n_cols
    row_counts = (9, 16, 17, 18, 24)
    layouts = [exchange.layout(rows, n_cols, torch.device("cuda")) for rows in row_counts]
    assert {
        (layout.zero_width, layout.splits, layout.buffer.stride(0)) for layout in layouts
    } == expected
    w1 = torch.zeros(tp_size * n_cols, rank, device="cuda")
    w2_rows = torch.zeros(n_cols, rank, device="cuda")
    kernels = (kern._markov_reduce_embed_kernel, kern._markov_add_local_argmax_kernel)

    def chain(rows: int) -> None:
        logits = torch.zeros(rows, block, n_cols, device="cuda")
        anchors = torch.zeros(rows, dtype=torch.int64, device="cuda")
        kern._markov_chain_tokens(logits, anchors, w1, w2_rows, col_base, exchange)

    chain(row_counts[0])
    compiled = [_num_compiled(kernel) for kernel in kernels]
    for rows in row_counts[1:]:
        chain(rows)
    assert [_num_compiled(kernel) for kernel in kernels] == compiled
