# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ngram_hash entry.

Exact integer arithmetic, gated bitwise against the entry's scalar reference.
Beyond the cells: the EOS reset that `ngram_token_history`'s EOS seed relies
on, the narrow-multiplier guard and graph capture.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.embedding.ngram_hash import (  # noqa: E501
    CKPT_CONTEXT_LEN,
    CKPT_EOS_TOKEN_ID,
    CKPT_HEAD_OFFSETS,
    CKPT_HEAD_VOCAB_SIZES,
    CKPT_MULTIPLIERS,
    CKPT_VOCAB_SIZE,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.embedding.ngram_hash import (  # noqa: E501
    ngram_hash as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "ngram_hash requires a CUDA device"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Bitwise: exact integer arithmetic, no tolerance to loosen."""
    assert out.dtype == ref.dtype == torch.int64, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    assert out.is_contiguous()
    assert torch.equal(out, ref), f"{int((out != ref).sum())} indices differ"


DEV = "cuda"
EOS = CKPT_EOS_TOKEN_ID
NGRAM_HEADS = len(CKPT_HEAD_VOCAB_SIZES)
PADDED_TABLE_ROWS = 320001536  # sum(head_vocab_sizes) rounded up to a multiple of 128


def _constants():
    return (
        torch.tensor(CKPT_MULTIPLIERS, dtype=torch.long, device=DEV),
        torch.tensor(CKPT_HEAD_VOCAB_SIZES, dtype=torch.long, device=DEV),
        torch.tensor(CKPT_HEAD_OFFSETS, dtype=torch.long, device=DEV),
    )


def _tokens(batch, length, seed, eos_positions=(), lead_eos=True, eos_density=0.0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, CKPT_VOCAB_SIZE, (batch, length), generator=g, dtype=torch.long)
    if eos_density > 0.0:
        ids[torch.rand(batch, length, generator=g) < eos_density] = EOS
    if lead_eos:
        ids[:, : min(CKPT_CONTEXT_LEN, length)] = EOS
    for position in eos_positions:
        ids[:, position] = EOS
    return ids.to(DEV)


def _hash(ids, num_query_tokens):
    return op(ids, *_constants(), EOS, num_query_tokens)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    ids = _tokens(
        spec["batch"],
        spec["length"],
        seed=zlib.crc32(cell.why.encode()),
        eos_positions=spec.get("eos_positions", ()),
        lead_eos=spec.get("lead_eos", True),
        eos_density=spec.get("eos_density", 0.0),
    )
    before = ids.clone()
    args = (ids, *_constants(), EOS, spec["num_query_tokens"])
    out = op(*args)
    assert out.shape == (spec["batch"], spec["num_query_tokens"], NGRAM_HEADS)
    assert out.data_ptr() != ids.data_ptr()
    assert torch.equal(ids, before), "the history is read-only"
    _compare(out, op.reference(*args))


def test_eos_restarts_the_window() -> None:
    """One past an EOS hashes as a sequence's first token, and only there."""
    token = 4242
    fresh = torch.tensor([[EOS] * CKPT_CONTEXT_LEN + [token]], device=DEV)
    after_eos = torch.tensor([[11, 22, 33, EOS, token]], device=DEV)
    no_eos = torch.tensor([[11, 22, 33, 44, token]], device=DEV)
    assert torch.equal(_hash(fresh, 1), _hash(after_eos, 1))
    assert not torch.equal(_hash(fresh, 1), _hash(no_eos, 1))


def test_guard_refuses_narrow_multipliers() -> None:
    ids = _tokens(1, CKPT_CONTEXT_LEN + 4, seed=47)
    multipliers, sizes, offsets = _constants()
    narrow = multipliers.to(torch.int32)
    unguarded = op(ids, narrow, sizes, offsets, EOS, 4)
    assert not torch.equal(unguarded, _hash(ids, 4)), "the hazard this guard covers is gone"
    assert int(unguarded.min()) >= 0 and int(unguarded.max()) < PADDED_TABLE_ROWS
    with pytest.raises(AssertionError, match="int64"):
        with validating(op):
            op(ids, narrow, sizes, offsets, EOS, 4)


def test_cuda_graph_capture_and_replay() -> None:
    ids = _tokens(8, CKPT_CONTEXT_LEN + 1, seed=53, lead_eos=False, eos_density=0.2)
    constants = _constants()
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            op(ids, *constants, EOS, 1)
    torch.cuda.current_stream().wait_stream(warmup)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(ids, *constants, EOS, 1)

    fresh = _tokens(8, CKPT_CONTEXT_LEN + 1, seed=54, lead_eos=False, eos_density=0.2)
    ids.copy_(fresh)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, op.reference(fresh, *constants, EOS, 1))
