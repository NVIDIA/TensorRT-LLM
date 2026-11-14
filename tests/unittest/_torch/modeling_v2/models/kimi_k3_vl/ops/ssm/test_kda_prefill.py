# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test of the kda_prefill entry: every cell of ``kda_prefill.CELLS``, built as the tp1 KDA
layer calls it (metadata hints, a chunk-padded single sequence, padded pool slots, in-forward
snapshot rows), against the entry's fp64 recurrence; and that recurrence against the family's
``_kda_reference.recurrent_reference``."""

import zlib

import pytest
import torch

pytest.importorskip("cutlass")

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.ssm.k3r_kda_prefill import (  # noqa: E501, E402
    kda_prefill,
)

__extra_import_path__ = ["../../../..", "."]
import _kda_reference as kr  # noqa: E402
from _validating import validating  # noqa: E402

HEAD_DIM, CHUNK = kr.HEAD_DIM, kr.CHUNK
SLOTS = 12
SLOT_PAD = 4096  # floats past each slot's state: the cache manager's slot stride is wider
# Bands relative to the reference's largest magnitude: bf16 tensor-core chunks vs an fp64
# token-by-token recurrence (the family prefill entries' bands).
OUT_TOL = 2e-2
STATE_TOL = 1e-2


def _driven_here() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}" in {arch.value for arch in kda_prefill.ARCHS}


pytestmark = pytest.mark.skipif(not _driven_here(), reason="kda_prefill.ARCHS")


def _chunk_indices(seq_lens) -> torch.Tensor:
    pairs = [(s, c) for s, n in enumerate(seq_lens) for c in range(-(-n // CHUNK))]
    return torch.tensor(pairs, dtype=torch.long, device="cuda").view(-1, 2)


def _call_args(spec: dict, seed: int) -> tuple[dict, torch.Tensor, int]:
    """The arguments K3KDAKernelDispatch.prefill_chunk_kda passes for this batch, the padded pool
    storage behind ``state_pool``, and the real token count."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    heads, seq_lens = spec["heads"], list(spec["seq_lens"])
    tokens = sum(seq_lens)
    raw = kr.kimi_k3_inputs(heads, tokens, gen)
    pool = torch.randn(SLOTS, heads * HEAD_DIM * HEAD_DIM + SLOT_PAD, generator=gen, device="cuda")
    state_pool = pool[:, : heads * HEAD_DIM * HEAD_DIM].view(SLOTS, heads, HEAD_DIM, HEAD_DIM)
    state_pool.mul_(0.05)
    order = torch.randperm(SLOTS, generator=torch.Generator().manual_seed(seed))
    slots = order[: len(seq_lens)]
    for slot, fresh in zip(slots.tolist(), spec["fresh"]):
        if fresh:
            state_pool[slot].zero_()  # the layer resets fresh rows before the op
    # The metadata's int32 state indices sit at a 16-byte aligned offset.
    state_indices = torch.empty(4 + len(seq_lens), dtype=torch.int32, device="cuda")[4:]
    state_indices.copy_(slots)
    single = tokens if len(seq_lens) == 1 else None
    q, k, v, beta = raw["q"], raw["k"], raw["v"], raw["beta"]
    if single is not None and tokens % CHUNK:
        # kda_prepad_prefill: Q/K/V/beta arrive zero-padded to the chunk, g does not.
        pad = CHUNK - tokens % CHUNK
        q, k, v = (torch.nn.functional.pad(t, (0, 0, 0, 0, 0, pad)) for t in (q, k, v))
        beta = torch.nn.functional.pad(beta, (0, 0, 0, pad))
    args = dict(
        q=q,
        k=k,
        v=v,
        g=raw["g"],
        beta=beta,
        state_pool=state_pool,
        state_indices=state_indices,
        scale=kr.SCALE,
        cu_seqlens=kr.cu_seqlens_of(seq_lens),
        chunk_indices=_chunk_indices(seq_lens),
        chunk_size=CHUNK,
        safe_gate=True,
        lower_bound=kr.LOWER_BOUND,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        A_log=raw["A_log"],
        dt_bias=raw["dt_bias"],
        varlen_is_aligned=all(n % CHUNK == 0 for n in seq_lens),
        single_sequence_length=single,
    )
    if spec.get("snapshot_chunks") is not None:
        args["snapshot_rows"] = order[len(seq_lens) : 2 * len(seq_lens)].to(torch.int32).cuda()
        args["snapshot_chunks"] = torch.tensor(
            spec["snapshot_chunks"], dtype=torch.int32, device="cuda"
        )
    return args, pool, tokens


@pytest.mark.parametrize("cell", kda_prefill.CELLS, ids=[c.why[:60] for c in kda_prefill.CELLS])
@torch.no_grad()
def test_cells(cell) -> None:
    args, pool, tokens = _call_args(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    pool_before = pool.clone()
    ref_out, ref_pool = kda_prefill.reference(**args)
    with validating(kda_prefill):
        out = kda_prefill(**args).clone()  # a view of the op's shape-keyed scratch buffer

    assert out.shape == args["v"].shape and out.dtype == args["v"].dtype, cell.why
    kr.assert_scaled_close(out[:, :tokens], ref_out[:, :tokens], OUT_TOL)
    written = set(args["state_indices"].tolist())
    if "snapshot_rows" in args:
        written |= {
            row
            for row, c in zip(args["snapshot_rows"].tolist(), args["snapshot_chunks"].tolist())
            if c >= 0
        }
    state_pool = args["state_pool"]
    for slot in range(SLOTS):
        if slot in written:
            kr.assert_scaled_close(state_pool[slot], ref_pool[slot], STATE_TOL)
        else:
            assert torch.equal(pool[slot], pool_before[slot]), f"{cell.why}: slot {slot} written"
    state_size = state_pool[0].numel()
    assert torch.equal(pool[:, state_size:], pool_before[:, state_size:]), "slot padding written"


@torch.no_grad()
def test_reference_is_the_family_recurrence() -> None:
    """The entry's reference on two sequences equals the family's fp64 recurrence."""
    spec = dict(heads=6, seq_lens=(70, 33), fresh=(True, False))
    args, _, _ = _call_args(spec, 5)
    ref_out, ref_pool = kda_prefill.reference(**args)
    starts = args["cu_seqlens"].tolist()
    for b, slot in enumerate(args["state_indices"].tolist()):
        lo, hi = starts[b], starts[b + 1]
        o, s = kr.recurrent_reference(
            args["q"][0, lo:hi],
            args["k"][0, lo:hi],
            args["v"][0, lo:hi],
            args["g"][0, lo:hi],
            args["beta"][0, lo:hi],
            args["state_pool"][slot],
            args["A_log"],
            args["dt_bias"],
        )
        kr.assert_scaled_close(ref_out[0, lo:hi], o.to(ref_out.dtype), 1e-2)
        kr.assert_scaled_close(ref_pool[slot], s.float(), 1e-6)
