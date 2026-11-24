# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the rubin_fp8_paged_mqa_logits entry (family FP8 paged MQA logits, Rubin).

The keys live on random physical pages of a [pages, P, 1, 132] cache view; the schedule comes from
DeepGEMM's schedule builder. Only each row's causal keys are compared. Every cell is gated against
the entry's reference: bit for bit on integer data, within the fp32 accumulation band otherwise.
The engine kernel (the family's sm_103 entry, default L2 policy) must give the same bits, and so
must the other L2 policy.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.attention.rubin_fp8_paged_mqa_logits import (  # noqa: E501
    RubinPagedMQALogitsRunner,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.cute_dsl_fp8_paged_mqa_logits import (  # noqa: E501
    cute_dsl_fp8_paged_mqa_logits as engine_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_fp8_paged_mqa_logits import (  # noqa: E501
    rubin_fp8_paged_mqa_logits as op,
)
from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as _dsl_ops
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata

__extra_import_path__ = ["../../../..", "."]
import _k31_dsa_inputs as inputs  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107,
    reason="the Rubin family entries are certified on sm_107",
)

_SENTINEL = 12345.0
_ROW_ALIGN = 256


def _out(rows: int, max_context_len: int, row_stride: int | None = None) -> torch.Tensor:
    """A sentinel-filled caller buffer with one spare row, and the [rows, max_context_len] view."""
    stride = -(-max_context_len // _ROW_ALIGN) * _ROW_ALIGN if row_stride is None else row_stride
    base = torch.full((rows + 1, stride), _SENTINEL, dtype=torch.float32, device="cuda")
    return base[:rows, :max_context_len]


def _build(spec, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    batch, next_n = spec["batch"], spec["next_n"]
    lo, hi = spec["context_lens"]
    lens = torch.randint(lo, hi + 1, (batch,), generator=gen, dtype=torch.int32)
    kv_cache, table = inputs.paged_cache(
        lens.tolist(), spec["tokens_per_block"], spec["data"], gen, spec["page_stride_pad"]
    )
    schedule = get_paged_mqa_logits_metadata(
        lens.reshape(-1, 1).cuda(), inputs.SCHEDULE_BLOCK_KV, inputs.num_sms()
    )
    q = inputs.queries(batch * next_n, spec["data"], gen).view(
        batch, next_n, inputs.NUM_INDEX_HEADS, inputs.HEAD_DIM
    )
    return dict(
        q=q,
        kv_cache=kv_cache,
        weights=inputs.weights(batch * next_n, spec["data"], gen),
        context_lens=lens.cuda(),
        block_table=table,
        schedule_meta=schedule,
        max_context_len=spec["max_context_len"],
        num_next_n_atoms=spec["num_next_n_atoms"],
        atom=spec["atom"],
        kv_evict_first=spec["kv_evict_first"],
        out=_out(batch * next_n, spec["max_context_len"]),
    )


def _engine(args) -> torch.Tensor:
    """The same inputs through the engine kernel (no L2 hint), into a fresh buffer."""
    kwargs = {k: v for k, v in args.items() if k != "kv_evict_first"}
    kwargs["out"] = _out(*args["out"].shape)
    return engine_op(**kwargs)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, inputs.seed_of(cell))
    out = op(**args)
    assert out.data_ptr() == args["out"].data_ptr(), "the entry must write the caller's buffer"
    first = out.clone()
    again = op(**dict(args, out=_out(*out.shape)))
    flipped = op(**dict(args, kv_evict_first=not args["kv_evict_first"], out=_out(*out.shape)))
    ref = op.reference(**args)
    engine = _engine(args)
    torch.cuda.synchronize()
    batch, next_n = args["q"].shape[:2]
    assert out.shape == (batch * next_n, args["max_context_len"]) and out.dtype == torch.float32
    causal = torch.isfinite(ref)
    assert torch.equal(first[causal], again[causal]), "two identical calls disagree"
    assert torch.equal(first[causal], flipped[causal]), "the L2 policy changed the logits"
    assert torch.equal(first[causal], engine[causal]), "the engine kernel computes otherwise"
    # Neither the slack columns of each row nor the spare row past the view are written.
    rows, width, stride = out.shape[0], out.shape[1], out.stride(0)
    if stride > width:
        assert (out.as_strided((rows, stride - width), (stride, 1), width) == _SENTINEL).all()
    assert (out.as_strided((1, stride), (stride, 1), rows * stride) == _SENTINEL).all()
    if cell.spec["data"] == "int":
        assert torch.equal(first[causal], ref[causal])
        return
    inputs.check_paged_logits(first, ref, args, next_n, args["context_lens"])


def test_each_l2_policy_compiles_its_own_kernel():
    """The hint is a compile option: one cache entry per policy, in the family runner's own
    cache, never in the engine runner's."""
    spec = next(c.spec for c in op.CELLS if c.spec["next_n"] == 8 and c.spec["batch"] == 4)
    args = _build(spec, seed=17)
    for flag in (False, True):
        with validating(op):
            op(**dict(args, kv_evict_first=flag, out=_out(*args["out"].shape)))
    torch.cuda.synchronize()
    family = RubinPagedMQALogitsRunner.kernel_cache
    ours = [k for k in family if k[4] == 8 and k[1] == spec["tokens_per_block"]]
    assert {k[-1] for k in ours} >= {False, True}
    assert family is not _dsl_ops.CuteDSLPagedMQALogitsRunner.kernel_cache
    assert not set(family) & set(_dsl_ops.CuteDSLPagedMQALogitsRunner.kernel_cache)


def test_row_stride_at_the_unrounded_width():
    """A row stride of exactly max_context_len (not a multiple of 256) computes the same logits."""
    spec = dict(op.CELLS[2].spec, max_context_len=10000)
    args = _build(spec, seed=11)
    rows = args["q"].shape[0] * args["q"].shape[1]
    with validating(op):
        tight = op(**dict(args, out=_out(rows, 10000, row_stride=10000))).clone()
        padded = op(**dict(args, out=_out(rows, 10000)))
    torch.cuda.synchronize()
    causal = torch.isfinite(op.reference(**args))
    assert torch.equal(tight[causal], padded[causal])


def test_guard_refuses_a_schedule_for_another_sm_count():
    args = _build(op.CELLS[2].spec, seed=8)
    fewer = get_paged_mqa_logits_metadata(
        args["context_lens"].reshape(-1, 1), inputs.SCHEDULE_BLOCK_KV, inputs.num_sms() - 8
    )
    with pytest.raises(AssertionError, match="schedule_meta"):
        with validating(op):
            op(**dict(args, schedule_meta=fewer))
