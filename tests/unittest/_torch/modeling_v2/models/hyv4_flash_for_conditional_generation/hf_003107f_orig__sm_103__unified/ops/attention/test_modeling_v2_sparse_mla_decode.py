# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the hyv4_flash sm_103 target's sparse_mla_decode entry.

The entry owns the contract: `CELLS` is the certified range, `reference` is what
the kernel is compared against, and `is_valid` is the guard. This file owns what
a cell cannot carry -- turning a spec into real tensors, the comparison band and
its derivation, and the inputs the guard exists to refuse.

Layout follows glm5_next's sparse_mla_decode test
(`tests/unittest/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/
attention/test_modeling_v2_sparse_mla_decode.py`), whose entry wraps the same
FlashInfer callable for the NoPE geometry; the tensors here are Hy4-Flash's
rope-64 rows and 4/8 heads per rank.

The length contract is measured, not assumed (sm_103, GB300): a row's `lengths[r]`
is its visible-token count and the kernel reads the first `min(lengths[r], top_k)`
columns of that row's selection. Each multi-row cell therefore carries the three
ends of that range -- a one-token prompt's row, a short request's row whose
selection is 700 of the 2048 slots, and a row whose physical length is past the
selection width, which the kernel clamps.
"""

import time
import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.ops.attention.sparse_mla_decode import (  # noqa: E501
    WORKSPACE_BYTES,
    counter_bytes,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.ops.attention.sparse_mla_decode import (  # noqa: E501
    sparse_mla_decode as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 103,
    reason="the hyv4_flash sm_103 target's entries are certified on sm_103",
)

assert torch.cuda.is_available(), "sparse_mla_decode requires a CUDA device"

DEVICE = "cuda"
CELLS = op.CELLS

#: Hy4-Flash's absorbed MLA geometry: one row is [ckv 512 | kpe 64], read as 32-row pages.
DIM, KV_LORA, PAGE, TOP_K = 576, 512, 32, 2048
#: The latent pool the selections index: 256 pages. Every pick is a distinct row of it.
POOL_ROWS = 256 * PAGE
#: The model's softmax scale, 1 / sqrt(qk_nope_head_dim 192 + qk_rope_head_dim 64).
SOFTMAX_SCALE = 0.0625
#: The FP8 cache's per-tensor dequantization scale: a power of two, so dequantization is exact
#: and the fp8 cell's only inexactness is the kernel's own.
KV_SCALE = 0.5

#: Relative-RMS band per dtype, measured on sm_103 (GB300) over the cells below: the core path
#: lands at 1.5e-3 - 2.2e-3 (2**-7 is 3.5x the worst of them, and tighter than torch's own
#: bf16 default of 1.6e-2), the FP8 path at 1.7e-2 (2**-4 is 3.8x). The derivation is the
#: entry's `note` plus the numbers this file prints.
REL_RMS_BAND = {"bf16": 2**-7, "e4m3": 2**-4}
#: A cosine floor far below the band: it catches a wrong result, not a rounding.
COS_FLOOR = {"bf16": 1 - 2**-14, "e4m3": 1 - 2**-10}


def _lengths(spec: dict) -> torch.Tensor:
    """Each row's visible-token count. A single-row cell selects the full top-k; wider cells
    also carry the three ends of the range -- a one-token prompt's row (length 1), a short
    request whose selection is 700 of the 2048 slots, and a row whose physical length is past
    the selection width (``TOP_K + 1234``), which the kernel clamps to ``TOP_K``."""
    rows = spec["rows"]
    lengths = torch.full((rows,), TOP_K, dtype=torch.int32, device=DEVICE)
    if rows > 2:
        lengths[0] = 1
        lengths[1] = 700
        lengths[2] = TOP_K + 1234
    return lengths


def _args(spec: dict, seed: int):
    """The op's arguments: a latent pool at the cell's dtype, each row's distinct selected pool
    rows in front of ``-1`` padding, queries at the same dtype, zeroed workspace and counter,
    and the scales.

    The FP8 cache holds ``value / KV_SCALE`` (the model's layout) and the scales carry the
    dequantization -- ``bmm1_scale = KV_SCALE * SOFTMAX_SCALE``, ``bmm2_scale = KV_SCALE`` as a
    device tensor -- so the kernel and the reference both attend over the same values."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    rows, heads = spec["rows"], spec["heads"]
    fp8 = spec["dtype"] == "e4m3"
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    pool = torch.randn(POOL_ROWS, DIM, generator=g, device=DEVICE)
    if fp8:
        pool = pool / KV_SCALE
    pool = pool.to(dtype)
    lengths = _lengths(spec)
    packed = torch.full((rows, TOP_K), -1, dtype=torch.int32, device=DEVICE)
    for r in range(rows):
        n = min(int(lengths[r]), TOP_K)
        packed[r, :n] = torch.randperm(POOL_ROWS, generator=g, device=DEVICE)[:n].to(torch.int32)
    query = torch.randn(rows, heads, DIM, generator=g, device=DEVICE).to(dtype)
    sm_count = torch.cuda.get_device_properties(DEVICE).multi_processor_count
    workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=DEVICE)
    counter = torch.zeros(counter_bytes(rows, heads, sm_count), dtype=torch.uint8, device=DEVICE)
    if fp8:
        bmm2 = torch.tensor([KV_SCALE], dtype=torch.float32, device=DEVICE)
        return query, pool, packed, lengths, workspace, counter, KV_SCALE * SOFTMAX_SCALE, bmm2
    return query, pool, packed, lengths, workspace, counter, SOFTMAX_SCALE, 1.0


def _judge(out, ref, dtype):
    """The cell's distribution against the reference: relative RMS and cosine inside the
    measured band. The worst element is returned, not asserted on -- a cell is judged by its
    distribution, and the negative control below is what separates a wrong result from a
    rounding."""
    err = (out.float() - ref.float()).abs()
    scale = float(ref.float().square().mean().sqrt())
    rel_rms = float(err.square().mean().sqrt() / scale)
    cos = float(
        torch.nn.functional.cosine_similarity(
            out.float().reshape(-1), ref.float().reshape(-1), dim=0
        )
    )
    worst = float(err.max()) / scale
    assert rel_rms < REL_RMS_BAND[dtype], f"relative RMS error {rel_rms:.3e}"
    assert cos > COS_FLOOR[dtype], f"cosine similarity {cos:.9f}"
    return rel_rms, cos, worst


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_sparse_mla_decode_matches_reference(cell, certifying):  # noqa: F811
    """Every cell's output matches the FP32 reference over its selected rows, by distribution.

    The kernel hands its multi-CTA counter back zeroed and reads no row outside the pool."""
    certifying(op)
    args = _args(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    started = time.perf_counter()
    out = op(*args)
    elapsed = time.perf_counter() - started
    ref = op.reference(*args)
    query, pool, packed, lengths, workspace, counter, bmm1, bmm2 = args
    assert out.shape == ref.shape == (query.shape[0], query.shape[1], KV_LORA)
    assert out.dtype == torch.bfloat16
    assert not bool(counter.any()), "the kernel leaves its counter at zero for the next launch"
    assert bool(torch.isfinite(out).all())
    rel_rms, cos, worst = _judge(out, ref, cell.spec["dtype"])
    print(
        f"\n{cell.why}: rel_rms={rel_rms:.4e} cos={cos:.9f} worst={worst:.3e} "
        f"kernel={elapsed * 1e3:.1f} ms"
    )


def test_a_wrong_selection_falls_outside_the_band():
    """The negative control: the same call judged against a different, equally plausible set of
    latent rows. The selection is a *set* -- reordering one row's picks is the identical result
    (measured 1.0x), so only a different set separates the band, and it misses by the margin
    the cell test relies on."""
    cell = CELLS[0]
    args = list(_args(cell.spec, zlib.crc32(b"wrong-selection") % 2**31))
    out = op(*args)
    ref = op.reference(*args)
    good = (out.float() - ref.float()).abs()
    g = torch.Generator(device=DEVICE).manual_seed(zlib.crc32(b"other-selection") % 2**31)
    rows, width = args[2].shape
    args[2] = torch.stack(
        [torch.randperm(POOL_ROWS, generator=g, device=DEVICE)[:width] for _ in range(rows)]
    ).to(torch.int32)
    wrong = (out.float() - op.reference(*args).float()).abs()
    good_rel = float(good.square().mean().sqrt() / ref.float().square().mean().sqrt())
    wrong_rel = float(wrong.square().mean().sqrt() / ref.float().square().mean().sqrt())
    print(f"\nnegative control: correct {good_rel:.3e}, other set {wrong_rel:.3e}")
    assert wrong_rel > 20 * max(good_rel, REL_RMS_BAND[cell.spec["dtype"]]), (
        f"a different selection is only {wrong_rel / good_rel:.1f}x the correct one's error"
    )


def test_guard_refuses_a_short_workspace():
    """The kernel's split-KV scratch is the rest of the workspace: a short one is overrun."""
    args = list(_args(dict(heads=8, rows=1, dtype="bf16"), seed=0))
    args[4] = torch.zeros(1 << 20, dtype=torch.uint8, device=DEVICE)
    with pytest.raises(AssertionError, match="workspace"):
        with validating(op):
            op(*args)


def test_guard_refuses_a_zero_length():
    """A zero-length row leaves the kernel no row to read: unsupported, not a no-op."""
    args = list(_args(dict(heads=8, rows=1, dtype="bf16"), seed=0))
    args[3] = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    with pytest.raises(AssertionError, match="length"):
        with validating(op):
            op(*args)


def test_guard_refuses_a_length_past_the_picks():
    """A length longer than the row's valid picks makes the kernel read the ``-1`` padding: the
    measured silent wrong answer, relative RMS 0.32 against the cell's 0.0019."""
    args = list(_args(dict(heads=8, rows=1, dtype="bf16"), seed=0))
    args[2] = args[2].clone()
    args[2][0, 700:] = -1
    with pytest.raises(AssertionError, match="selected id"):
        with validating(op):
            op(*args)


def test_guard_refuses_a_zero_width_selection():
    """``sparse_mla_top_k=0`` is dense MLA over the whole cache -- a different operation with
    the same call shape, so an empty selection is refused rather than run."""
    args = list(_args(dict(heads=8, rows=1, dtype="bf16"), seed=0))
    args[2] = torch.zeros(1, 0, dtype=torch.int32, device=DEVICE)
    with pytest.raises(AssertionError, match="dense"):
        with validating(op):
            op(*args)
