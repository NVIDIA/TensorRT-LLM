# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for ``draft_shard_topk`` (fast path ``draft_head``).

Gated on its cells with ``torch.equal`` against its torch reference, in both
output layouts: the int32 keys the all-gather takes and the fp32 halves the
fabric all-reduce takes.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec import (  # noqa: E501
    draft_shard_topk as shard_mod,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec import (  # noqa: E501
    draft_topk_merge as merge_mod,
)

__extra_import_path__ = ["../../../../..", "../../kernels/draft_head"]
from _draft_head_cases import (  # noqa: E402
    TP,
    VOCAB,
    draft_shard_topk,
    logits,
    packed_keys,
    seed_of,
    shard_keys,
)
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "the vocab-sharded draft head requires a CUDA device"


@pytest.mark.parametrize(
    "cell", shard_mod.DraftShardTopK.CELLS, ids=[c.why[:48] for c in shard_mod.DraftShardTopK.CELLS]
)
def test_shard_topk_cells(cell, certifying):  # noqa: F811
    certifying(draft_shard_topk)
    spec = cell.spec
    kind = "special_nan" if spec["values"] == "special" else spec["values"]
    x = logits(kind, spec["rows"], spec["shard"], seed_of(cell.why))
    out = draft_shard_topk(x, spec["top_k"], spec["chunk"])
    ref = draft_shard_topk.reference(x, spec["top_k"], spec["chunk"])
    assert out.dtype == torch.int32 and out.shape == ref.shape
    assert torch.equal(out, ref), f"{(out != ref).sum().item()} keys differ"


@pytest.mark.parametrize("rank", [0, 3, 7])
def test_shard_topk_allreduce_layout(rank):
    """The fp32-halves layout: this rank's slot exact, zeros elsewhere, and it sums to the gather."""
    full = logits("special_nan", 7, VOCAB, 17)
    shard = VOCAB // TP
    local = full[:, rank * shard : (rank + 1) * shard]
    packed = draft_shard_topk(local, rank=rank, num_ranks=TP)
    assert packed.dtype == torch.float32 and packed.shape == (7, TP * 112 * 2)
    assert torch.equal(packed, draft_shard_topk.reference(local, rank=rank, num_ranks=TP))
    assert not torch.signbit(packed).any(), "no -0.0: it is the fabric all-reduce's empty marker"
    assert torch.equal(merge_mod.unpack_halves(packed_keys(full, TP), TP), shard_keys(full, TP))


def test_shard_topk_reads_a_strided_row_view():
    full = logits("logits", 7, VOCAB, 5)
    shard = VOCAB // TP
    view = full[:, 3 * shard : 4 * shard]
    assert view.stride(0) == VOCAB
    assert torch.equal(draft_shard_topk(view), draft_shard_topk(view.contiguous()))


def test_shard_topk_guards_refuse_what_the_kernel_cannot_take():
    with validating(draft_shard_topk):
        with pytest.raises(ValueError):  # wider than the 16-bit column field
            draft_shard_topk(torch.zeros(2, 70000, device="cuda", dtype=torch.bfloat16))
        with pytest.raises(ValueError):  # not bf16
            draft_shard_topk(torch.zeros(2, 1024, device="cuda", dtype=torch.float32))
