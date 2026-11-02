# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for ops/moe/fp4_block_scale_moe_renorm_permuted.

Per cell: the emulated finalize of the three outputs is **bitwise** the
finalized entry's output on identical operands (so the fold a consumer makes
with the same arithmetic is bitwise what the runner's finalize stored), the
outputs have the documented shapes, dtypes and index map, and the finalized
rows sit inside the pre-routed entry's band against the fp32 reference. The
operand construction uses the shared NVFP4 test helpers.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.moe.fp4_block_scale_moe_renorm_permuted import (  # noqa: E501
    finalize,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.moe.fp4_block_scale_moe_renorm_permuted import (  # noqa: E501
    fp4_block_scale_moe_renorm_permuted as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.moe.fp4_block_scale_moe_renorm_routed import (  # noqa: E501
    fp4_block_scale_moe_renorm_routed as finalized,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.moe.renorm_moe_routing_op import (  # noqa: E501
    renorm_moe_routing_op,
)

__extra_import_path__ = ["../../../.."]
import _fp4_moe_test_utils as H  # noqa: E402
from _fp4_moe_test_utils import many_expert_layer  # noqa: E402,F401 — fixture by name
from _precision import true_fp32_matmul  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "fp4_block_scale_moe_renorm_permuted requires a CUDA device"


def _args(args, data, sf, logits, scalars, spec, top_k=None):
    return (
        logits,
        data,
        sf.view(torch.float8_e4m3fn),
        args["gemm1_weights"],
        args["gemm1_weights_scale"],
        args["gemm2_weights"],
        args["gemm2_weights_scale"],
        *scalars,
        spec["num_experts"],
        spec["top_k"] if top_k is None else top_k,
        spec["intermediate"],
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying, many_expert_layer):  # noqa: F811
    certifying(op)
    spec = cell.spec
    t, k, hidden = spec["tokens"], spec["top_k"], spec["hidden"]
    seed = zlib.crc32(cell.why.encode()) % 2**31
    args, ref, data, sf, x, logits, ids, wts, g2, scalars = H.renorm_operands(
        many_expert_layer(), spec, seed, renorm_moe_routing_op
    )
    call = _args(args, data, sf, logits, scalars, spec)

    rows, topk_w, topk_rows = op(*call)
    assert rows.dtype == torch.bfloat16 and rows.dim() == 2 and rows.shape[1] == hidden
    assert topk_w.shape == (t, k) and topk_w.dtype == torch.bfloat16
    assert topk_rows.shape == (t, k) and topk_rows.dtype == torch.int32
    # Every slot maps to its own row (no expert parallelism: no -1 slots).
    assert int(topk_rows.min()) >= 0 and int(topk_rows.max()) < rows.shape[0]
    assert topk_rows.flatten().unique().numel() == t * k, "two slots share a permuted row"
    # The weights are the routing's (as a set per token; the slot order is the runner's).
    assert torch.equal(topk_w.sort(-1).values, wts.sort(-1).values), "routing weights differ"

    # 1. Bitwise the finalized entry once finalized with the runner's arithmetic.
    y = finalize(rows, topk_w, topk_rows)
    y_fin = finalized(*call)
    differing = (y.view(torch.int16) != y_fin.view(torch.int16)).any(dim=1).sum().item()
    assert differing == 0, f"{differing} of {t} rows differ from the finalized entry"

    # 2. The fp32 reference, through the pre-routed entry's gate.
    with true_fp32_matmul():
        exp = op.reference(logits, x, k, ref["up"], ref["gate"], ref["down"], g2)
    H._compare(y, exp.to(torch.bfloat16))


def test_uncertified_pair_refused(many_expert_layer):  # noqa: F811
    spec = op.CELLS[0].spec
    args, _, data, sf, _, logits, _, _, _, scalars = H.renorm_operands(
        many_expert_layer(), spec, 13, renorm_moe_routing_op
    )
    with validating(op), pytest.raises(AssertionError, match="certified Renormalize"):
        op(*_args(args, data, sf, logits, scalars, spec, top_k=9))


def test_cuda_graph_capture_and_replay(many_expert_layer):  # noqa: F811
    spec = next(c.spec for c in op.CELLS if c.spec["tokens"] == 1)
    args, _, data, sf, _, logits, _, _, _, scalars = H.renorm_operands(
        many_expert_layer(), spec, 43, renorm_moe_routing_op
    )
    call = list(_args(args, data, sf, logits, scalars, spec))
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            op(*call)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_rows, g_w, g_idx = op(*call)
    logits.copy_(torch.randn_like(logits.float()).to(logits.dtype))
    graph.replay()
    torch.cuda.synchronize()
    want = finalized(*call)
    assert torch.equal(finalize(g_rows, g_w, g_idx), want)
