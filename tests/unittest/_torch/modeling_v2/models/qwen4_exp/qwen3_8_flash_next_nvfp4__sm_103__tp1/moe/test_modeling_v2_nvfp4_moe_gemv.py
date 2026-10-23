# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for ops/moe/nvfp4_moe_gemv.

Per cell, on the runner test's exact operands (512 experts in the trtllm-gen layout,
power-of-two block scales): the routing weights are bitwise the trtllm-gen runner's in its
slot order, every slot's FC2 row is bitwise the runner's row for that slot (with power-of-two
scales every accumulation is exact, so accumulation order cannot show), and the finalized
rows sit inside the pre-routed entry's band against the fp32 reference. Beyond the cells:
CUDA-graph replay and the row-count refusal. The operand construction is the routed entry's
test's, loaded by path.
"""

import importlib.util
import zlib
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.moe.fp4_block_scale_moe_renorm_permuted import (  # noqa: E501
    finalize,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.moe.fp4_block_scale_moe_renorm_permuted import (  # noqa: E501
    fp4_block_scale_moe_renorm_permuted as runner,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.moe.nvfp4_moe_gemv import (  # noqa: E501
    MAX_ROWS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.moe.nvfp4_moe_gemv import (  # noqa: E501
    nvfp4_moe_gemv_op as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "nvfp4_moe_gemv requires a CUDA device"


def _routed_test():
    path = Path(__file__).resolve().parent / "test_modeling_v2_fp4_block_scale_moe_renorm_routed.py"
    spec = importlib.util.spec_from_file_location("_fp4_moe_renorm_routed_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _routed_test()
H = R.H


def _args(args, data, sf, logits, scalars, spec):
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
        spec["top_k"],
        spec["intermediate"],
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    spec = cell.spec
    t, k, hidden = spec["tokens"], spec["top_k"], spec["hidden"]
    seed = zlib.crc32(cell.why.encode()) % 2**31
    args, ref, data, sf, x, logits, ids, wts, g2, scalars = R._operands(spec, seed)
    call = _args(args, data, sf, logits, scalars, spec)

    rows, topk_w, topk_rows = op(*call)
    assert rows.shape == (t * k, hidden) and rows.dtype == torch.bfloat16
    assert topk_w.shape == (t, k) and topk_w.dtype == torch.bfloat16
    assert torch.equal(topk_rows.cpu(), torch.arange(t * k, dtype=torch.int32).view(t, k))

    r_rows, r_w, r_idx = runner(*call)
    assert torch.equal(topk_w, r_w), "routing weights differ from the runner's"
    slot_rows = r_rows[r_idx.flatten().long()]
    differing = (slot_rows.view(torch.int16) != rows.view(torch.int16)).any(dim=1).sum().item()
    assert differing == 0, f"{differing} of {t * k} slot rows differ from the runner's"

    exp = op.reference(logits, x, k, ref["up"], ref["gate"], ref["down"], g2)
    H._compare(finalize(rows, topk_w, topk_rows), exp.to(torch.bfloat16))


def test_cuda_graph_capture_and_replay():
    spec = next(c.spec for c in op.CELLS if c.spec["tokens"] == 4)
    args, _, data, sf, _, logits, _, _, _, scalars = R._operands(spec, 43)
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
    want = op(*call)
    assert torch.equal(g_w, want[1])
    assert torch.equal(finalize(g_rows, g_w, g_idx), finalize(*want))


def test_too_many_rows_refused():
    spec = dict(op.CELLS[0].spec, tokens=MAX_ROWS + 1)
    args, _, data, sf, _, logits, _, _, _, scalars = R._operands(spec, 13)
    with validating(op), pytest.raises(AssertionError, match="rows"):
        op(*_args(args, data, sf, logits, scalars, spec))
