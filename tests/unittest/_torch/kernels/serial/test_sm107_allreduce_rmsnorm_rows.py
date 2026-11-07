# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import tensorrt_llm
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceStrategy

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _assert_rmsnorm_contract(
    outputs: list[torch.Tensor],
    x: torch.Tensor,
    residual: torch.Tensor | None,
    bias: torch.Tensor | None,
    weight: torch.Tensor,
) -> None:
    # Native add128b rounds after each bias/residual addition in the input
    # dtype. Evaluate the norm independently in FP64 after those additions.
    inter = x if bias is None else x + bias
    if residual is not None:
        inter = inter + residual
    inter_fp64 = inter.double()
    expected_norm = (
        inter_fp64
        * torch.rsqrt(inter_fp64.square().mean(dim=-1, keepdim=True) + 1e-5)
        * weight.double()
    )
    precision = torch.finfo(x.dtype)
    torch.testing.assert_close(
        outputs[0].double(),
        expected_norm,
        rtol=2 * precision.eps,
        atol=2 * precision.tiny * precision.eps,
    )
    # Without a residual, the native kernel leaves its intermediate untouched;
    # bias still contributes to the norm but is not materialized in that output.
    expected_inter = x if residual is None else inter
    assert torch.equal(outputs[1].view(torch.uint8), expected_inter.view(torch.uint8))


@pytest.mark.parametrize("rows_case", ["threshold", "tail", "multiple-waves"])
@pytest.mark.parametrize("hidden", [2880, 6144])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "with_bias,with_residual", [(False, False), (False, True), (True, False), (True, True)]
)
def test_sm107_allreduce_rmsnorm_rows_bitwise(
    rows_case: str,
    hidden: int,
    dtype: torch.dtype,
    with_bias: bool,
    with_residual: bool,
) -> None:
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Persistent RMSNorm rows dispatch requires SM107")
    sm_count = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    rows = {"threshold": 1024, "tail": 1025, "multiple-waves": max(1027, 2 * sm_count + 1)}[
        rows_case
    ]
    torch.manual_seed(20629)
    x = torch.randn(rows, hidden, dtype=dtype, device="cuda")
    residual = torch.randn_like(x) if with_residual else None
    bias = torch.randn(hidden, dtype=dtype, device="cuda") if with_bias else None
    weight = torch.randn(hidden, dtype=dtype, device="cuda")
    group = [tensorrt_llm.mpi_rank()]

    def run(start: int, end: int) -> list[torch.Tensor]:
        outputs = torch.ops.trtllm.allreduce(
            x[start:end],
            residual[start:end] if residual is not None else None,
            weight,
            None,
            bias,
            None,
            group,
            int(AllReduceStrategy.NCCL),
            int(AllReduceFusionOp.RESIDUAL_RMS_NORM),
            1e-5,
            False,
        )
        # A build without multi-device support returns only the input. Reject
        # that no-op even though this norm-dispatch test needs just one GPU.
        assert len(outputs) == 2
        return outputs

    whole = run(0, rows)
    _assert_rmsnorm_contract(whole, x, residual, bias, weight)
    chunks = [run(start, min(start + 1023, rows)) for start in range(0, rows, 1023)]
    for output_index in range(2):
        assert torch.equal(
            whole[output_index].view(torch.uint8),
            torch.cat([chunk[output_index].view(torch.uint8) for chunk in chunks]),
        )
