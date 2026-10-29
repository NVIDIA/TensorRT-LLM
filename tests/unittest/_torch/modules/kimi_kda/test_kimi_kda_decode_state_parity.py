# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sequential KDA decode parity for live W-1 convolution and recurrent pools.

Compare indexed decode with the production fallback using identical weights,
changing tokens, and independently cloned BF16/FP32 recurrent state. Check the
selected rows against the fallback and untouched rows exactly after every step.
Requires one GPU and fla-core; no checkpoint.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("fla")

from tensorrt_llm._torch.modules.kimi_kda import KimiKDALinearAttention  # noqa: E402
from tests.unittest._torch.modules.kimi_kda.kda_module_test_utils import compare  # noqa: E402

HIDDEN_SIZE = 512
NUM_HEADS = 4
HEAD_DIM = 128
CONV_WIDTH = 4


class _Cfg:
    hidden_size = HIDDEN_SIZE
    rms_norm_eps = 1e-6
    linear_attn_config = {
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "short_conv_kernel_size": CONV_WIDTH,
        "use_full_rank_gate": True,
        "gate_lower_bound": -5.0,
    }


def _make_runtime() -> KimiKDALinearAttention:
    if not torch.cuda.is_available():
        pytest.skip("the fused KDA decode kernel requires a CUDA device")
    if torch.cuda.get_device_capability(0) not in {(10, 0), (10, 3), (10, 7)}:
        pytest.skip("Kimi K3 requires SM100/SM103/SM107")
    runtime = KimiKDALinearAttention(_Cfg(), layer_idx=0).to("cuda")
    # Checkpoints initialize dt_bias; seed it before finalizing kernel constants.
    torch.nn.init.normal_(runtime.dt_bias, std=0.1)
    runtime.finalize_decode_weights()
    return runtime


@torch.no_grad()
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch", [1, 3])
def test_decode_state_matches_fallback_over_many_steps(
    batch: int, state_dtype: torch.dtype
) -> None:
    torch.manual_seed(0)
    slots, steps = batch + 3, 6
    dim = NUM_HEADS * HEAD_DIM
    runtime = _make_runtime()
    # Nonzero, reversed slots detect accidental batch-row addressing.
    slot_indices = torch.arange(slots - 1, slots - batch - 1, -1, dtype=torch.int32, device="cuda")
    slots_long = slot_indices.long()
    untouched = torch.arange(3, device="cuda")
    conv_pool = (
        torch.randn(slots, 3 * dim, CONV_WIDTH - 1, dtype=torch.bfloat16, device="cuda") * 0.05
    )
    ssm_pool = (
        torch.randn(slots, NUM_HEADS, HEAD_DIM, HEAD_DIM, dtype=state_dtype, device="cuda") * 0.05
    )
    direct_conv, direct_ssm = conv_pool.clone(), ssm_pool.clone()
    fallback_conv, fallback_ssm = conv_pool.clone(), ssm_pool.clone()
    hidden = torch.randn(batch, steps, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.1
    hidden_before = hidden.clone()
    metadata = SimpleNamespace(
        _arange_buffer=torch.arange(batch + 1, dtype=torch.int32, device="cuda")
    )

    for step in range(steps):
        actual = runtime.forward_decode(
            hidden[:, step],
            direct_conv,
            direct_ssm,
            slots_long,
            mamba_metadata=metadata,
            ssm_state_indices=slot_indices,
        )
        expected = runtime.forward_decode_fallback(
            hidden[:, step],
            fallback_conv,
            fallback_ssm,
            slots_long,
        )
        # Fused and separate projections can round differently; use the established
        # module parity criteria, while op-only pool updates stay bit-exact.
        for name, got, want in (
            ("output", actual, expected),
            (
                "conv",
                direct_conv.index_select(0, slots_long),
                fallback_conv.index_select(0, slots_long),
            ),
            (
                "state",
                direct_ssm.index_select(0, slots_long),
                fallback_ssm.index_select(0, slots_long),
            ),
        ):
            metrics = compare(got, want)
            assert metrics["passed"], (step, name, metrics)
        for changed, original in (
            (direct_conv, conv_pool),
            (fallback_conv, conv_pool),
            (direct_ssm, ssm_pool),
            (fallback_ssm, ssm_pool),
        ):
            torch.testing.assert_close(
                changed.index_select(0, untouched),
                original.index_select(0, untouched),
                rtol=0,
                atol=0,
            )
    torch.testing.assert_close(hidden, hidden_before, rtol=0, atol=0)
