# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OPROJ_PUSH_GEMM's o_proj GEMM (``kernels/lamport_push_gemm``) on the installed FlashInfer, one GPU.

The kernel is FlashInfer's CuTe DSL MXFP8 dense GEMM with a Lamport-push epilogue: it stores the product into the
current Lamport data slot's local and peer sections (selected by the flag word, mod 3) with every 32-bit word
0x80000000 replaced by +0. Here the six slot pointers are scratch buffers, so no collective is needed. At every pinned
o_proj tactic of 49..256 rows its bytes must equal FlashInfer's own GEMM at that tactic (the family's pinned table)
after the same -0 rule, land only in the flag's slot, and include rows that are all zero (signed zeros).

The prebuild byte check the modeling runs before engaging the GEMM must pass against the decode graph's stock GEMM.
The 2-rank boundary (the GEMM pushing into the real workspace, then the prepushed all-reduce) is certified by
``comm/_ar_prepushed_op_matrix.py``.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2.kernels import (  # noqa: E501
        lamport_push_gemm as lpg,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.gemm import (
    mxfp8_cute_pinned_tactics as pin,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP2 target runs on SM107 only",
)

N, K = 6144, 4096
ROWS = (49, 64, 65, 100, 128, 129, 200, 256)
NEG_ZERO_WORD = -(2**31)


def _tactic(rows: int):
    return pin.TACTICS[(N, K, max(64, 1 << (rows - 1).bit_length()))]


@pytest.fixture(scope="module")
def operands():
    torch.ops.trtllm.mxfp8_quantize  # noqa: B018 - the stock quantize must be registered
    g = torch.Generator(device="cuda").manual_seed(7)
    w_bf16 = (torch.randn(N, K, device="cuda", generator=g) * 0.05).to(torch.bfloat16)
    w, w_sf = torch.ops.trtllm.mxfp8_quantize(w_bf16, True)
    return w, w_sf


@pytest.fixture(scope="module")
def launcher():
    # Only the slot addressing reads the workspace; every call below passes its own slots and flag.
    ws = torch.zeros(8, dtype=torch.int64, device="cuda")
    return lpg.LamportPushGemmLauncher(ws, 0, N, K, lambda addr: 0)


@pytest.mark.parametrize("flag", [0, 4])
@pytest.mark.parametrize("rows", ROWS)
def test_push_gemm_bytes_are_flashinfers_after_the_push_rule(operands, launcher, rows, flag):
    w, w_sf = operands
    tactic = _tactic(rows)
    assert tactic[4] == 1, "the epilogue serves non-split-K tactics only"
    g = torch.Generator(device="cuda").manual_seed(rows)
    x = (torch.randn(rows, K, device="cuda", generator=g) * 4).to(torch.bfloat16)
    x[: rows // 8] = 0
    x[rows // 8 : rows // 4] = -0.0
    act, act_sf = torch.ops.trtllm.mxfp8_quantize(x, True)
    with pin.engaged_for("test_m31_lamport_push_gemm"):
        ref = pin.run(act, act_sf, w, w_sf, torch.bfloat16)
    assert ref is not None, f"FlashInfer's runner did not take the pinned tactic {tactic}"
    words = ref.view(torch.int32).clone()
    words[words == NEG_ZERO_WORD] = 0

    sentinel = torch.tensor(0x7FC07FC0, dtype=torch.int32, device="cuda")
    scratch = sentinel.expand(6, rows, N // 2).contiguous()
    flag_word = torch.full((1,), flag, dtype=torch.int32, device="cuda")
    dummy = torch.empty((rows, N), dtype=torch.bfloat16, device="cuda")
    launcher.run(
        tactic,
        act,
        act_sf,
        w,
        w_sf,
        dummy,
        slots=[scratch[i].data_ptr() for i in range(6)],
        flag_addr=flag_word.data_ptr(),
    )
    torch.cuda.synchronize()
    slot = flag % 3
    for i in range(6):
        if i // 2 == slot:
            bad = int((scratch[i] != words).sum().item())
            assert bad == 0, f"rows={rows} tactic={tactic} section {i}: {bad} words differ"
        else:
            assert bool((scratch[i] == sentinel).all().item()), f"section {i} outside slot {slot}"
    assert not bool((scratch[2 * slot] == NEG_ZERO_WORD).any().item())


def _modeling():
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2 import (  # noqa: E501
        modeling,
    )

    return modeling


@pytest.mark.parametrize("rows", (49, 128, 256))
def test_prebuild_byte_check_accepts_the_stock_decode_gemm(operands, launcher, rows):
    """The modeling engages OPROJ_PUSH_GEMM only when its prebuild check finds the epilogue GEMM's bytes equal to the
    decode graph's stock GEMM (m3_mxfp8_pinned_quantize + m3_mxfp8_pinned_gemm at the engine's tactic table, which the
    model's RuntimeHooks arm): the check must pass here, and leave the table's armed state as it found it."""
    modeling = _modeling()
    table = modeling._engine_pinned_tactics()
    before = table._ARMED[0]
    w, w_sf = operands
    assert modeling._push_gemm_tactic(rows) == _tactic(rows)
    assert modeling._push_gemm_check(launcher, _tactic(rows), rows, w, w_sf) is None
    assert table._ARMED[0] == before


def test_steps_take_the_gemm_only_while_the_engine_table_is_armed():
    modeling = _modeling()
    table = modeling._engine_pinned_tactics()
    before = table._ARMED[0]
    try:
        table._ARMED[0] = False
        assert not modeling._push_gemm_stock_is_pinned()
        table._ARMED[0] = True
        assert modeling._push_gemm_stock_is_pinned() == table.ENABLED
    finally:
        table._ARMED[0] = before
