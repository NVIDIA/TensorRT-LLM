# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.quantization import W4A16NVFP4CutlassFusedMoEMethod

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("gated", [False, True], ids=["relu2", "swiglu"])
@pytest.mark.parametrize("streamed", [False, True], ids=["whole_checkpoint", "streamed"])
def test_w4a16_eplb_migrates_linear_block_scales(gated: bool, streamed: bool) -> None:
    """Host expert scales must have the layout consumed by resident-slot dequantization."""
    method = W4A16NVFP4CutlassFusedMoEMethod()
    method.block_scales_vec_size = 4
    module = torch.nn.Module()
    module.tp_size = 1
    module.tp_rank = 0
    module.initial_local_expert_ids = [2, 0]
    module.layer_load_balancer = SimpleNamespace(get_load_expert_ids=lambda: [0, 1, 2, 3])
    fc1_rows = 256 if gated else 128
    module.w3_w1_weight_scale = torch.nn.Parameter(
        torch.zeros(2, fc1_rows, 4, dtype=torch.int32), requires_grad=False
    )
    module.w2_weight_scale = torch.nn.Parameter(
        torch.zeros(2, 256, 2, dtype=torch.int32), requires_grad=False
    )
    module.local_shared_w3_w1_scale_tensors = torch.zeros(4, fc1_rows, 4, dtype=torch.int32)
    module.local_shared_w2_scale_tensors = torch.zeros(4, 256, 2, dtype=torch.int32)
    module.fc31_input_scale = torch.nn.Parameter(torch.ones(()), requires_grad=False)
    module.fc2_input_scale = torch.nn.Parameter(torch.ones(()), requires_grad=False)
    module.fc31_alpha = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
    module.fc2_alpha = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
    module.tmp_raw_input_scales = {
        i: {name: torch.tensor(1.0) for name in ("w1", "w3", "w2")} for i in range(4)
    }
    global_scales = {
        i: {"w1": torch.tensor(i + 1.0), "w3": torch.tensor(i + 1.0), "w2": torch.tensor(i + 2.0)}
        for i in range(4)
    }
    module.tmp_weight_scale_2 = {
        slot: global_scales[expert] for slot, expert in enumerate(module.initial_local_expert_ids)
    }
    module.tmp_shared_weight_scale_2 = global_scales
    registered = {}
    module.register_all_parameter_slot_and_to_fix_weight_fns = registered.update

    def scales(rows: int, columns: int, offset: int) -> torch.Tensor:
        return ((torch.arange(rows * columns).reshape(rows, columns) + offset) % 13 + 1).to(
            torch.float8_e4m3fn
        )

    source_scales = [
        (scales(128, 16, i), scales(128, 16, i + 2) if gated else None, scales(256, 8, i + 4))
        for i in range(4)
    ]
    expected_fc1 = [
        (torch.cat([w3, w1]) if gated else w1).view(torch.int32) for w1, w3, _ in source_scales
    ]
    expected_fc2 = [w2.view(torch.int32) for _, _, w2 in source_scales]
    for expert_ids, fc1, fc2 in (
        (module.initial_local_expert_ids, module.w3_w1_weight_scale, module.w2_weight_scale),
        (
            list(range(4)),
            module.local_shared_w3_w1_scale_tensors,
            module.local_shared_w2_scale_tensors,
        ),
    ):
        for slot, expert in enumerate(expert_ids):
            w1, w3, w2 = source_scales[expert]
            method.load_expert_w3_w1_weight_scale_nvfp4(module, w1, w3, fc1[slot], expert_idx=slot)
            method.load_expert_w2_weight_scale_nvfp4(module, w2, fc2[slot])
            if streamed and fc1 is module.w3_w1_weight_scale:
                method.finalize_streamed_expert(module, slot)

    method.process_weights_after_loading(module)

    for slot, expert in enumerate(module.initial_local_expert_ids):
        assert torch.equal(module.w3_w1_weight_scale[slot], expected_fc1[expert])
        assert torch.equal(module.w2_weight_scale[slot], expected_fc2[expert])
    for expert in range(4):
        assert torch.equal(registered["w3_w1_weight_scale"][expert], expected_fc1[expert])
        assert torch.equal(registered["w2_weight_scale"][expert], expected_fc2[expert])

    # Replace a resident expert with one that was only present in host storage.
    for name, values in registered.items():
        getattr(module, name).data[0].copy_(values[3])
    assert torch.equal(module.w3_w1_weight_scale[0], expected_fc1[3])
    assert torch.equal(module.w2_weight_scale[0], expected_fc2[3])
    assert module.fc31_alpha[0].item() == 4.0
    assert module.fc2_alpha[0].item() == 5.0
