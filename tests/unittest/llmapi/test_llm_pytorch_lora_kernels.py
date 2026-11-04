# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import torch

from .lora_test_utils import CUDAGraphLoRATestParams, compare_cuda_graph_lora_params_filler


def test_lora_cuda_graph_params_filling_kernel_special_cases():
    torch.cuda.set_device(0)

    # test all requests have the same LoRA id case
    test_params = CUDAGraphLoRATestParams(
        batch_slot_ids=[0] * 10,
        input_hidden_size=4096,
        slot_ranks=[64] * 10,
        max_lora_rank=64,
        output_hidden_sizes=[123],
        layer_module_mask=None,
        dtype=torch.bfloat16,
        seed=42,
    )
    compare_cuda_graph_lora_params_filler(test_params)

    # test no LoRA in a batch case
    test_params2 = replace(test_params, batch_slot_ids=[len(test_params.slot_ranks)] * 10)
    compare_cuda_graph_lora_params_filler(test_params2)

    # test all having three modules case
    test_params3 = replace(test_params, output_hidden_sizes=[123, 456, 789])
    compare_cuda_graph_lora_params_filler(test_params3)

    # test some layer module have invalid weight pointers case
    mask = torch.full((test_params3.module_count, test_params3.slot_count), True, dtype=torch.bool)
    mask[0, 0] = False
    mask[1, 7] = False
    mask[2, 3] = False
    test_params4 = replace(test_params3, layer_module_mask=mask)
    compare_cuda_graph_lora_params_filler(test_params4)

    # test mixed slot ids case
    test_params5 = CUDAGraphLoRATestParams(
        batch_slot_ids=[6, 2, 0, 1, 1, 1, 5, 6],
        input_hidden_size=512,
        slot_ranks=[8, 12, 4] * 2,
        max_lora_rank=15,
        output_hidden_sizes=[123, 456, 789],
        layer_module_mask=None,
        dtype=torch.bfloat16,
        seed=42,
    )
    compare_cuda_graph_lora_params_filler(test_params5)

    # test mixed slot with invalid weight pointers
    mask = torch.full((test_params5.module_count, test_params5.slot_count), True, dtype=torch.bool)
    mask[1, 3] = False
    mask[2, 5] = False
    mask[-1, -4] = False
    test_params6 = replace(test_params5, layer_module_mask=mask)
    compare_cuda_graph_lora_params_filler(test_params6)
