# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from safetensors.torch import save_file

from tensorrt_llm._torch.models.checkpoints.base_weight_loader import MmappedSafetensorsWeights
from tensorrt_llm._torch.models.modeling_utils import materialize_meta_parameters


def test_mmapped_safetensors_weights_lazy_load_and_mark_consumed(tmp_path):
    weight_path = tmp_path / "model.safetensors"
    save_file({"layer.weight": torch.ones(2, 3)}, weight_path)

    weights = MmappedSafetensorsWeights([str(weight_path)])
    assert "layer.weight" in weights
    assert len(weights) == 1

    tensor = weights["layer.weight"]
    assert hasattr(tensor, "get_shape")

    deleted = weights.mark_consumed("layer")
    assert deleted == 1
    assert len(weights) == 0


def test_materialize_meta_parameters():
    module = torch.nn.Linear(4, 2, bias=False)
    module.weight = torch.nn.Parameter(
        torch.empty(2, 4, device="meta"),
        requires_grad=False,
    )

    materialize_meta_parameters(module)

    assert module.weight.is_cuda
    assert not module.weight.is_meta
    assert module.weight.shape == (2, 4)
