# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm.inputs.multimodal import MultimodalParams

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(3, 1, 11), (1,), (0, 3), ()])
def test_bytes_roundtrip_has_no_ipc_dependency(dtype, shape):
    original = torch.ones(shape, dtype=dtype)
    with patch(
        "tensorrt_llm._torch.shared_tensor.shared_tensor.reduce_tensor",
        side_effect=AssertionError("IPC must not be used"),
    ):
        payload = SharedTensorContainer.from_tensor_bytes(original).dump_to_dict()
    assert set(payload) == {"method_key", "tensor_size", "dtype", "data"}
    payload = json.loads(json.dumps(payload))
    original.zero_()
    restored = SharedTensorContainer.from_dict(payload).get_local_view()
    assert torch.equal(restored, torch.ones(shape, dtype=dtype))
    assert restored.dtype == dtype
    params = MultimodalParams(multimodal_data={"mrope_config": {"mrope_position_deltas": payload}})
    params.to_tensor("multimodal_data")
    assert torch.equal(params.multimodal_data["mrope_config"]["mrope_position_deltas"], restored)


def test_bytes_rejects_empty_nonempty_tensor():
    payload = SharedTensorContainer.from_tensor_bytes(torch.ones(3)).dump_to_dict()
    payload["data"] = ""
    with pytest.raises(RuntimeError):
        SharedTensorContainer.from_dict(payload).get_local_view()
