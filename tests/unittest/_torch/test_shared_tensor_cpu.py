# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from tensorrt_llm._torch.shared_tensor import SharedTensorContainer

pytestmark = pytest.mark.cpu_only


def test_concurrent_cpu_handle_exports_restore_values_and_strategy():
    strategy = torch.multiprocessing.get_sharing_strategy()

    def roundtrip(value):
        tensor = torch.full((3, 8), value, dtype=torch.int32)
        payload = SharedTensorContainer.from_tensor(tensor).dump_to_dict()
        restored = SharedTensorContainer.from_dict(payload).get_local_view()
        assert torch.equal(restored, tensor)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(roundtrip, range(40)))
    assert torch.multiprocessing.get_sharing_strategy() == strategy
