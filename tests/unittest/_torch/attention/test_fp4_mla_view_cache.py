# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP4 MLA cached pool views stay writable outside inference mode."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fp4_mla.cache_manager import _cacheable_view
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _pool() -> torch.Tensor:
    return torch.arange(64, dtype=torch.uint8, device="cuda")


def test_plain_view_built_in_inference_mode_is_an_inference_tensor():
    # Negative control: the failure the cache hit on GEN warmup (seen on Rubin).
    pool = _pool()
    with torch.inference_mode():
        view = convert_to_torch_tensor(TensorWrapper(pool.data_ptr(), torch.uint8, (8, 8)))
    assert view.is_inference()
    with pytest.raises(RuntimeError, match="Inplace update to inference tensor"):
        view.zero_()


def test_cacheable_view_built_in_inference_mode_can_be_scrubbed_outside_it():
    pool = _pool()
    with torch.inference_mode():
        view = _cacheable_view(TensorWrapper(pool.data_ptr(), torch.uint8, (8, 8)))
        assert not view.is_inference()
        assert view.data_ptr() == pool.data_ptr()
        view[0].fill_(7)  # inference-mode writes still work
    assert int(pool[:8].sum()) == 56
    view.zero_()  # the warmup scrub, outside inference mode
    torch.cuda.synchronize()
    assert int(pool.sum()) == 0
