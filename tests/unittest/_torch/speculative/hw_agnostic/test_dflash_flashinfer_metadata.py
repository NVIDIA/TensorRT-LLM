# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash on attention metadata that keeps no device-side KV lengths (FlashInfer)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.speculative.dflash import DFlashWorker

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("has_kv_lens", [True, False], ids=["trtllm", "flashinfer"])
@pytest.mark.parametrize("cuda_graph", [True, False], ids=["cuda_graph", "eager"])
def test_spec_dec_save_skips_kv_lens_the_metadata_lacks(has_kv_lens: bool, cuda_graph: bool):
    """A CUDA-graph step saves `kv_lens_cuda` only where the metadata has it.

    TRTLLM attention metadata keeps the KV lengths on the device; FlashInfer's
    does not, and asking it to save the field raised.
    """
    saved = []
    metadata = SimpleNamespace(prepare_for_spec_dec=lambda *fields: saved.append(fields))
    if has_kv_lens:
        metadata.kv_lens_cuda = torch.zeros(2, dtype=torch.int32)
    with patch.object(torch.cuda, "is_current_stream_capturing", return_value=False):
        DFlashWorker._prepare_attn_metadata_for_dflash(
            SimpleNamespace(), metadata, SimpleNamespace(is_cuda_graph=cuda_graph)
        )
    expected = ["_seq_lens", "_seq_lens_cuda"]
    if cuda_graph and has_kv_lens:
        expected.append("kv_lens_cuda")
    assert saved == [tuple(expected)]
