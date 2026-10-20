# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""per_token_valid_blocks: the NumPy pass returns the torch path's values bit for bit (TRTLLM_MSA_PTVB_NUMPY)."""

import random

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels import msa_utils


def _both(monkeypatch, *args, **kwargs):
    # The NumPy pass is SM107-only: these CPU checks pose as SM107 to compare the two paths.
    monkeypatch.setattr(msa_utils, "get_sm_version", lambda: 107)
    monkeypatch.setattr(msa_utils, "_PER_TOKEN_VALID_BLOCKS_NUMPY", False)
    ref = msa_utils.per_token_valid_blocks(*args, **kwargs)
    monkeypatch.setattr(msa_utils, "_PER_TOKEN_VALID_BLOCKS_NUMPY", True)
    got = msa_utils.per_token_valid_blocks(*args, **kwargs)
    return ref, got


@pytest.mark.cpu_only
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("block_size", [64, 128])
def test_numpy_pass_matches_torch_on_random_batches(monkeypatch, dtype, causal, block_size):
    rng = random.Random(f"{dtype}-{causal}-{block_size}")
    for _ in range(300):
        batch = rng.randint(1, 64)
        qo = [rng.choice([0, 1, 4, rng.randint(1, 4096)]) for _ in range(batch)]
        kv = [q + rng.randint(0, 300_000) for q in qo]
        offset = (
            None
            if rng.random() < 0.3
            else torch.tensor([k - q for q, k in zip(qo, kv)], dtype=dtype)
        )
        ref, got = _both(
            monkeypatch,
            torch.tensor(qo, dtype=dtype),
            torch.tensor(kv, dtype=dtype),
            offset,
            causal=causal,
            block_size=block_size,
        )
        assert got.dtype == torch.long
        assert torch.equal(ref, got), (qo, kv, offset)


@pytest.mark.cpu_only
def test_numpy_pass_takes_strided_inputs_and_empty_batches(monkeypatch):
    qo = torch.arange(1, 41, dtype=torch.int32)[::2]
    ref, got = _both(monkeypatch, qo, qo * 1000, None, causal=True, block_size=128)
    assert torch.equal(ref, got)
    zeros = torch.zeros(3, dtype=torch.int32)
    ref, got = _both(monkeypatch, zeros, zeros + 5, None, causal=True, block_size=128)
    assert ref.numel() == got.numel() == 0 and got.dtype == torch.long


@pytest.mark.cpu_only
def test_kill_env_name_unchanged():
    import inspect

    assert '"TRTLLM_MSA_PTVB_NUMPY", "1"' in inspect.getsource(msa_utils)
