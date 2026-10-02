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
"""Unit tests for ``NemotronHForCausalLM``'s Mamba2 tensor-parallelism gate.

The old code whitelisted ``tp_size in {1, 2, 4, 8}``. It now delegates to
``_validate_mamba2_tensor_parallelism``, which calls the shared
``validate_mamba2_tp`` (see ``mamba2_tp.py``) so that TP degrees Mamba2 group
replication makes legal -- e.g. 16 and 32 for the Nemotron-Ultra shapes below
-- are accepted too.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.models import modeling_nemotron_h

pytestmark = pytest.mark.cpu_only

# Nemotron-Ultra shapes: every Nemotron-3 checkpoint uses n_groups=8.
MAMBA_NUM_HEADS = 256
N_GROUPS = 8


def _make_config(tp_size: int, enable_attention_dp: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        mapping=SimpleNamespace(enable_attention_dp=enable_attention_dp, tp_size=tp_size),
        pretrained_config=SimpleNamespace(mamba_num_heads=MAMBA_NUM_HEADS, n_groups=N_GROUPS),
    )


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8, 16, 32])
def test_valid_tp_sizes_pass(tp_size):
    # tp_size <= n_groups (1, 2, 4, 8) splits groups evenly (historical
    # layout); tp_size > n_groups (16, 32) replicates groups across the
    # ranks whose heads own them. Both must be accepted without raising.
    modeling_nemotron_h._validate_mamba2_tensor_parallelism(_make_config(tp_size))


@pytest.mark.parametrize("tp_size", [3, 12])
def test_invalid_tp_sizes_raise(tp_size):
    with pytest.raises(ValueError, match="Valid tp_size values"):
        modeling_nemotron_h._validate_mamba2_tensor_parallelism(_make_config(tp_size))


def test_attention_dp_skips_validation():
    # Attention-DP runs the Mamba2 mixers unsharded, so even a tp_size that
    # would otherwise be illegal (3 divides neither heads nor groups) must
    # pass through untouched.
    modeling_nemotron_h._validate_mamba2_tensor_parallelism(
        _make_config(3, enable_attention_dp=True)
    )
