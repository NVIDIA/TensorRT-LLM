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
"""``locality_domain.placement``: domain-split copies and the SM -> domain table (two-domain
devices only)."""

import pytest
import torch

from tensorrt_llm._torch.locality_domain_utils import (
    get_locality_domain_compute_sm_counts,
    is_locality_domain_enabled,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_locality_domain_enabled(),
    reason="needs a device with two locality domains (SM107)",
)


@pytest.mark.parametrize("nbytes", [1 << 20, 5 << 20, 37 * (1 << 20) + 123])
def test_split_domain_copy_keeps_bytes_and_owns_memory(nbytes):
    from tensorrt_llm._torch.locality_domain.placement import split_domain_copy

    src = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device="cuda")
    copy = split_domain_copy(src.view(-1, 1), nbytes // 2)
    assert copy is not None
    assert copy.shape == (nbytes, 1) and copy.dtype == src.dtype
    assert copy.data_ptr() != src.data_ptr()
    assert torch.equal(copy.flatten(), src)
    view = copy[nbytes // 3 :]
    del copy
    torch.cuda.synchronize()
    # a view keeps the mapping alive
    assert torch.equal(view.flatten(), src[nbytes // 3 :])


def test_sm_locality_domains_matches_partition_sizes():
    from tensorrt_llm._torch.locality_domain.placement import sm_locality_domains

    device = torch.cuda.current_device()
    domains = sm_locality_domains(device)
    assert len(domains) == torch.cuda.get_device_properties(device).multi_processor_count
    counts = get_locality_domain_compute_sm_counts(1)
    assert counts is not None
    # SMs outside both partitions read as domain 0
    assert sum(domains) == counts[0]
