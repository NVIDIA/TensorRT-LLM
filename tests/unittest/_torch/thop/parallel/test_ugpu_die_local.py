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
"""Unit tests for the die-local weight placement helpers in tensorrt_llm._torch.ugpu.die_local."""

import pytest
import torch

from tensorrt_llm._torch.ugpu import die_local

_G = 2 << 20


@pytest.mark.parametrize(
    "nbytes, split_at, homes",
    [
        (4 * _G, 2 * _G, [0, 0, 1, 1]),  # split on a chunk boundary
        (4 * _G, 2 * _G + _G // 4, [0, 0, 1, 1]),  # straddling chunk: mostly die 1
        (4 * _G, 2 * _G + 3 * _G // 4, [0, 0, 0, 1]),  # straddling chunk: mostly die 0
        (3 * _G + _G // 4, 3 * _G + _G // 4, [0, 0, 0, 0]),  # all die 0, short last chunk
        (3 * _G + _G // 4, 0, [1, 1, 1, 1]),  # all die 1
    ],
)
def test_chunk_homes_follow_the_majority_of_used_bytes(nbytes, split_at, homes):
    assert die_local._chunk_homes(nbytes, split_at, _G) == homes


def _require_two_dies() -> int:
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    device = torch.cuda.current_device()
    torch.empty(1, device="cuda")  # initialize CUDA before the driver query
    if die_local.locality_domain_count(device) != 2:
        pytest.skip("needs a GPU with two memory locality domains")
    return device


def test_stack_halves_moves_each_n_half_into_its_own_expert_block():
    _require_two_dies()
    weight = torch.randint(0, 256, (6, 512, 96), dtype=torch.uint8, device="cuda")
    out = die_local.stack_halves(weight)
    assert out.shape == (12, 256, 96)
    assert out.dtype == weight.dtype
    assert torch.equal(out[:6], weight[:, :256])
    assert torch.equal(out[6:], weight[:, 256:])


def _locality_domain_of(address: int) -> int:
    cu = die_local._driver()
    handle = die_local._check(
        cu.cuMemRetainAllocationHandle(address), "cuMemRetainAllocationHandle"
    )
    try:
        prop = die_local._check(
            cu.cuMemGetAllocationPropertiesFromHandle(handle),
            "cuMemGetAllocationPropertiesFromHandle",
        )
    finally:
        cu.cuMemRelease(handle)
    assert prop.location.type == cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN
    return int(prop.location.localized.localityDomainId)


def test_stack_halves_homes_each_half_on_its_die():
    _require_two_dies()
    weight = torch.randint(0, 256, (2, 4096, 1024), dtype=torch.uint8, device="cuda")  # 8 MiB
    out = die_local.stack_halves(weight)
    half_bytes = out.numel() * out.element_size() // 2  # 4 MiB: two 2 MiB chunks per die
    for offset, die in ((0, 0), (_G, 0), (half_bytes, 1), (half_bytes + _G, 1)):
        assert _locality_domain_of(out.data_ptr() + offset) == die
    assert torch.equal(out[:2], weight[:, :2048]) and torch.equal(out[2:], weight[:, 2048:])


def test_localized_range_is_released_with_its_owner_but_not_at_exit():
    device = _require_two_dies()
    owner = die_local._LocalizedRange([0, 1], die_local._granularity(device), device)
    base, finalizer = owner.base, owner._finalizer
    assert finalizer.alive and not finalizer.atexit  # no device sync during interpreter exit
    del owner
    assert not finalizer.alive
    cu = die_local._driver()
    status = cu.cuMemRetainAllocationHandle(base)[0]  # nothing is mapped there any more
    assert status != cu.CUresult.CUDA_SUCCESS


def test_smid_die_table_is_none_without_two_locality_domains(monkeypatch):
    monkeypatch.setattr(die_local, "locality_domain_count", lambda device: 1)
    assert die_local.smid_die_table(0) is None


def test_stack_halves_rejects_odd_n():
    with pytest.raises(ValueError, match="even N"):
        die_local.stack_halves(torch.empty((2, 3, 4), dtype=torch.uint8))


def test_smid_die_table_covers_every_sm_with_both_dies():
    device = _require_two_dies()
    table = die_local.smid_die_table(device)
    assert table is not None
    assert table.dtype == torch.int32 and table.device.type == "cuda"
    values = table.cpu()
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    assert int((values >= 0).sum()) == sms
    assert set(values[values >= 0].tolist()) == {0, 1}


def test_stream_scratch_gives_each_stream_its_own_zeroed_set():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    scratch = die_local.StreamScratch((4, 4), torch.device("cuda"), max_streams=2)
    main = scratch.get()
    assert [t.shape[0] for t in main] == [4, 4]
    assert all(t.dtype == torch.int32 and t.count_nonzero().item() == 0 for t in main)
    assert scratch.get() is main  # one stream keeps its set
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        other = scratch.get()
        assert scratch.get() is other
    assert other[0].data_ptr() != main[0].data_ptr()
    # A high-priority stream comes from another pool, so it differs from `side`.
    with torch.cuda.stream(torch.cuda.Stream(priority=-1)):
        extra = scratch.get()  # past max_streams: fresh zeroed tensors on every call
        assert all(t.count_nonzero().item() == 0 for t in extra)
        assert extra[0].data_ptr() not in {main[0].data_ptr(), other[0].data_ptr()}
        assert scratch.get() is not extra
    assert len(scratch.all_sets()) == 2


def test_stream_scratch_rejects_empty_layout():
    with pytest.raises(ValueError, match="bad scratch layout"):
        die_local.StreamScratch((), torch.device("cpu"))
