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
"""Shared CPU stand-in for the CUDA VMM driver used by the MegaMoE uGPU tests."""

import torch


class MockVmm:
    """Track driver resources while the placed tensor uses real CPU storage."""

    def __init__(self, *, granularity=4096, fail_operation=None, fail_index=1):
        self.gran = granularity
        self.fail_operation = fail_operation
        self.fail_index = fail_index
        self.counts = {}
        self.calls = []
        self.addresses = {}
        self.handles = {}
        self.mappings = {}
        self.next_address = 0x400000
        self.next_handle = 1

    def _call(self, operation):
        self.calls.append(operation)
        self.counts[operation] = self.counts.get(operation, 0) + 1
        if operation == self.fail_operation and self.counts[operation] == self.fail_index:
            raise RuntimeError(f"injected {operation} failure")

    def granularity(self):
        self._call("granularity")
        return self.gran

    def reserve(self, size, granularity):
        self._call("reserve")
        assert size % granularity == 0
        address = self.next_address
        self.next_address += size
        self.addresses[address] = size
        return address

    def create(self, size, die):
        self._call("create")
        handle = self.next_handle
        self.next_handle += 1
        self.handles[handle] = (size, die)
        return handle

    def map(self, address, size, handle):
        self._call("map")
        assert self.handles[handle][0] == size
        self.mappings[address] = (size, handle)

    def allow_access(self, address, size):
        self._call("access")
        assert self.addresses[address] == size

    def alias(self, address, size):
        self._call("alias")
        assert self.addresses[address] == size
        return torch.empty(size, dtype=torch.uint8)

    def locality(self, address):
        self._call("locality")
        for base, (size, handle) in self.mappings.items():
            if base <= address < base + size:
                return self.handles[handle][1]
        raise RuntimeError("unmapped locality probe")

    def synchronize(self):
        self._call("synchronize")

    def unmap(self, address, size):
        self._call("unmap")
        assert self.mappings[address][0] == size
        del self.mappings[address]

    def release(self, handle):
        self._call("release")
        assert all(mapping[1] != handle for mapping in self.mappings.values())
        del self.handles[handle]

    def free_address(self, address, size):
        self._call("free_address")
        assert self.addresses[address] == size
        assert all(not address <= base < address + size for base in self.mappings)
        del self.addresses[address]

    def assert_empty(self):
        assert not self.addresses
        assert not self.handles
        assert not self.mappings


def packed_weights(experts=4):
    generator = torch.Generator().manual_seed(421)
    return tuple(
        torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator)
        for shape in (
            (experts, 1024, 512),
            (experts, 65536),
            (experts, 1024, 256),
            (experts, 32768),
        )
    )
