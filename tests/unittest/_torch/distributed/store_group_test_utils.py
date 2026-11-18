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
"""Helpers of the tests of the membership and the collectives over the rendezvous store."""

import threading

from torch.distributed import DistStoreError


class InMemoryStore:
    """In-memory store with the TCPStore calls of the membership, with signed 64-bit counters."""

    def __init__(self):
        self.values = {}
        self._lock = threading.Condition()

    def set(self, key, value):
        with self._lock:
            self.values[key] = bytes(value)
            self._lock.notify_all()

    def wait(self, keys, timeout):
        with self._lock:
            if not self._lock.wait_for(
                lambda: all(key in self.values for key in keys), timeout.total_seconds()
            ):
                raise DistStoreError(f"wait timeout after {timeout}, keys: {keys}")

    def get(self, key):
        return self.values[key]

    def check(self, keys):
        with self._lock:
            return all(key in self.values for key in keys)

    def multi_get(self, keys):
        with self._lock:
            return [self.values[key] for key in keys]

    def add(self, key, amount):
        with self._lock:
            value = int(self.values.get(key, b"0")) + amount
            assert -(1 << 63) <= value < 1 << 63, f"counter {key} overflows"
            self.values[key] = str(value).encode()
            self._lock.notify_all()
            return value

    def delete_key(self, key):
        with self._lock:
            return self.values.pop(key, None) is not None


class Unpickled:
    """A value whose unpickling runs code, as a forged pickle in the store would."""

    calls = []

    @staticmethod
    def record():
        Unpickled.calls.append("unpickled")

    def __reduce__(self):
        return (Unpickled.record, ())
