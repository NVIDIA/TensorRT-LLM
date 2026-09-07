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

import gc
import unittest

import pytest
from gc_utils import assert_resource_freed

pytestmark = pytest.mark.cpu_only

# A global list to simulate a leak
LEAKY_HOLD = []


class TestObject:

    def __init__(self, value):
        self.value = value


class LeakyObject(TestObject):

    def __init__(self, value):
        super().__init__(value)
        LEAKY_HOLD.append(self)


class ShutdownObject(TestObject):

    shutdown_calls: int = 0

    def shutdown(self) -> None:
        type(self).shutdown_calls += 1


class TestAssertResourceFreed(unittest.TestCase):

    def setUp(self):
        # Clear any previous leaks and force a clean GC
        LEAKY_HOLD.clear()
        gc.collect()

    def test_simple_object_freed(self):
        """A plain TestObject should be freed without error."""

        def factory():
            return TestObject("foo")

        # generator‐based
        with assert_resource_freed(factory) as obj:
            self.assertEqual(obj.value, "foo")

        # class‐based
        with assert_resource_freed(factory) as obj2:
            self.assertEqual(obj2.value, "foo")

    def test_resource_is_shutdown(self) -> None:
        """A resource with a shutdown method should be shut down before release."""
        ShutdownObject.shutdown_calls = 0

        def factory() -> ShutdownObject:
            return ShutdownObject("foo")

        with assert_resource_freed(factory):
            pass

        self.assertEqual(ShutdownObject.shutdown_calls, 1)

    def test_leaky_object_raises(self):
        """LeakyObject holds itself in a global list → should raise."""

        def factory():
            return LeakyObject("bar")

        with self.assertRaises(AssertionError) as cm:
            with assert_resource_freed(factory):
                pass

    def test_diagnostic_message(self):
        """Check that the AssertionError message includes a count and type."""

        def factory():
            return LeakyObject("baz")

        with self.assertRaises(AssertionError) as cm:
            with assert_resource_freed(factory):
                pass

        msg = str(cm.exception)
        # e.g. "1 referrer(s) still alive"
        self.assertRegex(msg, r"\d+\s+referrer")
        # and something like "- list at 0x"
        self.assertIn("list", msg)

    def test_no_false_positive_from_generator_cell(self):
        """Ensure that our filter skips the internal cell, so no leak is reported."""

        def factory():
            return TestObject("qux")

        # If the internal cell weren’t filtered, this would raise—
        # so no exception means our filter worked.
        with assert_resource_freed(factory):
            pass


if __name__ == "__main__":
    unittest.main()
