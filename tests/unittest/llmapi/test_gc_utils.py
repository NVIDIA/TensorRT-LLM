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
import weakref

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

    lifecycle_events: list[str] = []
    shutdown_calls: int = 0
    shutdown_fails: bool = False

    def __init__(self, value):
        """Create a resource with an observable finalization event."""
        super().__init__(value)
        weakref.finalize(self, type(self).lifecycle_events.append, "released")

    def shutdown(self) -> None:
        """Record shutdown and optionally fail for cleanup-path tests."""
        type(self).lifecycle_events.append("shutdown")
        type(self).shutdown_calls += 1
        if type(self).shutdown_fails:
            raise RuntimeError("shutdown failed")


class TestAssertResourceFreed(unittest.TestCase):

    def setUp(self) -> None:
        """Reset global state used by the resource-lifetime tests."""
        # Clear any previous leaks and force a clean GC
        LEAKY_HOLD.clear()
        ShutdownObject.lifecycle_events.clear()
        ShutdownObject.shutdown_calls = 0
        ShutdownObject.shutdown_fails = False
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
        with assert_resource_freed(ShutdownObject, "foo"):
            pass

        self.assertEqual(ShutdownObject.shutdown_calls, 1)
        self.assertEqual(ShutdownObject.lifecycle_events,
                         ["shutdown", "released"])

    def test_shutdown_exception_restores_gc_state(self) -> None:
        """A shutdown failure should propagate after restoring the GC state."""
        gc_was_enabled = gc.isenabled()
        ShutdownObject.shutdown_fails = True

        try:
            for expected_gc_enabled in (False, True):
                with self.subTest(gc_enabled=expected_gc_enabled):
                    ShutdownObject.lifecycle_events.clear()
                    ShutdownObject.shutdown_calls = 0
                    if expected_gc_enabled:
                        gc.enable()
                    else:
                        gc.disable()

                    with self.assertRaisesRegex(RuntimeError,
                                                "shutdown failed"):
                        with assert_resource_freed(ShutdownObject, "foo"):
                            pass

                    self.assertEqual(gc.isenabled(), expected_gc_enabled)
                    self.assertEqual(ShutdownObject.shutdown_calls, 1)
                    self.assertEqual(ShutdownObject.lifecycle_events,
                                     ["shutdown", "released"])
        finally:
            if gc_was_enabled:
                gc.enable()
            else:
                gc.disable()

    def test_shutdown_exception_does_not_mask_body_exception(self) -> None:
        """A context-body exception should take priority over shutdown errors."""
        expected_gc_enabled = gc.isenabled()
        ShutdownObject.shutdown_fails = True

        with self.assertRaisesRegex(ValueError, "body failed"):
            with assert_resource_freed(ShutdownObject, "foo"):
                raise ValueError("body failed")

        self.assertEqual(ShutdownObject.shutdown_calls, 1)
        self.assertEqual(ShutdownObject.lifecycle_events,
                         ["shutdown", "released"])
        self.assertEqual(gc.isenabled(), expected_gc_enabled)

    def test_shutdown_exception_propagates_inside_exception_handler(
            self) -> None:
        """An outer handled exception should not suppress a shutdown error."""
        ShutdownObject.shutdown_fails = True

        try:
            raise ValueError("outer error")
        except ValueError:
            with self.assertRaisesRegex(RuntimeError, "shutdown failed"):
                with assert_resource_freed(ShutdownObject, "foo"):
                    pass

        self.assertEqual(ShutdownObject.shutdown_calls, 1)
        self.assertEqual(ShutdownObject.lifecycle_events,
                         ["shutdown", "released"])

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
