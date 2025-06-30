import gc
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from gc_utils import assert_resource_freed

# A global list to simulate a leak
LEAKY_HOLD = []


class TestObject:

    def __init__(self, value):
        self.value = value


class LeakyObject(TestObject):

    def __init__(self, value):
        super().__init__(value)
        LEAKY_HOLD.append(self)


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
