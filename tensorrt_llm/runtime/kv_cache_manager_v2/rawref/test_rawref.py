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

"""Test script for the rawref module."""

import os
import sys

# Add parent directory to path to import the rawref package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from rawref import NULL, ReferenceType, ref
except ImportError as e:
    print(f"Error importing rawref: {e}")
    print("Make sure to build the extension first with: python setup.py build_ext --inplace")
    sys.exit(1)


class TestObject:
    """Test class with __del__ that invalidates references."""

    def __init__(self, value):
        self.value = value
        self.refs = []  # Store references to invalidate

    def __del__(self):
        print(f"TestObject({self.value}).__del__ called, invalidating {len(self.refs)} references")
        for ref in self.refs:
            ref.invalidate()


def test_basic_reference():
    """Test basic reference creation and dereferencing."""
    print("\n=== Test 1: Basic Reference ===")

    obj = TestObject(42)
    r = ref(obj)
    obj.refs.append(r)

    print(f"Created object with value: {obj.value}")
    print(f"Reference is_valid: {r.is_valid}")

    # Dereference
    dereferenced = r()
    print(f"Dereferenced object: {dereferenced}")
    if dereferenced:
        print(f"Dereferenced value: {dereferenced.value}")

    assert r.is_valid, "Reference should be valid"
    assert dereferenced is obj, "Dereferenced object should be the same as original"


def test_invalidation():
    """Test manual invalidation."""
    print("\n=== Test 2: Manual Invalidation ===")

    obj = TestObject(123)
    r = ref(obj)

    print(f"Before invalidation - is_valid: {r.is_valid}")
    print(f"Dereferenced: {r()}")

    r.invalidate()

    print(f"After invalidation - is_valid: {r.is_valid}")
    print(f"Dereferenced: {r()}")

    assert not r.is_valid, "Reference should be invalid after invalidate()"
    assert r() is None, "Dereferencing invalid reference should return None"


def test_del_invalidation():
    """Test invalidation via __del__."""
    print("\n=== Test 3: Invalidation via __del__ ===")

    r = None

    # Create object in a scope
    def create_and_ref():
        obj = TestObject(999)
        nonlocal r
        r = ref(obj)
        obj.refs.append(r)

        print(f"Inside scope - is_valid: {r.is_valid}")
        print(f"Inside scope - dereferenced: {r()}")

    create_and_ref()

    # Object should be deleted and reference invalidated
    print(f"After scope - is_valid: {r.is_valid}")
    print(f"After scope - dereferenced: {r()}")

    assert not r.is_valid, "Reference should be invalid after object deletion"
    assert r() is None, "Dereferencing should return None after object deletion"


def test_multiple_references():
    """Test that singleton pattern returns same reference."""
    print("\n=== Test 4: Singleton Pattern - Same Reference ===")

    obj = TestObject(555)
    r1 = ref(obj)
    r2 = ref(obj)
    obj.refs.extend([r1, r2])

    print(f"r1 is r2: {r1 is r2}")

    assert r1 is r2, "With singleton pattern, should return the same reference"

    # Invalidate r1 (which is the same as r2)
    r1.invalidate()

    print("After invalidating r1:")
    print(f"  r1 is_valid: {r1.is_valid}, dereferenced: {r1()}")
    print(f"  r2 is_valid: {r2.is_valid}, dereferenced: {r2()}")

    # Since they're the same object, both are invalid
    assert not r1.is_valid, "r1 should be invalid"
    assert not r2.is_valid, "r2 should also be invalid (same object)"


def test_alias_equivalence():
    """Test that ref and ReferenceType are the same."""
    print("\n=== Test 5: ref and ReferenceType are equivalent ===")

    obj = TestObject(777)
    r1 = ref(obj)
    r2 = ReferenceType(obj)

    print(f"ref is ReferenceType: {ref is ReferenceType}")
    print(f"type(r1): {type(r1)}")
    print(f"type(r2): {type(r2)}")
    print(f"r1 is r2: {r1 is r2}")

    assert ref is ReferenceType, "ref should be an alias for ReferenceType"
    assert r1 is r2, "Should return the same reference object (singleton pattern)"


def test_singleton_pattern():
    """Test that ref(obj) returns the same reference if __rawref__ is valid."""
    print("\n=== Test 6: Singleton Pattern ===")

    obj = TestObject(888)

    # First call creates a new reference
    r1 = ref(obj)
    print(f"First ref: {r1}, valid: {r1.is_valid}")

    # Second call returns the same reference
    r2 = ref(obj)
    print(f"Second ref: {r2}, valid: {r2.is_valid}")
    print(f"r1 is r2: {r1 is r2}")

    assert r1 is r2, "Should return the same reference object"

    # After invalidation, a new call creates a new reference
    r1.invalidate()
    print(f"After invalidation: r1.valid={r1.is_valid}, r2.valid={r2.is_valid}")

    r3 = ref(obj)
    print(f"Third ref (after invalidation): {r3}, valid: {r3.is_valid}")
    print(f"r1 is r3: {r1 is r3}")

    assert r1 is not r3, "Should create a new reference after invalidation"
    assert r3.is_valid, "New reference should be valid"


def test_null_constant():
    """Test the NULL constant."""
    print("\n=== Test 7: NULL Constant ===")

    print(f"NULL: {NULL}")
    print(f"NULL.is_valid: {NULL.is_valid}")
    print(f"NULL(): {NULL()}")
    print(f"type(NULL): {type(NULL)}")

    assert not NULL.is_valid, "NULL should be invalid"
    assert NULL() is None, "NULL() should return None"

    # Test using NULL to initialize __rawref__
    class MyClass:
        __rawref__ = NULL

    obj = MyClass()
    r = ref(obj)
    print("After creating ref for obj with __rawref__=NULL:")
    print(f"  obj.__rawref__ is r: {obj.__rawref__ is r}")
    print(f"  r.is_valid: {r.is_valid}")

    assert obj.__rawref__ is r, "Should update __rawref__ to new reference"
    assert r.is_valid, "New reference should be valid"


def test_hidden_object_id():
    """Test that object_id is hidden."""
    print("\n=== Test 8: Hidden object_id ===")

    obj = TestObject(123)
    r = ref(obj)

    # Try to access object_id - should raise AttributeError
    try:
        _ = r.object_id
        assert False, "Should not be able to access object_id"
    except AttributeError:
        print("✓ object_id is hidden (AttributeError raised)")

    print()


def test_is_valid_readonly():
    """Test that is_valid is read-only."""
    print("\n=== Test 9: is_valid is read-only ===")

    obj = TestObject(456)
    r = ref(obj)

    print(f"r.is_valid: {r.is_valid}")

    # Try to set is_valid - should raise AttributeError
    try:
        r.is_valid = False
        assert False, "Should not be able to set is_valid"
    except AttributeError:
        print("✓ is_valid is read-only (AttributeError raised)")

    print()


if __name__ == "__main__":
    print("Testing rawref module...")

    try:
        test_basic_reference()
        test_invalidation()
        test_del_invalidation()
        test_multiple_references()
        test_alias_equivalence()
        test_singleton_pattern()
        test_null_constant()
        test_hidden_object_id()
        test_is_valid_readonly()

        print("\n" + "=" * 50)
        print("All tests passed!")
        print("=" * 50)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
