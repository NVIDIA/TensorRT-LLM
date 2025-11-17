"""Pytest configuration for disagg tests.

Only collects tests in this directory when --disagg parameter is provided.
Can share options like --disagg-test-list defined in this conftest.py.
"""

import pytest


def pytest_addoption(parser):
    """Add disagg-specific command line options."""
    parser.addoption(
        "--disagg",
        action="store_true",
        default=False,
        help="Enable disaggregated tests collection. Example: pytest --disagg",
    )
    parser.addoption(
        "--disagg-test-list",
        action="store",
        default=None,
        help="Path to a file containing test IDs (one per line) to run. "
        "Example: pytest --disagg --disagg-test-list=testlist/testlist_gb200.txt",
    )


def pytest_collect_directory(path, parent):
    """Only collect tests in this directory when --disagg parameter is provided.

    This hook executes earliest in the collection phase to avoid loading unnecessary test files.

    Args:
        path: Current directory path
        parent: Parent collector

    Returns:
        True: Skip collection of this directory
        None: Proceed with normal collection
    """
    disagg_enabled = parent.config.getoption("--disagg", default=False)

    if not disagg_enabled:
        # No --disagg parameter, skip collection
        return True

    # With --disagg parameter, proceed with normal collection
    # Can subsequently use --disagg-test-list and other options from main conftest.py for filtering
    return None


def pytest_collection_modifyitems(config, items):
    """Filter tests based on --disagg-test-list option.

    Args:
        config: pytest config object
        items: list of collected test items
    """
    test_list_file = config.getoption("--disagg-test-list")

    if not test_list_file:
        # No filtering needed if --disagg-test-list is not provided
        return

    # Read test IDs from file
    try:
        with open(test_list_file, "r", encoding="utf-8") as f:
            # Read non-empty lines and strip whitespace
            wanted_tests = set(
                line.strip() for line in f
                if line.strip() and not line.strip().startswith("#"))
    except FileNotFoundError:
        pytest.exit(f"❌ Error: Test list file not found: {test_list_file}")
        return
    except Exception as e:
        pytest.exit(f"❌ Error reading test list file {test_list_file}: {e}")
        return

    if not wanted_tests:
        pytest.exit(
            f"❌ Error: Test list file {test_list_file} is empty or contains no valid test IDs"
        )
        return

    # Filter items based on test list
    selected = []
    deselected = []

    for item in items:
        # item.nodeid is the full test identifier like:
        # "test_disagg_simple.py::TestDisaggBenchmark::test_benchmark[deepseek-r1-fp4:1k1k:...]"
        if item.nodeid in wanted_tests:
            selected.append(item)
        else:
            deselected.append(item)

    # Apply the filtering
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected

    # Print summary
    print(f"\n{'=' * 70}")
    print("✅ Test List Filter Active")
    print(f"   File: {test_list_file}")
    print(f"   Requested: {len(wanted_tests)} test(s)")
    print(f"   Selected:  {len(selected)} test(s)")
    print(f"   Deselected: {len(deselected)} test(s)")

    if len(selected) == 0:
        print("\n⚠️  Warning: No tests matched the test list!")
        print(
            f"   Please check that the test IDs in {test_list_file} are correct."
        )

    print(f"{'=' * 70}\n")
