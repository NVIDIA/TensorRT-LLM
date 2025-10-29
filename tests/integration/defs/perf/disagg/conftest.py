"""
Pytest configuration file
Provides custom command line options and test collection modifications
"""

import pytest

def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--test-list",
        action="store",
        default=None,
        help="Path to a file containing test IDs (one per line) to run. "
             "Example: pytest --test-list=testlist_gb200.txt"
    )


def pytest_collection_modifyitems(config, items):
    """
    Filter tests based on --test-list option
    
    Args:
        config: pytest config object
        items: list of collected test items
    """
    test_list_file = config.getoption("--test-list")
    
    if not test_list_file:
        # No filtering needed if --test-list is not provided
        return
    
    # Read test IDs from file
    try:
        with open(test_list_file, 'r', encoding='utf-8') as f:
            # Read non-empty lines and strip whitespace
            wanted_tests = set(line.strip() for line in f if line.strip() and not line.strip().startswith('#'))
    except FileNotFoundError:
        pytest.exit(f"❌ Error: Test list file not found: {test_list_file}")
        return
    except Exception as e:
        pytest.exit(f"❌ Error reading test list file {test_list_file}: {e}")
        return
    
    if not wanted_tests:
        pytest.exit(f"❌ Error: Test list file {test_list_file} is empty or contains no valid test IDs")
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
    print(f"\n{'='*70}")
    print(f"✅ Test List Filter Active")
    print(f"   File: {test_list_file}")
    print(f"   Requested: {len(wanted_tests)} test(s)")
    print(f"   Selected:  {len(selected)} test(s)")
    print(f"   Deselected: {len(deselected)} test(s)")
    
    if len(selected) == 0:
        print(f"\n⚠️  Warning: No tests matched the test list!")
        print(f"   Please check that the test IDs in {test_list_file} are correct.")
    
    print(f"{'='*70}\n")
