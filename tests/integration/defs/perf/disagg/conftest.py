"""
Pytest configuration for disagg tests
Only collects tests in this directory when --disagg parameter is provided
Can share options like --test-list defined in the main conftest.py
"""

import pytest


def pytest_addoption(parser):
    """Add disagg-specific command line options"""
    parser.addoption(
        "--disagg",
        action="store_true",
        default=False,
        help="Enable disaggregated tests collection. "
             "Example: pytest --disagg"
    )


def pytest_collect_directory(path, parent):
    """
    Only collect tests in this directory when --disagg parameter is provided
    This hook executes earliest in the collection phase to avoid loading unnecessary test files
    
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
    # Can subsequently use --test-list and other options from main conftest.py for filtering
    return None
