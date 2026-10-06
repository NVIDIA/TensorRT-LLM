from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_addoption(parser):
    parser.addoption(
        "--run-live-backends",
        action="store_true",
        help="Run integration checks that launch installed agent CLIs and use local account config.",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "live_backend: requires a configured local agent CLI")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-live-backends"):
        skip = pytest.mark.skip(
            reason="requires --run-live-backends and configured local agent CLIs"
        )
        for item in items:
            if "live_backend" in item.keywords:
                item.add_marker(skip)
