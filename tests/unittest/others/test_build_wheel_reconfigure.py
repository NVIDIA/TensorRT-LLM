# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``build_wheel.py`` must reconfigure when its cmake arguments change.

The cmake configure step is skipped unless ``--clean``/``--configure_cmake``
is passed or the build dir is fresh, so changing a configuration-affecting
flag (``--cuda_architectures``, ``--nvrtc_dynamic_linking``,
``--extra-cmake-vars``) used to silently build the OLD configuration. The
fix fingerprints the configure arguments into
``.cmake_configure_args.sha256`` inside the build dir and forces a
reconfigure when the fingerprint changes.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_BUILD_WHEEL = Path(__file__).resolve().parents[3] / "scripts" / "build_wheel.py"


@pytest.fixture(scope="module")
def build_wheel():
    """``scripts/build_wheel.py`` loaded by path.

    It is a script rather than a package member. Importing it runs only its
    imports and constants; ``main()`` is behind the usual ``__main__`` guard.
    """
    assert _BUILD_WHEEL.is_file(), _BUILD_WHEEL
    spec = importlib.util.spec_from_file_location("_build_wheel_under_test", _BUILD_WHEEL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def test_fingerprint_ignores_argument_order(build_wheel):
    args = ["-DFAST_BUILD=ON", "-DNVRTC_DYNAMIC_LINKING=ON", "-GNinja"]
    assert build_wheel.configure_args_fingerprint(args) == build_wheel.configure_args_fingerprint(
        list(reversed(args))
    )


def test_fingerprint_changes_when_an_argument_changes(build_wheel):
    base = ["-DCMAKE_CUDA_ARCHITECTURES=100-real"]
    with_nvrtc = base + ["-DNVRTC_DYNAMIC_LINKING=ON"]
    assert build_wheel.configure_args_fingerprint(base) != build_wheel.configure_args_fingerprint(
        with_nvrtc
    )


def test_fingerprint_is_a_sha256_hex_digest(build_wheel):
    fingerprint = build_wheel.configure_args_fingerprint(["-DFAST_BUILD=ON"])
    assert len(fingerprint) == 64
    assert set(fingerprint) <= set("0123456789abcdef")


def test_stored_fingerprint_missing_file_is_none(build_wheel, tmp_path):
    # Pre-existing build dirs without a fingerprint keep the old skip
    # behavior (no forced reconfigure); the helper reports None for them.
    assert build_wheel.stored_configure_fingerprint(tmp_path) is None


def test_stored_fingerprint_empty_file_is_none(build_wheel, tmp_path):
    (tmp_path / build_wheel.CONFIGURE_FINGERPRINT_FILENAME).write_text("\n")
    assert build_wheel.stored_configure_fingerprint(tmp_path) is None


def test_stored_fingerprint_round_trip(build_wheel, tmp_path):
    fingerprint = build_wheel.configure_args_fingerprint(["-DFAST_BUILD=ON"])
    (tmp_path / build_wheel.CONFIGURE_FINGERPRINT_FILENAME).write_text(fingerprint + "\n")
    assert build_wheel.stored_configure_fingerprint(tmp_path) == fingerprint
