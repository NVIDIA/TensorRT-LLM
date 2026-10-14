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


def test_fingerprint_distinguishes_effective_value_of_duplicate_keys(build_wheel):
    # cmake applies repeated -DKEY in order (last wins), so these lists
    # configure FOO=2 vs FOO=1. Sorting the raw args used to collapse them to
    # the same fingerprint, letting the stale-config guard miss a real change.
    assert build_wheel.configure_args_fingerprint(
        ["-DFOO=1", "-DFOO=2"]
    ) != build_wheel.configure_args_fingerprint(["-DFOO=2", "-DFOO=1"])


def test_fingerprint_keeps_only_last_definition_of_a_key(build_wheel):
    # An earlier definition overridden by a later one does not change the
    # effective configuration, so it must not change the fingerprint.
    assert build_wheel.configure_args_fingerprint(
        ["-DFOO=1", "-DFOO=2"]
    ) == build_wheel.configure_args_fingerprint(["-DFOO=2"])


def test_fingerprint_changes_when_an_argument_changes(build_wheel):
    base = ["-DCMAKE_CUDA_ARCHITECTURES=100-real"]
    with_nvrtc = base + ["-DNVRTC_DYNAMIC_LINKING=ON"]
    assert build_wheel.configure_args_fingerprint(base) != build_wheel.configure_args_fingerprint(
        with_nvrtc
    )


def test_fingerprint_is_a_sha256_hex_digest(build_wheel):
    # A golden literal, not a recomputation with hashlib.sha256 in the test
    # (which would just re-implement the function): this pins the serialization
    # so a future change to it fails loudly instead of silently invalidating
    # every developer's stored fingerprint.
    assert (
        build_wheel.configure_args_fingerprint(["-DFAST_BUILD=ON"])
        == "41e0e41eb5c37990d53366cdf090648a9ec747cf22fd828986a39db7a0baea41"
    )


def test_fingerprint_serialization_is_unambiguous(build_wheel):
    # A newline inside a value must not collide with the separator between
    # values. Under a plain "\n".join these two sort to the same string.
    embedded_newline = ["-DNCCL_ROOT=x\n-DNIXL_ROOT=y"]
    two_values = ["-DNCCL_ROOT=x", "-DNIXL_ROOT=y"]
    assert build_wheel.configure_args_fingerprint(
        embedded_newline
    ) != build_wheel.configure_args_fingerprint(two_values)


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


def test_fingerprint_tracks_the_source_directory(build_wheel):
    # An explicit build_dir can be reused across checkouts with every other
    # argument equal; only -S changes. If the source dir were left out, the
    # reused tree would build the previous checkout's sources.
    base = ["-DFAST_BUILD=ON"]
    assert build_wheel.configure_args_fingerprint(
        base + ['-S "/work/a/cpp"']
    ) != build_wheel.configure_args_fingerprint(base + ['-S "/work/b/cpp"'])


def test_user_override_of_a_builtin_wins_in_command_order(build_wheel):
    # The configure command passes built-in definitions before cmake_def_args,
    # so a user override (e.g. --extra-cmake-vars BUILD_PYT=OFF) is applied
    # after the built-in default and wins. The fingerprint must be built in the
    # same order so its last-wins reflects the user value, not the built-in;
    # otherwise a user override could leave the fingerprint unchanged.
    builtin_then_user = build_wheel.configure_args_fingerprint(
        ['-DBUILD_PYT="ON"', "-DBUILD_PYT=OFF"]
    )
    assert builtin_then_user == build_wheel.configure_args_fingerprint(
        ["-DBUILD_PYT=OFF"]
    )  # user value wins
    assert builtin_then_user != build_wheel.configure_args_fingerprint(
        ['-DBUILD_PYT="ON"']
    )  # built-in is shadowed, not kept


# The reconfigure decision (configure_reason) is the flow the guard drives:
# whether a fingerprint change forces a cmake configure. main() itself needs a
# full toolchain, so the behavior is pinned here on the decision instead.
def _reason(
    build_wheel,
    tmp_path,
    fingerprint="cur",
    *,
    first_build=False,
    configure_cmake=False,
    configure_only=False,
    clean=False,
):
    return build_wheel.configure_reason(
        tmp_path,
        fingerprint,
        first_build=first_build,
        configure_cmake=configure_cmake,
        configure_only=configure_only,
        clean=clean,
    )


def _record(build_wheel, tmp_path, fingerprint):
    (tmp_path / build_wheel.CONFIGURE_FINGERPRINT_FILENAME).write_text(fingerprint + "\n")


def test_matching_fingerprint_skips_configure(build_wheel, tmp_path):
    _record(build_wheel, tmp_path, "cur")
    assert _reason(build_wheel, tmp_path, "cur") is None


def test_changed_fingerprint_forces_configure(build_wheel, tmp_path):
    _record(build_wheel, tmp_path, "old")
    assert _reason(build_wheel, tmp_path, "cur") is not None


def test_missing_marker_on_existing_dir_forces_one_configure(build_wheel, tmp_path):
    # Pre-existing configured build dir from before fingerprinting: no marker
    # yet, so reconfigure once to record one instead of silently skipping.
    assert build_wheel.stored_configure_fingerprint(tmp_path) is None
    assert _reason(build_wheel, tmp_path, "cur") is not None


@pytest.mark.parametrize("mode", ["first_build", "configure_cmake", "configure_only", "clean"])
def test_explicit_configure_modes_are_left_to_the_caller(build_wheel, tmp_path, mode):
    # These already configure on their own; the fingerprint guard stays out of
    # the way (and must not fire on a mismatch it did not need to handle).
    _record(build_wheel, tmp_path, "old")
    assert _reason(build_wheel, tmp_path, "cur", **{mode: True}) is None
