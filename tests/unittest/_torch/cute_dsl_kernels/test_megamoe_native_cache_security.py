# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import stat
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[4]
_MODULE_PATH = _ROOT / "tensorrt_llm/_torch/cute_dsl_kernels/megamoe_scheduler_v2/native.py"
_SPEC = importlib.util.spec_from_file_location("megamoe_scheduler_native", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_native_cache_permissions_and_unsafe_entries(tmp_path):
    cache = tmp_path / "native-cache"
    cache.mkdir(mode=0o755)
    _MODULE._secure_cache_directory(cache)
    assert stat.S_IMODE(cache.stat().st_mode) == 0o700

    lock_path = cache / "build.lock"
    with _MODULE._open_cache_lock(lock_path):
        pass
    assert stat.S_ISREG(lock_path.lstat().st_mode)
    assert stat.S_IMODE(lock_path.lstat().st_mode) == 0o600

    public_output = cache / "extension.so"
    public_output.touch(mode=0o644)
    assert not _MODULE._is_private_regular_file(public_output)

    symlink_output = cache / "linked.so"
    symlink_output.symlink_to(public_output)
    assert not _MODULE._is_private_regular_file(symlink_output)
    _MODULE._remove_unsafe_cache_file(symlink_output)
    assert not symlink_output.exists()
    assert public_output.exists()
    _MODULE._remove_unsafe_cache_file(public_output)
    assert not public_output.exists()


def test_native_cache_rejects_symlink_directory(tmp_path):
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    cache = tmp_path / "native-cache"
    cache.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="not owned by this user"):
        _MODULE._secure_cache_directory(cache)


def test_default_cache_rejects_symlink_parent(tmp_path, monkeypatch):
    target = tmp_path / "attacker-controlled"
    target.mkdir(mode=0o700)
    default_root = tmp_path / "tensorrt_llm-user"
    default_root.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(_MODULE, "_DEFAULT_BUILD_ROOT", default_root)
    monkeypatch.delenv("UNSET_MEGAMOE_TEST_BUILD_DIR", raising=False)

    with pytest.raises(RuntimeError, match="not owned by this user"):
        _MODULE._build_and_load(
            module_name="unused",
            argv=lambda output: ["false"],
            sources=(),
            build_dir_env="UNSET_MEGAMOE_TEST_BUILD_DIR",
            default_build_dir=str(default_root / "module"),
            failure="unused",
            validate=lambda module: None,
            force=False,
        )
