# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The user cache directory exists before an engine compiles kernels at run time.

Without it the kernel generator keeps run-time compiled attention kernels in `/tmp/jitify2_cache`,
shared by every process of the host and found by name, and a generation kernel's name does not say
whether it was built for one query token per sequence or for several. A harness that exports
`XDG_CACHE_HOME=/tmp/some-new-path` and never creates the path is how that happens.
"""

import pytest

from tensorrt_llm._torch.pyexecutor import py_executor_creator
from tensorrt_llm._torch.pyexecutor.user_cache_directory import (
    ensure_user_cache_directory,
    user_cache_directory,
)

pytestmark = pytest.mark.cpu_only


def test_a_named_cache_home_that_nobody_created_is_created(tmp_path):
    named = tmp_path / "xdg" / "of-this-run"
    environ = {"XDG_CACHE_HOME": str(named), "HOME": str(tmp_path / "home")}
    assert not named.exists()
    assert ensure_user_cache_directory(environ) == named
    assert named.is_dir()
    # The home's own cache directory is not what the rules name here: left alone.
    assert not (tmp_path / "home").exists()


def test_without_a_cache_home_it_is_the_homes_dot_cache(tmp_path):
    home = tmp_path / "home"
    for environ in ({"HOME": str(home)}, {"HOME": str(home), "XDG_CACHE_HOME": ""}):
        assert user_cache_directory(environ) == home / ".cache"
    assert ensure_user_cache_directory({"HOME": str(home)}) == home / ".cache"
    assert (home / ".cache").is_dir()


def test_an_existing_directory_is_left_as_it_is(tmp_path):
    (tmp_path / "kept").write_text("x")
    assert ensure_user_cache_directory({"XDG_CACHE_HOME": str(tmp_path)}) == tmp_path
    assert (tmp_path / "kept").read_text() == "x"


def test_no_known_directory_and_one_that_cannot_be_made_do_not_raise(tmp_path):
    assert user_cache_directory({}) is None
    assert ensure_user_cache_directory({}) is None
    in_the_way = tmp_path / "a-file"
    in_the_way.write_text("x")
    assert ensure_user_cache_directory({"XDG_CACHE_HOME": str(in_the_way / "below")}) is None


def test_executor_creation_makes_it_first(monkeypatch):
    """`create_py_executor` creates the directory before anything of the engine is built."""
    order = []
    monkeypatch.setattr(
        py_executor_creator, "ensure_user_cache_directory", lambda: order.append("cache directory")
    )

    def build(**kwargs):
        order.append("engine")
        return "executor"

    monkeypatch.setattr(py_executor_creator, "_create_py_executor_impl", build)
    assert py_executor_creator.create_py_executor(llm_args=None) == "executor"
    assert order == ["cache directory", "engine"]
