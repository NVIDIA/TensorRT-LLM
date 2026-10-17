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
"""Run a chosen set of the target's fast paths in a test.

``select(dispatch, paths)`` resolves and installs the plan that runs exactly ``paths``: every
other fast path is listed in ``TRTLLM_STAIRCASE_GLM_LL_DISABLE``, so the plan is built by the
same ``resolve`` and ``install`` as the target's at construction. ``paths`` must hold every
path that one of them extends (``depends_on``), since ``DISABLE`` would turn such a path off;
``select`` raises otherwise. ``dispatch`` is the selector module the code under test reads
(some harnesses load the target's modules under a stand-in package).

With ``monkeypatch`` the selector state of the process (the installed plan, its validation
switch and the ``require`` mode) is restored when the test ends; without it the selection
stays installed, as a process-level harness wants.

A test that runs its case in a child interpreter passes the paths in ``FAST_PATHS_ENV``; the
child installs them with ``select_from_env``.
"""

import os
from typing import FrozenSet, Iterable, Mapping, Optional, Union

Paths = Union[str, Iterable[str]]

FAST_PATHS_ENV = "GLM_MOE_DSA_TEST_FAST_PATHS"


def names(paths: Paths) -> FrozenSet[str]:
    """The fast-path names of a comma-separated string or an iterable of names."""
    if isinstance(paths, str):
        return frozenset(name.strip() for name in paths.split(",") if name.strip())
    return frozenset(paths)


def plan_running(dispatch, paths: Paths, mode=None):
    """The resolved plan that runs exactly ``paths``.

    Raises:
        ValueError: if a name is not a fast path, or if ``paths`` lacks a path one of them
            extends.
    """
    wanted = names(paths)
    unknown = sorted(wanted - set(dispatch.FAST_PATH_NAMES))
    if unknown:
        raise ValueError(f"{unknown} are not fast paths of the target")
    bases = {base for path in wanted for base in dispatch.REQUIRED_FEATURES.get(path, ())}
    missing = sorted(bases - wanted)
    if missing:
        raise ValueError(f"{sorted(wanted)} lacks {missing}, which they extend")
    off = sorted(set(dispatch.FAST_PATH_NAMES) - wanted)
    environ = {dispatch.DISABLE_ENV: ",".join(off)} if off else {}
    plan = dispatch.resolve(mode if mode is not None else dispatch.ModelingV2Mode.AUTO, environ)
    running = frozenset(name for name in dispatch.FAST_PATH_NAMES if plan.enabled(name))
    assert running == wanted, (sorted(running), sorted(wanted))
    return plan


def select(
    dispatch,
    paths: Paths,
    monkeypatch=None,
    environ: Optional[Mapping[str, str]] = None,
    mode=None,
) -> FrozenSet[str]:
    """Install the plan that runs exactly ``paths`` and return the installed paths.

    ``environ`` is what the record line reads (the process environment by default), so a
    variant knob set there selects the numerical variant of a selected path.
    """
    if monkeypatch is not None:
        for name in ("_INSTALLED", "_VALIDATE", "_require"):
            monkeypatch.setattr(dispatch, name, list(getattr(dispatch, name)))
    dispatch.install(plan_running(dispatch, paths, mode), environ)
    return dispatch.installed()


def select_from_env(dispatch, environ: Optional[Mapping[str, str]] = None) -> FrozenSet[str]:
    """Install, for this process, the plan that runs exactly the paths in ``FAST_PATHS_ENV``."""
    environ = os.environ if environ is None else environ
    return select(dispatch, environ.get(FAST_PATHS_ENV, ""), environ=environ)
