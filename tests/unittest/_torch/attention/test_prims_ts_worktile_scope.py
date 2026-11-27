# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PrimTS keeps task scheduling's WorkTileInfo rebind inside its own traces once a model arms the scope.

``arm()`` changes a CUTLASS class for the whole process, so each case runs in its own child process.
"""

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("cutlass")

_CHILD = """
import json

from tensorrt_llm._torch.attention.backends.prims_ts import _worktile_scope as scope


def same(a, b):
    return all(a[name] is b[name] for name in scope._ATTRS)


original = scope._ORIGINAL
result = {"saw_original": original is not None}
# Stand in for task scheduling's rebind with a distinct WorkTileInfo form, installed globally the way
# the rebind is: CUTLASS releases differ in whether importing task scheduling rebinds at all.
scalar = dict(original, update_from=lambda self, *args, **kwargs: None)
scope._SCALAR = scalar
scope._install(scalar)
result["global_before_arm"] = same(scope._snapshot(), scalar)
result["armed"] = scope.arm()
result["outside_after_arm"] = same(scope._snapshot(), original if result["armed"] else scalar)


@scope.with_scalar_work_tile_info
def traced(depth):
    inside = same(scope._snapshot(), scalar)
    if depth:
        inner = traced(depth - 1)
        return inside and inner and same(scope._snapshot(), scalar)
    return inside


result["inside"] = traced(2)
result["outside_after_trace"] = same(scope._snapshot(), original if result["armed"] else scalar)
print(json.dumps(result))
"""


def _run(env_value):
    env = {k: v for k, v in os.environ.items() if k != "TRTLLM_PRIMS_TS_SCOPED_WORKTILE"}
    if env_value is not None:
        env["TRTLLM_PRIMS_TS_SCOPED_WORKTILE"] = env_value
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD], env=env, capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("env_value", [None, "0"], ids=["armed", "kill_switch"])
def test_armed_scope_installs_the_scalar_form_only_while_tracing(env_value):
    result = _run(env_value)
    armed = env_value is None
    assert result["armed"] is armed
    if armed:
        # Until armed, the rebind stays global (the behavior without the scope).
        assert result["saw_original"] and result["global_before_arm"]
    assert result["outside_after_arm"]
    # Re-entrant: nested traces see the scalar form, and the originals return after the outermost one.
    assert result["inside"]
    assert result["outside_after_trace"]
