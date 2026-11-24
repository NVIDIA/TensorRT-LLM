# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import boundaries of the specialized SM107 targets."""

import json
import subprocess
import sys
import textwrap

import pytest

__extra_import_path__ = ["."]
from test_rubin_no_k3_rubin import EXCEPTIONS, FAMILY, K3_PACKAGE, REPO  # noqa: E402

TARGETS = [
    *(
        f"tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__{t}.modeling"
        for t in ("dep16", "helix8", "tep8")
    ),
]

_RECORDER = textwrap.dedent(
    """
    import importlib.abc, json, sys

    K3 = {k3!r}
    seen = []

    class Recorder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == K3 or fullname.startswith(K3 + "."):
                frame, importer = sys._getframe(1), None
                while frame is not None:
                    name = frame.f_globals.get("__name__", "")
                    if not (name.startswith("importlib") or name.startswith("_frozen_importlib")
                            or name == "__main__"):
                        importer = name
                        break
                    frame = frame.f_back
                seen.append([fullname, importer])
            return None

    sys.meta_path.insert(0, Recorder())
    for target in {targets!r}:
        __import__(target)
    print("K3_RUBIN_IMPORTS " + json.dumps(seen))
    """
)


def test_target_import_closure_has_no_k3_rubin():
    pytest.importorskip("torch")
    missing = [t for t in TARGETS if not (FAMILY / t.split(".")[-2]).is_dir()]
    assert not missing, f"Rubin targets missing from the tree: {missing}"
    script = _RECORDER.format(k3=K3_PACKAGE, targets=TARGETS)
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO, capture_output=True, text=True, timeout=900
    )
    assert result.returncode == 0, result.stderr[-4000:]
    line = next(line for line in result.stdout.splitlines() if line.startswith("K3_RUBIN_IMPORTS "))
    seen = json.loads(line.split(" ", 1)[1])
    allowed = {"tensorrt_llm." + rel[: -len(".py")].replace("/", ".") for rel in EXCEPTIONS}
    offenders = [
        (module, importer)
        for module, importer in seen
        if importer not in allowed and not (importer or "").startswith(K3_PACKAGE)
    ]
    assert not offenders, offenders
