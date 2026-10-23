# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What ``_native`` does with a manifest, and that the installed libraries load.

The first half drives the loader's decisions with synthetic manifests: which
outcomes are ``NoTarget`` (this build or this device cannot run the target, so
``auto`` keeps the built-in model) and which are ``NativeLoadError`` (the
installation is broken, which is never a fallback). Nothing is opened.

The second half is the load smoke test of the installed wheel: every family
library the manifest lists opens in a fresh process, reports the stock contract
fingerprint, and defines no op its target's fake_ops.py does not fake. A family
library aborts the process from a static initializer if two libraries claim one
torch namespace, so each one is opened in a child process.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tensorrt_llm._torch._experimental.modeling_v2 import _native
from tensorrt_llm._torch._experimental.modeling_v2._native import (
    NativeLoadError,
    arch_covers,
    available_targets,
    camel,
    identifier,
    load_target_ops,
    namespace_of,
    read_manifest,
    split_target_package,
    torch_namespace,
)
from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget

pytestmark = pytest.mark.cpu_only

_FIXTURE = "tensorrt_llm._torch._experimental.modeling_v2.models.fixture.scale_add__sm_80__default"


def _manifest(status="built", archs=("100f-real",), library="libtrtllm_mv2_fixture.so", **extra):
    return {
        "schema": _native.MANIFEST_SCHEMA,
        "enabled": True,
        "requested_archs": ["100", "103"],
        "families": {
            "fixture": {
                "library": library,
                "namespace": "mv2_fixture",
                "shared": None,
                "targets": {
                    "scale_add__sm_80__default": {
                        "status": status,
                        "namespace": "mv2_fixture_scale_add_sm_80_default",
                        "declared_archs": ["100f"],
                        "archs": list(archs),
                        "reason": extra.get("reason", ""),
                    }
                },
            }
        },
    }


# --- Naming ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("fixture", "fixture"),
        ("minimax_m3", "minimaxM3"),
        ("glm_moe_dsa", "glmMoeDsa"),
        ("minimax_m3_nvfp4__sm_107__tp2", "minimaxM3Nvfp4Sm107Tp2"),
        ("r1_0528_nvfp4__sm_10x__default", "r10528Nvfp4Sm10xDefault"),
    ],
)
def test_camel_is_the_native_directory(name, expected):
    assert camel(name) == expected


def test_namespaces_collapse_double_underscores():
    assert identifier("scale_add__sm_80__default") == "scale_add_sm_80_default"
    assert torch_namespace("fixture") == "mv2_fixture"
    assert namespace_of(_FIXTURE) == "mv2_fixture_scale_add_sm_80_default"
    assert split_target_package(_FIXTURE) == ("fixture", "scale_add__sm_80__default")
    with pytest.raises(ValueError):
        split_target_package("tensorrt_llm._torch.models")


@pytest.mark.parametrize(
    "arch, capability, runs",
    [
        ("100f-real", (10, 0), True),
        ("100f-real", (10, 3), True),
        ("100f-real", (10, 7), True),
        ("100f-real", (12, 0), False),
        ("103a-real", (10, 3), True),
        ("103a-real", (10, 0), False),
        ("80-real", (8, 6), True),
        ("80-real", (9, 0), False),
        ("90a-real", (9, 0), True),
        ("120f", (12, 1), True),
    ],
)
def test_arch_coverage(arch, capability, runs):
    assert arch_covers(arch, capability) is runs


# --- Decisions -----------------------------------------------------------------


def test_a_missing_manifest_is_a_broken_installation(tmp_path):
    with pytest.raises(NativeLoadError, match="no modeling_v2"):
        read_manifest(tmp_path / "absent.json")


def test_a_manifest_from_another_schema_is_refused(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({**_manifest(), "schema": 999}))
    with pytest.raises(NativeLoadError, match="schema"):
        read_manifest(path)


def test_a_build_without_modeling_v2_is_no_target():
    manifest = {
        "schema": _native.MANIFEST_SCHEMA,
        "enabled": False,
        "reason": "off",
        "families": {},
    }
    with pytest.raises(NoTarget, match="no modeling_v2 libraries"):
        load_target_ops(_FIXTURE, manifest)


@pytest.mark.parametrize("status", ["skipped", "excluded"])
def test_a_target_this_build_skipped_is_no_target(status):
    manifest = _manifest(status=status, archs=(), reason="needs sm_107")
    with pytest.raises(NoTarget, match="needs sm_107"):
        load_target_ops(_FIXTURE, manifest)


def test_a_target_the_manifest_does_not_know_is_a_broken_installation():
    manifest = _manifest()
    manifest["families"]["fixture"]["targets"] = {}
    with pytest.raises(NativeLoadError, match="different builds"):
        load_target_ops(_FIXTURE, manifest)
    with pytest.raises(NativeLoadError, match="different builds"):
        load_target_ops(_FIXTURE, {**manifest, "families": {}})


def test_a_device_the_target_was_not_built_for_is_no_target(monkeypatch):
    monkeypatch.setattr(_native, "_device_capability", lambda: (8, 0))
    with pytest.raises(NoTarget, match="does not run on this sm_80"):
        load_target_ops(_FIXTURE, _manifest(archs=("100f-real",)))


def test_a_listed_library_that_is_missing_is_a_broken_installation(monkeypatch, tmp_path):
    monkeypatch.setattr(_native, "_device_capability", lambda: None)
    monkeypatch.setattr(_native, "_LIBS_DIR", tmp_path)
    with pytest.raises(NativeLoadError, match="missing"):
        load_target_ops(_FIXTURE, _manifest())


def test_available_targets_lists_every_target():
    rows = available_targets(_manifest(status="skipped", archs=(), reason="needs sm_107"))
    assert rows == [
        {
            "family": "fixture",
            "target": "scale_add__sm_80__default",
            "status": "skipped",
            "archs": [],
            "reason": "needs sm_107",
            "library": "libtrtllm_mv2_fixture.so",
        }
    ]


# --- The installed wheel ---------------------------------------------------------

_LOAD_ONE_FAMILY = textwrap.dedent(
    """
    import json, sys
    import torch
    import tensorrt_llm  # noqa: F401 -- loads libth_common.so
    from tensorrt_llm._torch._experimental.modeling_v2 import _native

    family = sys.argv[1]
    _native.load_family_library(family)
    schemas = [s.name for s in torch._C._jit_get_all_schemas()]
    print(json.dumps(sorted(n for n in schemas if n.startswith("mv2_"))))
    """
)


def _installed_manifest():
    path = _native._LIBS_DIR / _native.MANIFEST_NAME
    assert path.is_file(), (
        f"{path} is missing: scripts/build_wheel.py installs it with the family libraries"
    )
    return read_manifest(path)


def test_every_built_family_library_loads():
    manifest = _installed_manifest()
    families = list(_native.built_families(manifest))
    if not families:
        pytest.skip(f"this build has no modeling_v2 family library: {manifest.get('reason')}")
    models = Path(_native.__file__).resolve().parent / "models"
    for family in families:
        result = subprocess.run(
            [sys.executable, "-c", _LOAD_ONE_FAMILY, family],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"libtrtllm_mv2_{family}.so failed to load:\n{result.stdout}\n{result.stderr}"
        )
        ops = json.loads(result.stdout.strip().splitlines()[-1])
        for target, unit in manifest["families"][family]["targets"].items():
            if unit["status"] != "built":
                continue
            prefix = f"{unit['namespace']}::"
            target_ops = [op[len(prefix) :] for op in ops if op.startswith(prefix)]
            fake_ops = models / family / target / "fake_ops.py"
            text = fake_ops.read_text() if fake_ops.is_file() else ""
            unfaked = [op for op in target_ops if f'::{op}"' not in text]
            assert not unfaked, f"{family}/{target} defines {unfaked} with no fake in fake_ops.py"
