# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ULL target's GEN-step boundary (step prep, layer 0, the chain entry mix, the MLA layers'
mixes, the decoder loop), host-side: the measured run's settings as constants, no environment
reads, self-containment and the provenance manifest ``FLAT_BOUNDARY_SOURCE.json`` (no GPU)."""

import ast
import json
import pathlib
import re

import pytest

from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4__sm_107__helix16_moetp16ep1 import (  # noqa: E501
    modeling as layer0,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.ops.common import (  # noqa: E501
    k3_ull_features as features,
)

pytestmark = pytest.mark.cpu_only

_REPO = pathlib.Path(__file__).resolve().parents[7]
_T_DIR = (
    _REPO / "tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl" / pathlib.Path(__file__).parent.name
)
_MANIFEST = _T_DIR.parent / "FLAT_BOUNDARY_SOURCE.json"

#: The boundary's files: no environment reads, no imports of the shared Kimi K3 VL code.
BOUNDARY_FILES = (
    "modeling.py",
    "weights.py",
    "../ops/norm/attn_res_fwd_ull.py",
    "../ops/norm/k3_attn_res_norm_ull.py",
    "../ops/comm/k3_attn_res_lamport_ull.py",
    "../ops/comm/k3_lamport_finish.py",
    "../ops/comm/k3_lamport_push.py",
)


def _tree(rel):
    return ast.parse((_T_DIR / rel).read_text())


def test_the_measured_runs_settings_are_constants():
    """Deployment settings: TRTLLM_MNNVL_AR_ONESHOT_MAX_BYTES=2097152; TRTLLM_ENABLE_PDL,
    KIMI_K3_AUX_ATTN_RES_STREAM and TLLM_SPEC_HEAD_FUSION unset."""
    from importlib import import_module

    layers = import_module(f"{layer0.__package__}.modeling")
    modeling = import_module(f"{layer0.__package__}.modeling")
    assert layers.MNNVL_ONE_SHOT_MAX_BYTES == 2097152
    assert layers._K3_AR_PUSH_ONE_SHOT_MAX_BYTES == 2097152
    assert layers._K3_FUSED_FINISH_ONE_SHOT_MAX_BYTES == 2097152
    assert layers.USE_PDL is True
    assert modeling.TAP_MIX is True
    assert modeling.ENTRY_RELEASE is True
    assert modeling.MLAIN_MXFP8_MAX_ROWS == 16
    assert features.PROFILE.embedding_gather is True
    assert features.PROFILE.pack_step_inputs is True
    assert features.PROFILE.attn_res_pdl_max_m >= 32  # every GEN step takes the PDL mixes


def _module_constants(rel):
    """Top-level NAME = <literal> assignments of a target file: name -> [values, in file order]."""
    out = {}
    for node in _tree(rel).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        value = ast.literal_eval(node.value)
                    except ValueError:
                        value = None
                    out.setdefault(target.id, []).append(value)
    return out


def test_no_module_constant_is_assigned_twice():
    """A second top-level assignment silently overrides the first. A duplicate REDUCE_UNROLL
    assignment previously kept FP8 MLA decode at one split per round instead of eight."""
    twice = {}
    for path in sorted(_T_DIR.rglob("*.py")):
        rel = path.relative_to(_T_DIR).as_posix()
        names = [n for n, v in _module_constants(rel).items() if n.isupper() and len(v) > 1]
        if names:
            twice[rel] = names
    assert not twice


@pytest.mark.parametrize(
    "rel, name, value",
    [
        # Generation settings (TRTLLM_K3_DRAFT3_REDUCE_UNROLL=8, TRTLLM_FP4_MLA_LPT_PLAN=1,
        # TRTLLM_K3_ULL3_MLA_TAIL_TRIGGER=epilogue, the step-upload trio).
        ("../ops/attention/k3_mla_decode_fp8.py", "REDUCE_UNROLL", 8),
        ("../ops/attention/_fp4_mla_decode_ull.py", "_LPT_PLAN", True),
        ("../ops/attention/_fp4_mla_decode_ull.py", "_ULL3_MLA_TAIL_TRIGGER_EPILOGUE", True),
        ("../kernels/k3r_step/step_upload_kernels.py", "_BLOCK_OFFSETS_TRITON", True),
        ("../kernels/k3r_step/step_upload_kernels.py", "_SPEC_SAMPLER_FUSED_SCATTER", True),
        ("../kernels/k3r_step/step_upload_kernels.py", "_STATE_INDEX_COPY_KERNEL", True),
    ],
)
def test_the_measured_runs_gen_switches_are_pinned(rel, name, value):
    assert _module_constants(rel).get(name) == [value]


def test_one_shot_switch_at_the_measured_threshold():
    """8 rows (1.84 MB over 16 ranks) are one-shot, 16 rows two-shot: the C++ kernel served ran."""
    from importlib import import_module

    import torch

    layers = import_module(f"{layer0.__package__}.modeling")
    h, world = 7168, 16
    assert layers.mnnvl_workspace_bytes(8, h, world, torch.bfloat16) == 8 * h * world * 2
    assert layers.mnnvl_workspace_bytes(16, h, world, torch.bfloat16) == 2 * 16 * h * 2


@pytest.mark.parametrize("rel", BOUNDARY_FILES)
def test_no_environment_reads(rel):
    bad = []
    for node in ast.walk(_tree(rel)):
        if isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv"):
            bad.append(node.lineno)
        elif isinstance(node, ast.Name) and node.id in ("getenv", "get_env_enable_pdl"):
            bad.append(node.lineno)
    assert not bad, f"{rel}: environment read at lines {bad}"


@pytest.mark.parametrize("rel", BOUNDARY_FILES)
def test_no_shared_kimi_imports(rel):
    """The boundary imports nothing of the shared Kimi K3 VL ops or the stock Kimi models."""
    for node in ast.walk(_tree(rel)):
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            module = node.module or ""
            assert ".kimi_k3_vl.ops" not in module, (rel, node.lineno, module)
            assert not module.startswith("tensorrt_llm._torch.models.modeling_kimi"), (
                rel,
                node.lineno,
                module,
            )


def _qualified(tree, name):
    """Whether ``name`` (``f``, ``C``, ``C.m`` or a module constant) is defined in ``tree``."""
    head, _, attr = name.partition(".")
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == head:
            if not attr:
                return True
            return isinstance(node, ast.ClassDef) and any(
                isinstance(f, ast.FunctionDef) and f.name == attr for f in node.body
            )
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and not attr:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == head for t in targets):
                return True
    return False


def test_manifest_names_existing_target_functions():
    """Keys name files relative to the family directory, where the manifest is."""
    data = json.loads(_MANIFEST.read_text())
    assert data["functions"]
    missing = []
    for key in data["functions"]:
        rel, _, name = key.partition(":")
        name = name.split(" (")[0]
        if not _qualified(ast.parse((_MANIFEST.parent / rel).read_text()), name):
            missing.append(key)
    assert not missing


def test_manifest_source_references_have_digests():
    """Every reference records a source symbol, line number and file digest."""
    data = json.loads(_MANIFEST.read_text())
    for path, digest in data["source_sha256"].items():
        assert re.fullmatch(r"[0-9a-f]{64}", digest), path
    for key, refs in data["functions"].items():
        assert refs, key
        for ref in refs:
            path, line = ref["source"].rsplit(":", 1)
            assert path in data["source_sha256"], key
            assert line.isdecimal() and int(line) > 0, key
            assert ref["symbol"].strip(), key
