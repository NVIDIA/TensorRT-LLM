# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The surface survey: what the checkpoint demands, what the build supplies."""

from __future__ import annotations

import json

import pytest

from agent_flow.workflows.trtllm_modeling_v2.infra.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.onboard.survey import (
    MARKER,
    RECEIPT_ABSENT,
    RECEIPT_PASSED,
    Inventory,
    format_surface,
    parse_inventory,
    probe_command,
    read_checkpoint_surface,
    run_inventory_probe,
    write_surface_file,
)

# --------------------------------------------------------------- the demands


def _dense():
    return dict(
        architectures=["LlamaForCausalLM"],
        model_type="llama",
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
        hidden_size=4096,
        num_hidden_layers=32,
        vocab_size=128256,
    )


def _names(config):
    return read_checkpoint_surface(config).demand_names


def test_a_dense_decoder_reports_only_what_it_has():
    names = _names(_dense())
    assert "attention.shape" in names
    assert "mlp.activation" in names
    assert "norm.style" in names
    # Nothing it does not have.
    assert not any(n.startswith(("moe.", "recurrent.", "stack.", "speculation.")) for n in names)


def test_a_hybrid_stack_is_reported_as_hybrid_not_as_a_window():
    config = _dense() | {"layer_types": ["linear_attention"] * 3 + ["full_attention"]}
    names = _names(config)
    assert "stack.hybrid" in names
    assert "attention.windowing" not in names


def test_moe_is_read_through_any_of_its_spellings():
    for key in ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts"):
        config = _dense() | {key: 128, "num_experts_per_tok": 8}
        demand = next(d for d in read_checkpoint_surface(config).demands if d.name == "moe.routing")
        assert "128 experts" in demand.detail, key
        assert "top-8" in demand.detail, key


def test_a_multimodal_checkpoint_is_read_through_its_text_tower():
    config = {
        "architectures": ["SomeVLForConditionalGeneration"],
        "model_type": "some_vl",
        "vision_config": {"num_attention_heads": 16, "hidden_size": 1408},
        "text_config": _dense() | {"num_attention_heads": 40, "num_key_value_heads": 8},
    }
    surface = read_checkpoint_surface(config)
    assert surface.submodels == ("vision_config", "text_config")
    demand = next(d for d in surface.demands if d.name == "attention.shape")
    assert "40 query heads" in demand.detail


# ------------------------------------------------------- the unknown column


def test_an_unclaimed_structural_key_is_surfaced_not_dropped():
    config = _dense() | {"query_pre_attn_scalar": 224, "no_rope_layers": [0, 1, 1]}
    surface = read_checkpoint_surface(config)
    assert "query_pre_attn_scalar" in surface.unrecognized
    assert "no_rope_layers" in surface.unrecognized


def test_bookkeeping_keys_are_not_reported_as_discoveries():
    config = _dense() | {"bos_token_id": 1, "transformers_version": "4.51", "_name_or_path": "x"}
    assert read_checkpoint_surface(config).unrecognized == ()


# ------------------------------------------------------------- the inventory


def _inventory_payload(**overrides):
    payload = {
        "trtllm_ops": ["fused_qk_norm_rope", "flashinfer_silu_and_mul", "brand_new_op"],
        "catalog": [
            {
                "path": "attention/fused_qk_norm_rope.py",
                "impl": "torch.ops.trtllm.fused_qk_norm_rope",
                "receipt": "passed",
                "impl_resolves": True,
            },
            {
                "path": "activation/flashinfer_silu_and_mul.py",
                "impl": "torch.ops.trtllm.flashinfer_silu_and_mul",
                "receipt": "absent",
                "impl_resolves": True,
            },
            {
                "path": "gemm/gone.py",
                "impl": "torch.ops.trtllm.gone",
                "receipt": "passed",
                "impl_resolves": False,
            },
        ],
        "modules": ["rms_norm.RMSNorm"],
        "versions": {"torch": "2.12"},
        "error": "",
    }
    payload.update(overrides)
    return payload


def test_the_payload_is_found_in_a_log_full_of_engine_chatter():
    stdout = (
        "[TRT-LLM] loading plugins\nrank 0 initialized\n"
        + MARKER
        + json.dumps(_inventory_payload())
        + "\n[TRT-LLM] shutting down\n"
    )
    inventory = parse_inventory(stdout, arch="sm_103")
    assert inventory.ok
    assert len(inventory.catalog) == 3


def test_a_silent_probe_is_an_error_carrying_the_tail():
    inventory = parse_inventory("import failed\nTraceback\nImportError: no module", arch="sm_103")
    assert not inventory.ok
    assert "ImportError" in inventory.error


def test_entries_are_sorted_into_spendable_uncertified_and_rotted():
    inventory = parse_inventory(MARKER + json.dumps(_inventory_payload()), arch="sm_103")
    assert [e.name for e in inventory.spendable] == ["fused_qk_norm_rope"]
    assert [e.name for e in inventory.uncertified] == ["flashinfer_silu_and_mul"]
    assert [e.name for e in inventory.rotted] == ["gone"]


def test_the_probe_is_never_launched_as_engine_work():
    env = RunEnvironment(kind="session_spool", python="/venv/bin/python", world_size=4, spool="/s")
    command = probe_command(env, repo="/repo", arch="sm_103")
    assert "mpirun" not in command
    assert "trtllm-llmapi-launch" not in command
    assert "__SURVEY_JSON__" in command


def test_a_failed_probe_is_reported_rather_than_raised():
    class _Result:
        ok, returncode, timed_out = False, 1, False
        stdout, stderr = "", "ModuleNotFoundError: tensorrt_llm"

    inventory = run_inventory_probe(
        {"arch": "sm_103", "trtllm_repo_path": "/repo"},
        RunEnvironment(kind="local", python="python3"),
        shell_runner=lambda *a, **k: _Result(),
    )
    assert not inventory.ok
    assert "exited 1" in inventory.error


# ---------------------------------------------------------------- rendering


def test_the_block_lays_both_columns_out_without_joining_them():
    surface = read_checkpoint_surface(_dense() | {"query_pre_attn_scalar": 224})
    inventory = parse_inventory(MARKER + json.dumps(_inventory_payload()), arch="sm_103")
    block = format_surface(surface, inventory)
    assert "Joining them is your job" in block
    assert "query_pre_attn_scalar" in block
    assert "fused_qk_norm_rope" in block
    assert "brand_new_op" in block
    assert "gone" in block


def test_a_failed_probe_is_not_presented_as_a_measurement():
    surface = read_checkpoint_surface(_dense())
    measured = parse_inventory(MARKER + json.dumps(_inventory_payload()), arch="sm_103")
    assert format_surface(surface, measured).startswith("## Surface survey\n\nMeasured")

    failed = format_surface(surface, Inventory(arch="sm_103", error="probe exited 127"))
    assert "The supply column was NOT" in failed
    assert "an assumption and not a reading" in failed
    assert "evidence of absence" in failed


def test_the_workspace_file_is_unabridged(tmp_path):
    payload = _inventory_payload(
        trtllm_ops=[f"op_{i}" for i in range(200)],
        modules=[f"mod.Class{i}" for i in range(90)],
    )
    inventory = parse_inventory(MARKER + json.dumps(payload), arch="sm_103")
    path = write_surface_file(tmp_path / "surface.md", read_checkpoint_surface(_dense()), inventory)
    text = path.read_text()
    assert "op_199" in text
    assert "mod.Class89" in text
    assert "torch: 2.12" in text
    assert "op_199" not in format_surface(read_checkpoint_surface(_dense()), inventory)


# ------------------------------------------------- the probe, actually run


def _entry_source(impl, archs):
    arch_set = ", ".join(f"Arch.{a}" for a in archs)
    cell_list = "".join(f"Cell(why='c{i}', spec={{}}), " for i in range(2))
    return (
        "from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper\n\n\n"
        "class Entry(OpWrapper):\n"
        '    """Mentions torch.ops.trtllm.neighbour in prose, which is not the op."""\n\n'
        f"    ARCHS = frozenset({{{arch_set}}})\n"
        f"    CELLS = ({cell_list})\n\n"
        "    def __call__(self, x):\n"
        f"        return {impl}(x)\n\n"
        "    def reference(self, x):\n"
        "        return x\n"
    )


def _fake_repo(tmp_path, entries, model_ops=()):
    """A repo tree with catalog entries, and nothing else.

    ``entries`` are ``(category/name.py, impl, archs)`` written under
    ``catalog/``; ``model_ops`` are ``(family/target, category/name.py, impl,
    archs)`` kept in a target's ``ops/``.
    """
    catalog = tmp_path / "tensorrt_llm/_torch/_experimental/modeling_v2/catalog"
    catalog.mkdir(parents=True)
    (catalog / "_op.py").write_text("class OpWrapper:\n    pass\n", encoding="utf-8")
    for path, impl, archs in entries:
        py = catalog / path
        py.parent.mkdir(parents=True, exist_ok=True)
        (py.parent / "__init__.py").write_text("", encoding="utf-8")
        py.write_text(_entry_source(impl, archs), encoding="utf-8")
    models = tmp_path / "tensorrt_llm/_torch/_experimental/modeling_v2/models"
    for scope, path, impl, archs in model_ops:
        py = models / scope / "ops" / path
        py.parent.mkdir(parents=True, exist_ok=True)
        py.write_text(_entry_source(impl, archs), encoding="utf-8")
        (py.parent / "__init__.py").write_text("", encoding="utf-8")
    modules = tmp_path / "tensorrt_llm/_torch/modules"
    modules.mkdir(parents=True)
    (modules / "rms_norm.py").write_text("class RMSNorm:\n    pass\n\ndef _hidden():\n    pass\n")
    return tmp_path


def _run_probe(repo, arch="sm_103"):
    import subprocess
    import sys

    from agent_flow.workflows.trtllm_modeling_v2.onboard.survey import PROBE_SOURCE

    proc = subprocess.run(
        [sys.executable, "-c", PROBE_SOURCE, str(repo), arch],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return parse_inventory(proc.stdout + proc.stderr, arch=arch)


_INLINE_PASS = ("SM_103",)


@pytest.mark.parametrize(
    "archs,expected",
    [(_INLINE_PASS, RECEIPT_PASSED), (("SM_100",), RECEIPT_ABSENT)],
    ids=["this-arch", "other-arch"],
)
def test_the_probe_reads_the_entrys_archs(tmp_path, archs, expected):
    repo = _fake_repo(tmp_path, [("norm/x.py", "torch.ops.trtllm.x", archs)])
    inventory = _run_probe(repo)
    assert inventory.catalog[0].receipt == expected
    assert inventory.catalog[0].cells == 2


def test_another_targets_ops_are_promotable_not_spendable(tmp_path):
    repo = _fake_repo(
        tmp_path,
        [("norm/x.py", "torch.ops.trtllm.x", _INLINE_PASS)],
        model_ops=[
            ("fam/ckpt__sm_103__tp1", "moe/router.py", "torch.ops.trtllm.router", _INLINE_PASS)
        ],
    )
    inventory = _run_probe(repo)
    (op,) = inventory.promotable
    assert (op.scope, op.path, op.impl) == (
        "fam/ckpt__sm_103__tp1",
        "moe/router.py",
        "torch.ops.trtllm.router",
    )
    assert op.receipt == RECEIPT_PASSED
    assert op not in inventory.spendable and op not in inventory.uncertified
    assert "router" not in inventory.unwrapped_ops
    block = format_surface(read_checkpoint_surface(_dense()), inventory)
    assert "`fam/ckpt__sm_103__tp1/ops/moe/router`" in block


def test_an_engine_that_will_not_import_does_not_condemn_the_whole_catalog(tmp_path):
    """The probe here runs on an interpreter with no engine."""
    repo = _fake_repo(tmp_path, [("norm/x.py", "torch.ops.trtllm.x", _INLINE_PASS)])
    inventory = _run_probe(repo)
    assert inventory.error
    assert inventory.rotted == ()
    assert inventory.catalog[0].receipt == RECEIPT_PASSED


# ------------------------------------------------- how the op plane is enumerated

#: A stand-in `torch` whose `ops.trtllm` is LAZY, exactly as the real one is:
#: `dir()` lists only what something has already resolved. The schema registry
#: knows the rest.
_FAKE_TORCH = '''
import sys, types

class _Schema:
    def __init__(self, name):
        self.name = name

class _LazyNamespace:
    """dir() reports only the two names somebody happened to touch."""
    def __dir__(self):
        return ["already_resolved_a", "already_resolved_b"]

torch = types.ModuleType("torch")
torch.__version__ = "9.9.9-fake"
torch.ops = types.SimpleNamespace(trtllm=_LazyNamespace())
torch._C = types.SimpleNamespace(
    _jit_get_all_schemas=lambda: [
        _Schema("trtllm::allgather"),
        _Schema("trtllm::fp4_quantize"),
        _Schema("trtllm::fused_qk_norm_rope"),
        _Schema("aten::add"),
    ]
)
sys.modules["torch"] = torch

trtllm = types.ModuleType("tensorrt_llm")
trtllm.__version__ = "0.0.0-fake"
sys.modules["tensorrt_llm"] = trtllm
'''


def _run_probe_with_fake_torch(repo, arch="sm_103"):
    import subprocess
    import sys

    from agent_flow.workflows.trtllm_modeling_v2.onboard.survey import PROBE_SOURCE

    proc = subprocess.run(
        [sys.executable, "-c", _FAKE_TORCH + PROBE_SOURCE, str(repo), arch],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return parse_inventory(proc.stdout + proc.stderr, arch=arch)


def test_the_op_plane_is_read_from_the_schema_registry_not_from_dir(tmp_path):
    repo = _fake_repo(tmp_path, [("comm/allgather.py", "torch.ops.trtllm.allgather", _INLINE_PASS)])
    inventory = _run_probe_with_fake_torch(repo)

    assert "allgather" in inventory.trtllm_ops
    assert "fp4_quantize" in inventory.trtllm_ops
    assert "add" not in inventory.trtllm_ops
    # And the lazily-resolved names are still there, so the registry is a
    # superset rather than a replacement.
    assert "already_resolved_a" in inventory.trtllm_ops
    entry = inventory.catalog[0]
    assert entry.impl_resolves
    assert entry.spendable
