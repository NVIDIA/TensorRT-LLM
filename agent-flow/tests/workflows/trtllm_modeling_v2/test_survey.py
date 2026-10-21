# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The surface survey: what the checkpoint demands, what the build supplies.

The demand half is tested against five *different families* on purpose. A
reader that only ever sees one checkpoint converges on that checkpoint, and
the whole reason this module exists is the next model — the one with a layer
nobody has a word for.
"""

from __future__ import annotations

import json

import pytest

from agent_flow.workflows.trtllm_modeling_v2.infra.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.onboard.survey import (
    MARKER,
    RECEIPT_ABSENT,
    RECEIPT_PASSED,
    CatalogEntry,
    CheckpointSurface,
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
    """An ordinary dense decoder — the boring baseline."""
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


@pytest.mark.parametrize(
    "kv_heads,shape",
    [(32, "MHA"), (8, "GQA"), (1, "MQA")],
)
def test_attention_shape_is_read_from_the_head_counts(kv_heads, shape):
    config = _dense() | {"num_key_value_heads": kv_heads}
    demand = next(d for d in read_checkpoint_surface(config).demands if d.name == "attention.shape")
    assert shape in demand.detail


def test_mla_is_not_confused_with_ordinary_attention():
    config = _dense() | {"kv_lora_rank": 512, "q_lora_rank": 1536}
    assert "attention.mla" in _names(config)


def test_a_hybrid_stack_is_reported_as_hybrid_not_as_a_window():
    """`layer_types` that interleaves attention kinds is a hybrid stack, not a window.

    A checkpoint that interleaves linear and full attention has no window
    anywhere in it, and reporting one sends the plan looking for a windowing
    op that the model never calls.
    """
    config = _dense() | {"layer_types": ["linear_attention"] * 3 + ["full_attention"]}
    names = _names(config)
    assert "stack.hybrid" in names
    assert "attention.windowing" not in names


def test_a_real_window_is_still_reported():
    config = _dense() | {"sliding_window": 4096}
    assert "attention.windowing" in _names(config)


def test_a_uniform_layer_types_list_is_not_a_hybrid():
    config = _dense() | {"layer_types": ["full_attention"] * 8}
    assert "stack.hybrid" not in _names(config)


def test_moe_is_read_through_any_of_its_spellings():
    for key in ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts"):
        config = _dense() | {key: 128, "num_experts_per_tok": 8}
        demand = next(d for d in read_checkpoint_surface(config).demands if d.name == "moe.routing")
        assert "128 experts" in demand.detail, key
        assert "top-8" in demand.detail, key


def test_a_recurrent_plane_is_called_out_as_a_different_cache():
    config = _dense() | {"mamba_d_state": 128, "conv_kernel": 4}
    demand = next(d for d in read_checkpoint_surface(config).demands if d.name == "recurrent.state")
    assert "state cache manager" in demand.detail


def test_quantization_absence_is_stated_rather_than_omitted():
    """An absent demand and an unmeasured one read the same; say which."""
    demand = next(d for d in read_checkpoint_surface(_dense()).demands if d.name == "quantization")
    assert "none declared" in demand.detail


def test_quantization_reports_the_held_out_modules():
    config = _dense() | {
        "quantization_config": {
            "quant_algo": "NVFP4",
            "kv_cache_quant_algo": "FP8",
            "exclude_modules": ["lm_head", "vision"],
        }
    }
    demand = next(d for d in read_checkpoint_surface(config).demands if d.name == "quantization")
    assert "NVFP4" in demand.detail
    assert "kv=FP8" in demand.detail
    assert "2 module(s) held out" in demand.detail


def test_a_multimodal_checkpoint_is_read_through_its_text_tower():
    """Reading the outer config would describe the vision encoder."""
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
    """The whole point, for a model this reader has never seen.

    `query_pre_attn_scalar` changes the attention scale. A reader that drops
    what it does not recognize produces a target that is quietly wrong, and
    the discovery moves to a smith at tier 4 with no budget left.
    """
    config = _dense() | {"query_pre_attn_scalar": 224, "no_rope_layers": [0, 1, 1]}
    surface = read_checkpoint_surface(config)
    assert "query_pre_attn_scalar" in surface.unrecognized
    assert "no_rope_layers" in surface.unrecognized


def test_bookkeeping_keys_are_not_reported_as_discoveries():
    config = _dense() | {"bos_token_id": 1, "transformers_version": "4.51", "_name_or_path": "x"}
    assert read_checkpoint_surface(config).unrecognized == ()


def test_a_key_a_probe_already_read_is_not_also_unrecognized():
    config = _dense() | {"rope_scaling": {"rope_type": "yarn"}, "num_experts": 8}
    surface = read_checkpoint_surface(config)
    assert "rope_scaling" not in surface.unrecognized
    assert "num_experts" not in surface.unrecognized


# ------------------------------------------------------------- the inventory


def _inventory_payload(**overrides):
    payload = {
        "trtllm_ops": ["fused_qk_norm_rope", "flashinfer_silu_and_mul", "brand_new_op"],
        "catalog": [
            {
                "path": "attention/fused_qk_norm_rope.py",
                "impl": "torch.ops.trtllm.fused_qk_norm_rope",
                "summary": "s",
                "receipt": "passed",
                "impl_resolves": True,
            },
            {
                "path": "activation/flashinfer_silu_and_mul.py",
                "impl": "torch.ops.trtllm.flashinfer_silu_and_mul",
                "summary": "s",
                "receipt": "absent",
                "impl_resolves": True,
            },
            {
                "path": "gemm/gone.py",
                "impl": "torch.ops.trtllm.gone",
                "summary": "s",
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


def test_a_rotted_entry_is_never_spendable_however_good_its_receipt():
    """It carries a passing receipt and still cannot be called."""
    entry = CatalogEntry(path="gemm/gone.py", receipt=RECEIPT_PASSED, impl_resolves=False)
    assert not entry.spendable


def test_an_op_no_entry_wraps_is_reported_as_the_tier_one_rung():
    inventory = parse_inventory(MARKER + json.dumps(_inventory_payload()), arch="sm_103")
    assert inventory.unwrapped_ops == ("brand_new_op",)


def test_the_probe_command_carries_the_runs_launch_prefix():
    env = RunEnvironment(kind="local", python="/venv/bin/python", env={"TRTLLM_MODELING_V2": "off"})
    command = probe_command(env, repo="/repo", arch="sm_103")
    assert "TRTLLM_MODELING_V2=off" in command
    assert "/venv/bin/python" in command
    assert "/repo" in command and "sm_103" in command


def test_the_probe_is_never_launched_as_engine_work():
    """Read-only introspection, so one process at any world size.

    Launched the way a gate is, as `mpirun -n 4 ... trtllm-llmapi-launch`, it
    exits 127 wherever the run's own `PATH` carries no `mpirun`, and the plan
    phase drafts against an empty inventory -- every certified entry reads as
    a gap.
    """
    env = RunEnvironment(kind="session_spool", python="/venv/bin/python", world_size=4, spool="/s")
    command = probe_command(env, repo="/repo", arch="sm_103")
    assert "mpirun" not in command
    assert "trtllm-llmapi-launch" not in command
    assert "__SURVEY_JSON__" in command


def test_a_failed_probe_is_reported_rather_than_raised():
    """A run survives a survey failure; it just plans on assumption."""

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


def test_a_timeout_says_so():
    class _Result:
        ok, returncode, timed_out = False, None, True
        stdout, stderr = "", ""

    inventory = run_inventory_probe(
        {"arch": "sm_103", "trtllm_repo_path": "/repo"},
        RunEnvironment(kind="local", python="python3"),
        shell_runner=lambda *a, **k: _Result(),
        timeout=7,
    )
    assert "timed out after 7s" in inventory.error


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


def test_a_failed_inventory_tells_the_drafter_it_is_planning_blind():
    block = format_surface(
        read_checkpoint_surface(_dense()), Inventory(arch="sm_103", error="probe died")
    )
    assert "probe died" in block
    assert "you do not know what exists" in block


def test_a_failed_probe_is_not_presented_as_a_measurement():
    """The block opens by saying which of its two columns is a reading.

    Without that, a supply column that failed to land renders with the same
    shape as one that landed empty, and the drafter -- which is told it is
    reading a measurement -- plans every entry as a gap.
    """
    surface = read_checkpoint_surface(_dense())
    measured = parse_inventory(MARKER + json.dumps(_inventory_payload()), arch="sm_103")
    assert format_surface(surface, measured).startswith("## Surface survey\n\nMeasured")

    failed = format_surface(surface, Inventory(arch="sm_103", error="probe exited 127"))
    assert "The supply column was NOT" in failed
    assert "an assumption and not a reading" in failed
    assert "evidence of absence" in failed


def test_the_workspace_file_is_unabridged(tmp_path):
    """The prompt block is capped; the audit record is not."""
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


def _flashinfer_inventory(**overrides):
    payload = _inventory_payload(
        flashinfer_api=[
            "flashinfer.norm.rmsnorm",
            "flashinfer.norm.gemma_rmsnorm",
            "flashinfer.gemm.mm_fp4",
            "flashinfer.fused_moe.trtllm_fp4_block_scale_moe",
        ],
        flashinfer_refs={"rmsnorm": ["tensorrt_llm/_torch/custom_ops/flashinfer_custom_ops.py"]},
        versions={"torch": "2.12", "flashinfer": "0.6.18"},
    )
    payload.update(overrides)
    return parse_inventory(MARKER + json.dumps(payload), arch="sm_103")


def test_flashinfer_kernels_no_op_wraps_are_offered_as_the_library_rung():
    """The op inventory sees only trtllm ops; FlashInfer's own API is most of the library.

    Unlisted, a FlashInfer kernel no trtllm op wraps is invisible to the plan,
    which then writes the kernel (tier 4-5) or declares a ceiling.
    """
    inventory = _flashinfer_inventory()
    assert inventory.flashinfer_unreferenced == (
        "flashinfer.norm.gemma_rmsnorm",
        "flashinfer.gemm.mm_fp4",
        "flashinfer.fused_moe.trtllm_fp4_block_scale_moe",
    )
    block = format_surface(read_checkpoint_surface(_dense()), inventory)
    assert "FlashInfer 0.6.18: 4 public callables, 3 referenced nowhere in the engine" in block
    assert "tier 3" in block and "before any ceiling" in block
    assert "- `flashinfer.gemm`: `mm_fp4`" in block
    assert "- `flashinfer.norm`: `gemma_rmsnorm`" in block
    assert "`rmsnorm`" not in block  # the engine already wraps it as a trtllm op


def test_the_workspace_file_names_where_the_engine_uses_each_flashinfer_callable(tmp_path):
    inventory = _flashinfer_inventory()
    path = write_surface_file(tmp_path / "surface.md", read_checkpoint_surface(_dense()), inventory)
    text = path.read_text()
    assert (
        "- `flashinfer.norm.rmsnorm` — referenced in "
        "`tensorrt_llm/_torch/custom_ops/flashinfer_custom_ops.py`"
    ) in text
    assert "- `flashinfer.gemm.mm_fp4` — referenced nowhere in the engine" in text


def test_an_unread_flashinfer_api_is_unmeasured_not_empty():
    inventory = _flashinfer_inventory(flashinfer_api=[], versions={"flashinfer": "not installed"})
    block = format_surface(read_checkpoint_surface(_dense()), inventory)
    assert "FlashInfer's API was not read** (not installed)" in block
    assert "unmeasured, not empty" in block


def test_an_empty_surface_still_renders():
    block = format_surface(CheckpointSurface(architecture="", model_type=""), Inventory())
    assert "Surface survey" in block


def test_receipt_absent_is_the_default_not_a_pass():
    assert CatalogEntry(path="x/y.py").receipt == RECEIPT_ABSENT
    assert not CatalogEntry(path="x/y.py").spendable


# ------------------------------------------------- the probe, actually run


def _entry_source(impl, archs, cells=2):
    """An entry as modeling_v2 states it: the OpWrapper class is the contract."""
    arch_set = ", ".join(f"Arch.{a}" for a in archs)
    cell_list = "".join(f"Cell(why='c{i}', spec={{}}), " for i in range(cells))
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

    The catalog is its entry files: ``entries`` are ``(category/name.py, impl,
    archs)`` written under ``catalog/``, beside the catalog's own ``_op.py``.
    ``model_ops`` are ``(family/target, category/name.py, impl, archs)`` kept in
    a target's ``ops/``.
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


#: Makes `import yaml` raise: the probe reads the entries without it, and this
#: keeps that true.
_BLOCK_YAML = (
    "import sys\n"
    "class _NoYaml:\n"
    "    def find_module(self, name, path=None):\n"
    "        return self if name == 'yaml' else None\n"
    "    def load_module(self, name):\n"
    "        raise ImportError('blocked for test')\n"
    "sys.meta_path.insert(0, _NoYaml())\n"
)


def _run_probe(repo, arch="sm_103", *, block_yaml=False):
    import subprocess
    import sys

    from agent_flow.workflows.trtllm_modeling_v2.onboard.survey import PROBE_SOURCE

    source = (_BLOCK_YAML + PROBE_SOURCE) if block_yaml else PROBE_SOURCE
    proc = subprocess.run(
        [sys.executable, "-c", source, str(repo), arch],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return parse_inventory(proc.stdout + proc.stderr, arch=arch)


#: An entry's certification is its ``ARCHS``: the architectures its cells were
#: driven on.
_INLINE_PASS = ("SM_103",)
_OTHER_ARCH = ("SM_100",)


@pytest.mark.parametrize(
    "archs,expected",
    [
        (_INLINE_PASS, RECEIPT_PASSED),
        (("SM_100", "SM_103"), RECEIPT_PASSED),
        (_OTHER_ARCH, RECEIPT_ABSENT),
        ((), RECEIPT_ABSENT),
    ],
    ids=["this-arch", "two-archs", "other-arch", "none"],
)
@pytest.mark.parametrize("block_yaml", [False, True], ids=["yaml", "no-yaml"])
def test_the_probe_reads_the_entrys_archs(tmp_path, archs, expected, block_yaml):
    repo = _fake_repo(tmp_path, [("norm/x.py", "torch.ops.trtllm.x", archs)])
    inventory = _run_probe(repo, block_yaml=block_yaml)
    assert inventory.catalog[0].receipt == expected
    assert inventory.catalog[0].cells == 2


def test_an_entry_with_no_cells_certifies_nothing_on_any_arch(tmp_path):
    repo = _fake_repo(tmp_path, [("norm/x.py", "torch.ops.trtllm.x", _INLINE_PASS)])
    py = repo / "tensorrt_llm/_torch/_experimental/modeling_v2/catalog/norm/x.py"
    py.write_text(_entry_source("torch.ops.trtllm.x", _INLINE_PASS, cells=0), encoding="utf-8")
    assert _run_probe(repo).catalog[0].receipt == RECEIPT_ABSENT


def test_another_targets_ops_are_promotable_not_spendable(tmp_path):
    """Targets share the catalog and nothing else of each other's."""
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


def test_the_probe_reads_the_entries_from_the_catalog_files(tmp_path):
    """Every `catalog/<category>/<name>.py` is an entry; the files are the catalog.

    The catalog's own modules (`_op.py`, a category's `__init__.py`) are not
    entries, and an entry's summary is its module docstring's first line.
    """
    repo = _fake_repo(
        tmp_path,
        [
            ("norm/x.py", "torch.ops.trtllm.x", _INLINE_PASS),
            ("gemm/y.py", "torch.ops.trtllm.y", _INLINE_PASS),
        ],
    )
    (repo / "tensorrt_llm/_torch/_experimental/modeling_v2/catalog/gemm/y.py").write_text(
        '"""A y GEMM."""\n' + _entry_source("torch.ops.trtllm.y", _INLINE_PASS), encoding="utf-8"
    )
    inventory = _run_probe(repo)
    assert [e.path for e in inventory.catalog] == ["gemm/y.py", "norm/x.py"]
    assert [e.impl for e in inventory.catalog] == ["torch.ops.trtllm.y", "torch.ops.trtllm.x"]
    assert inventory.catalog[0].summary == "A y GEMM."
    assert all(e.receipt == RECEIPT_PASSED for e in inventory.catalog)


def test_a_receipt_for_another_arch_is_not_this_arch(tmp_path):
    """The architecture is the whole key; a pass elsewhere says nothing here."""
    repo = _fake_repo(tmp_path, [("norm/x.py", "torch.ops.trtllm.x", _OTHER_ARCH)])
    assert not _run_probe(repo, arch="sm_103").catalog[0].spendable


def test_the_probe_reads_the_kernel_plane_without_importing_it(tmp_path):
    repo = _fake_repo(tmp_path, [("norm/x.py", "torch.ops.trtllm.x", _INLINE_PASS)])
    inventory = _run_probe(repo)
    assert "rms_norm.RMSNorm" in inventory.modules
    assert not any(m.endswith("._hidden") for m in inventory.modules)


def test_an_engine_that_will_not_import_does_not_condemn_the_whole_catalog(tmp_path):
    """The probe here runs on an interpreter with no engine.

    Without the guard every entry would be reported as rotted, which is the
    most alarming possible way to say "I could not measure this".
    """
    repo = _fake_repo(tmp_path, [("norm/x.py", "torch.ops.trtllm.x", _INLINE_PASS)])
    inventory = _run_probe(repo)
    # This interpreter has no engine, whether or not it has torch — so the
    # op inventory cannot be taken either way, which is the condition.
    assert inventory.error
    assert inventory.rotted == ()
    assert inventory.catalog[0].receipt == RECEIPT_PASSED


# ------------------------------------------------- no category is exempt


def test_no_category_is_spendable_without_certification():
    """Every entry, `torch/` included, is an OpWrapper with cells."""
    entry = CatalogEntry(path="torch/add.py", impl="torch.add")
    assert entry.certifiable
    assert not entry.spendable


def test_an_uncertified_entry_is_listed_as_needing_a_run():
    payload = _inventory_payload(
        catalog=[
            {
                "path": "norm/y.py",
                "impl": "torch.ops.trtllm.y",
                "receipt": "absent",
                "impl_resolves": True,
            },
            {
                "path": "moe/z.py",
                "impl": "torch.ops.trtllm.z",
                "receipt": "passed",
                "impl_resolves": True,
                "scope": "fam/ckpt__sm_103__tp1",
            },
        ]
    )
    inventory = parse_inventory(MARKER + json.dumps(payload), arch="sm_103")
    assert [e.name for e in inventory.uncertified] == ["y"]
    assert [e.name for e in inventory.promotable] == ["z"]
    assert inventory.spendable == ()


def test_a_partial_inventory_is_still_rendered():
    """An import failure still leaves the catalog, which is most of the value."""
    payload = _inventory_payload(trtllm_ops=[], error="ModuleNotFoundError: torch")
    inventory = parse_inventory(MARKER + json.dumps(payload), arch="sm_103")
    block = format_surface(read_checkpoint_surface(_dense()), inventory)
    assert "ModuleNotFoundError" in block
    assert "fused_qk_norm_rope" in block  # measured, so still shown
    assert "nothing can be called rotted" in block


# ------------------------------------------- the linter cannot see this code


def test_the_probe_source_compiles():
    """`ruff` and `ruff format` do not look inside a string literal.

    A `\"\"\"` docstring written inside the `r\"\"\"...\"\"\"` literal terminates
    it and the module stops importing, which a linter catches on a real file.
    `compile()` is the cheapest possible stand-in for the check ruff would run.
    """
    from agent_flow.workflows.trtllm_modeling_v2.onboard.survey import PROBE_SOURCE

    compile(PROBE_SOURCE, "<probe>", "exec")


def test_the_probe_declares_the_marker_the_parser_looks_for():
    """Two literals, in two modules, that must agree."""
    from agent_flow.workflows.trtllm_modeling_v2.onboard.survey import MARKER, PROBE_SOURCE

    assert MARKER in PROBE_SOURCE


# ------------------------------------------------- FlashInfer, read off disk

#: A stand-in FlashInfer install: a re-exporting package, a plain module with
#: its own helper import, a package-style submodule, and infrastructure.
_FAKE_FLASHINFER = {
    "__init__.py": (
        "from .norm import rmsnorm as rmsnorm\n"
        "from .autotuner import autotune as autotune\n"
        "from . import gemm as gemm\n"
        "try:\n"
        "    from .quantization.fp4_quantization import fp4_quantize as fp4_quantize\n"
        "except ImportError:\n"
        "    pass\n"
    ),
    "norm.py": (
        "from .utils import register_custom_op\n\n"
        "def rmsnorm(x, w, eps):\n    return x\n\n"
        "def gemma_rmsnorm(x, w, eps):\n    return x\n\n"
        "def _private(x):\n    return x\n"
    ),
    "gemm/__init__.py": "from .core import mm_fp4 as mm_fp4\n",
    "gemm/core.py": "def mm_fp4(a, b):\n    return a\n\ndef _tile():\n    return 1\n",
    "quantization/__init__.py": "",
    "quantization/fp4_quantization.py": "def fp4_quantize(x):\n    return x\n",
    "autotuner.py": "def autotune():\n    pass\n",
    "utils.py": "def register_custom_op(f):\n    return f\n",
}


def _run_probe_with_flashinfer(tmp_path, repo):
    import subprocess
    import sys

    from agent_flow.workflows.trtllm_modeling_v2.onboard.survey import PROBE_SOURCE

    site = tmp_path / "site"
    for rel, body in _FAKE_FLASHINFER.items():
        path = site / "flashinfer" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    prefix = f"import sys\nsys.path.insert(0, {str(site)!r})\n"
    proc = subprocess.run(
        [sys.executable, "-c", prefix + PROBE_SOURCE, str(repo), "sm_103"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return parse_inventory(proc.stdout + proc.stderr, arch="sm_103")


def test_the_probe_reads_flashinfers_public_api_without_importing_it(tmp_path):
    """No torch, no GPU: the package's files say what it offers."""
    repo = _fake_repo(tmp_path, [("norm/x.py", "torch.ops.trtllm.x", _INLINE_PASS)])
    inventory = _run_probe_with_flashinfer(tmp_path, repo)

    assert inventory.flashinfer_api == (
        "flashinfer.gemm.mm_fp4",
        "flashinfer.norm.gemma_rmsnorm",
        "flashinfer.norm.rmsnorm",
        "flashinfer.quantization.fp4_quantization.fp4_quantize",
    )
    # Infrastructure, private names and a module's own helper imports are not API.
    assert not any(
        "autotune" in p or "_private" in p or "_tile" in p for p in inventory.flashinfer_api
    )
    assert "flashinfer.norm.register_custom_op" not in inventory.flashinfer_api


def test_the_probe_records_which_flashinfer_callables_the_engine_already_calls(tmp_path):
    repo = _fake_repo(tmp_path, [("norm/x.py", "torch.ops.trtllm.x", _INLINE_PASS)])
    custom_ops = repo / "tensorrt_llm/_torch/custom_ops"
    custom_ops.mkdir(parents=True)
    (custom_ops / "flashinfer_custom_ops.py").write_text(
        "import flashinfer\n\ndef op(x, w):\n    return flashinfer.norm.rmsnorm(x, w, 1e-6)\n",
        encoding="utf-8",
    )
    (repo / "tensorrt_llm/_torch/modules/gemm.py").write_text(
        "from flashinfer.gemm import mm_fp4\n", encoding="utf-8"
    )
    inventory = _run_probe_with_flashinfer(tmp_path, repo)

    assert inventory.flashinfer_refs["rmsnorm"] == (
        "tensorrt_llm/_torch/custom_ops/flashinfer_custom_ops.py",
    )
    assert inventory.flashinfer_refs["mm_fp4"] == ("tensorrt_llm/_torch/modules/gemm.py",)
    assert inventory.flashinfer_unreferenced == (
        "flashinfer.norm.gemma_rmsnorm",
        "flashinfer.quantization.fp4_quantization.fp4_quantize",
    )


def test_an_entry_calling_flashinfer_directly_reports_that_callable_as_its_impl(tmp_path):
    repo = _fake_repo(
        tmp_path,
        [],
        model_ops=[("fam/tgt", "norm/gemma.py", "flashinfer.norm.gemma_rmsnorm", _INLINE_PASS)],
    )
    inventory = _run_probe(repo)
    assert inventory.catalog[0].impl == "flashinfer.norm.gemma_rmsnorm"


# ------------------------------------------------- how the op plane is enumerated

#: A stand-in `torch` whose `ops.trtllm` is LAZY, exactly as the real one is:
#: `dir()` lists only what something has already resolved. The schema registry
#: knows the rest: on a real build `dir()` lists about a third of what
#: `_jit_get_all_schemas()` does.
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
    """`torch.ops.<ns>` is a lazy proxy and `dir()` under-reports it badly.

    Read through `dir()`, perfectly good catalog entries such as `allgather`,
    `fp4_quantize`, `fused_qk_norm_rope` and `causal_conv1d_fwd` are reported
    as "Rotted --- do not spend these". A plan drafted against that routes
    around the exact vocabulary a quantized hybrid MoE target needs, and
    dispatches a smith to re-onboard each one.
    """
    repo = _fake_repo(tmp_path, [("comm/allgather.py", "torch.ops.trtllm.allgather", _INLINE_PASS)])
    inventory = _run_probe_with_fake_torch(repo)

    assert "allgather" in inventory.trtllm_ops
    assert "fp4_quantize" in inventory.trtllm_ops
    # And the lazily-resolved names are still there, so the registry is a
    # superset rather than a replacement.
    assert "already_resolved_a" in inventory.trtllm_ops


def test_ops_from_other_namespaces_are_not_claimed_as_trtllms(tmp_path):
    repo = _fake_repo(tmp_path, [("comm/allgather.py", "torch.ops.trtllm.allgather", _INLINE_PASS)])
    assert "add" not in _run_probe_with_fake_torch(repo).trtllm_ops


def test_an_entry_whose_op_is_registered_is_not_reported_as_rotted(tmp_path):
    """The consequence, stated as the thing a plan reads."""
    repo = _fake_repo(tmp_path, [("comm/allgather.py", "torch.ops.trtllm.allgather", _INLINE_PASS)])
    entry = _run_probe_with_fake_torch(repo).catalog[0]
    assert entry.impl_resolves
    assert entry.spendable
