# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nothing a worker reads, and no anchor, names a place in the engine.

The engine's layout moves. What the workflow reads off it is found by what a file
defines (``operators/tree_anchors.py``), and what a worker is told is behaviour, symbols
and searches to run. The one place the workflow does name is its subject: the ModelingV2
package it adds a target to (``coverage.SUBJECT_PATHS``).
"""

from __future__ import annotations

import dataclasses
import re
import subprocess
from pathlib import Path

import yaml

from agent_flow.workflows.trtllm_modeling_v2.agents import decisions, prompts, roles
from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import card as card_mod
from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import kernel as kernel_prompts
from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import paths as paths_mod
from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import survey as survey_prompts
from agent_flow.workflows.trtllm_modeling_v2.agents.prompts.tags import TAG_PROMPTS
from agent_flow.workflows.trtllm_modeling_v2.infra import git as git_mod
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.operators import (
    coverage,
    inventory,
    prior_art,
    tree_anchors,
)
from agent_flow.workflows.trtllm_modeling_v2.operators import rows as rows_mod
from agent_flow.workflows.trtllm_modeling_v2.run import graph as graph_mod
from agent_flow.workflows.trtllm_modeling_v2.run.execution import handoff
from agent_flow.workflows.trtllm_modeling_v2.run.execution import tasks as tasks_mod
from agent_flow.workflows.trtllm_modeling_v2.spec import advice

# One file per anchor, each holding what the anchor's definition looks for and nothing
# else an anchor could take for its own. The directories and file names are invented:
# no anchor could know them.
_FIXTURE: dict[str, tuple[str, str]] = {
    "modeling_v2_guide": ("handbook/python/guide.md", "<!-- license -->\n# Modeling V2\n\nText.\n"),
    "attention_backends": (
        "engine/attn/sparse_gqa.py",
        "class SparseGqaBackend(AttentionBackend):\n    pass\n",
    ),
    "custom_ops": (
        "engine/registered/rope.py",
        '@torch.library.custom_op("trtllm::rope_rows", mutates_args=())\ndef rope_rows(x):\n    return x\n',
    ),
    "kernels": (
        "engine/gpu/topk_select.py",
        "import triton\n\n\n@triton.jit\ndef topk_select(x):\n    pass\n",
    ),
    "moe_modules": ("engine/experts/fused.py", "class FusedExperts(MoE):\n    pass\n"),
    "builtin_models": (
        "engine/zoo/toy.py",
        '@register_auto_model("ToyForCausalLM")\nclass Toy:\n    pass\n',
    ),
    "kv_cache_dtypes": (
        "engine/executor/loading.py",
        '_VALID_KV_CACHE_DTYPES = ("fp8", "nvfp4", "auto")\n',
    ),
    "fmha_head_dims": (
        "native/attn/fmha_kernels.h",
        'return {false, "HeadDimV must be 32, 64, 128, or a multiple of 128"};\n',
    ),
    "unit_tests": ("checks/unit/pytest.ini", "[pytest]\nmarkers =\n    cpu_only: needs no GPU\n"),
    "accuracy_tests": (
        "checks/scored/test_toy_accuracy.py",
        "class TestToy(LlmapiAccuracyTestHarness):\n    MODEL_NAME = 'org/toy'\n",
    ),
    "accuracy_references": (
        "checks/scored/scores/arithmetic.yaml",
        "org/toy:\n  - accuracy: 41.77\n  - quant_algo: FP8\n    accuracy: 40.1\n",
    ),
    "test_lists": (
        "checks/lists/premerge_one_gpu.yml",
        "version: 0.0.1\npremerge_one_gpu:\n- condition:\n    terms:\n      stage: pre_merge\n  tests:\n  - x\n",
    ),
}


def _plant(repo: Path, layout: dict[str, tuple[str, str]]) -> None:
    for path, text in layout.values():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text, encoding="utf-8")


def _moved(path: str) -> str:
    """``path`` after the tree moved: every directory renamed, the file too where its kind allows."""
    *dirs, name = path.split("/")
    stem, dot, suffix = name.rpartition(".")
    # A tool finds `pytest.ini` by that name, and pytest collects `test_*.py`: both stay.
    if name != "pytest.ini":
        name = f"{stem}_renamed{dot}{suffix}"
    return "/".join(["relocated", *(f"{d}_v2" for d in dirs), name])


def _commit_all(repo: Path) -> None:
    for args in (["init", "-q"], ["add", "-A"]):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.test"]
        + ["commit", "-q", "-m", "fixture"],
        check=True,
        capture_output=True,
    )


def test_every_anchor_is_a_definition_and_is_found_again_in_a_tree_that_moved(tmp_path):
    """The moved-file fixture: every directory renamed, every file renamed, every anchor found."""
    fields = {f.name for f in dataclasses.fields(tree_anchors.Anchor)}
    assert not fields & {"hint", "roots", "dirname", "path", "paths"}
    for key, anchor in tree_anchors.ANCHORS.items():
        assert anchor.marker and anchor.literals and anchor.search, key
        # A file-name pattern, never a path; and the search names what to look for, not where.
        assert "/" not in anchor.name and "/" not in anchor.search, key
    assert set(_FIXTURE) == set(tree_anchors.ANCHORS)
    after = tmp_path / "after"
    layout = {key: (_moved(path), text) for key, (path, text) in _FIXTURE.items()}
    _plant(after, layout)
    moved = tree_anchors.resolve_all(after)
    assert {key: one.paths for key, one in moved.items()} == {
        key: (path,) for key, (path, _) in layout.items()
    }
    assert all(one.via == "definition" and one.line for one in moved.values())
    # What code reads through the anchors is read from the moved tree as well.
    config = {"architectures": ["ToyForCausalLM"], "head_dim": 192, "num_attention_heads": 8}
    assert prior_art.engine_model(after, config).cls == "Toy"
    assert prior_art.engine_kv_dtypes(after) == (
        ("fp8", "nvfp4", "auto"),
        f"{layout['kv_cache_dtypes'][0]}:1",
    )
    listed, multiple, where = prior_art.fmha_rule(after)
    assert (listed, multiple) == ((32, 64, 128), 128) and where.startswith("relocated/")
    assert "192" in prior_art.engine_constraints(config, after)[0]
    # A definition no file holds is missing, and never a remembered place.
    (after / layout["kv_cache_dtypes"][0]).write_text("KV_DTYPES = ()\n")  # the engine renamed it
    found = tree_anchors.resolve_all(after)
    assert found["kv_cache_dtypes"].via == "missing" and not found["kv_cache_dtypes"].paths
    assert prior_art.engine_kv_dtypes(after) is None
    assert tree_anchors.resolve(None, "kernels").via == "missing"
    written = tree_anchors.render_map(after, list(found.values()))
    assert "the KV-cache dtypes the PyTorch loader accepts**: missing. Searched for: " in written


def test_a_test_file_or_the_workflows_own_copy_is_never_taken_for_the_engine(tmp_path):
    """A definition quoted by a test, or by this workflow's prompts in the checkout, is not the engine's."""
    repo = tmp_path / "repo"
    _plant(repo, _FIXTURE)
    quoted = "class Quoted(AttentionBackend):\n    pass\n"
    own = "agent_flow/workflows/trtllm_modeling_v2"
    for stray in (
        "checks/unit/test_backend_double.py",
        # A copy of this workflow beside its own tests, and one at the top of the checkout.
        f"tools/{own}/agents/prompts/example.py",
        "tools/tests/workflows/fixture_backend.py",
        f"{own}/notes.py",
    ):
        (repo / stray).parent.mkdir(parents=True, exist_ok=True)
        (repo / stray).write_text(quoted, encoding="utf-8")
    assert tree_anchors.resolve(repo, "attention_backends").paths == (
        _FIXTURE["attention_backends"][0],
    )
    # A directory of the engine that only shares the distribution's name is the engine's.
    (repo / "engine/agent_flow").mkdir()
    (repo / "engine/agent_flow/routed.py").write_text(quoted, encoding="utf-8")
    assert tree_anchors.resolve(repo, "attention_backends").paths == (
        "engine/agent_flow/routed.py",
        _FIXTURE["attention_backends"][0],
    )
    (repo / "engine/agent_flow/routed.py").unlink()
    # In a git checkout the same files are found, through git's own search.
    _commit_all(repo)
    tracked = tree_anchors.resolve_all(repo)
    assert {key: one.paths for key, one in tracked.items()} == {
        key: (path,) for key, (path, _) in _FIXTURE.items()
    }
    # The map says what each definition found and at which commit; many files as a count.
    for n in range(3):
        (repo / f"engine/gpu/k{n}.py").write_text(f"@cute.kernel\ndef k{n}(x):\n    pass\n")
    (repo / "elsewhere.py").write_text("import triton\n")  # imports a library: defines no kernel
    _commit_all(repo)
    found = tree_anchors.resolve_all(repo)
    written = tree_anchors.render_map(repo, list(found.values()))
    assert found["kernels"].commit and f"at commit `{found['kernels'].commit}`" in written
    assert "Nothing here is a fixed place" in written
    assert (
        "- **Python kernels (Triton, CuTe DSL)**: 4 files, nearly all under `engine/gpu`."
        in written
    )


#: Every line of the flow's code (the files below) that names a directory of the engine,
#: by file, with why it stays. A line more than these fails the case under it.
_KEPT_IN_CODE: dict[str, tuple[int, str]] = {
    "operators/coverage.py": (2, "the subject: where a run writes (SUBJECT_PATHS)"),
    "operators/rows.py": (1, "which of a row's files are tests: the coordinator's, by directory"),
}


def test_the_subject_is_named_in_one_place_and_the_flows_code_names_no_other_directory():
    """The ModelingV2 package and its tests are where a run writes: the only places named."""
    assert coverage.SUBJECT_PATHS == (coverage.MODELING_V2_PACKAGE, coverage.CATALOG_TESTS_DIR)
    package = Path(coverage.__file__).resolve().parents[1]
    # The files the path tables of the redesign listed, and the Python that fills prompts.
    listed = [
        package / "__init__.py",
        *sorted((package / "operators").glob("*.py")),
        *sorted((package / "agents").rglob("*.py")),
        package / "infra/git.py",
        package / "run/workflow.py",
        package / "run/startup/bootstrap.py",
        package / "run/execution/kernel_task.py",
    ]
    engine_dir = re.compile(r"""["'`( ](?:tensorrt_llm|tests|cpp|examples|docs|scripts)/""")
    found = {}
    for path in listed:
        hits = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if engine_dir.search(line)
        ]
        if hits:
            found[path.relative_to(package).as_posix()] = hits
    assert {name: len(hits) for name, hits in found.items()} == {
        name: count for name, (count, _) in _KEPT_IN_CODE.items()
    }, found
    # The probe that runs under the engine's interpreter is told both by its caller.
    assert "_torch" not in inventory.PROBE_SOURCE
    assert "modeling_v2" not in inventory.PROBE_SOURCE


def test_no_directory_of_the_engine_is_where_a_run_may_commit_by_default(tmp_path):
    """A spec that states no list allows the repository: a task's write scope is what bounds it."""
    repo = tmp_path / "repo"
    for path in ("native/core/CMakeLists.txt", "handbook/guide.md", "samples/run.py"):
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text("x\n", encoding="utf-8")
    _commit_all(repo)
    workspace = repo / "runs/ws"
    policy = git_mod.CommitPolicy.from_task({"trtllm_repo_path": str(repo)}, workspace=workspace)
    assert policy.allowed_paths == ()
    for path in ("native/core/kernel.cu", "samples/run.py", "handbook/guide.md"):
        assert policy.path_problem(path) == "", path
    # What is always refused stays refused: the run's own records.
    assert policy.path_problem("runs/ws/journal.md")
    # A worker is told its write scope, not "anywhere".
    said = git_mod.commit_instruction(policy)
    assert "Commit only the files your card's write scope names" in said
    assert "anywhere in the repository" not in said
    # A spec that wants a fence states it, and a worker is told it.
    stated = git_mod.CommitPolicy.from_task(
        {"trtllm_repo_path": str(repo), "commit_policy": {"allowed_paths": ["samples"]}}
    )
    assert stated.allowed_paths == ("samples/",)
    assert "outside the paths this run may commit" in stated.path_problem("handbook/guide.md")
    assert "Commit only under `samples/`" in git_mod.commit_instruction(stated)
    # The task list's own check of a row's files reads the same: no list, nothing outside.
    assert graph_mod.inside("samples/run.py", policy.allowed_paths)


def test_the_probe_imports_the_modules_that_register_ops_wherever_they_are(tmp_path):
    repo = tmp_path / "repo"
    _plant(repo, _FIXTURE)
    home = repo / "engine/registered"
    for name in ("norm", "gemm"):
        (home / f"{name}.py").write_text(
            f'@torch.library.custom_op("trtllm::{name}", mutates_args=())\ndef {name}(x):\n    return x\n'
        )
    assert inventory.registering_modules(repo) == ("engine.registered",)
    assert inventory.registering_modules(tmp_path / "nowhere") == ()

    class Environment:
        def python_command(self, argv, single_process=False):
            return argv

    argv = inventory.probe_command(Environment(), repo=repo, arch="sm_103")
    assert argv[2:] == [str(repo), "sm_103", coverage.MODELING_V2_PACKAGE, "engine.registered"]


# ------------------------------------------------------------------ what a worker reads

_PACKAGE = Path(coverage.__file__).resolve().parents[1]
_TARGET = f"{coverage.MODELING_V2_PACKAGE}/models/toy/toy_1b__sm_103__unified"
_TESTS = f"{coverage.CATALOG_TESTS_DIR}/models/toy/toy_1b__sm_103__unified"


def _static_texts() -> dict[str, str]:
    """Every text a worker reads that is written ahead of a run: templates, skills, sections."""
    texts = {}
    for path in sorted((_PACKAGE / "agents/prompts/templates").rglob("*.j2")):
        texts[f"template {path.name}"] = path.read_text(encoding="utf-8")
    for path in sorted(Path(roles.PACKAGED_SKILLS_DIR).rglob("*.md")):
        texts[f"skill {path.parent.name}/{path.name}"] = path.read_text(encoding="utf-8")
    for role, text in dataclasses.asdict(prompts.DEFAULT_PROMPTS).items():
        texts[f"system prompt {role}"] = text
    for tag, text in TAG_PROMPTS.items():
        texts[f"section {tag}"] = text
    for role in decisions.WORKER_ROLES:
        texts[f"ending {role}"] = decisions.ending_section(role)
    for world in (1, 4):
        environment = RunEnvironment(kind="slurm", python="/p", world_size=world, spool="/s")
        texts[f"pool text, {world} rank(s)"] = environment.describe()
    texts["capture lists"] = survey_prompts.capture_lists_block({})
    texts["card reference defaults"] = "\n".join(card_mod.REFERENCE.values())
    texts["scope note"] = card_mod.scope_note(["a/b"], ["c/d"])
    return texts


def _task(row: rows_mod.Row, kind: str, touches: list[str], **more) -> graph_mod.Task:
    return graph_mod.Task(
        id=row.id if not row.id.isdigit() else f"k{row.id}",
        kind=kind,
        goal=row.outcome,
        depends_on=["survey"],
        touches=touches,
        ref=row.id,
        check=row.check,
        **more,
    )


def _card(task: graph_mod.Task, row: rows_mod.Row, **more) -> str:
    command = tasks_mod.check_command(
        task.check,
        task_path="/ws/task.yaml",
        tree="/trees/mine",
        tests=rows_mod.test_files(task.touches),
        workdir="/ws/gates/mine",
    )
    return card_mod.render_card(
        task, row, command=command, notes="/ws/operator_notes.md", spec="/ws/task.yaml", **more
    )


def _cards() -> dict[str, tuple[graph_mod.Task, rows_mod.Row, str]]:
    """A kernel row and a weights row, as a survey writes them."""
    cell = {
        "use_paged_context_fmha": True, "num_heads": 16, "num_kv_heads": 8, "head_size": 128,
        "batch": 1, "tokens": 512, "cached": 0, "kv_dtype": "bf16", "blocks": 64, "phase": "prefill",
    }  # fmt: skip
    cells = [
        cell,
        {**cell, "cached": 4096, "phase": "decode"},
        {**cell, "batch": 64, "blocks": 10240},
    ]
    kernel = rows_mod.Row(
        id="3",
        kind=rows_mod.OPERATOR,
        outcome="certify the entry `attention/qk_norm_rope` at this checkpoint's values on this "
        "architecture, with a GPU test that drives the cells it claims",
        config_axes={"head_dim": 128, "norm_eps": 1e-6, "dtype": "bf16"},
        leads=(
            "tensorrt_llm/_torch/attention/backends/fmha/qk_rope.py (QkNormRope): same fused "
            "normalization and rotation",
            "How does the backend receive the cached length?",
        ),
        required=tuple(", ".join(f"{k}: {v}" for k, v in c.items()) for c in cells),
        optional=("one FP8 KV cell",),
        check="kernel:3",
        gpus=1,
    )
    weights = rows_mod.Row(
        id="toy-weights",
        kind=rows_mod.WEIGHTS,
        outcome="every checkpoint tensor is loaded, transformed, shared out or deliberately skipped",
        config_axes={"quant": "none", "layers": 28},
        leads=("the loader in tensorrt_llm/_torch/models/modeling_qwen3.py: the same layer names",),
        required=("every production key is accounted for", "the vision keys are skipped by name"),
        optional=("an unrequested quantized export",),
        check="cpu:toy-weights",
    )
    made = {}
    scopes = {
        "a kernel row": ["ops/attention/qk__r3.py", "attention/test_modeling_v2_qk__r3.py"],
        "a weights row": ["weights.py", "test_modeling_v2_weights.py"],
    }
    for name, row, kind in (
        ("a kernel row", kernel, "kernel"),
        ("a weights row", weights, "weights"),
    ):
        module, test = scopes[name]
        touches = [f"{_TARGET}/{module}", f"{_TESTS}/{test}"]
        task = _task(row, kind, touches)
        made[name] = (task, row, _card(task, row))
    return made


#: The flow's own directories: a document's code map names its modules, which are not the engine's.
_FLOW_DIRS = ("agents", "checks", "infra", "operators", "run", "spec", "prompts")


def test_no_text_a_worker_or_a_reader_is_given_names_a_place_in_the_engine():
    """The rule, over every text: a prompt, a skill, a card or a document that gains an engine path fails.

    What is kept is listed in one place (``agents/prompts/paths.py``): an entry's name
    inside the subject package, and a run's own files.
    """
    texts = _static_texts()
    assert len(texts) > 30
    found = {name: hits for name, text in texts.items() if (hits := paths_mod.engine_paths(text))}
    assert not found, found
    # The subject itself is not named in them either: a worker's write scope says it.
    for name, text in texts.items():
        assert not [s for s in coverage.SUBJECT_PATHS if s in text], name
    # What a run fills in names only where the run writes: its system prompts, a kernel
    # dispatch, a card.
    policy = git_mod.CommitPolicy(allowed_paths=tuple(f"{p}/" for p in coverage.SUBJECT_PATHS))
    environment = RunEnvironment(kind="slurm", python="/p", world_size=1, spool="/s")
    bundle = prompts.build_trtllm_modeling_v2_prompts(
        environment=environment, scope="text_only", commit_policy=policy, notes="Architecture: toy."
    )
    for role, text in dataclasses.asdict(bundle).items():
        assert paths_mod.engine_paths(text, allowed=policy.allowed_paths) == [], role
    ctx = kernel_prompts.KernelContext(
        item=coverage.QueueItem(4, "verify", "attention/sparse_gqa", surface="bf16", note="rows"),
        workspace="/ws",
        repo="/repo",
        arch="sm_103",
        target_rel=_TARGET,
        vocabulary_path="/ws/vocabulary.yaml",
        note="",
        goal="## Your task: `k4` (kernel, attempt 1 of 2)",
        task_id="k4",
        task_kind="kernel",
        landing="ops/attention/sparse_gqa__r4",
        paths=(f"{_TARGET}/ops/attention/sparse_gqa__r4.py", f"{_TESTS}/attention/test_x.py"),
        shared=coverage.shared_entry_paths("attention/sparse_gqa"),
    )
    dispatch = kernel_prompts.kernel_prompt(ctx)
    assert paths_mod.engine_paths(dispatch, allowed=coverage.SUBJECT_PATHS) == []
    for name, (task, _, text) in _cards().items():
        assert paths_mod.engine_paths(text, allowed=task.touches) == [], name
    # A dependency's receipt hands on what it left, not the places it read: in two whole
    # runs the survey's receipt put seven engine paths into every later dispatch.
    receipt = {
        "task": "survey",
        "kind": "survey",
        "decision": "ready",
        "summary": "mapped 14 steps",
        "references": [
            "tensorrt_llm/_torch/models/modeling_toy.py (ToyForCausalLM): the layer names",
            "tests/integration/defs/accuracy/references/gsm8k.yaml: the anchor row",
        ],
        "not_done": ["the FP8 cells of the attention row: not measured"],
        "decision_file": "/ws/decisions/plan-surveyor-1.yaml",
    }
    handed = handoff.render([receipt])
    assert "- it lists 2 reference(s) it read, in its file" in handed
    assert "- not done: the FP8 cells of the attention row: not measured" in handed
    assert paths_mod.engine_paths(handed) == []
    # The documents: no place in the engine, links that resolve, and no site in the example.
    names = ("README.md", "DESIGN.md", "OPERATING.md", "spec/INTAKE.md", "task.example.yaml")
    docs = {name: (_PACKAGE / name).read_text(encoding="utf-8") for name in names}
    for name, text in docs.items():
        assert paths_mod.engine_paths(text, allowed=_FLOW_DIRS) == [], name
        for target in re.findall(r"\]\(([\w./-]+\.(?:md|yaml))\)", text):
            assert (_PACKAGE / name).parent.joinpath(target).is_file(), (name, target)
    example = docs["task.example.yaml"]
    spec = yaml.safe_load(example)
    for key in ("checkpoint_path", "trtllm_repo_path"):
        assert spec[key].startswith("/path/to/"), key
    assert spec["environment"]["python"].startswith("/path/to/")
    for placeholder in ("account: <account>", "partition: <partition>", "qos: <qos>",
                        "container_image: /path/to/", "model_key: <org>/<model>"):  # fmt: skip
        assert placeholder in example, placeholder
    # The scanner sees what it is for, however a place is wrapped, and leaves what is kept.
    planted = (
        "Read `tensorrt_llm/_torch/modules/linear.py`, then imitate `tests/unittest/x/test_y.py`."
    )
    assert paths_mod.engine_paths(planted) == [
        "tensorrt_llm/_torch/modules/linear.py",
        "tests/unittest/x/test_y.py",
    ]
    for hidden in (
        "`_torch/pyexecutor/`",
        "**tensorrt_llm/_torch/modules/attention.py**",
        "<repo>/tensorrt_llm/_torch/attention/backends",
        "./cpp/tensorrt_llm/kernels",
        "${REPO}/tensorrt_llm/_torch/x.py",
        "x=tests/unittest/y.py",
    ):
        assert len(paths_mod.engine_paths(f"read {hidden} now")) == 1, hidden
    for kept_form in (
        "entry: catalog/<category>/<name>.py",
        "`ops/norm/rmsnorm__r4.py`",
        "<workspace>/decisions/<name>-surveyor-<turn>.yaml",
        "/abs/checkout/tensorrt_llm/x.py",
        "`task.yaml`, `plan.md`, `tree_map.md`, prefill/decode, TP/EP",
    ):
        assert paths_mod.engine_paths(kept_form) == [], kept_form


def test_a_card_says_what_its_task_is_for_what_to_hand_back_and_how_it_is_judged():
    """A kernel row and a weights row: each card under 600 words."""
    cards = _cards()
    for name, (task, row, text) in cards.items():
        assert len(text.split()) < card_mod.CARD_WORDS, f"{name}: {len(text.split())} words"
        # What it is for, what to hand back, how it is judged.
        for part in ("**What it is for.**", "**Hand back.**", "**How it is judged.**"):
            assert part in text, (name, part)
    kernel = cards["a kernel row"][2]
    # A lead somebody wrote as a place reaches the card as the name to search for.
    assert "- lead: qk_rope (QkNormRope): same fused normalization and rotation" in kernel
    weights = cards["a weights row"][2]
    assert "- lead: the loader in modeling_qwen3: the same layer names" in weights
    # A row of the target ends at its own check: the assembly that follows wires it in. A
    # whole run's row did its files, passed, and blocked asking for the model's file.
    wired = "the assembly that follows runs after your row and wires what you wrote into the model"
    assert wired in " ".join(weights.split()) and wired not in " ".join(kernel.split())
    task, row, _ = cards["a weights row"]
    late = dataclasses.replace(task, depends_on=["assemble-tp1"])
    assert "runs after your row" not in _card(late, row)
    further = dataclasses.replace(row, deployment="kv_fp8")
    assert "the assembly of `kv_fp8` runs after your row" in " ".join(_card(task, further).split())
    # Said only of a scope that leaves the model's file out. The state row code makes for a
    # disaggregated deployment has the whole target, and its card said the files were not in it.
    whole = dataclasses.replace(task, touches=[_TARGET, _TESTS])
    owns = dataclasses.replace(task, touches=[f"{_TARGET}/modeling.py", task.touches[1]])
    assert not any("not in it on purpose" in _card(t, row) for t in (whole, owns))
    # A row with no check says what judges it instead: the delivery's task has none, and
    # its worker read the flow's source to learn how its files are judged.
    unchecked = _card(dataclasses.replace(task, check=""), row)
    assert "**How it is judged.** By no command of yours, and `ready` needs none." in unchecked
    assert "**Required** (each must hold before you write `ready`):" in unchecked
    # A task's own notes have one place, beside its envelope, which every attempt is told:
    # a survey of a large reference lost all it had read to seven compactions and two cuts.
    assert "Yours, to read first and keep: `/ws/tasks/toy-weights.notes.md`" in weights
    # A row with no lead of its own is still told what kind of reference to look for.
    assert card_mod.REFERENCE["weights"] in cards["a weights row"][2]
    # Cells are shown once whole, then by what differs.
    assert "- use_paged_context_fmha: True, num_heads: 16" in kernel
    assert "- as the first cell, but cached: 4096, phase: decode" in kernel
    assert "- as the first cell, but batch: 64, blocks: 10240" in kernel


def test_a_row_too_long_for_a_card_is_cut_by_code_and_the_card_says_where_the_whole_is():
    """Ten cells of twenty-five values each is one card; so is a row with everything too long."""
    cell = {f"axis_{n}": n * 7 for n in range(25)}
    cells = [{**cell, "axis_3": n, "axis_9": f"value-{n}"} for n in range(10)]
    required = tuple(", ".join(f"{k}: {v}" for k, v in c.items()) for c in cells)
    touches = [f"{_TARGET}/ops/attention/thop__r5.py", f"{_TESTS}/attention/test_x.py"]
    # What a real survey wrote for one attention entry: 533 words of cells on the old card.
    real = rows_mod.Row(
        id="5",
        kind=rows_mod.OPERATOR,
        outcome="certify the entry `attention/thop_attention` at this checkpoint's values",
        leads=("How does the backend receive the cached length?",),
        required=required,
        check="kernel:5",
        gpus=1,
    )
    assert sum(len(line.split()) for line in required) > 450
    text = _card(_task(real, "kernel", touches), real)
    assert len(text.split()) < card_mod.CARD_WORDS
    # The first cell, then each other one by what differs: nothing a worker needs is lost.
    assert "- axis_0: 0, axis_1: 7, axis_2: 14" in text
    for n in range(1, card_mod.CARD_ITEMS):
        assert f"- as the first cell, but axis_3: {n}, axis_9: value-{n}" in text
    assert "- and 2 more" in text
    assert "**Cut to fit.**" in text and "row `5` of `/ws/work.yaml`" in text
    # Everything too long at once: still one card, and it says where the whole is.
    row = rows_mod.Row(
        id="5",
        kind=rows_mod.OPERATOR,
        outcome="certify the entry `attention/thop_attention` " + "at every value " * 40,
        config_axes={f"key_{n}": "v" * 12 for n in range(30)},
        leads=tuple("a lead that " + "may well fit " * 25 for _ in range(9)),
        required=required,
        optional=tuple("an optional cell " * 20 for _ in range(9)),
        check="kernel:5",
        gpus=1,
    )
    task = _task(row, "kernel", touches, attempts=2, last="the check failed: " + "x " * 200)
    text = _card(task, row, skipped=["`k1`"])
    assert len(text.split()) < card_mod.CARD_WORDS
    assert "**Cut to fit.**" in text and "row `5` of `/ws/work.yaml`" in text
    for part in ("**Required**", "**Optional**", "**Reference code**", "A receipt: `ready` carries",
                 "**How it is judged.** By the check `kernel:5`", "**Your previous attempt.**",
                 "**Skipped before you.**"):  # fmt: skip
        assert part in text
    # A short row is cut nowhere and says nothing about cutting.
    assert "**Cut to fit.**" not in _cards()["a weights row"][2]


def test_required_lines_are_the_last_thing_a_card_cuts():
    """Leads, questions and optional lines give way first: a real row once lost its required ones.

    Rendering the rows of two whole runs showed a card that kept its bound by cutting each
    required line to ten words while the row's required lines were ninety-four words in all.
    """
    required = tuple(
        f"requirement {n}: " + " ".join(f"word{n}_{i}" for i in range(28)) for n in range(4)
    )
    row = rows_mod.Row(
        id="toy-weights",
        kind=rows_mod.WEIGHTS,
        outcome="every checkpoint tensor is loaded or named as intentionally skipped " * 3,
        config_axes={f"axis_{n}": f"value_{n}" for n in range(24)},
        leads=tuple("a loader that " + "may well fit " * 20 for _ in range(5)),
        required=required,
        optional=tuple("an optional export " * 15 for _ in range(4)),
        check="cpu:toy-weights",
    )
    task = _task(row, "weights", [f"{_TARGET}/weights.py", f"{_TESTS}/test_modeling_v2_w.py"])
    text = _card(task, row)
    assert len(text.split()) < card_mod.CARD_WORDS
    for line in required:
        assert f"- {line}" in text
    # What gave way is said, and the row's file has it whole.
    assert "**Cut to fit.**" in text and "- lead: a loader that may well fit" in text
    assert text.count("treat this key") < 5 * 20


def test_a_place_becomes_a_name_without_damage_to_what_stands_around_it():
    """What a lead written as a place reaches a card as: bold markup, brackets, a test's node id, a pattern."""
    cases = {
        "see **tensorrt_llm/_torch/modules/attention.py** first": "see **attention** first",
        "[tensorrt_llm/_torch/a.py, tests/unittest/b.py]": "[a, b]",
        "tests/unittest/x/test_y.py::TestA::test_b passes": "test_y::TestA::test_b passes",
        "(see cpp/tensorrt_llm/kernels/fmha.h).": "(see fmha).",
        # A run's placeholder, an entry's name, an absolute path: none is a place to rewrite.
        "<your tree>/ops/attention/foo.py": "<your tree>/ops/attention/foo.py",
        "entry: catalog/<category>/<name>.py": "entry: catalog/<category>/<name>.py",
        "/abs/checkout/tensorrt_llm/x.py": "/abs/checkout/tensorrt_llm/x.py",
        # Two plain words with a slash are prose; a pattern keeps its last component.
        "1000 examples/sec at tests/gates": "1000 examples/sec at tests/gates",
        "the files models/*.py": "the files *.py",
    }
    for text, expected in cases.items():
        assert paths_mod.as_search(text) == expected, text


#: A zoo whose toy model is built from two modules of its own and from what every model uses.
_ZOO = {
    "engine/zoo/base.py": "def register_auto_model(name):\n    return lambda cls: cls\n\n"
    "class DecoderModelForCausalLM:\n    pass\n",
    "engine/layers/hashed.py": "class HashedEmbedding:\n    pass\n\ndef hashed_gate(x):\n    return x\n",
    "engine/layers/norm.py": "class RMSNorm:\n    pass\n",
    "engine/zoo/toy.py": "import torch\nfrom typing import Any\n\n"
    "from ..layers.hashed import HashedEmbedding, hashed_gate\n"
    "from ..layers.norm import RMSNorm\n"
    "from .base import DecoderModelForCausalLM, register_auto_model\n\n"
    "class ToyDecoderLayer:\n    pass\n\nclass ToyWeightLoader:\n    pass\n\n"
    "class _Private:\n    pass\n\n"
    '@register_auto_model("ToyForCausalLM")\n'
    "class ToyForCausalLM(DecoderModelForCausalLM):\n    pass\n",
    **{
        f"engine/zoo/other_{n}.py": "from ..layers.norm import RMSNorm\n"
        "from .base import DecoderModelForCausalLM, register_auto_model\n\n"
        f'@register_auto_model("Other{n}ForCausalLM")\n'
        f"class Other{n}ForCausalLM(DecoderModelForCausalLM):\n    pass\n"
        for n in (1, 2, 3)
    },
}


def test_the_engines_own_model_is_found_by_its_registration_and_named_first(tmp_path):
    """A large model the engine already serves: 106 leads were ranked and none was its own model."""
    repo = tmp_path / "repo"
    _plant(repo, _FIXTURE)
    for path, text in _ZOO.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text, encoding="utf-8")
    config = {"architectures": ["ToyForCausalLM"], "sparse_region": True, "topk_indexer": 8}
    found = prior_art.find_prior_art(repo, config)
    model = found.engine_model
    assert found.registered is True and found.architectures == ("ToyForCausalLM",)
    # The class by its registration, and what its module defines.
    assert (model.architecture, model.cls, model.path) == (
        "ToyForCausalLM", "ToyForCausalLM", "engine/zoo/toy.py",
    )  # fmt: skip
    assert model.defines == ("ToyForCausalLM", "ToyDecoderLayer", "ToyWeightLoader")
    # It is what the intake report and the survey's dispatch say of the tree, and all they say.
    for text in (
        advice.Advice(prior_art=found).render(),
        prior_art.survey_block(found, ["a limit"]),
    ):
        assert text.count("**The engine registers a model for `ToyForCausalLM`**") == 1
        assert "- its module also defines: `ToyDecoderLayer`, `ToyWeightLoader`" in text
        assert "shares" not in text and "ranked" not in text.lower()
    # Kept as data, and a card that has room names it first.
    workspace = tmp_path / "ws"
    workspace.mkdir()
    prior_art.save(found, workspace / prior_art.PRIOR_ART_DATA)
    assert prior_art.load(workspace / prior_art.PRIOR_ART_DATA) == model
    row = rows_mod.Row(
        id="hashed-embedding",
        kind=rows_mod.FEATURE,
        outcome="the hashed embedding lookup of the checkpoint",
        required=("a hashed lookup matches the reference",),
        check="cpu:hashed-embedding",
    )
    task = _task(row, "feature", [f"{_TARGET}/hashed.py"])
    text = card_mod.render_card(
        task, row, command="gate cpu", notes=workspace / "operator_notes.md", spec="/ws/t.yaml"
    )
    assert (
        "- `ToyForCausalLM` (the engine's own implementation of this architecture; search "
        '`register_auto_model("ToyForCausalLM")`): copy what you need from it into the '
        "target; do not import it. Compare against it\n"
    ) in text
    assert paths_mod.engine_paths(text, allowed=task.touches) == []
    # A card that has no room for leads leaves it out; the required lines are not cut for it.
    long = dataclasses.replace(
        row, required=tuple(f"requirement {n}: " + "word " * 60 for n in range(8))
    )
    tight = card_mod.render_card(
        task, long, command="gate cpu", notes=workspace / "operator_notes.md", spec="/ws/t.yaml"
    )
    assert "the engine's own implementation" not in tight
    assert len(tight.split()) < card_mod.CARD_WORDS
    # An architecture the engine registers no model for: said, and nothing is offered to copy.
    none = prior_art.find_prior_art(repo, {**config, "architectures": ["NewForCausalLM"]})
    assert none.engine_model is None and none.registered is False
    assert none.reference().startswith("**The engine registers no model for `NewForCausalLM`**")
    assert prior_art.reference_for(None) == ""
    # A checkpoint that states no architecture: nothing is known, and nothing is said.
    assert prior_art.find_prior_art(repo, {"sparse_region": True}).reference() == ""
    assert prior_art.survey_block(prior_art.find_prior_art(repo, {}), []) == ""
    # A model registered under a name that is not written out may be the one: it is named.
    (repo / "engine/zoo/named_elsewhere.py").write_text(
        "@register_auto_model(ElsewhereConfig.architecture)\nclass Elsewhere:\n    pass\n"
    )
    unnamed = prior_art.find_prior_art(repo, {**config, "architectures": ["NewForCausalLM"]})
    assert unnamed.reference().endswith(
        "(Registered under a name that is not written out: `Elsewhere`.)"
    )


def test_each_verb_a_worker_may_name_is_one_the_resolve_command_has():
    from agent_flow.workflows.trtllm_modeling_v2.run import escalation

    text = prompts.DEFAULT_PROMPTS.kernel_writer
    for verb in ("retry", "allow-shared", "edit-env", "split", "change-requirement", "skip"):
        assert f"- `{verb}" in text and verb in escalation.VERBS, verb
