# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The benchmark datasets are staged in the background and handed to the gates that read them."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agent_flow.workflows.trtllm_modeling_v2.run.startup import datasets


def _files(path: Path) -> Path:
    (path / "main").mkdir(parents=True, exist_ok=True)
    (path / "main" / "test-00000-of-00001.parquet").write_text("rows")
    return path


def test_a_dataset_the_models_root_holds_is_used_where_it_is(tmp_path):
    root = tmp_path / "models"
    _files(root / "datasets" / "openai" / "gsm8k")
    task = {"environment": {"env": {"LLM_MODELS_ROOT": str(root)}}, "accuracy": {}}
    entry = datasets.stage(task, tmp_path / "ws", "gsm8k", run=None)
    assert entry == {"state": "ready", "path": str(root / "datasets/openai/gsm8k"),
                     "source": "models root"}  # fmt: skip


def test_a_missing_dataset_is_downloaded_into_the_workspace(tmp_path):
    asked = []

    def run(argv, **kwargs):
        asked.append(argv[-2:])
        _files(Path(argv[-1]))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    entry = datasets.stage({"accuracy": {}}, tmp_path / "ws", "gsm8k", python="/py", run=run)
    assert entry["state"] == "ready" and entry["source"] == "downloaded openai/gsm8k"
    assert asked == [["openai/gsm8k", str(tmp_path / "ws/datasets/openai/gsm8k")]]


def test_a_dataset_that_cannot_be_staged_is_reported_and_not_waited_for(tmp_path):
    def run(argv, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="OfflineModeIsEnabled")

    entry = datasets.stage({"accuracy": {}}, tmp_path / "ws", "gsm8k", run=run)
    assert entry["state"] == "failed" and "OfflineModeIsEnabled" in entry["detail"]
    datasets._set(tmp_path / "ws", "gsm8k", **entry)
    assert datasets.wait(tmp_path / "ws", "gsm8k", timeout=0) == ""


def test_the_tree_tests_read_a_models_root_that_holds_the_staged_copy(tmp_path):
    root = tmp_path / "models"
    (root / "stepfun").mkdir(parents=True)
    (root / "datasets" / "mmlu").mkdir(parents=True)
    staged = _files(tmp_path / "ws" / "datasets" / "openai" / "gsm8k")
    overlay = Path(datasets.models_root(tmp_path / "ws", str(root), "gsm8k", str(staged)))
    assert overlay != root
    assert (overlay / "stepfun").resolve() == (root / "stepfun").resolve()
    assert (overlay / "datasets" / "mmlu").resolve() == (root / "datasets" / "mmlu").resolve()
    assert (overlay / "datasets/openai/gsm8k/main").is_dir()
    # A root that holds it already is used as it is.
    assert datasets.models_root(tmp_path / "ws", str(overlay), "gsm8k", str(staged)) == str(overlay)
