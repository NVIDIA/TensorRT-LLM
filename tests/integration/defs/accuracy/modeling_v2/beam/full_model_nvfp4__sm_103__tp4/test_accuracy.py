# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Named TP4 baseline gate: pinned first-100 GSM8K through trtllm-serve.

Stage an authorized NVFP4 target below LLM_MODELS_ROOT and name it with
TRTLLM_TEST_TARGET_RELATIVE_PATH. Stage the pinned dataset at datasets/openai/gsm8k.
Missing inputs fail a selected run. This manual qualification uses four GB300s,
BF16 KV, no drafting, eager execution and no overlap; it never runs full GSM8K.
"""

import os
import sys
from pathlib import Path

import pytest
import yaml

EXAMPLES_DIR = Path(__file__).resolve().parents[7] / "examples" / "llm-api"
sys.path.insert(0, str(EXAMPLES_DIR))
from gsm8k_eval import runner  # noqa: E402


def _run_first100(models_root, output_dir, time_budget_s):
    root = Path(models_root).resolve()
    relative = Path(os.environ["TRTLLM_TEST_TARGET_RELATIVE_PATH"])
    if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        raise ValueError(
            "TRTLLM_TEST_TARGET_RELATIVE_PATH must name a checkpoint below LLM_MODELS_ROOT"
        )
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or not target.is_dir():
        raise ValueError("the staged target must be a directory below LLM_MODELS_ROOT")
    config = {
        "modeling_v2": "require",
        "checkpoint_format": "beam_nvfp4",
        "custom_tokenizer": "tensorrt_llm._torch.models.checkpoints.beam.tokenizer.BeamFastTokenizer",
        "reasoning_parser": "beam_harmony_text",
        "tensor_parallel_size": 4,
        "max_batch_size": 1,
        "max_num_tokens": 8192,
        "max_seq_len": 131072,
        "cuda_graph_config": None,
        "disable_overlap_scheduler": True,
        "kv_cache_config": {"dtype": "auto", "free_gpu_memory_fraction": 0.5},
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "beam-baseline.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    summary = runner.run(
        dataset_dir=root / "datasets" / "openai" / "gsm8k",
        output_dir=output,
        num_questions=100,
        concurrency=1,
        time_budget_s=min(2700, time_budget_s),
        max_seq_len=config["max_seq_len"],
        min_correct=95,
        managed=runner.ManagedServer(
            argv=("trtllm-serve", str(target), "--config", str(config_path)),
            env=dict(os.environ),
            settings=config,
        ),
        identity_extra={"target": str(target), "server_config": config},
    )
    assert summary["verdict"]["passed"], summary["verdict"]


@pytest.mark.skip_less_device(4)
@pytest.mark.skip_device_not_contain(["GB300"])
class TestModelingV2BeamFullModelNvfp4Sm103Tp4:
    def test_gsm8k_first100(self, tmp_path, timeout_from_marker, timeout_from_command_line):
        from .....conftest import get_device_count, llm_models_root

        if int(get_device_count()) != 4:
            pytest.skip("the TP4 baseline gate requires exactly four visible GB300 GPUs")
        timeout = timeout_from_marker or timeout_from_command_line or 2700
        _run_first100(llm_models_root(), tmp_path, timeout)
