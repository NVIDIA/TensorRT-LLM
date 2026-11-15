# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end scoring regression with a local, deterministic tiny checkpoint."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


def _make_checkpoint(path: Path) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        config = LlamaConfig(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=512,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            rms_norm_eps=1e-6,
        )
        model = LlamaForCausalLM(config).to(torch.bfloat16)
        model.save_pretrained(path)
    vocabulary = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
    vocabulary.update({f"t{index}": index for index in range(4, 256)})
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocabulary, unk_token="[UNK]")),
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(path)


def _configure_determinism(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    if enabled and torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Deterministic TRTLLM attention requires SM100 or SM103")
    monkeypatch.setenv("FORCE_DETERMINISTIC", "1" if enabled else "0")
    monkeypatch.delenv("FORCE_ATTENTION_KERNEL_DETERMINISTIC", raising=False)


@pytest.mark.parametrize("deterministic", [False, True], ids=["default", "deterministic"])
def test_prefill_decode_consistency_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deterministic: bool
) -> None:
    _configure_determinism(monkeypatch, deterministic)
    source = (
        Path(__file__).resolve().parents[3] / "examples/llm-api/llm_prefill_decode_consistency.py"
    )
    spec = importlib.util.spec_from_file_location("consistency_probe_gpu", source)
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    model = tmp_path / "model"
    _make_checkpoint(model)
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps([{"id": "boundary", "prompt_token_ids": [17, 63, 129, 12, 71]}]))
    output = tmp_path / "report.json"
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "attn_backend": "TRTLLM" if deterministic else "FlashInfer",
                "kv_cache_config": {"free_gpu_memory_fraction": 0.05},
                "env_overrides": {
                    "FORCE_DETERMINISTIC": "1" if deterministic else "0",
                    "FORCE_ATTENTION_KERNEL_DETERMINISTIC": "0",
                },
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(source),
            "--model",
            str(model),
            "--output",
            str(output),
            "--prompts",
            str(prompts),
            "--config",
            str(config),
            "--prompt-lengths",
            "63",
            "64",
            "65",
            "127",
            "128",
            "129",
            "--max-tokens",
            "8",
            "--temperature",
            "0",
            "--max-abs-dlogprob",
            "0" if deterministic else "0.01",
        ],
    )
    probe.main()
    report = json.loads(output.read_text())
    assert report["status"] == "completed"
    assert [len(case["prompt_token_ids"]) for case in report["cases"]] == [
        63,
        64,
        65,
        127,
        128,
        129,
    ]
    assert all(len(case["response_token_ids"]) == 8 for case in report["cases"])
    summary = report["summary"]
    assert summary["decode_vs_prefill"]["token_count"] == 42
    assert summary["prefill_boundary_vs_full_prefill"]["token_count"] == 6
    assert summary["prefill_repeatability"]["token_count"] == 48
    assert summary["prefill_repeatability"]["max_abs_delta"] == 0
    if deterministic:
        assert summary["decode_vs_prefill"]["max_abs_delta"] == 0
        assert summary["prefill_boundary_vs_full_prefill"]["max_abs_delta"] == 0
    else:
        # BF16 tolerance for this fixed two-layer checkpoint only.
        assert summary["decode_vs_prefill"]["max_abs_delta"] < 0.01
        assert summary["prefill_boundary_vs_full_prefill"]["max_abs_delta"] < 0.01
