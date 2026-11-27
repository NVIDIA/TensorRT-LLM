# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request-scoped decoder adapters on the four-rank deployment."""

import json

import pytest
import torch
from safetensors.torch import save_file

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch.peft.lora.config import LoraConfig
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.llmapi import KvCacheConfig

from ._engine import shortcut_text_engine


def _write_adapter(directory, text_config, adapter_id):
    hidden = text_config["hidden_size"]
    rank = 8
    generator = torch.Generator().manual_seed(adapter_id)
    weights = {}
    for i, kind in enumerate(text_config["layer_types"]):
        if kind == "shortcut_moe_attn":
            query = text_config["num_attention_heads"] * text_config["head_dim"]
            kv = text_config["num_key_value_heads"] * text_config["head_dim"]
            dimensions = {"self_attn.o_proj": (query, hidden)}
            if adapter_id == 1:
                dimensions.update(
                    {"self_attn.q_proj": (hidden, query), "self_attn.v_proj": (hidden, kv)}
                )
            else:
                dimensions["self_attn.k_proj"] = (hidden, kv)
        elif kind == "mlp":
            intermediate = text_config["intermediate_size"]
            dimensions = {
                "mlp.up_proj": (hidden, intermediate),
                "mlp.down_proj": (intermediate, hidden),
            }
        else:
            continue
        for proj, (in_size, out_size) in dimensions.items():
            prefix = f"base_model.model.model.language_model.layers.{i}.{proj}"
            for suffix, shape in (("A", (rank, in_size)), ("B", (out_size, rank))):
                weights[f"{prefix}.lora_{suffix}.weight"] = (
                    torch.randn(shape, generator=generator).mul_(0.1).bfloat16()
                )
    directory.mkdir()
    (directory / "adapter_config.json").write_text(json.dumps({"r": rank, "lora_alpha": rank}))
    save_file(weights, str(directory / "adapter_model.safetensors"))
    return LoRARequest(directory.name, adapter_id, str(directory))


@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("checkpoint", ["nvfp4"], indirect=True)
def test_multi_adapter_generation(checkpoint, tmp_path):
    if not checkpoint.is_dir():
        pytest.skip(f"Checkpoint unavailable: {checkpoint}")
    cfg = json.loads((checkpoint / "config.json").read_text())["text_config"]
    a = _write_adapter(tmp_path / "adapter_a", cfg, 1)
    b = _write_adapter(tmp_path / "adapter_b", cfg, 2)
    options = dict(
        max_batch_size=3,
        batch_wait_timeout_ms=100,
        enable_iter_perf_stats=True,
        gather_generation_logits=True,
        max_num_tokens=128,
        max_seq_len=128,
        lora_config=LoraConfig(lora_dir=[a.path], max_lora_rank=8, max_loras=2, max_cpu_loras=2),
        kv_cache_config=KvCacheConfig(use_kv_cache_manager_v2=True, enable_block_reuse=False),
    )
    prompt = "The capital of France is"
    sampling = SamplingParams(
        temperature=0, max_tokens=4, min_tokens=4, return_generation_logits=True
    )
    adapters = [a, None, b]

    def result(output):
        sequence = output.outputs[0]
        first_logits = sequence.generation_logits.reshape(-1, cfg["vocab_size"])[0].cpu()
        return tuple(sequence.token_ids), first_logits

    def generate_together(llm, requests):
        llm.get_stats(timeout=0.1)
        outputs = llm.generate([prompt] * len(requests), sampling, lora_request=requests)
        assert any(
            step.get("inflightBatchingStats", {}).get("numScheduledRequests", 0) == len(requests)
            for step in llm.get_stats(timeout=0.1)
        ), "The adapters and base request must share a scheduler iteration"
        return [result(output) for output in outputs]

    with shortcut_text_engine(checkpoint, **options) as llm:
        isolated = [
            result(llm.generate(prompt, sampling, lora_request=adapter))[0] for adapter in adapters
        ]
        assert len(set(isolated)) == len(adapters)
        plain = [result(output) for output in llm.generate([prompt] * len(adapters), sampling)]
        mixed = generate_together(llm, adapters)
        mixed_tokens = [tokens for tokens, _ in mixed]
        assert mixed_tokens[::2] == isolated[::2]
        assert mixed_tokens[1][0] == isolated[1][0]
        # A request can have different BF16 rounding in a batch. Match an entire
        # homogeneous control row at the same concurrency, including its first logits.
        assert any(
            tokens == mixed_tokens[1] and torch.equal(logits, mixed[1][1])
            for tokens, logits in plain
        )

    # Starting a fresh engine forces both adapters to reload from disk.
    with shortcut_text_engine(checkpoint, **options) as llm:
        for adapter, expected in ((b, isolated[2]), (a, isolated[0])):
            assert result(llm.generate(prompt, sampling, lora_request=adapter))[0] == expected
        reloaded = generate_together(llm, adapters[::-1])
        assert [tokens for tokens, _ in reloaded] == mixed_tokens[::-1]
