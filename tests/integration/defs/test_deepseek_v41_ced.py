# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full-checkpoint CED E2E, including a strict >91% sampled GSM8K gate.

Run on four Blackwell GPUs through trtllm-llmapi-launch. Set DSV41_MODEL_PATH,
GSM8K_DATASET_PATH and DSV41_CED_RESULTS to local checkpoint/data/output paths.
Set TRTLLM_V41_DECODER_BOUNDED_REPLAY=1 before workers are launched.
GSM8K uses 256 requests by default; DSV41_CED_GSM8K_SAMPLES overrides the count.
"""

import json
import os
from functools import partial
from pathlib import Path

import pytest

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.evaluate import GSM8K
from tensorrt_llm.scheduling_params import SchedulingParams


@pytest.fixture(scope="module")
def ced_llm():
    assert os.environ.get("TRTLLM_V41_DECODER_BOUNDED_REPLAY") == "1", (
        "This suite must execute decoder bounded replay"
    )
    model_path = Path(os.environ["DSV41_MODEL_PATH"])
    kv_max_tokens = int(os.environ.get("DSV41_CED_KV_MAX_TOKENS", 65536))
    args = dict(
        tensor_parallel_size=4,
        moe_expert_parallel_size=4,
        gpus_per_node=4,
        enable_attention_dp=os.environ.get("DSV41_CED_ADP") == "1",
        custom_tokenizer="deepseek_v41",
        trust_remote_code=True,
        max_seq_len=8192,
        max_num_tokens=int(os.environ.get("DSV41_CED_MAX_NUM_TOKENS", 1024)),
        max_batch_size=int(os.environ.get("DSV41_CED_MAX_BATCH_SIZE", 8)),
        enable_chunked_prefill=True,
        disable_overlap_scheduler=os.environ.get("DSV41_CED_DISABLE_OVERLAP", "0") == "1",
        enable_iter_perf_stats=os.environ.get("DSV41_CED_ITER_STATS") == "1",
        cuda_graph_config=(
            {
                "batch_sizes": json.loads(
                    os.environ.get("DSV41_CED_GRAPH_BATCH_SIZES", "[1, 2, 4, 8, 16, 32, 64]")
                ),
                "enable_padding": True,
            }
            if os.environ.get("DSV41_CED_CUDA_GRAPH") == "1"
            else None
        ),
        moe_config={"backend": "TRTLLM"},
        allreduce_strategy=os.environ.get("DSV41_CED_ALLREDUCE_STRATEGY", "AUTO"),
        kv_cache_config=dict(
            dtype="fp8",
            enable_block_reuse=True,
            enable_swa_scratch_reuse=False,
            tokens_per_block=128,
            max_tokens=kv_max_tokens or None,
            free_gpu_memory_fraction=float(os.environ.get("DSV41_CED_KV_MEMORY_FRACTION", 0.5)),
        ),
        sparse_attention_config=dict(algorithm="csa2"),
    )
    out = Path(os.environ["DSV41_CED_RESULTS"])
    if os.environ.get("DSV41_CED_DSPARK") == "1":
        args["speculative_config"] = dict(
            decoding_type="DSpark", max_draft_len=5, speculative_model=str(model_path)
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "model_config.json").write_text(json.dumps(args, indent=2))
    with LLM(str(model_path), **args) as llm:
        yield llm


@pytest.fixture(autouse=True)
def pin_adp_cache_tests(ced_llm, request, monkeypatch):
    if os.environ.get("DSV41_CED_ADP") != "1" or "gsm8k" in request.node.name:
        return
    # Reuse assertions target one rank. The asymmetric test overrides this
    # default; GSM8K uses ordinary load balancing across all ADP ranks.
    scheduling = SchedulingParams(attention_dp_rank=0, attention_dp_relax=False)
    for name in ("generate", "generate_async"):
        monkeypatch.setattr(
            ced_llm, name, partial(getattr(ced_llm, name), scheduling_params=scheduling)
        )


def prompt_ids(llm, length):
    seed = llm.tokenizer.encode("A short story about a traveler. ")
    return (seed * ((length + len(seed) - 1) // len(seed)))[:length]


def test_ced_mixed_length_smoke(ced_llm):
    # Keep one full-checkpoint liveness gate; numerical boundary coverage is
    # in the unit tests. Unique salts keep these requests on the cold path.
    params = SamplingParams(max_tokens=8, temperature=0, ignore_eos=True)
    futures = [
        ced_llm.generate_async(
            prompt_ids(ced_llm, length), params, cache_salt=f"ced-smoke-{length}"
        )
        for length in (1, 127, 128, 129, 1025, 2049, 5377)
    ]
    assert all(len(future.result().outputs[0].token_ids) == 8 for future in futures)
    followup = ced_llm.generate(prompt_ids(ced_llm, 129), params, cache_salt="ced-smoke-129")
    assert len(followup.outputs[0].token_ids) == 8


@pytest.mark.skipif(os.environ.get("DSV41_CED_ADP") != "1", reason="Attention DP only")
def test_ced_adp_asymmetric_replay(ced_llm):
    params = SamplingParams(max_tokens=32, temperature=0, ignore_eos=True)
    futures = [
        ced_llm.generate_async(
            prompt_ids(ced_llm, length),
            params,
            cache_salt=f"ced-adp-{rank}-{length}",
            scheduling_params=SchedulingParams(attention_dp_rank=rank, attention_dp_relax=False),
        )
        for rank, length in ((0, 4097), (0, 1025), (1, 129), (2, 1))
    ]
    # Rank 3 has no real requests; short ranks finish prefill while rank 0 is
    # still chunking, so every Decoder MoE must handle different local shapes.
    assert all(len(future.result().outputs[0].token_ids) == 32 for future in futures)


@pytest.mark.skipif(
    os.environ.get("TRTLLM_V41_ENCODER_REPLAY") != "1", reason="Optional Encoder recovery only"
)
def test_ced_snapshot_short_fallback_and_long_suffix_restore(ced_llm):
    salt = "ced-m4-snapshot-short-suffix"
    # Finish the longer prompt with a question: an indefinitely repeated story
    # has competing EOS/continuation answers sensitive to numerical variation.
    question = ced_llm.tokenizer.encode("\n\nWhat is 2 + 2? Answer with only the number.\nAnswer:")
    assert len(question) < 128
    long_tokens = prompt_ids(ced_llm, 384 - len(question)) + question
    tokens = long_tokens[:257]
    # Keep the common prefix exactly block aligned even with partial matching,
    # so the long request below exercises checkpoint restore without replay.
    tokens[-1] += 1
    # A raw repeated-text prompt can emit EOS as its single token. Keep it
    # visible when checking completion. Concurrent requests may run in different
    # batch shapes; greedy tokens need not match for this ambiguous continuation.
    params = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True, skip_special_tokens=False)
    cold = ced_llm.generate(tokens, params, cache_salt=salt)
    assert cold.cached_tokens == 0
    # Short suffixes must rebuild decoder inputs even with an encoder snapshot.
    futures = [ced_llm.generate_async(tokens, params, cache_salt=salt) for _ in range(3)]
    hits = [future.result() for future in futures]
    for hit in hits:
        assert hit.cached_tokens == 256
        assert len(hit.outputs[0].token_ids) == 1
        assert hit.outputs[0].text
    longer = ced_llm.generate(long_tokens, params, cache_salt=salt)
    assert longer.cached_tokens == 256
    assert len(longer.outputs[0].token_ids) == 1 and longer.outputs[0].text
    isolated = ced_llm.generate(long_tokens, params, cache_salt=salt + "-isolated")
    assert isolated.cached_tokens == 0
    assert longer.outputs[0].token_ids == isolated.outputs[0].token_ids


@pytest.mark.skipif(
    os.environ.get("TRTLLM_V41_ENCODER_REPLAY") == "1", reason="Required Encoder state only"
)
def test_ced_required_encoder_snapshot(ced_llm):
    params = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
    salt = "ced-required-encoder"
    window = 128
    warm = ced_llm.generate(prompt_ids(ced_llm, 257), params, cache_salt=salt)
    assert warm.cached_tokens == 0
    # The final aligned Encoder state is required and adopted by ordinary matching.
    tokens = prompt_ids(ced_llm, 256 + window)
    hit = ced_llm.generate(tokens, params, cache_salt=salt)
    assert hit.cached_tokens == 256
    cold = ced_llm.generate(tokens, params, cache_salt=salt + "-cold")
    assert cold.cached_tokens == 0
    assert hit.outputs[0].token_ids == cold.outputs[0].token_ids
    short = ced_llm.generate(prompt_ids(ced_llm, 257), params, cache_salt=salt)
    assert short.cached_tokens <= max(0, 257 - window)
    assert len(short.outputs[0].token_ids) == 1


@pytest.mark.skipif(
    os.environ.get("TRTLLM_V41_ENCODER_REPLAY") != "1", reason="Optional Encoder recovery only"
)
@pytest.mark.parametrize("prefix,suffix", [(255, 1), (383, 129), (4097, 903)])
def test_ced_partial_global_hit_replays_encoder(ced_llm, prefix, suffix):
    salt = f"ced-partial-global-{prefix}-{suffix}"
    params = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
    tokens = prompt_ids(ced_llm, prefix + suffix)
    warm_tokens = tokens[:prefix] + [tokens[prefix] + 1]
    cold = ced_llm.generate(warm_tokens, params, cache_salt=salt)
    assert cold.cached_tokens == 0
    # An incomplete OPTIONAL checkpoint must preserve the nonaligned Global
    # match and replay across the page containing the first historical row.
    hit = ced_llm.generate(tokens, params, cache_salt=salt)
    assert hit.cached_tokens == prefix
    assert len(hit.outputs[0].token_ids) == 1


def test_ced_gsm8k_full_accuracy(ced_llm):
    out = Path(os.environ["DSV41_CED_RESULTS"])
    num_samples = int(os.environ.get("DSV41_CED_GSM8K_SAMPLES", 256))
    assert 0 < num_samples <= 1319
    protocol = dict(
        num_samples=num_samples,
        num_fewshot=5,
        random_seed=0,
        apply_chat_template=True,
        chat_template_kwargs={"thinking": False},
        system_prompt=(
            "Solve the problem carefully. End your response with a final line exactly "
            "in the form #### <answer>, using the simplest numeric form without units "
            "or trailing zeros."
        ),
    )
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    evaluator = GSM8K(
        dataset_path=os.environ["GSM8K_DATASET_PATH"],
        **protocol,
        log_samples=True,
        output_path=str(out / "results"),
        output_dir=str(out / "outputs"),
    )
    score = evaluator.evaluate(
        ced_llm,
        SamplingParams(max_tokens=256, truncate_prompt_tokens=4096, temperature=0, seed=0),
        sampling_override=True,
        scores_filter="exact_match,flexible-extract",
    )
    saved = json.loads((out / "results/samples_gsm8k.json").read_text())
    assert saved["n-samples"]["gsm8k"]["effective"] == num_samples
    for name in ("strict-match", "flexible-extract"):
        records = [r for r in saved["samples"]["gsm8k"] if r["filter"] == name]
        assert len(records) == num_samples and len({r["doc_id"] for r in records}) == num_samples
    assert score > 91.0, f"CED GSM8K flexible exact match {score:.4f}% must exceed 91%"
