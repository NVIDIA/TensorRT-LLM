# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end prompt_embeds tests on the PyTorch backend.

Rows taken from the checkpoint's own input embedding must reproduce the greedy
output of the equivalent token prompt exactly, whichever positions carry them.
"""

import asyncio
import json
import os
import pickle
import uuid
from types import SimpleNamespace
from typing import Iterator, List, Optional

import pytest
import torch
from safetensors import safe_open

from tensorrt_llm import LLM
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.executor import RequestError
from tensorrt_llm.executor.base_worker import BaseWorker
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.inputs.prompt_embeds import build_prompt_embeds, prompt_embeds_cache_salt
from tensorrt_llm.llmapi import (
    CacheTransceiverConfig,
    CapacitySchedulerPolicy,
    CudaGraphConfig,
    KvCacheConfig,
    SchedulerConfig,
    TorchCompileConfig,
)
from tensorrt_llm.sampling_params import SamplingParams

# isort: off
from .test_llm import get_model_path
# isort: on

# Module-scoped LLM instances keep their proxy threads across tests.
pytestmark = pytest.mark.threadleak(enabled=False)

QWEN3 = "Qwen3/Qwen3-0.6B"
TINYLLAMA = "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"

PROMPTS = [
    "The capital of France is",
    "Write a haiku about the ocean. The waves",
    "In 1969, the first humans to walk on the moon were",
]
LONG_PROMPT = " ".join(
    f"Item {i} of the inventory is a {c} box."
    for i, c in enumerate(["red", "green", "blue", "yellow", "black", "white", "orange"] * 4)
)


def _greedy() -> SamplingParams:
    # Fresh per call: generate() fills in model-specific fields such as end_id.
    return SamplingParams(max_tokens=16, temperature=0.0)


def _run(llm: LLM, requests: list, salt: Optional[str] = None) -> List[List[int]]:
    """Generate greedily; a fresh salt keeps every prompt off cached blocks.

    Prefix hits change which kernels run the prompt, which can flip a greedy
    choice between otherwise identical requests.
    """
    outputs = llm.generate(requests, _greedy(), cache_salt=salt or uuid.uuid4().hex)
    return [list(o.outputs[0].token_ids) for o in outputs]


def _embedding_table(model_dir: str) -> torch.Tensor:
    """Read the input embedding table from the checkpoint."""
    for name in sorted(os.listdir(model_dir)):
        if not name.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(model_dir, name), "pt") as f:
            if "model.embed_tokens.weight" in f.keys():
                return f.get_tensor("model.embed_tokens.weight")
    raise FileNotFoundError(f"no embed_tokens.weight under {model_dir}")


def _token_ids(llm: LLM, prompts: List[str]) -> List[List[int]]:
    return [llm.tokenizer.encode(p) for p in prompts]


def _placeholder_request(prompt_ids: List[int], mask: List[bool], table: torch.Tensor) -> dict:
    """Schedule id 0 at masked positions; only the supplied rows carry the prompt there."""
    return {
        "prompt_token_ids": [0 if m else t for t, m in zip(prompt_ids, mask)],
        "prompt_embeds": table[torch.tensor([t for t, m in zip(prompt_ids, mask) if m])],
        "prompt_embeds_mask": mask,
    }


def _max_paused_requests(llm: LLM) -> int:
    """Largest per-iteration paused-request count in the stats drained so far."""
    paused = 0
    for entry in llm.get_stats(timeout=2):
        stats = json.loads(entry) if isinstance(entry, str) else entry
        paused = max(paused, stats["inflightBatchingStats"]["numPausedRequests"])
    return paused


@pytest.fixture(scope="module", params=[QWEN3, TINYLLAMA])
def model_dir(request) -> str:
    path = get_model_path(request.param)
    if not os.path.isdir(path):
        pytest.skip(f"{path} not found")
    return path


@pytest.fixture(scope="module")
def llm(model_dir: str) -> Iterator[LLM]:
    with LLM(
        model=model_dir,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.3, enable_block_reuse=True),
        max_batch_size=8,
    ) as instance:
        yield instance


@pytest.fixture(scope="module")
def table(model_dir: str) -> torch.Tensor:
    return _embedding_table(model_dir)


@pytest.mark.parametrize(
    "v2, connector, keys_rows",
    [(True, None, True), (False, None, False), (True, object(), False)],
    ids=["v2", "v1", "v2_connector"],
)
def test_worker_cache_salt(v2: bool, connector: Optional[object], keys_rows: bool) -> None:
    # Where blocks are keyed by ids alone (V1, external KV stores) the salt scopes the prompt.
    _, pe = build_prompt_embeds([1, 2, 3], torch.ones(3, 4))
    params = SamplingParams(max_tokens=1)
    worker = SimpleNamespace(
        llm_args=SimpleNamespace(kv_connector_config=connector),
        _uses_kv_cache_manager_v2=lambda: v2,
    )
    worker._kv_cache_keys_prompt_embeds = lambda: BaseWorker._kv_cache_keys_prompt_embeds(worker)
    plain = GenerationRequest([1, 2, 3], params, cache_salt="s")
    embeds = GenerationRequest([1, 2, 3], params, cache_salt="s", prompt_embeds=pe)
    assert embeds.cache_salt == "s"
    assert BaseWorker._effective_cache_salt(worker, plain) == "s"
    salt = BaseWorker._effective_cache_salt(worker, embeds)
    assert salt == ("s" if keys_rows else prompt_embeds_cache_salt("s", pe.digest))
    # A plain request cannot claim the derived salt of an embedding request.
    with pytest.raises(ValueError, match="reserved prefix"):
        GenerationRequest([1, 2, 3], params, cache_salt=prompt_embeds_cache_salt("s", pe.digest))
    restored = pickle.loads(pickle.dumps(embeds))
    assert torch.equal(restored.prompt_embeds.embeds, pe.embeds)
    assert restored.prompt_embeds.row_digests == pe.row_digests


MASKS = {
    "prefix": lambda p, n: p < n // 2,
    "suffix": lambda p, n: p >= n // 2,
    "interleaved": lambda p, n: p % 3 != 0,
}


@pytest.mark.parametrize("pattern", ["whole", *MASKS])
def test_matches_tokens(llm: LLM, table: torch.Tensor, pattern: str) -> None:
    # The last prompt stays plain, so every batch mixes embedding and token prompts.
    ids = _token_ids(llm, PROMPTS)
    expected = _run(llm, ids)
    if pattern == "whole":
        requests = [{"prompt_embeds": table[torch.tensor(i)]} for i in ids[:-1]]
    else:
        requests = [
            _placeholder_request(i, [MASKS[pattern](p, len(i)) for p in range(len(i))], table)
            for i in ids[:-1]
        ]
    assert _run(llm, requests + [ids[-1]]) == expected


def test_reuse_scoped_to_embeddings(llm: LLM, table: torch.Tensor) -> None:
    ids = _token_ids(llm, [LONG_PROMPT])[0]
    other_ids = _token_ids(llm, [LONG_PROMPT.replace("box", "crate")])[0]
    n = min(len(ids), len(other_ids))
    ids, other_ids = ids[:n], other_ids[:n]
    assert ids != other_ids
    salt = uuid.uuid4().hex
    expected = _run(llm, [ids], salt)[0]
    expected_other = _run(llm, [other_ids], salt)[0]

    # Same placeholder ids as the cached `ids` prompt, different rows: must
    # not reuse that prompt's blocks.
    request = {
        "prompt_token_ids": ids,
        "prompt_embeds": table[torch.tensor(other_ids)],
    }
    first = llm.generate([request], _greedy(), cache_salt=salt)[0]
    assert list(first.outputs[0].token_ids) == expected_other
    assert first.cached_tokens == 0

    # Identical ids and rows reuse every full block, leaving a short final
    # context chunk whose position is an embedding.
    second = llm.generate([request], _greedy(), cache_salt=salt)[0]
    assert list(second.outputs[0].token_ids) == expected_other
    assert second.cached_tokens > 0

    # Same rows under different placeholder ids share keys only up to the
    # first differing id.
    other_placeholders = {**request, "prompt_token_ids": other_ids}
    first_diff = next(p for p, (a, b) in enumerate(zip(ids, other_ids)) if a != b)
    third = llm.generate([other_placeholders], _greedy(), cache_salt=salt)[0]
    assert third.cached_tokens <= first_diff

    # The plain prompt still reuses its own blocks and is unaffected.
    plain = llm.generate([ids], _greedy(), cache_salt=salt)[0]
    assert list(plain.outputs[0].token_ids) == expected


def test_reuse_shares_plain_prefix(llm: LLM, table: torch.Tensor) -> None:
    # Blocks before the first embedding row are keyed by token ids alone, so
    # they are shared with the plain prompt; later blocks still match only
    # identical rows.
    ids = _token_ids(llm, [LONG_PROMPT])[0]
    salt = uuid.uuid4().hex
    expected = _run(llm, [ids], salt)[0]
    first_embed = len(ids) // 2
    mask = [p >= first_embed for p in range(len(ids))]
    request = _placeholder_request(ids, mask, table)
    shared = llm.generate([request], _greedy(), cache_salt=salt)[0]
    assert list(shared.outputs[0].token_ids) == expected
    assert 0 < shared.cached_tokens <= first_embed
    again = llm.generate([request], _greedy(), cache_salt=salt)[0]
    assert list(again.outputs[0].token_ids) == expected
    assert again.cached_tokens > first_embed


def test_v1_reuse_scoped_to_embeddings(model_dir: str, table: torch.Tensor) -> None:
    # V1 keys blocks by ids alone; the embedding identity joins the salt.
    with LLM(
        model=model_dir,
        kv_cache_config=KvCacheConfig(
            free_gpu_memory_fraction=0.3, enable_block_reuse=True, use_kv_cache_manager_v2=False
        ),
        max_batch_size=4,
    ) as v1:
        ids = _token_ids(v1, [LONG_PROMPT])[0]
        other_ids = _token_ids(v1, [LONG_PROMPT.replace("box", "crate")])[0]
        n = min(len(ids), len(other_ids))
        ids, other_ids = ids[:n], other_ids[:n]
        salt = uuid.uuid4().hex
        # Same placeholder ids and user salt, two row payloads.
        payloads = [(ids, _run(v1, [ids], salt)[0]), (other_ids, _run(v1, [other_ids])[0])]
        for repeat in (False, True):
            for rows, expected in payloads:
                request = {"prompt_token_ids": ids, "prompt_embeds": table[torch.tensor(rows)]}
                out = v1.generate([request], _greedy(), cache_salt=salt)[0]
                assert list(out.outputs[0].token_ids) == expected
                assert (out.cached_tokens > 0) == repeat


def test_reuse_with_embedding_final_token(llm: LLM, table: torch.Tensor) -> None:
    # A repeated short prompt is cached up to its last token; that one-token
    # final context must not run as a decode row, which embeds the placeholder.
    ids = _token_ids(llm, PROMPTS[1:2])[0]
    salt = uuid.uuid4().hex
    expected = _run(llm, [ids])[0]
    request = {"prompt_embeds": table[torch.tensor(ids)]}
    for _ in range(3):
        assert _run(llm, [request], salt)[0] == expected


def test_n_greater_than_one(llm: LLM, table: torch.Tensor, monkeypatch: pytest.MonkeyPatch) -> None:
    # Child requests share the parent's rows.
    monkeypatch.setenv("TLLM_ALLOW_N_GREEDY_DECODING", "1")
    ids = _token_ids(llm, PROMPTS[:1])[0]

    def run(request: object) -> List[List[int]]:
        params = SamplingParams(max_tokens=8, n=2, temperature=0.0)
        out = llm.generate(request, params, cache_salt=uuid.uuid4().hex)
        return [list(o.token_ids) for o in out.outputs]

    expected = run(ids)
    assert len(expected) == 2
    assert run({"prompt_embeds": table[torch.tensor(ids)]}) == expected


def test_rejects_bad_requests(llm: LLM, table: torch.Tensor) -> None:
    ids = _token_ids(llm, PROMPTS[:1])[0]
    rows = table[torch.tensor(ids)]
    with pytest.raises((RequestError, ValueError), match="hidden size"):
        llm.generate({"prompt_embeds": rows[:, :-1]}, _greedy())
    with pytest.raises(ValueError, match="prompt_embeds_mask"):
        llm.generate(
            {
                "prompt_token_ids": ids,
                "prompt_embeds": rows[:2],
                "prompt_embeds_mask": [True] * len(ids),
            },
            _greedy(),
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        llm.generate({"prompt": PROMPTS[0], "prompt_embeds": rows}, _greedy())
    encoder = SimpleNamespace(max_seq_len=4096, max_num_tokens=4096, max_batch_size=8)
    encoder.batch_forward = lambda *args, **kwargs: pytest.fail("encode() ran the model")
    encode_only = SimpleNamespace(_encode_only=True, _encoder_executor=encoder)
    with pytest.raises(ValueError, match="not supported by encode"):
        LLM.encode(encode_only, {"prompt_token_ids": ids, "prompt_embeds": rows})
    # The engine keeps serving after rejected requests.
    assert llm.generate(ids, _greedy()).outputs[0].token_ids


@pytest.mark.parametrize("disable_overlap_scheduler", [False, True], ids=["overlap", "no_overlap"])
def test_chunked_prefill(
    model_dir: str, table: torch.Tensor, disable_overlap_scheduler: bool
) -> None:
    with LLM(
        model=model_dir,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.3, enable_block_reuse=False),
        enable_chunked_prefill=True,
        max_num_tokens=64,
        disable_overlap_scheduler=disable_overlap_scheduler,
        max_batch_size=4,
    ) as chunked:
        ids = _token_ids(chunked, [LONG_PROMPT, PROMPTS[0]])
        assert len(ids[0]) > 2 * 64
        expected = _run(chunked, ids)
        n = len(ids[0])
        mask = [(p // 7) % 2 == 1 for p in range(n)]
        # One run of embedding rows straddles the first chunk boundary.
        assert mask[63] and mask[64]
        requests = [
            _placeholder_request(ids[0], mask, table),
            {"prompt_embeds": table[torch.tensor(ids[1])]},
        ]
        assert _run(chunked, requests) == expected


@pytest.mark.gpu2
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("tp_size, pp_size", [(2, 1), (1, 2)], ids=["tp2", "pp2"])
def test_multi_gpu(model_dir: str, table: torch.Tensor, tp_size: int, pp_size: int) -> None:
    with LLM(
        model=model_dir,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.3, enable_block_reuse=True),
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        max_batch_size=4,
    ) as parallel:
        ids = _token_ids(parallel, PROMPTS)
        expected = _run(parallel, ids)
        n = len(ids[0])
        mask = [p % 2 == 1 for p in range(n)]
        requests = [
            _placeholder_request(ids[0], mask, table),
            {"prompt_embeds": table[torch.tensor(ids[1])]},
            ids[2],
        ]
        assert _run(parallel, requests) == expected


def test_preemption_with_graphs_and_compile(model_dir: str, table: torch.Tensor) -> None:
    # A small KV pool under MAX_UTILIZATION suspends running requests; their
    # context, embedding rows included, must come back intact. CUDA graphs and
    # torch.compile stay on for passes without embedding rows.
    with LLM(
        model=model_dir,
        kv_cache_config=KvCacheConfig(max_tokens=640, enable_block_reuse=False),
        scheduler_config=SchedulerConfig(
            capacity_scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION
        ),
        cuda_graph_config=CudaGraphConfig(batch_sizes=[1, 2, 4]),
        torch_compile_config=TorchCompileConfig(),
        enable_iter_perf_stats=True,
        max_batch_size=4,
    ) as pressured:
        base = _token_ids(pressured, [LONG_PROMPT])[0][:200]
        ids = [base[: 200 - 7 * i] for i in range(4)]
        params = SamplingParams(max_tokens=96, temperature=0.0)
        expected = [list(o.outputs[0].token_ids) for o in pressured.generate(ids, params)]
        _max_paused_requests(pressured)
        requests = [
            _placeholder_request(prompt_ids, [p % 3 == 1 for p in range(len(prompt_ids))], table)
            for prompt_ids in ids
        ]
        got = [list(o.outputs[0].token_ids) for o in pressured.generate(requests, params)]
        assert _max_paused_requests(pressured) > 0, "no request was preempted"
        assert got == expected


def test_disaggregated_handoff() -> None:
    # Context and generation engines share one GPU; both key the transferred
    # blocks by the embedding identity.
    model = get_model_path(QWEN3)
    if not os.path.isdir(model):
        pytest.skip(f"{model} not found")
    table = _embedding_table(model)

    def engine() -> LLM:
        return LLM(
            model=model,
            kv_cache_config=KvCacheConfig(
                max_tokens=4096, enable_block_reuse=True, use_kv_cache_manager_v2=True
            ),
            cache_transceiver_config=CacheTransceiverConfig(
                backend="NIXL", transceiver_runtime="PYTHON"
            ),
            disable_overlap_scheduler=True,
            max_batch_size=4,
        )

    with engine() as ctx, engine() as gen:
        ids = _token_ids(ctx, [LONG_PROMPT])[0]
        mask = [p % 3 == 1 for p in range(len(ids))]
        request = _placeholder_request(ids, mask, table)
        other = {**request, "prompt_embeds": request["prompt_embeds"] * 1.5}
        salt = uuid.uuid4().hex

        def handoff(req: dict) -> tuple[List[int], int]:
            first = ctx.generate_async(
                req,
                sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
                disaggregated_params=DisaggregatedParams(request_type="context_only"),
                cache_salt=salt,
            ).result()
            params = first.disaggregated_params
            params.request_type = "generation_only"
            out = gen.generate_async(
                req, sampling_params=_greedy(), disaggregated_params=params, cache_salt=salt
            ).result()
            return list(out.outputs[0].token_ids), out.cached_tokens

        assert handoff(request) == (_run(ctx, [ids])[0], 0)
        tokens, cached = handoff(request)
        assert tokens == _run(ctx, [ids])[0] and cached > mask.index(True)
        tokens, cached = handoff(other)
        assert tokens == _run(ctx, [other])[0] and cached <= mask.index(True)


@pytest.mark.parametrize("request_type", ["context_only", "generation_only"])
@pytest.mark.parametrize("v2", [True, False], ids=["v2", "v1"])
def test_disaggregated_deployment_checks(request_type: str, v2: bool) -> None:
    # Both roles pass the full checks; the V1 transceiver cannot key blocks by rows.
    _, pe = build_prompt_embeds([1, 2, 3], torch.ones(3, 4))
    request = GenerationRequest(
        [1, 2, 3],
        SamplingParams(max_tokens=1),
        prompt_embeds=pe,
        disaggregated_params=DisaggregatedParams(request_type=request_type),
    )
    model = SimpleNamespace(
        prompt_embeds_unsupported_reason=lambda: None, config=SimpleNamespace(hidden_size=4)
    )
    worker = SimpleNamespace(
        llm_args=SimpleNamespace(
            speculative_config=None,
            parallel_config=SimpleNamespace(enable_attention_dp=False, cp_size=1),
        ),
        engine=SimpleNamespace(model_engine=SimpleNamespace(model=model)),
        _uses_kv_cache_manager_v2=lambda: v2,
    )
    if v2:
        BaseWorker._check_prompt_embeds(worker, request)
    else:
        with pytest.raises(RequestError, match="use_kv_cache_manager_v2"):
            BaseWorker._check_prompt_embeds(worker, request)


def test_cancel_releases_rows(llm: LLM, table: torch.Tensor) -> None:
    ids = _token_ids(llm, PROMPTS[:1])[0]
    params = SamplingParams(max_tokens=200, temperature=0.0, end_id=-1)

    async def run() -> None:
        result = llm.generate_async(
            {"prompt_embeds": table[torch.tensor(ids)]}, sampling_params=params, streaming=True
        )
        result.abort()
        outputs = [output async for output in result]
        assert outputs[-1].outputs[0].finish_reason == "cancelled"
        # The client keeps only the identity once the request is done.
        assert result._generation_request.prompt_embeds.embeds is None

    asyncio.run(run())
    assert _run(llm, [ids]) == _run(llm, [{"prompt_embeds": table[torch.tensor(ids)]}])


@pytest.mark.parametrize("backend", ["piecewise", "breakable"])
def test_prefill_cuda_graph(model_dir: str, table: torch.Tensor, backend: str) -> None:
    # Passes with embedding rows run eagerly; plain prefills keep replaying graphs.
    with LLM(
        model=model_dir,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.3, enable_block_reuse=False),
        prefill_cuda_graph_backend=backend,
        torch_compile_config=TorchCompileConfig() if backend == "piecewise" else None,
        max_batch_size=4,
    ) as graphs:
        ids = _token_ids(graphs, PROMPTS)
        expected = _run(graphs, ids)
        mixed = [ids[0], {"prompt_embeds": table[torch.tensor(ids[1])]}, ids[2]]
        assert _run(graphs, mixed) == expected
        assert _run(graphs, ids) == expected
