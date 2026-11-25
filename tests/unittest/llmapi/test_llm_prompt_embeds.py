# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end prompt_embeds tests on the PyTorch backend.

Rows taken from the checkpoint's own input embedding must reproduce the greedy
output of the equivalent token prompt exactly, whichever positions carry them.
"""

import asyncio
import gc
import json
import os
import uuid
import weakref
from types import SimpleNamespace
from typing import Iterator, List, Optional

import pytest
import torch
from safetensors import safe_open

import tensorrt_llm.llmapi.llm as llm_module
from tensorrt_llm import LLM
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph_runner import BreakableCUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner import DecoderRunner
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.executor import RequestError
from tensorrt_llm.inputs.prompt_embeds import build_prompt_embeds
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


@pytest.fixture(scope="module")
def model_dir() -> str:
    # Engine paths do not depend on the architecture; test_matches_tokens_llama adds a second one.
    path = get_model_path(QWEN3)
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


MASKS = {
    "prefix": lambda p, n: p < n // 2,
    "suffix": lambda p, n: p >= n // 2,
    "interleaved": lambda p, n: p % 3 != 0,
}


@pytest.mark.parametrize("pattern", ["whole", *MASKS])
def test_matches_tokens(llm: LLM, table: torch.Tensor, pattern: str) -> None:
    _check_matches_tokens(llm, table, pattern)


def test_matches_tokens_llama() -> None:
    model = get_model_path(TINYLLAMA)
    if not os.path.isdir(model):
        pytest.skip(f"{model} not found")
    with LLM(
        model=model, kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.3), max_batch_size=8
    ) as llama:
        table = _embedding_table(model)
        for pattern in ["whole", *MASKS]:
            _check_matches_tokens(llama, table, pattern)


def _check_matches_tokens(llm: LLM, table: torch.Tensor, pattern: str) -> None:
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
    for repeat in range(3):
        out = llm.generate([request], _greedy(), cache_salt=salt)[0]
        assert list(out.outputs[0].token_ids) == expected
        assert out.cached_tokens == (len(ids) - 1 if repeat else 0)


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


def test_chunked_prefill(model_dir: str, table: torch.Tensor) -> None:
    with LLM(
        model=model_dir,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.3, enable_block_reuse=False),
        enable_chunked_prefill=True,
        max_num_tokens=64,
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
    # Chunked prefill sends rows to the other TP rank in several passes.
    with LLM(
        model=model_dir,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.3, enable_block_reuse=True),
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        enable_chunked_prefill=True,
        max_num_tokens=64,
        max_batch_size=4,
    ) as parallel:
        ids = _token_ids(parallel, PROMPTS + [LONG_PROMPT])
        assert len(ids[3]) > 2 * 64
        expected = _run(parallel, ids)
        requests = [
            _placeholder_request(ids[0], [p % 2 == 1 for p in range(len(ids[0]))], table),
            {"prompt_embeds": table[torch.tensor(ids[1])]},
            ids[2],
            _placeholder_request(ids[3], [(p // 7) % 2 == 1 for p in range(len(ids[3]))], table),
        ]
        assert _run(parallel, requests) == expected


@pytest.mark.parametrize("host_cache_size", [None, 0], ids=["host_tier", "recompute"])
def test_preemption_with_graphs_and_compile(
    model_dir: str, table: torch.Tensor, host_cache_size: Optional[int]
) -> None:
    # Without a host tier a preempted request recomputes its context, rows included.
    _check_preemption(model_dir, table, tp_size=1, host_cache_size=host_cache_size)


@pytest.mark.gpu2
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
def test_preemption_multi_gpu(model_dir: str, table: torch.Tensor) -> None:
    # Every TP rank must preempt the same requests, though only rank 0 holds the rows.
    _check_preemption(model_dir, table, tp_size=2)


def _check_preemption(
    model_dir: str, table: torch.Tensor, tp_size: int, host_cache_size: Optional[int] = None
) -> None:
    # A small KV pool under MAX_UTILIZATION suspends running requests; their
    # context, embedding rows included, must come back intact. CUDA graphs and
    # torch.compile stay on for passes without embedding rows.
    with LLM(
        model=model_dir,
        tensor_parallel_size=tp_size,
        kv_cache_config=KvCacheConfig(
            max_tokens=640, enable_block_reuse=False, host_cache_size=host_cache_size
        ),
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
            # The generation request takes the rows' identity from the context result.
            gen_request = {"prompt_token_ids": req["prompt_token_ids"]}
            out = gen.generate_async(
                gen_request, sampling_params=_greedy(), disaggregated_params=params, cache_salt=salt
            ).result()
            return list(out.outputs[0].token_ids), out.cached_tokens

        assert handoff(request) == (_run(ctx, [ids])[0], 0)
        tokens, cached = handoff(request)
        assert tokens == _run(ctx, [ids])[0] and cached > mask.index(True)
        tokens, cached = handoff(other)
        assert tokens == _run(ctx, [other])[0] and cached <= mask.index(True)
        # Rows differing from the context request's, or added to a plain one.
        for context_prompt, gen_prompt in ((request, other), ({"prompt_token_ids": ids}, request)):
            params = (
                ctx.generate_async(
                    context_prompt,
                    sampling_params=SamplingParams(max_tokens=1),
                    disaggregated_params=DisaggregatedParams(request_type="context_only"),
                )
                .result()
                .disaggregated_params
            )
            params.request_type = "generation_only"
            with pytest.raises(ValueError, match="must match"):
                gen.generate_async(
                    gen_prompt, sampling_params=_greedy(), disaggregated_params=params
                )


@pytest.mark.parametrize("model_dtype", [torch.float16, torch.bfloat16])
def test_rows_staged_in_model_dtype(model_dtype: torch.dtype) -> None:
    # Rows of every caller dtype upload in the model dtype, so a mixed batch is
    # not promoted; float32 rows at the model dtype's limit stay finite.
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs import DecoderInputPreparer

    limit = torch.finfo(model_dtype).max
    caller_rows = [
        torch.tensor([[limit, -limit, 1 / 3, 0.1]], dtype=torch.float32),
        torch.tensor([[0.1, 0.2, 0.3, 0.4], [1.5, -2.5, 3.5, 4.5]], dtype=torch.bfloat16),
        torch.tensor([[-0.1, 0.7, 1e-3, 2.0]], dtype=torch.float16),
    ]
    snapshots = [build_prompt_embeds(None, rows)[1] for rows in caller_rows]
    host = SimpleNamespace(
        lists=SimpleNamespace(
            prompt_embeds_src_rows=[(pe, 0, pe.num_rows) for pe in snapshots],
            prompt_embeds_dst_rows=[pe.positions for pe in snapshots],
        )
    )
    preparer = SimpleNamespace(
        _config=SimpleNamespace(dtype=model_dtype), mapping=SimpleNamespace(tp_size=1, tp_rank=0)
    )
    _, rows = DecoderInputPreparer._ship_prompt_embeds(preparer, host)
    assert rows.dtype == model_dtype
    assert torch.equal(rows.cpu(), torch.cat([r.to(model_dtype) for r in caller_rows]))
    assert bool(torch.isfinite(rows).all())


def test_cancel_releases_worker_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    # Cancelling an active request (n=2: the children share its rows) and one
    # still queued behind it frees the rows the worker owns. The worker runs in
    # this process, so the test can hold weak references to them.
    model = get_model_path(QWEN3)
    if not os.path.isdir(model):
        pytest.skip(f"{model} not found")
    table = _embedding_table(model)
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    monkeypatch.setenv("TLLM_ALLOW_N_GREEDY_DECODING", "1")
    owned_rows = []

    def build_and_track(*args, **kwargs) -> tuple:
        prompt_token_ids, prompt_embeds = build_prompt_embeds(*args, **kwargs)
        owned_rows.append(weakref.ref(prompt_embeds.embeds))
        return prompt_token_ids, prompt_embeds

    monkeypatch.setattr(llm_module, "build_prompt_embeds", build_and_track)
    # Two slots: the n=2 request fills both, so the second one waits in the queue.
    with LLM(
        model=model,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.3),
        max_batch_size=2,
    ) as llm:
        ids = _token_ids(llm, PROMPTS[:1])[0]
        request = {"prompt_embeds": table[torch.tensor(ids)]}

        def params(n: int) -> SamplingParams:
            return SamplingParams(max_tokens=1000, n=n, temperature=0.0, end_id=-1)

        async def run() -> None:
            active = llm.generate_async(request, sampling_params=params(2), streaming=True)
            stream = active.__aiter__()
            await stream.__anext__()
            queued = llm.generate_async(request, sampling_params=params(1), streaming=True)
            for _ in range(4):
                await stream.__anext__()
            assert not queued.outputs[0].token_ids
            # The client keeps only the identity; the worker still owns both payloads.
            assert active._generation_request.prompt_embeds.embeds is None
            assert all(ref() is not None for ref in owned_rows)
            for result in (active, queued):
                result.abort()
                outputs = [output async for output in result]
                assert outputs[-1].outputs[0].finish_reason == "cancelled"

        # A cancellation that never ends the streams fails here instead of hanging.
        asyncio.run(asyncio.wait_for(run(), timeout=300))
        assert len(owned_rows) == 2
        # A later request runs to completion, so the executor has moved past both.
        assert llm.generate(ids, _greedy()).outputs[0].token_ids
        gc.collect()
        assert all(ref() is None for ref in owned_rows)


@pytest.mark.parametrize("backend", ["piecewise", "breakable"])
def test_prefill_cuda_graph(
    model_dir: str, table: torch.Tensor, backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Passes with embedding rows run eagerly; plain prefills replay graphs
    # before and after them. The worker runs in this process, so the test
    # counts graph replays per context pass.
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    replays = [0]
    for owner in (torch.cuda.CUDAGraph, BreakableCUDAGraphRunner):

        def counted(self, *args, _replay=owner.replay, **kwargs):
            replays[0] += 1
            return _replay(self, *args, **kwargs)

        monkeypatch.setattr(owner, "replay", counted)
    passes = []  # (carries embedding rows, graph replays) per context pass
    forward_step = DecoderRunner._forward_step

    def recorded(self, inputs: dict, **kwargs) -> dict:
        has_rows, before = "prompt_embeds_overlay" in inputs, replays[0]
        outputs = forward_step(self, inputs, **kwargs)
        if not kwargs.get("is_dummy") and inputs["attn_metadata"].num_contexts:
            passes.append((has_rows, replays[0] - before))
        return outputs

    monkeypatch.setattr(DecoderRunner, "_forward_step", recorded)
    with LLM(
        model=model_dir,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.3, enable_block_reuse=False),
        prefill_cuda_graph_backend=backend,
        torch_compile_config=TorchCompileConfig() if backend == "piecewise" else None,
        max_batch_size=4,
    ) as graphs:
        ids = _token_ids(graphs, PROMPTS)
        mixed_requests = [ids[0], {"prompt_embeds": table[torch.tensor(ids[1])]}, ids[2]]
        per_run = []
        for requests in (ids, mixed_requests, ids):
            passes.clear()
            _run(graphs, requests)
            per_run.append(list(passes))
    # Outputs are not compared: with the worker in this process, plain piecewise
    # replays flip near-tie greedy tokens from run to run without any embedding pass.
    plain_before, mixed, plain_after = per_run
    assert any(count for _, count in plain_before)
    assert any(has_rows for has_rows, _ in mixed)
    assert all(count == 0 for has_rows, count in mixed if has_rows)
    assert any(count for _, count in plain_after)
