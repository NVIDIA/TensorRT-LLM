# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end prompt_embeds tests on the PyTorch backend.

Rows taken from the checkpoint's own input embedding must reproduce the greedy
output of the equivalent token prompt exactly, whichever positions carry them.
"""

import os
import pickle
import uuid
from typing import Iterator, List, Optional

import pytest
import torch
from safetensors import safe_open

from tensorrt_llm import LLM
from tensorrt_llm.executor import RequestError
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.inputs.prompt_embeds import build_prompt_embeds, prompt_embeds_cache_salt
from tensorrt_llm.llmapi import KvCacheConfig
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


def _generated(outputs) -> List[List[int]]:
    return [list(o.outputs[0].token_ids) for o in outputs]


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


def test_request_salt_carries_identity() -> None:
    # Direct executor callers get the same reuse scoping as LLM.generate.
    _, pe = build_prompt_embeds([1, 2, 3], torch.ones(3, 4))
    _, other = build_prompt_embeds([1, 2, 3], torch.zeros(3, 4))
    params = SamplingParams(max_tokens=1)
    plain = GenerationRequest([1, 2, 3], params, cache_salt="s")
    a = GenerationRequest([1, 2, 3], params, cache_salt="s", prompt_embeds=pe)
    b = GenerationRequest([1, 2, 3], params, cache_salt="s", prompt_embeds=other)
    assert a.cache_salt == prompt_embeds_cache_salt("s", pe.digest)
    assert len({plain.cache_salt, a.cache_salt, b.cache_salt}) == 3
    restored = pickle.loads(pickle.dumps(a))
    assert restored.cache_salt == a.cache_salt
    assert torch.equal(restored.prompt_embeds.embeds, pe.embeds)


def test_whole_prompt_matches_tokens(llm: LLM, table: torch.Tensor) -> None:
    ids = _token_ids(llm, PROMPTS)
    expected = _run(llm, ids)
    embeds = [{"prompt_embeds": table[torch.tensor(i)]} for i in ids]
    assert _run(llm, embeds) == expected


@pytest.mark.parametrize("pattern", ["prefix", "suffix", "interleaved"])
def test_partial_mask_matches_tokens(llm: LLM, table: torch.Tensor, pattern: str) -> None:
    ids = _token_ids(llm, PROMPTS)
    expected = _run(llm, ids)
    requests = []
    for prompt_ids in ids:
        n = len(prompt_ids)
        if pattern == "prefix":
            mask = [p < n // 2 for p in range(n)]
        elif pattern == "suffix":
            mask = [p >= n // 2 for p in range(n)]
        else:
            mask = [p % 3 != 0 for p in range(n)]
        positions = [p for p in range(n) if mask[p]]
        # Placeholders at embedding positions must not matter.
        placeholder_ids = [0 if m else t for t, m in zip(prompt_ids, mask)]
        requests.append(
            {
                "prompt_token_ids": placeholder_ids,
                "prompt_embeds": table[torch.tensor([prompt_ids[p] for p in positions])],
                "prompt_embeds_mask": mask,
            }
        )
    assert _run(llm, requests) == expected


def test_mixed_batch_matches_tokens(llm: LLM, table: torch.Tensor) -> None:
    ids = _token_ids(llm, PROMPTS)
    expected = _run(llm, ids)
    mixed = [
        ids[0],
        {"prompt_embeds": table[torch.tensor(ids[1])]},
        {"prompt_token_ids": ids[2]},
    ]
    assert _run(llm, mixed) == expected


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

    # The plain prompt still reuses its own blocks and is unaffected.
    plain = llm.generate([ids], _greedy(), cache_salt=salt)[0]
    assert list(plain.outputs[0].token_ids) == expected


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
        requests = [
            {
                "prompt_token_ids": ids[0],
                "prompt_embeds": table[torch.tensor([t for t, m in zip(ids[0], mask) if m])],
                "prompt_embeds_mask": mask,
            },
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
            {
                "prompt_token_ids": ids[0],
                "prompt_embeds": table[torch.tensor([t for t, m in zip(ids[0], mask) if m])],
                "prompt_embeds_mask": mask,
            },
            {"prompt_embeds": table[torch.tensor(ids[1])]},
            ids[2],
        ]
        assert _run(parallel, requests) == expected
