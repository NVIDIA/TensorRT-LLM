"""Postprocess parallelism under the Ray orchestrator.

The rank-0 RayGPUWorker spawns a local PostprocWorker pool; finished
``PostprocWorker.Output`` records re-enter the RPC response stream via the
collector thread (RpcWorkerMixin.init_postproc_workers). These tests pin the
two user-visible contracts:

  1. workers>0 produces token-for-token identical text to the inline
     (workers=0) path, sync and streaming;
  2. the stream/proxy demux finalizes Output records (no leaked results,
     no hang on completion).
"""

import asyncio
import os

from utils.llm_data import llm_models_root

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


def _model_path() -> str:
    override = os.environ.get("POSTPROC_TEST_MODEL")
    if override:
        return override
    return str(llm_models_root() / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0")


def _make_llm(num_postprocess_workers: int) -> LLM:
    model = _model_path()
    extra = {}
    if num_postprocess_workers > 0:
        extra = dict(
            num_postprocess_workers=num_postprocess_workers, postprocess_tokenizer_dir=model
        )
    return LLM(
        model=model,
        kv_cache_config=KvCacheConfig(enable_block_reuse=False, max_tokens=16384),
        **extra,
    )


def _generate_sync(num_postprocess_workers: int) -> list[str]:
    sampling_params = SamplingParams(temperature=0, max_tokens=32)
    with _make_llm(num_postprocess_workers) as llm:
        outputs = llm.generate(PROMPTS, sampling_params)
        return [output.outputs[0].text for output in outputs]


def test_postproc_workers_match_inline():
    """Greedy outputs with workers=2 must equal the inline workers=0 path."""
    baseline = _generate_sync(0)
    assert all(text for text in baseline)
    with_postproc = _generate_sync(2)
    assert with_postproc == baseline


def test_streaming_control_no_postproc():
    """Control: streaming with workers=0 must work (isolates any pre-existing
    ray-streaming issue from the postproc path)."""
    _streaming_body(num_postprocess_workers=0)


def test_postproc_workers_streaming():
    """Streaming rides the same Output records; final text must match sync."""
    _streaming_body(num_postprocess_workers=2)


def _streaming_body(num_postprocess_workers: int):
    sampling_params = SamplingParams(temperature=0, max_tokens=32)

    with _make_llm(num_postprocess_workers) as llm:
        sync_texts = [output.outputs[0].text for output in llm.generate(PROMPTS, sampling_params)]

        async def collect(prompt: str) -> str:
            final = None
            async for output in llm.generate_async(prompt, sampling_params, streaming=True):
                final = output
            return final.outputs[0].text

        async def run_all() -> list[str]:
            return await asyncio.gather(*[collect(p) for p in PROMPTS])

        streamed_texts = asyncio.run(run_all())

    assert streamed_texts == sync_texts
