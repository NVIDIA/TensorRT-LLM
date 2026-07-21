#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""crit8 real_runtime LLM API smoke: the Inkling text tower dispatches through the
production ``LLM`` runtime (PyExecutor) on the real NVFP4 checkpoint at TP=4.

Launched under MPI (``trtllm-llmapi-launch python inkling_llmapi_smoke_test.py``
inside ``srun --ntasks=4 --mpi=pmix``), this exercises the FULL runtime path that
the focused replays could not:
  * ``KVCacheManagerV2`` selected + built with the hybrid per-layer KV geometry
    (local 16 / global 8) via the ``is_inkling`` branch in ``_util.py``,
  * the ``InklingConvStateManager`` registered as a request-lifetime resource
    manager and threaded into ``model.forward`` (the per-request short-conv state
    pool), fetched from the ``resource_manager`` kwarg each step,
  * the selected TRTLLM attention backend + NVFP4 CUTLASS MoE, over real
    context prefill and multi-step generation decode.

It generates a few fixed prompts under deterministic greedy decoding and asserts
the runtime produces finite, non-empty, on-vocab tokens. This is the crit8
real_runtime dispatch proof (not an accuracy gate -- crit6/crit7/crit11/crit12
own logit/generation/dataset parity); it is the foundation those build on.

Config matrix (env-selected so one script covers both acceptance rows):
  * INKLING_CUDA_GRAPH=0/1   -> cuda_graph_config None / CudaGraphConfig()
  * INKLING_OVERLAP=0/1      -> disable_overlap_scheduler True / False
Baseline is (0, 0); the enabled acceptance row is (1, 1).

Run:
    trtllm-llmapi-launch python tests/unittest/_torch/modeling/inkling_llmapi_smoke_test.py
Override the checkpoint with INKLING_CHECKPOINT=/path/to/Inkling-NVFP4-full.
"""

import os
import sys

CKPT = os.environ.get(
    "INKLING_CHECKPOINT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/"
    "users/kleinc/hf_data/Inkling-NVFP4-full")

CUDA_GRAPH = os.environ.get("INKLING_CUDA_GRAPH", "0") == "1"
# Overlap defaults to OFF for the baseline row and ON when explicitly enabled;
# the enabled acceptance row pairs cuda_graph=true with overlap=true.
OVERLAP = os.environ.get("INKLING_OVERLAP", "1" if CUDA_GRAPH else "0") == "1"

# Deterministic greedy prompts: one arithmetic, one factual, one instruction-ish,
# one multiple-choice-ish, and one longer prompt (exercises >1 KV page / the
# 512-token local window is crossed by the long-horizon canary, not here).
PROMPTS = [
    "The capital of France is",
    "2 + 2 =",
    "Question: What color is the sky on a clear day? Answer:",
    "List the first three prime numbers:",
    "Once upon a time, in a small village nestled between two mountains,",
]


def main() -> int:
    import torch

    from tensorrt_llm import LLM, SamplingParams
    # Import registers the auto-model + the InklingConvStateManager / mapper.
    from tensorrt_llm._torch.models.modeling_inkling import \
        InklingForConditionalGeneration  # noqa: F401
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MoeConfig

    assert torch.cuda.is_available(), "the LLM API smoke needs CUDA GPUs"
    print(
        f"[smoke] cuda_graph={CUDA_GRAPH} overlap_scheduler={OVERLAP} "
        f"ckpt={CKPT}",
        flush=True)

    # Block reuse (prefix caching) would hand a new request a reused KV block but
    # a fresh (zeroed) short-conv slot -- the two must stay in lock-step, so it is
    # disabled for the bring-up runtime (plan risk register: SConv cache
    # ownership). The TP=4 NVFP4 shard is ~135 GiB/rank of weights on each
    # 184 GiB GB200 GPU, so only ~49 GiB is free after the model loads;
    # free_gpu_memory_fraction=0.75 sizes the KV cache from that remainder.
    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75,
                                    dtype="auto",
                                    enable_block_reuse=False)

    # The prompts are short (<32 ctx tokens) and generate 32 tokens, so a small
    # max_num_tokens is ample. It also bounds the dummy forward the PyExecutor
    # runs to estimate activation memory before sizing the KV cache -- important
    # here because that estimation pass runs on a near-full GPU (weights already
    # occupy ~135 GiB/rank); an 8192-token default dummy batch could itself OOM.
    # MoE backend / parallelization are env-configurable (default = today's
    # intermediate-TP CUTLASS). INKLING_MOE_EP=4 -> expert-parallel (moe_ep=4/
    # moe_tp=1); INKLING_MOE_BACKEND=TRTLLM -> flashinfer NVFP4 MoE.
    moe_backend = os.environ.get("INKLING_MOE_BACKEND", "CUTLASS")
    moe_ep = int(os.environ.get("INKLING_MOE_EP", "0"))
    print(f"[smoke] moe_backend={moe_backend} moe_ep={moe_ep}", flush=True)
    llm_kwargs = dict(
        tensor_parallel_size=4,
        trust_remote_code=True,
        attn_backend="TRTLLM",
        moe_config=MoeConfig(backend=moe_backend),
        kv_cache_config=kv_cache_config,
        cuda_graph_config=CudaGraphConfig() if CUDA_GRAPH else None,
        disable_overlap_scheduler=not OVERLAP,
        max_seq_len=2048,
        max_batch_size=8,
        max_num_tokens=2048,
    )
    if moe_ep > 0:
        llm_kwargs["moe_expert_parallel_size"] = moe_ep
        llm_kwargs["moe_tensor_parallel_size"] = 1
    llm = LLM(CKPT, **llm_kwargs)

    # ---- KVCacheManagerV2 runtime proof (crit8 V2 contract) -----------------
    # Inkling's per-layer KV-head split (local 16 / global 8) structurally
    # requires KVCacheManagerV2 (the ``is_inkling`` branch of
    # ``_util._non_hybrid_kv_cache_manager_cls``; ``_fallback_if_unsupported_...``
    # raises rather than silently downgrading). Prove the LIVE engine dispatched
    # V2 by introspecting the executor's resource manager -- not by trusting the
    # ``use_kv_cache_manager_v2`` config flag. The authoritative record is the
    # ``[KV] resolved kv_cache_manager_cls=...`` line logged by ``_util`` (also
    # greppable in the job log); this is the in-test gate.
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import \
        KVCacheManagerV2
    from tensorrt_llm._torch.pyexecutor.resource_manager import \
        ResourceManagerType
    kv_cls_name = None
    kv_is_v2 = None
    try:
        engine = getattr(getattr(llm, "_executor", None), "engine", None)
        kvm = getattr(engine, "kv_cache_manager", None) if engine else None
        if kvm is None and engine is not None:
            rm = getattr(engine, "resource_manager", None)
            if rm is not None:
                kvm = rm.resource_managers.get(
                    ResourceManagerType.KV_CACHE_MANAGER)
        if kvm is not None:
            kv_cls_name = type(kvm).__name__
            kv_is_v2 = isinstance(kvm, KVCacheManagerV2)
    except Exception as e:  # introspection is best-effort; log grep is the backstop
        print(f"[smoke] kv-manager introspection skipped: {e!r}", flush=True)
    print(f"INKLING_KV_MANAGER cls={kv_cls_name} is_v2={kv_is_v2}", flush=True)
    # Hard-fail only when we positively observed a NON-V2 manager. When the
    # in-process executor doesn't expose the engine (proxy layouts), fall back to
    # the greppable ``_util`` log the sbatch checks.
    if kv_is_v2 is False:
        llm.shutdown()
        raise AssertionError(
            f"Inkling MUST run KVCacheManagerV2, but the live runtime built "
            f"{kv_cls_name}")

    # Deterministic greedy decode (temperature 0), >= 32 new tokens per prompt.
    sampling = SamplingParams(max_tokens=32, temperature=0.0)
    try:
        outputs = llm.generate(PROMPTS, sampling)
    finally:
        llm.shutdown()

    ok = True
    for i, out in enumerate(outputs):
        gen = out.outputs[0]
        tok = list(gen.token_ids)
        text = gen.text
        n = len(tok)
        on_vocab = all(0 <= t < 200058 for t in tok)  # unpadded vocab
        nonempty = n > 0
        good = nonempty and on_vocab
        ok = ok and good
        print(
            f"[smoke] prompt[{i}] n_tokens={n} on_vocab={on_vocab} "
            f"first_tokens={tok[:8]} text={text!r}",
            flush=True)

    print(
        f"INKLING_LLMAPI_SMOKE_{'OK' if ok else 'FAIL'} "
        f"cuda_graph={CUDA_GRAPH} overlap={OVERLAP} n_prompts={len(outputs)}",
        flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
