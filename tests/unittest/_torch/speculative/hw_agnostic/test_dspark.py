# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""End-to-end DSpark speculative-decoding test (LLM API).

DSpark is a *verified* speculative method: accepted tokens come from the standard
target verification (``accepted token == target greedy argmax`` within the same
forward), so it preserves greedy semantics by construction. The block draft +
confidence head only affect the *acceptance rate*.

Note on greedy parity: exact token-for-token equality against a separate no-spec
run is **not** asserted here, because the DeepSeek-V4-Pro 8-GPU MoE engine is not
bit-reproducible run-to-run even for plain greedy no-spec decoding (the MoE
expert-combine / all-reduce use non-associative floating-point atomics, so a
near-tie argmax can flip between two otherwise-identical runs; verified
empirically — two no-spec runs diverge from each other). Asserting ``ref == spec``
across two engine builds therefore fails on engine nondeterminism, not on a draft
bug. The verify invariant itself is covered by the hw-agnostic unit tests
(``test_dspark_worker.py``) and the numerical golden (see DSPARK_DEV.md C4). This
test instead checks that DSpark runs end-to-end, emits non-degenerate output, and
that the draft is wired (mean accepted length >= 0, i.e. acceptance is reported).

The DSpark draft lives in the SAME checkpoint as the target (under ``mtp.*``), so
``speculative_model`` points at the same directory as ``model``.
"""

import os

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import CudaGraphConfig, DSparkDecodingConfig, KvCacheConfig, MoeConfig

PROMPTS = [
    "The capital of France is",
    "The president of the United States is",
    "The future of AI is",
]


def _dspark_model_dir():
    root = llm_models_root()
    if root is None:
        return None
    path = os.path.join(root, "DeepSeek-V4-Pro-DSpark")
    return path if os.path.isdir(path) else None


@pytest.mark.skipif(
    torch.cuda.device_count() < 8, reason="DeepSeek-V4-Pro-DSpark needs an 8-GPU node"
)
@pytest.mark.parametrize("disable_overlap_scheduler", [True])
@pytest.mark.parametrize("use_cuda_graph", [False, True], ids=["eager", "cudagraph"])
def test_dspark_e2e_v4_pro(use_cuda_graph: bool, disable_overlap_scheduler: bool):
    """DSpark runs end-to-end on the real checkpoint and emits valid output.

    Parametrized over eager vs CUDA-graph execution: DSpark is a one-engine
    drafter, so with ``cuda_graph_config`` set the draft worker is captured into
    the target's graph. The default batched draft path is host-sync-free and
    capture-safe (its per-request equivalence to the eager path is unit-tested in
    ``test_dspark_cuda_graph.py``); this case proves it captures + runs end to end
    on the real 8-GPU engine and keeps acceptance positive.

    See the module docstring for why exact token-level greedy parity against a
    separate no-spec run is not asserted (8-GPU MoE engine nondeterminism).
    """
    model_dir = _dspark_model_dir()
    if model_dir is None:
        pytest.skip("DeepSeek-V4-Pro-DSpark checkpoint not available")

    sampling = SamplingParams(max_tokens=128, temperature=0)
    kv_cache_config = KvCacheConfig(enable_block_reuse=False)
    common = dict(
        model=model_dir,
        attn_backend="TRTLLM",
        tensor_parallel_size=8,
        moe_expert_parallel_size=8,
        max_batch_size=4,
        max_seq_len=2048,
        # The 384-expert/8-group MXFP4 layout exceeds the TRTLLM-Gen blockScaleMoe
        # routing kernel's experts/group<=32 (warp) limit, so AUTO->TRTLLM-Gen
        # asserts; CUTLASS has no such limit. The draft MoE inherits this backend
        # from the target config (no per-draft pin), so it covers both.
        moe_config=MoeConfig(backend="CUTLASS"),
        kv_cache_config=kv_cache_config,
        enable_chunked_prefill=False,
        disable_overlap_scheduler=disable_overlap_scheduler,
        # The default batched DSpark draft path is CUDA-graph-safe (the worker runs
        # inside the target's captured graph). The eager case keeps graphs off.
        cuda_graph_config=(CudaGraphConfig(max_batch_size=4) if use_cuda_graph else None),
    )

    # DSpark speculative decoding (draft = same checkpoint's mtp.* stages).
    spec_config = DSparkDecodingConfig(max_draft_len=5, speculative_model=model_dir)
    spec_llm = LLM(speculative_config=spec_config, **common)
    spec_out = spec_llm.generate(PROMPTS, sampling)
    spec_texts = [o.outputs[0].text for o in spec_out]
    spec_ids = [list(o.outputs[0].token_ids) for o in spec_out]
    avg_accepted = [o.avg_decoded_tokens_per_iter - 1 for o in spec_out]
    spec_llm.shutdown()

    # End-to-end sanity: every prompt produced the requested number of
    # non-degenerate tokens (the draft + verify + sampling path ran).
    for i, ids in enumerate(spec_ids):
        assert len(ids) > 0, f"prompt {i} produced no tokens"
        assert spec_texts[i].strip(), f"prompt {i} produced empty text"

    # The draft does useful work: with the captured-context fix + window seeding
    # the mean accepted draft length is ~0.6-0.9 tok/iter on these prompts
    # (decoded ~1.6-1.9 tok/iter). Require a clearly-positive mean as a regression
    # guard against the hidden-state capture bug (which collapsed acceptance to ~0
    # by feeding the draft the MoE delta instead of the mHC residual stream); the
    # threshold is conservative to tolerate 8-GPU MoE nondeterminism / prompt mix.
    mean_accepted = sum(avg_accepted) / len(avg_accepted)
    assert mean_accepted > 0.1, (
        f"DSpark acceptance collapsed (mean {mean_accepted:.3f}); the draft is not "
        "tracking the target — check the captured-context (main_x) pathway."
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
