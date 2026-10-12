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
"""Whole-model gates for the deepseek_v41 modeling_v2 target.

One file per model family, beside the other accuracy suites rather than inside
test_llm_api_pytorch.py: these gate a parallel implementation, and reading them
next to the built-in DeepSeek-V4.1 tests (``TestDeepSeekV41*`` there) would
invite treating one as a variant of the other.

The target is ``ModelingV2DeepseekV41FlashNvfp4Sm103Unified``: one modeling
codebase that folds three 4-rank members at runtime from the mapping -- tep4
(attention TP4 + MoE EP4), dep4 (attention DP4 + MoE EP4) and tp4 (attention
TP4 + MoE TP4). Each case gates against the anchor in
``references/{gsm8k,mmmu}.yaml``, keyed by this checkpoint's name.

Every case runs the target's default feature set: the checkpoint's DSpark
drafter (its ``mtp.*`` stages) at ``max_draft_len`` 5, CUDA graphs, chunked
prefill, the overlap scheduler and KV block reuse, with the Engram tables in
host memory and the KV pools in native (FP8/FP4) storage. The encoder and
decoder SWA bounded replays are left to the cache manager, which turns them on
for every member (MMMU's reuse-off engine has no prefix hit to replay). MMMU
turns block reuse off, because multimodal requests with reuse on are recorded
as crashing and the anchor was measured with prefix caching off.

Every test here runs under ``TRTLLM_MODELING_V2=require``, which the case puts
in place itself through ``modeling_v2_env``: the switch travels as an
``LLM(...)`` field, so it reaches every rank whether the LLM API spawned the
ranks or ``trtllm-llmapi-launch`` started them. Under ``"auto"`` a configuration
that missed the target's criteria would fall back to the built-in
implementation instead of failing, and a pass would then be reported as the
target's. Each case also asks the target to check its step contract on its
first forward, as the other modeling_v2 gates do. The feature switches are left
to the launch; a case only checks that each one is at its default, which is
what ``_require_default_features`` does. On a host where the LLM API cannot
spawn its workers, launch the file under
``mpirun -n 4 --oversubscribe trtllm-llmapi-launch pytest ...``.

The protocol is the one the anchors were measured under: ``trtllm-eval
gsm8k`` 5-shot plain completion with no chat template, and ``trtllm-eval
mmmu`` 0-shot with the V4.1 non-thinking chat rendering; ``random_seed`` 0,
greedy, 512 output tokens, ``max_seq_len`` 16384 and ``max_batch_size`` 32.
"""

import os

import pytest

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.engram_host_table import (  # noqa: E501
    HOST_OFFLOAD_ENV,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.v41_layout import (
    DEEPSEEK_V41_DECODER_REPLAY_ENV,
    DEEPSEEK_V41_ENCODER_REPLAY_ENV,
    DEEPSEEK_V41_KV_STORAGE_ENV,
    DEEPSEEK_V41_KV_STORAGE_NATIVE,
)
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import CudaGraphConfig, DSparkDecodingConfig, KvCacheConfig

from ..conftest import llm_models_root
from .accuracy_core import GSM8K, MMMU, LlmapiAccuracyTestHarness
from .modeling_v2_env import modeling_v2_llm_args

_MODEL_PATH = f"{llm_models_root()}/DeepSeek-V4.1-Flash-NVFP4"

# The target asserts its own SM at construction: certification is per GPU
# architecture, and a receipt from another one says nothing here.
skip_not_sm103 = pytest.mark.skipif(
    get_sm_version() != 103,
    reason="the deepseek_v41 modeling_v2 target is certified on sm_103 only",
)
skip_no_checkpoint = pytest.mark.skipif(
    not os.path.isdir(_MODEL_PATH), reason=f"checkpoint not staged at {_MODEL_PATH}"
)

# Values each feature switch may hold for a case to measure the default
# feature set. Unset is always the default. The two replays accept nothing
# else: unset lets the cache manager resolve them per engine, while an explicit
# value turns one off or forces it on.
_DEFAULT_FEATURE_VALUES = {
    HOST_OFFLOAD_ENV: ("1", "true", "yes", "on"),
    DEEPSEEK_V41_KV_STORAGE_ENV: (DEEPSEEK_V41_KV_STORAGE_NATIVE,),
    DEEPSEEK_V41_ENCODER_REPLAY_ENV: (),
    DEEPSEEK_V41_DECODER_REPLAY_ENV: (),
}


def _require_default_features() -> None:
    """Skip unless the ranks were started with every feature at its default.

    Not a failure: the ranks read these switches as the launch set them, so a
    launch that overrode one measures a different engine than the one these
    anchors gate, and has nothing to say about them.
    """
    for name, accepted in _DEFAULT_FEATURE_VALUES.items():
        value = os.environ.get(name)
        if value is not None and value.strip().lower() not in accepted:
            pytest.skip(f"{name}={value!r}; these gates measure the target's default features")


@skip_not_sm103
@skip_no_checkpoint
@pytest.mark.skip_less_device(4)
class TestModelingV2DeepseekV41FlashNvfp4Sm103Unified(LlmapiAccuracyTestHarness):
    """deepseek-v4.1-flash-nvfp4 / sm_103 / unified (tep4, dep4, tp4)."""

    MODEL_NAME = "nvidia/DeepSeek-V4.1-Flash-NVFP4"
    MODEL_PATH = _MODEL_PATH

    # The three members of the unified target. Routing derives the member from
    # the mapping, and the target asserts it against the one the engine built.
    # tp4 leaves the MoE split unset, as its gate did: it resolves to MoE TP4
    # (moe_tp_size 4, moe_ep_size 1).
    MEMBERS = {
        "tep4": dict(tensor_parallel_size=4, moe_expert_parallel_size=4),
        "dep4": dict(tensor_parallel_size=4, moe_expert_parallel_size=4, enable_attention_dp=True),
        "tp4": dict(tensor_parallel_size=4),
    }

    # max_position_embeddings is 1048576; left to default, the engine would
    # size its KV cache for a million-token sequence.
    MAX_SEQ_LEN = 16384
    MAX_BATCH_SIZE = 32

    # The anchors' protocol: greedy, 512 output tokens. The input caps are the
    # tasks' own (GSM8K 4096, MMMU 8192), and the MMMU stop string is the one
    # `trtllm-eval mmmu` passes.
    MAX_OUTPUT_LEN = 512
    MMMU_STOP = "<|endoftext|>"

    # The tokenizer ships no chat template; the DeepSeek-V4 tokenizer renders
    # the V4.1 non-thinking chat prompt, byte-identical to the checkpoint's
    # encoding/encoding.py on all 900 MMMU prompts.
    MMMU_TOKENIZER = "deepseek_v4"
    MMMU_EVALUATOR_KWARGS = dict(chat_template_kwargs={"thinking": False})

    def _llm(
        self, member: str, monkeypatch: pytest.MonkeyPatch, *, multimodal: bool = False
    ) -> LLM:
        return LLM(
            self.MODEL_PATH,
            max_seq_len=self.MAX_SEQ_LEN,
            max_batch_size=self.MAX_BATCH_SIZE,
            max_num_tokens=8192,
            enable_chunked_prefill=True,
            disable_overlap_scheduler=False,
            cuda_graph_config=CudaGraphConfig(
                enable_padding=True, max_batch_size=self.MAX_BATCH_SIZE
            ),
            kv_cache_config=KvCacheConfig(
                free_gpu_memory_fraction=0.5, enable_block_reuse=not multimodal
            ),
            # The checkpoint's mtp.* stages are its DSpark block drafter, loaded
            # from the target checkpoint itself.
            speculative_config=DSparkDecodingConfig(max_draft_len=5),
            custom_tokenizer=self.MMMU_TOKENIZER if multimodal else None,
            **self.MEMBERS[member],
            **modeling_v2_llm_args("require", monkeypatch, **{MODELING_V2_VALIDATE_ENV: "1"}),
        )

    @pytest.mark.parametrize("member", ["tep4", "dep4", "tp4"])
    def test_gsm8k(self, member, monkeypatch):
        _require_default_features()
        with self._llm(member, monkeypatch) as llm:
            task = GSM8K(self.MODEL_NAME)
            # No scores filter: the anchor is the mean of strict-match and
            # flexible-extract, which is what `trtllm-eval gsm8k` reports and
            # what this task averages by default.
            task.evaluate(
                llm,
                sampling_params=SamplingParams(
                    max_tokens=self.MAX_OUTPUT_LEN, truncate_prompt_tokens=GSM8K.MAX_INPUT_LEN
                ),
            )

    @pytest.mark.parametrize("member", ["tep4"])
    def test_mmmu(self, member, monkeypatch):
        _require_default_features()
        with self._llm(member, monkeypatch, multimodal=True) as llm:
            task = MMMU(self.MODEL_NAME)
            task.evaluate(
                llm,
                sampling_params=SamplingParams(
                    max_tokens=self.MAX_OUTPUT_LEN,
                    truncate_prompt_tokens=MMMU.MAX_INPUT_LEN,
                    stop=self.MMMU_STOP,
                ),
                extra_evaluator_kwargs=self.MMMU_EVALUATOR_KWARGS,
            )
