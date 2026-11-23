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
"""Engine, layout and GSM8K protocol the HY4-Flash target's accuracy tests share.

The engine options are the deployment's own (`intake/llm_api/<unit>.yaml`):
``max_seq_len`` 5120, ``max_batch_size`` 64, a 0.7 KV-cache fraction with block
reuse off, chunked prefill off, and the padded CUDA-graph batch list that
``max_batch_size`` 64 generates -- ``[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]``.
``core`` runs no speculation; every other member runs the checkpoint's DFly2
drafting through one ``DFlashDecodingConfig`` (five drafts, mask token 120009),
which is part of those members' deployment rather than a separate one.

GSM8K is the protocol the reference rows were measured under: the chat template
with few-shot turns, ``reasoning_effort no_think``, 4096 prompt tokens, 1024
output tokens, greedy (temperature 0), 1319 samples. The rows
are keyed by the model name plus the resolved quantization
(``FP8_BLOCK_SCALES``) and, where the member drafts, ``DFlash``.
"""

import math
import os

import pytest

from tensorrt_llm._torch.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import CudaGraphConfig, DFlashDecodingConfig, KvCacheConfig, SamplingParams
from tensorrt_llm.quantization import QuantAlgo

from ....accuracy_core import GSM8K

# The target asserts its own SM at construction: certification is per GPU
# architecture, and a receipt from another one says nothing here.
skip_not_sm103 = pytest.mark.skipif(
    get_sm_version() != 103, reason="modeling_v2 targets in this batch are certified on sm_103 only"
)

# The layouts routing admits, as LLM API parallel arguments (the keys of
# ``routing.py``'s ``_LAYOUTS``). tp4 splits the routed experts by intermediate
# and everything else by expert; dep4 replicates attention and lets each rank
# serve its own requests; tep8 spans two nodes.
TP4 = dict(
    tensor_parallel_size=4,
    moe_tensor_parallel_size=4,
    moe_expert_parallel_size=1,
    enable_attention_dp=False,
)
TEP4 = dict(tensor_parallel_size=4, moe_expert_parallel_size=4, enable_attention_dp=False)
DEP4 = dict(tensor_parallel_size=4, moe_expert_parallel_size=4, enable_attention_dp=True)
TEP8 = dict(tensor_parallel_size=8, moe_expert_parallel_size=8, enable_attention_dp=False)

MAX_SEQ_LEN = 5120
MAX_BATCH_SIZE = 64
KV_CACHE_FRACTION = 0.7

# Check the step contract on the first forward; the flag has to reach the worker ranks.
VALIDATE = {MODELING_V2_VALIDATE_ENV: "1"}

# The member's GSM8K evaluator settings: the protocol the reference rows were
# measured under.
GSM8K_EVALUATOR_KWARGS = dict(
    apply_chat_template=True,
    fewshot_as_multiturn=True,
    chat_template_kwargs=dict(reasoning_effort="no_think"),
)


def engine(**overrides) -> dict:
    """The member's LLM arguments, with fresh config objects and ``overrides`` applied.

    Fresh objects rather than module-level ones: an engine resolves fields of
    the configs it is given, and a test that builds two engines must not let the
    first one's resolution decide the second one's.
    """
    options = dict(
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE,
        enable_chunked_prefill=False,
        kv_cache_config=KvCacheConfig(
            free_gpu_memory_fraction=KV_CACHE_FRACTION, enable_block_reuse=False
        ),
        cuda_graph_config=CudaGraphConfig(enable_padding=True, max_batch_size=MAX_BATCH_SIZE),
    )
    options.update(overrides)
    return options


def dfly2() -> DFlashDecodingConfig:
    """The checkpoint's own drafting: five drafts, the checkpoint's mask token."""
    return DFlashDecodingConfig(max_draft_len=5, mask_token_id=120009)


def sampling() -> SamplingParams:
    """The anchor's generation settings: 4096 prompt tokens, 1024 new, greedy."""
    return SamplingParams(max_tokens=1024, truncate_prompt_tokens=4096, temperature=0.0)


def gsm8k_samples() -> int:
    """How many GSM8K items an evaluation runs.

    One under ``INTEGRATION_TEST``, as ``AccuracyTask.evaluate`` does: the smoke
    mode runs a single item and skips the reference check.
    """
    return 1 if os.getenv("INTEGRATION_TEST", "0") == "1" else GSM8K.NUM_SAMPLES


def assert_target_engine(llm, label: str) -> None:
    """The engine took the modeling_v2 route, and the harness reads this checkpoint's rows.

    One that did not take the route under ``require`` would not have built at all.
    The checkpoint publishes its quantization in ``text_config.quantization_config``,
    which the LLM API front end does not read (``ModelLoader._update_from_hf_quant_config``
    reads the top-level block only), so ``llm.args`` leaves ``quant_algo`` unset; the
    workers' ``ModelConfig`` reads it, and ``test_modeling_v2_hyv4_flash_config.py``
    holds that to FP8_BLOCK_SCALES. ``quant_algo`` is set here to the key the
    reference rows are stored under, as the disaggregated test does for its launcher.
    """
    assert llm.args.modeling_v2 == "require", (
        f"{label}: the engine was built with modeling_v2={llm.args.modeling_v2!r}"
    )
    quant_algo = llm.args.quant_config.quant_algo
    assert quant_algo in (None, QuantAlgo.FP8_BLOCK_SCALES), (
        f"{label}: the front end resolved quant_algo {quant_algo}; the references are "
        "this checkpoint's FP8_BLOCK_SCALES rows"
    )
    llm.args.quant_config.quant_algo = QuantAlgo.FP8_BLOCK_SCALES


def gsm8k_score(llm, model_name: str) -> float:
    """GSM8K on ``llm``, scored without the reference lookup.

    ``AccuracyTask.evaluate`` resolves a reference row from the engine's
    quantization and speculation and asserts against it. The core variants have
    no row of their own -- their anchor is core's own score at the same member
    and target -- so they are measured with the task's own evaluator and
    settings instead, which returns the number ``evaluate`` would.
    """
    task = GSM8K(model_name)
    evaluator = task.create_evaluator(gsm8k_samples(), GSM8K_EVALUATOR_KWARGS)
    return evaluator.evaluate(llm, sampling(), False, **GSM8K.EVALUATE_KWARGS)


def relative_band(score: float, samples: int, k: float = 2.0) -> float:
    """``k`` standard errors of the difference of two measurements of ``score``.

    The run's rule for a variant is the relative floor: the anchor is the unit's
    own score, measured as the variant is, so both legs carry binomial error and
    the error of their difference is ``sqrt(2)`` times one score's. The
    proportion is clamped half a sample from 0 and 1, so a 100-point score still
    has a nonzero band.
    """
    p = min(max(score / 100.0, 0.5 / samples), 1.0 - 0.5 / samples)
    return k * math.sqrt(2.0) * 100.0 * math.sqrt(p * (1.0 - p) / samples)
