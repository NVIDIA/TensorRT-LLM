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
"""Shared harness for the kimi_k31 modeling_v2 accuracy targets.

Each target's gate class is in ``<target>/test_accuracy.py``. It names the deployments each test runs at and lists its
tests, which run the ``_test_*`` implementations here; the deployments are here too.

Kimi K3.1 Pro: DSA sparse attention on the MLA layers, MXFP4 routed experts, FP8 index keys and the exact radix top-k.
A deployment is ``<layout><ranks>``: ``tep8`` (attention TP, 12 heads per rank), ``dep16`` (attention DP, 96 query
heads per rank padded to 128 in the sparse MLA). Attention DP keeps every rank's BF16 attention, shared-expert and
embedding weights whole, so at 8 ranks the weights alone fill a GB300 or VR200; attention DP runs 16 ranks. On sm_103
the FP8 latent KV cache routes TEP8 and DEP16 to their perf targets. The ``tep8_bf16kv`` case uses the checkpoint's
default BF16 latent KV cache and runs the general target.

Every test passes ``modeling_v2="require"`` to ``LLM``: under ``"auto"`` a deployment that missed the target would be
refused (Kimi K3.1 has no built-in implementation), but ``require`` names the reason. The engine also checks its step
contract on the first forward (``MODELING_V2_VALIDATE_ENV``).

With DFlash2 speculation (``_test_gsm8k_dflash2``) a gate can also check the acceptance length: block verification
keeps the text correct when the drafter regresses, so only the acceptance length shows it. The floors are in
``references/acceptance_length.yaml``, keyed by target and verify width.

The sm_107 targets of the ModelOpt NVFP4 export run their own recipe; its helpers are at the end of this file.
"""

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import (
    CudaGraphConfig,
    DeepSeekSparseAttentionConfig,
    DFlashDecodingConfig,
    KvCacheConfig,
    MoeConfig,
    SamplingParams,
)

from ....conftest import llm_models_root
from ...accuracy_core import (
    GSM8K,
    ForceTokenLogitsProcessor,
    GPQADiamond,
    LlmapiAccuracyTestHarness,
    assert_acceptance_length,
    compute_acceptance_length,
)
from .._env import scoped_env_overrides

# Certification is per GPU architecture: the perf targets on sm_103, the general target on sm_103 and sm_107.
skip_not_sm103 = pytest.mark.skipif(
    get_sm_version() != 103,
    reason="the kimi_k31 sm_103 modeling_v2 targets are certified on sm_103 only",
)
skip_not_sm103_or_sm107 = pytest.mark.skipif(
    get_sm_version() not in (103, 107),
    reason="the kimi_k31 general modeling_v2 target is certified on sm_103 and sm_107 only",
)
ON_SM103 = get_sm_version() == 103


def assert_logits_processor(llm: LLM) -> None:
    """A logits post-processor (``SamplingParams.logits_processor``) reaches greedy decoding: one that forces a token
    makes every generated token that token."""
    forced_token_id = 22
    output_length = 4
    outputs = llm.generate(
        [[1, 42, 43]],
        sampling_params=SamplingParams(
            max_tokens=output_length,
            temperature=0,
            end_id=-1,
            logits_processor=ForceTokenLogitsProcessor(forced_token_id),
        ),
        use_tqdm=False,
    )
    assert isinstance(outputs, list)
    assert len(outputs) == 1
    assert outputs[0].outputs[0].token_ids == [forced_token_id] * output_length


def _kv_cache_config(dtype: str) -> KvCacheConfig:
    return KvCacheConfig(
        use_kv_cache_manager_v2=True,
        dtype=dtype,
        tokens_per_block=64,
        enable_block_reuse=False,
        free_gpu_memory_fraction=0.5,
    )


# Parallel arguments and engine bounds by layout. TEP8: attention TP with experts EP8. DEP: attention DP with experts
# EP over every rank; its bounds are per rank.
_TEP8 = dict(
    tensor_parallel_size=8,
    moe_expert_parallel_size=8,
    enable_attention_dp=False,
    max_batch_size=8,
    max_num_tokens=8192,
    cuda_graph_config=CudaGraphConfig(max_batch_size=8),
    moe_config=MoeConfig(max_num_tokens=33024),
)


def _dep(ranks: int) -> dict:
    return dict(
        tensor_parallel_size=ranks,
        moe_expert_parallel_size=ranks,
        enable_attention_dp=True,
        max_batch_size=8,
        max_num_tokens=8192,
        cuda_graph_config=CudaGraphConfig(max_batch_size=8, enable_padding=True),
        moe_config=MoeConfig(max_num_tokens=32768),
    )


_LAYOUTS = {"tep8": _TEP8, "dep16": _dep(16)}


def pytest_generate_tests(metafunc):
    """Run each gate test at the deployments its class names for it (``DEPLOYMENTS``).

    The test ids do not depend on the GPU: a deployment the class leaves to the perf targets on sm_103
    (``SM103_PERF_TARGET_DEPLOYMENTS``) is skipped there.
    """
    if "deployment" not in metafunc.fixturenames:
        return
    names = metafunc.cls.DEPLOYMENTS.get(metafunc.function.__name__, ())
    perf_target_deployments = getattr(metafunc.cls, "SM103_PERF_TARGET_DEPLOYMENTS", frozenset())
    params = []
    for name in names:
        ranks = int(name.split("_")[0].lstrip("tepd"))
        marks = [pytest.mark.skip_less_mpi_world_size(ranks)]
        if ON_SM103 and name in perf_target_deployments:
            marks.append(
                pytest.mark.skip(reason="on sm_103 the perf targets serve this deployment")
            )
        params.append(pytest.param(name, marks=marks))
    metafunc.parametrize("deployment", params)


class _K31Target(LlmapiAccuracyTestHarness):
    """The gate of one target: each test runs at the deployments ``DEPLOYMENTS`` names for it."""

    MODEL_NAME = "moonshotai/Kimi-K3.1-Pro"
    MODEL_PATH = f"{llm_models_root()}/Kimi-K3.1-Pro"
    # The DFlash2 drafter the perf targets' speculation is written for.
    DRAFTER_PATH = f"{llm_models_root()}/Kimi-K3.1-Dflash2-Preview"
    # Test name -> the deployments it runs at.
    DEPLOYMENTS = {}

    def _llm_kwargs(self, deployment: str, max_seq_len: int, rank_env: dict, **kwargs) -> dict:
        """The LLM arguments of ``deployment`` after the model path."""
        layout = deployment.removesuffix("_bf16kv")
        return dict(
            modeling_v2="require",
            max_seq_len=max_seq_len,
            trust_remote_code=True,
            kv_cache_config=_kv_cache_config("auto" if deployment.endswith("_bf16kv") else "fp8"),
            sparse_attention_config=DeepSeekSparseAttentionConfig(indexer_k_dtype="fp8"),
            env_overrides=rank_env,
            **_LAYOUTS[layout],
            **kwargs,
        )

    def _test_gsm8k(self, deployment: str, logits_processor: bool = False) -> None:
        """GSM8K; with ``logits_processor``, first the logits post-processor check (``assert_logits_processor``)."""
        with (
            scoped_env_overrides({MODELING_V2_VALIDATE_ENV: "1"}) as rank_env,
            LLM(self.MODEL_PATH, **self._llm_kwargs(deployment, 8192, rank_env)) as llm,
        ):
            if deployment.endswith("_bf16kv"):
                assert llm.args.quant_config.kv_cache_quant_algo is None
            if logits_processor:
                assert_logits_processor(llm)
            GSM8K(self.MODEL_NAME).evaluate(llm)

    def _test_gsm8k_dflash2(
        self, deployment: str, max_draft_len: int, acceptance_key: str | None = None
    ) -> None:
        """GSM8K with DFlash2 at verify width ``max_draft_len + 1`` and block verification (rejection sampling), then,
        with ``acceptance_key``, the acceptance length against that key in ``acceptance_length.yaml``."""
        spec_config = DFlashDecodingConfig(
            max_draft_len=max_draft_len,
            speculative_model=self.DRAFTER_PATH,
            use_rejection_sampling=True,
        )
        with (
            scoped_env_overrides({MODELING_V2_VALIDATE_ENV: "1"}) as rank_env,
            LLM(
                self.MODEL_PATH,
                **self._llm_kwargs(
                    deployment,
                    8192,
                    rank_env,
                    speculative_config=spec_config,
                    enable_iter_perf_stats=True,
                ),
            ) as llm,
        ):
            GSM8K(self.MODEL_NAME).evaluate(llm)
            if acceptance_key is not None:
                acceptance_length = compute_acceptance_length(llm)
                print(f"[AL] {acceptance_key} acceptance_length = {acceptance_length:.3f}")
                assert_acceptance_length(acceptance_key, acceptance_length)

    def _test_gpqa_diamond(self, deployment: str) -> None:
        """GPQA Diamond with the model's chat template."""
        sampling_params = SamplingParams(
            max_tokens=GPQADiamond.MAX_OUTPUT_LEN,
            truncate_prompt_tokens=GPQADiamond.MAX_INPUT_LEN,
        )
        max_seq_len = GPQADiamond.MAX_INPUT_LEN + GPQADiamond.MAX_OUTPUT_LEN
        with (
            scoped_env_overrides({MODELING_V2_VALIDATE_ENV: "1"}) as rank_env,
            LLM(self.MODEL_PATH, **self._llm_kwargs(deployment, max_seq_len, rank_env)) as llm,
        ):
            GPQADiamond(self.MODEL_NAME).evaluate(
                llm,
                sampling_params=sampling_params,
                extra_evaluator_kwargs=dict(apply_chat_template=True),
            )


# The sm_107 targets of the ModelOpt NVFP4 export (``kimi_k31_pro_nvfp4__sm_107__{dep16,helix8,tep8,default}``).
#
# The specialized targets run their one recipe: the export as stored, an FP8 latent KV cache, FP8 index keys with the
# GVR top-k, an FP32 KDA state and DFlash2 speculative decoding. DEP16 and TEP8 run as aggregated servers, which take
# the generation server's role (its MoE backend and draft length 7); helix8 serves generation only, so its gate runs
# disaggregated behind a DEP16 context server at draft length 5. The default target serves the export on the same
# layouts with the settings the specialized targets do not take; its gates run the specialized TEP8 aggregated server
# on one such recipe each. A Rubin host has 4 GPUs, so these gates run from the QA multi-node list.

#: The ModelOpt NVFP4 export (nvidia/kimi_k31_pro_nvfp4) and the DFlash2 MLA drafter
#: (RadixArk/Kimi-K3.1-Dflash2-Preview) the sm_107 targets are validated with.
NVFP4_PATH = f"{llm_models_root()}/Kimi-K3.1-Pro-NVFP4"
#: The export's key in ``references/*.yaml``.
NVFP4_MODEL_NAME = "nvidia/Kimi-K3.1-Pro-NVFP4"
#: The drafter is the MXFP4 targets' one.
NVFP4_DFLASH2_PATH = _K31Target.DRAFTER_PATH


def nvfp4_sparse_attention() -> dict:
    """FP8 index keys and the self-sampling GVR top-k of the recipe."""
    return dict(
        algorithm="dsa",
        indexer_k_dtype="fp8",
        enable_heuristic_topk=True,
        use_self_sampling_topk=True,
        gvr_deterministic_topk=True,
    )


def nvfp4_kv_cache() -> dict:
    """The FP8 latent KV cache, its 64-token pages and the FP32 KDA state."""
    return dict(
        use_kv_cache_manager_v2=True,
        dtype="fp8",
        tokens_per_block=64,
        enable_block_reuse=False,
        mamba_ssm_cache_dtype="float32",
        free_gpu_memory_fraction=0.5,
    )


def nvfp4_dflash2(max_draft_len: int) -> DFlashDecodingConfig:
    return DFlashDecodingConfig(max_draft_len=max_draft_len, speculative_model=NVFP4_DFLASH2_PATH)


def nvfp4_aggregated(
    tensor_parallel_size: int, enable_attention_dp: bool, moe_backend: str, max_batch_size: int
) -> dict:
    """An aggregated server of the recipe: the generation role, draft length 7, with iteration stats for the
    acceptance length.

    The overlap scheduler stays off: DFlash2 under it is not validated on these targets.
    """
    return dict(
        tensor_parallel_size=tensor_parallel_size,
        moe_expert_parallel_size=tensor_parallel_size,
        enable_attention_dp=enable_attention_dp,
        modeling_v2="require",
        max_batch_size=max_batch_size,
        max_num_tokens=8192,
        max_seq_len=8192,
        trust_remote_code=True,
        disable_overlap_scheduler=True,
        cuda_graph_config=CudaGraphConfig(max_batch_size=max_batch_size, enable_padding=True),
        kv_cache_config=KvCacheConfig(**nvfp4_kv_cache()),
        moe_config=MoeConfig(backend=moe_backend, use_low_precision_moe_combine=True),
        sparse_attention_config=DeepSeekSparseAttentionConfig(**nvfp4_sparse_attention()),
        speculative_config=nvfp4_dflash2(7),
        enable_iter_perf_stats=True,
    )
