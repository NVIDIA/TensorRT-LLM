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
"""Whole-model gates for the hyv4_flash target on sm_103.

Every test here passes ``modeling_v2="require"`` to ``LLM``. Under ``"auto"`` a
configuration that missed a target's criteria would quietly fall back to another
implementation, pass, and report that implementation's numbers as the target's.

``TestModelingV2Hy4FlashFp8Sm103Default``: GSM8K on each layout the target's
routing admits (tp4, tep4, dep4, tep8), with the checkpoint's five-draft DFly2
engine on each feature member (dflash, multimodal, host_offload), the core
variants (FP8 KV, 512-token chunked prefill, block reuse) judged against core's
own score, and the stated disaggregated deployment (two TP2/EP2 context servers
and one TP4/EP4 generation server with the same DFly2 engine).

The anchor is ``tencent/Hy4-flash-FP8``'s row in ``references/gsm8k.yaml``: 89.69
for the FP8_BLOCK_SCALES core engine and 89.69 for the same engine with DFlash,
measured under the protocol ``_common`` states. The variants have no row of their
own, so each is measured beside core in one test and held to the difference's
sampling error (``_common.relative_band``): paired, the two legs share the
session's variance, and a delta against a number recorded in an earlier session
would carry that session's variance into the judgement.
"""

import pytest

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import DFlashDecodingConfig, KvCacheConfig
from tensorrt_llm.quantization import QuantAlgo

from .....conftest import llm_models_root
from ....accuracy_core import GSM8K, LlmapiAccuracyTestHarness
from ....test_disaggregated_serving import launch_disaggregated_llm, run_accuracy_test
from ..._env import scoped_env_overrides
from ._common import (
    DEP4,
    GSM8K_EVALUATOR_KWARGS,
    KV_CACHE_FRACTION,
    TEP4,
    TEP8,
    TP4,
    VALIDATE,
    assert_target_engine,
    dfly2,
    engine,
    gsm8k_samples,
    gsm8k_score,
    relative_band,
    sampling,
    skip_not_sm103,
)

# The reference rows are flexible-extract scores; the harness default averages it with
# strict-match.
_SCORES_FILTER = {"scores_filter": "exact_match,flexible-extract"}


@pytest.fixture(autouse=True)
def _score_flexible_extract(mocker):
    """Score GSM8K by flexible-extract, the filter the reference rows were measured with.

    Strict-match reads only a final ``#### N`` line, which this checkpoint seldom
    writes: greedy at tep4 it scored 57.09 against flexible-extract's 90.14, so the
    harness's default average of the two (73.62) would measure the answer format.
    """
    mocker.patch.dict(GSM8K.EVALUATE_KWARGS, _SCORES_FILTER)


# The layouts routing admits (the keys of routing.py's _LAYOUTS), with the
# hardware each needs. tep8 spans two nodes, so its guard is the MPI world size.
_LAYOUTS = [
    pytest.param(TP4, id="tp4", marks=pytest.mark.skip_less_device(4)),
    pytest.param(TEP4, id="tep4", marks=pytest.mark.skip_less_device(4)),
    pytest.param(DEP4, id="dep4", marks=pytest.mark.skip_less_device(4)),
    pytest.param(TEP8, id="tep8", marks=pytest.mark.skip_less_mpi_world_size(8)),
]

# The checkpoint's Engram tables live in host memory, so the host needs room for
# them beside the weights; the disaggregated deployment runs three engines.
_HOST_MEMORY_MB = 204800
_DISAGG_HOST_MEMORY_MB = 409600

# Workers spawn their second rank over MPI in a root container (docker/Dockerfile.multi).
_DISAGG_WORKER_ENV = {
    **VALIDATE,
    "PRTE_ALLOW_RUN_AS_ROOT": "1",
    "PRTE_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
}


def _disagg_server_config() -> dict:
    """Two context servers and one generation server, as intake/disagg states it."""
    return {
        "hostname": "localhost",
        "backend": "pytorch",
        "context_servers": {"num_instances": 2},
        "generation_servers": {"num_instances": 1},
    }


def _disagg_worker_configs() -> tuple[dict, dict]:
    """The context and generation worker configs of intake/disagg, as server dicts.

    The context engine is TP2/EP2 without a drafter: context does not draft. The
    generation engine is TP4/EP4 with the checkpoint's five-draft DFly2 config, and
    both exchange KV through the Python NIXL transceiver.
    """
    common = {
        "modeling_v2": "require",
        "pipeline_parallel_size": 1,
        "max_seq_len": 5120,
        "max_batch_size": 64,
        "max_num_tokens": 8192,
        "enable_chunked_prefill": False,
        "enable_attention_dp": False,
        "kv_cache_config": {
            "free_gpu_memory_fraction": KV_CACHE_FRACTION,
            "enable_block_reuse": False,
        },
        "cache_transceiver_config": {"backend": "NIXL", "transceiver_runtime": "PYTHON"},
    }
    ctx = {
        **common,
        "tensor_parallel_size": 2,
        "moe_expert_parallel_size": 2,
        "disable_overlap_scheduler": True,
        "cuda_graph_config": {"enable_padding": True, "max_batch_size": 64},
        "speculative_config": None,
    }
    gen = {
        **common,
        "tensor_parallel_size": 4,
        "moe_expert_parallel_size": 4,
        "disable_overlap_scheduler": False,
        "cuda_graph_config": {"enable_padding": True, "max_batch_size": 64},
        "speculative_config": {
            "decoding_type": "DFlash",
            "max_draft_len": 5,
            "mask_token_id": 120009,
        },
    }
    return ctx, gen


class TestModelingV2Hy4FlashFp8Sm103Default(LlmapiAccuracyTestHarness):
    """hyv4-flash-FP8 / hf-003107f_orig / sm_103 / unified, one case per served member."""

    # References are keyed by this name plus the resolved quantization
    # (FP8_BLOCK_SCALES) and, where the engine drafts, DFlash.
    MODEL_NAME = "tencent/Hy4-flash-FP8"
    MODEL_PATH = f"{llm_models_root()}/Hy4-flash-FP8"

    @skip_not_sm103
    @pytest.mark.timeout(5400)
    @pytest.mark.skip_less_device_memory(145000)
    @pytest.mark.skip_less_host_memory(_HOST_MEMORY_MB)
    @pytest.mark.parametrize("parallel", _LAYOUTS)
    def test_gsm8k(self, parallel):
        """GSM8K on one layout, against the FP8_BLOCK_SCALES core row.

        Each layout cuts the weights and crosses ranks differently, and a wrong
        cut still produces fluent text, so routing admits only the layouts that
        were measured. The run's core load is the boot evidence of tp4 and dep4;
        tep4 is the full member and tep8 the two-node engine.
        """
        with (
            scoped_env_overrides(VALIDATE) as rank_env,
            LLM(
                self.MODEL_PATH,
                **engine(),
                **parallel,
                modeling_v2="require",
                env_overrides=rank_env,
            ) as llm,
        ):
            assert_target_engine(llm, "gsm8k")
            GSM8K(self.MODEL_NAME).evaluate(
                llm,
                extra_evaluator_kwargs=GSM8K_EVALUATOR_KWARGS,
                sampling_params=sampling(),
            )

    @skip_not_sm103
    @pytest.mark.timeout(5400)
    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_less_device_memory(145000)
    @pytest.mark.skip_less_host_memory(_HOST_MEMORY_MB)
    @pytest.mark.parametrize("member", ["dflash", "multimodal", "host_offload"])
    def test_spec_gsm8k(self, member):
        """GSM8K with the checkpoint's five-draft DFly2 engine, against the DFlash row.

        The three members share one body because ``intake/llm_api`` gives them the
        same engine: what distinguishes them -- the vision tower's inputs, the
        Engram tables in pinned host memory -- is outside what a text GSM8K
        drives, and each member's boot, serve and acceptance gates carry it.

        The draft length is asserted, not assumed: routing accepts five drafts
        and nothing else, and an engine that resolved fewer would be scoring a
        path the deployment does not serve.
        """
        with (
            scoped_env_overrides(VALIDATE) as rank_env,
            LLM(
                self.MODEL_PATH,
                speculative_config=dfly2(),
                **engine(),
                **TEP4,
                modeling_v2="require",
                env_overrides=rank_env,
            ) as llm,
        ):
            assert_target_engine(llm, member)
            speculative = llm.args.speculative_config
            assert isinstance(speculative, DFlashDecodingConfig), (
                f"{member}: the engine resolved speculative_config {speculative!r}; the "
                "deployment's DFly2 engine is one DFlashDecodingConfig"
            )
            assert speculative.max_draft_len == 5, (
                f"{member}: the engine resolved {speculative.max_draft_len} drafts; the "
                "checkpoint's DFly2 engine drafts five"
            )
            GSM8K(self.MODEL_NAME).evaluate(
                llm,
                extra_evaluator_kwargs=GSM8K_EVALUATOR_KWARGS,
                sampling_params=sampling(),
            )

    def _paired_gsm8k(self, label: str, variant: dict, expect) -> None:
        """Core's score and the variant's, measured beside each other in one test.

        The core leg asserts the registered row on the way past, so this is the
        core gate as well as the comparison. The variant leg is scored without
        the reference lookup -- it has no row of its own -- and ``expect``
        asserts the engine resolved the variant's fields before it runs. Both
        legs are measured under the same protocol and on the same layout, so the
        two scores differ by sampling noise alone, and the band is the error of
        that difference (``_common.relative_band``).
        """
        task = GSM8K(self.MODEL_NAME)
        samples = gsm8k_samples()
        with (
            scoped_env_overrides(VALIDATE) as rank_env,
            LLM(
                self.MODEL_PATH,
                **engine(),
                **TEP4,
                modeling_v2="require",
                env_overrides=rank_env,
            ) as llm,
        ):
            assert_target_engine(llm, f"{label} core")
            core = task.evaluate(
                llm,
                extra_evaluator_kwargs=GSM8K_EVALUATOR_KWARGS,
                sampling_params=sampling(),
            )
        with (
            scoped_env_overrides(VALIDATE) as rank_env,
            LLM(
                self.MODEL_PATH,
                **engine(**variant),
                **TEP4,
                modeling_v2="require",
                env_overrides=rank_env,
            ) as llm,
        ):
            assert_target_engine(llm, label)
            expect(llm)
            measured = gsm8k_score(llm, self.MODEL_NAME)
        band = relative_band(core, samples)
        delta = measured - core
        print(
            f"[modeling_v2] gsm8k {label} core={core:.4f} variant={measured:.4f} "
            f"delta={delta:+.4f} band={band:.4f}"
        )
        assert abs(delta) <= band, (
            f"{label} moved gsm8k by {delta:+.4f} (core={core:.4f}, "
            f"variant={measured:.4f}); the band at {samples} samples is {band:.4f}, "
            "so this is not sampling noise"
        )

    @skip_not_sm103
    @pytest.mark.timeout(10800)
    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_less_device_memory(145000)
    @pytest.mark.skip_less_host_memory(_HOST_MEMORY_MB)
    def test_kv_fp8_gsm8k(self):
        """FP8 KV cache at tep4, judged against core's auto/BF16 KV score.

        An FP8 pool is a different quantization of what attention reads, so it
        has no reference row of its own; the run's rule is core's own score at
        the same member and target.
        """

        def expect(llm):
            assert llm.args.quant_config.kv_cache_quant_algo == QuantAlgo.FP8, (
                "kv_cache_config.dtype=fp8 resolved kv_cache_quant_algo "
                f"{llm.args.quant_config.kv_cache_quant_algo}"
            )

        self._paired_gsm8k(
            "kv_fp8",
            dict(
                kv_cache_config=KvCacheConfig(
                    free_gpu_memory_fraction=KV_CACHE_FRACTION,
                    enable_block_reuse=False,
                    dtype="fp8",
                )
            ),
            expect,
        )

    @skip_not_sm103
    @pytest.mark.timeout(10800)
    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_less_device_memory(145000)
    @pytest.mark.skip_less_host_memory(_HOST_MEMORY_MB)
    def test_chunked_gsm8k(self):
        """512-token chunked prefill at tep4, judged against core's score.

        Chunking splits a prompt across steps and reassembles the state each
        step reads, so a defect in the split shows up in the answers rather than
        in a shape check.
        """

        def expect(llm):
            assert llm.args.enable_chunked_prefill is True, "chunked prefill did not resolve on"
            assert llm.args.max_num_tokens == 512, (
                f"max_num_tokens resolved {llm.args.max_num_tokens}; the variant chunks at 512"
            )

        self._paired_gsm8k("chunked", dict(enable_chunked_prefill=True, max_num_tokens=512), expect)

    @skip_not_sm103
    @pytest.mark.timeout(10800)
    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_less_device_memory(145000)
    @pytest.mark.skip_less_host_memory(_HOST_MEMORY_MB)
    def test_reuse_gsm8k(self):
        """KV block reuse at tep4, judged against core's score.

        Reuse serves a second request from blocks a first one wrote, so a block
        that carries the wrong state changes the answers.
        """

        def expect(llm):
            assert llm.args.kv_cache_config.enable_block_reuse is True, (
                "kv_cache_config.enable_block_reuse did not resolve on"
            )

        self._paired_gsm8k(
            "reuse",
            dict(
                kv_cache_config=KvCacheConfig(
                    free_gpu_memory_fraction=KV_CACHE_FRACTION, enable_block_reuse=True
                )
            ),
            expect,
        )

    @skip_not_sm103
    @pytest.mark.timeout(7200)
    @pytest.mark.skip_less_device(8)
    @pytest.mark.skip_less_host_memory(_DISAGG_HOST_MEMORY_MB)
    def test_disagg_gsm8k(self):
        """The stated disaggregated deployment's GSM8K, against the DFlash row.

        Two TP2/EP2 context servers and one TP4/EP4 generation server, all on this
        host: the launcher hands each server a contiguous local device range, so
        this needs eight visible devices even though the generation engine is the
        member under test. The generation engine drafts with the same five-draft
        DFly2 config as the aggregated dflash member, and the reference row for
        it is the same 89.69.

        ``llm.args`` here is the launcher's stand-in for the deployment, not a
        worker's resolved arguments: it is given the quantization and the
        speculative config the generation server runs so the harness resolves the
        deployment's row.
        """
        ctx_config, gen_config = _disagg_worker_configs()
        with launch_disaggregated_llm(
            _disagg_server_config(),
            ctx_config,
            gen_config,
            self.MODEL_PATH,
            max_workers=128,
            extra_env=_DISAGG_WORKER_ENV,
        ) as llm:
            llm.args.quant_config.quant_algo = QuantAlgo.FP8_BLOCK_SCALES
            llm.args.speculative_config = dfly2()
            run_accuracy_test(
                llm,
                self.MODEL_NAME,
                ["GSM8K"],
                extra_evaluator_kwargs={GSM8K: GSM8K_EVALUATOR_KWARGS},
                sampling_params=sampling(),
                timeout=7200,
            )
