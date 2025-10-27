# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os

import pytest
import torch
from defs.conftest import get_sm_version

from tensorrt_llm import LLM
from tensorrt_llm._torch.modules.fused_moe.fused_moe_triton import \
    IS_TRITON_KERNELS_AVAILABLE
from tensorrt_llm._torch.pyexecutor.config import MoeLoadBalancerConfig
from tensorrt_llm.llmapi import (AutoDecodingConfig, CudaGraphConfig,
                                 EagleDecodingConfig, KvCacheConfig, MoeConfig,
                                 MTPDecodingConfig, NGramDecodingConfig,
                                 SamplingParams, TorchCompileConfig)
from tensorrt_llm.quantization import QuantAlgo

from ..conftest import (get_device_count, get_device_memory, llm_models_root,
                        parametrize_with_ids, skip_no_hopper,
                        skip_post_blackwell, skip_pre_ada, skip_pre_blackwell,
                        skip_pre_hopper, skip_ray)
from .accuracy_core import (GSM8K, MMLU, MMMU, CnnDailymail, GPQADiamond,
                            JsonModeEval, LlmapiAccuracyTestHarness)


class TestLlama3_1_8B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.1-8B"
    MODEL_PATH = f"{llm_models_root()}/llama-3.1-model/Meta-Llama-3.1-8B"

    @pytest.mark.skip_less_device_memory(32000)
    def test_auto_dtype(self):
        with LLM(self.MODEL_PATH) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    def test_nvfp4(self):
        model_path = f"{llm_models_root()}/nvfp4-quantized/Meta-Llama-3.1-8B"
        with LLM(model_path) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @pytest.mark.parametrize("stream_interval", [4, 64],
                             ids=["stream_interval_4", "stream_interval_64"])
    def test_nvfp4_streaming(self, stream_interval):
        # When stream_interval < TLLM_STREAM_INTERVAL_THRESHOLD, hf incremental detokenization is used.
        # When stream_interval >= TLLM_STREAM_INTERVAL_THRESHOLD, trtllm implemented incremental detokenization is used.
        # The behavior is due to perf considerations, while both paths need to be tested.
        with LLM(f"{llm_models_root()}/nvfp4-quantized/Meta-Llama-3.1-8B",
                 stream_interval=stream_interval) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4
            assert llm.args.stream_interval == stream_interval
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm, streaming=True)


class TestLlama3_1_8BInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct"

    @pytest.mark.skip_less_device_memory(32000)
    @parametrize_with_ids("attn_backend", ["TRTLLM", "FLASHINFER"])
    def test_chunked_prefill(self, attn_backend):
        with LLM(self.MODEL_PATH,
                 attn_backend=attn_backend,
                 enable_chunked_prefill=True,
                 max_num_tokens=512) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device_memory(32000)
    def test_dummy_load_format(self):
        llm = LLM(self.MODEL_PATH, load_format="dummy")
        with llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm, is_integration_test=True)

    @pytest.mark.skip_less_device_memory(32000)
    @parametrize_with_ids("torch_compile", [False, True])
    @parametrize_with_ids("attn_backend", ["TRTLLM", "FLASHINFER"])
    def test_bfloat16(self, attn_backend, torch_compile):
        torch_compile_config = TorchCompileConfig(
            enable_fullgraph=True,
            enable_piecewise_cuda_graph=True,
            capture_num_tokens=[2048, 8192],
            max_num_streams=3) if torch_compile else None
        pytorch_config = dict(
            torch_compile_config=torch_compile_config,
            cuda_graph_config=CudaGraphConfig(enable_padding=torch_compile,
                                              batch_sizes=[4]),
            attn_backend=attn_backend,
            disable_overlap_scheduler=torch_compile,
        )
        with LLM(self.MODEL_PATH, **pytorch_config) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @parametrize_with_ids("torch_compile", [False, True])
    @parametrize_with_ids("attn_backend", ["TRTLLM", "FLASHINFER"])
    @pytest.mark.skip_less_device(4)
    @pytest.mark.parametrize("tp_size,pp_size", [(4, 1), (2, 2), (1, 4)],
                             ids=["tp4", "tp2pp2", "pp4"])
    def test_bfloat16_4gpus(self, tp_size, pp_size, attn_backend,
                            torch_compile):
        if torch_compile and pp_size > 1:
            pytest.skip(
                "Pipeline parallel with torch.compile is not supported yet.\n"
                "Issue: Unfusing flashinfer_fused_add_rmsnorm causes outputs to be "
                "discarded at graph breaks.")
        torch_compile_config = TorchCompileConfig(
            enable_fullgraph=True,
            enable_piecewise_cuda_graph=True,
            capture_num_tokens=[2048, 8192],
            max_num_streams=3) if torch_compile else None
        pytorch_config = dict(
            torch_compile_config=torch_compile_config,
            cuda_graph_config=CudaGraphConfig(enable_padding=torch_compile,
                                              batch_sizes=[4]),
            attn_backend=attn_backend,
            disable_overlap_scheduler=torch_compile,
        )
        with LLM(self.MODEL_PATH,
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 **pytorch_config) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_ada
    @parametrize_with_ids("torch_compile", [False, True])
    @parametrize_with_ids("attn_backend", ["TRTLLM", "FLASHINFER"])
    @parametrize_with_ids("fp8kv", [False, True])
    def test_fp8(self, fp8kv, attn_backend, torch_compile):
        torch_compile_config = TorchCompileConfig(
            enable_fullgraph=True,
            enable_piecewise_cuda_graph=True,
            capture_num_tokens=[2048, 8192],
            max_num_streams=3) if torch_compile else None
        pytorch_config = dict(
            torch_compile_config=torch_compile_config,
            cuda_graph_config=CudaGraphConfig(enable_padding=torch_compile,
                                              batch_sizes=[4]),
            attn_backend=attn_backend,
            disable_overlap_scheduler=torch_compile,
        )
        if fp8kv:
            pytorch_config["kv_cache_config"] = KvCacheConfig(dtype="fp8")
        with LLM(
                f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct-FP8",
                **pytorch_config) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_ada
    @parametrize_with_ids("torch_compile", [False, True])
    @parametrize_with_ids("attn_backend", ["TRTLLM", "FLASHINFER"])
    @parametrize_with_ids("fp8kv", [False, True])
    @pytest.mark.skip_less_device(4)
    @pytest.mark.parametrize("tp_size,pp_size", [(4, 1), (2, 2), (1, 4)],
                             ids=["tp4", "tp2pp2", "pp4"])
    def test_fp8_4gpus(self, tp_size, pp_size, fp8kv, attn_backend,
                       torch_compile):
        if pp_size > 1 and torch_compile:
            pytest.skip(
                "Pipeline parallel with torch.compile is not supported yet.\n"
                "Issue: Unfusing flashinfer_fused_add_rmsnorm causes outputs to be "
                "discarded at graph breaks.")
        torch_compile_config = TorchCompileConfig(
            enable_fullgraph=True,
            enable_piecewise_cuda_graph=True,
            capture_num_tokens=[2048, 8192],
            max_num_streams=3) if torch_compile else None
        pytorch_config = dict(
            torch_compile_config=torch_compile_config,
            cuda_graph_config=CudaGraphConfig(enable_padding=torch_compile,
                                              batch_sizes=[4]),
            attn_backend=attn_backend,
            disable_overlap_scheduler=torch_compile,
        )
        if fp8kv:
            pytorch_config["kv_cache_config"] = KvCacheConfig(dtype="fp8")
        with LLM(
                f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct-FP8",
                tensor_parallel_size=tp_size,
                pipeline_parallel_size=pp_size,
                **pytorch_config) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_hopper
    def test_fp8_llm_sampler(self):
        model_path = f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct-FP8"
        with LLM(model_path, max_batch_size=256) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8

            sampling_params = SamplingParams(
                temperature=0.8,
                top_p=0.95,
            )

            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm,
                          sampling_params=sampling_params,
                          extra_acc_spec="temperature=0.8,top_p=0.95")
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm,
                          sampling_params=sampling_params,
                          extra_acc_spec="temperature=0.8,top_p=0.95")

    @skip_pre_hopper
    @parametrize_with_ids("overlap_scheduler", [True, False])
    @parametrize_with_ids("eagle3_one_model", [True, False])
    def test_eagle3(self, overlap_scheduler, eagle3_one_model):
        pytorch_config = dict(
            max_batch_size=
            1,  # add max_batch_size to avoid error in overlap scheduler
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig(max_batch_size=1,
                                              enable_padding=True),
        )
        kv_cache_config = KvCacheConfig(
            enable_block_reuse=True, free_gpu_memory_fraction=0.8
        )  # both one-model and two-model supports this feature

        eagle_model_dir = f"{llm_models_root()}/EAGLE3-LLaMA3.1-Instruct-8B"
        target_model_dir = f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct"

        draft_len = 4
        spec_config = EagleDecodingConfig(max_draft_len=draft_len,
                                          speculative_model_dir=eagle_model_dir,
                                          eagle3_one_model=eagle3_one_model)

        with LLM(model=target_model_dir,
                 **pytorch_config,
                 kv_cache_config=kv_cache_config,
                 speculative_config=spec_config,
                 build_config=None) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_hopper
    def test_ngram(self):
        pytorch_config = dict(
            disable_overlap_scheduler=True,
            cuda_graph_config=CudaGraphConfig(batch_sizes=[1]),
        )

        kv_cache_config = KvCacheConfig(enable_block_reuse=False,
                                        free_gpu_memory_fraction=0.8)

        spec_config = NGramDecodingConfig(
            max_draft_len=4,
            max_matching_ngram_size=2,
            is_keep_all=True,
            is_use_oldest=True,
            is_public_pool=True,
        )

        with LLM(model=self.MODEL_PATH,
                 **pytorch_config,
                 kv_cache_config=kv_cache_config,
                 speculative_config=spec_config,
                 max_batch_size=16) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @parametrize_with_ids("torch_compile", [False, True])
    @parametrize_with_ids("attn_backend", ["TRTLLM"])
    def test_nvfp4_kv(self, attn_backend, torch_compile):
        torch_compile_config = TorchCompileConfig(
            enable_fullgraph=True,
            enable_piecewise_cuda_graph=True,
            max_num_streams=3) if torch_compile else None
        pytorch_config = dict(
            torch_compile_config=torch_compile_config,
            cuda_graph_config=CudaGraphConfig(enable_padding=torch_compile,
                                              batch_sizes=[4]),
            attn_backend=attn_backend,
            disable_overlap_scheduler=torch_compile,
        )
        pytorch_config["kv_cache_config"] = KvCacheConfig(dtype="nvfp4")
        with LLM(f"{llm_models_root()}/Llama-3_1-8B-Instruct_nvfp4_fp8_hf",
                 **pytorch_config) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8
            assert llm.args.quant_config.kv_cache_quant_algo == QuantAlgo.NVFP4
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.parametrize("backend", ["xgrammar", "llguidance"])
    def test_guided_decoding(self, backend: str, mocker):
        mocker.patch.dict(os.environ, {"TRTLLM_XGUIDANCE_LENIENT": "1"})
        llm = LLM(self.MODEL_PATH, guided_decoding_backend=backend)
        with llm:
            task = JsonModeEval(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.timeout(7200)
    @pytest.mark.skip_less_device(4)
    @pytest.mark.parametrize("backend", ["xgrammar", "llguidance"])
    def test_guided_decoding_4gpus(self, backend: str, mocker):
        mocker.patch.dict(os.environ, {"TRTLLM_XGUIDANCE_LENIENT": "1"})
        with LLM(self.MODEL_PATH,
                 guided_decoding_backend=backend,
                 tensor_parallel_size=2,
                 pipeline_parallel_size=2) as llm:
            task = JsonModeEval(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_hopper
    @parametrize_with_ids("eagle3_one_model", [True, False])
    @pytest.mark.parametrize("backend", ["xgrammar", "llguidance"])
    def test_guided_decoding_with_eagle3(self, backend: str,
                                         eagle3_one_model: bool, mocker):
        mocker.patch.dict(os.environ, {"TRTLLM_XGUIDANCE_LENIENT": "1"})
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.8)
        cuda_graph_config = CudaGraphConfig(enable_padding=True)
        spec_config = EagleDecodingConfig(
            max_draft_len=3,
            speculative_model_dir=
            f"{llm_models_root()}/EAGLE3-LLaMA3.1-Instruct-8B",
            eagle3_one_model=eagle3_one_model)
        llm = LLM(
            self.MODEL_PATH,
            guided_decoding_backend=backend,
            kv_cache_config=kv_cache_config,
            cuda_graph_config=cuda_graph_config,
            enable_chunked_prefill=True,
            max_num_tokens=256,
            speculative_config=spec_config,
            # Two-model eagle3 does not support overlap scheduler
            disable_overlap_scheduler=not eagle3_one_model)
        with llm:
            task = JsonModeEval(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_hopper
    @pytest.mark.parametrize("backend", ["xgrammar", "llguidance"])
    def test_guided_decoding_with_ngram(self, backend: str, mocker):
        mocker.patch.dict(os.environ, {"TRTLLM_XGUIDANCE_LENIENT": "1"})
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.8)
        cuda_graph_config = CudaGraphConfig(enable_padding=True)
        spec_config = NGramDecodingConfig(max_draft_len=3,
                                          max_matching_ngram_size=3)
        llm = LLM(self.MODEL_PATH,
                  guided_decoding_backend=backend,
                  kv_cache_config=kv_cache_config,
                  cuda_graph_config=cuda_graph_config,
                  enable_chunked_prefill=True,
                  max_num_tokens=256,
                  speculative_config=spec_config,
                  disable_overlap_scheduler=True)
        with llm:
            task = JsonModeEval(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_hopper
    def test_auto_spec_decode(self):
        pytorch_config = {
            "cuda_graph_config":
            CudaGraphConfig(batch_sizes=[1, 32, 64], enable_padding=True)
        }
        kv_cache_config = KvCacheConfig(enable_block_reuse=False,
                                        free_gpu_memory_fraction=0.5)
        spec_config = AutoDecodingConfig()
        with LLM(model=self.MODEL_PATH,
                 **pytorch_config,
                 kv_cache_config=kv_cache_config,
                 speculative_config=spec_config,
                 max_batch_size=64) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @parametrize_with_ids("disable_overlap_scheduler", [False, True])
    @parametrize_with_ids(
        "enable_cuda_graph,enable_padding",
        [
            (False, False),  # No CUDA Graph (padding irrelevant)
            (True, False),  # CUDA Graph without padding
            (True, True),  # CUDA Graph with padding
        ])
    def test_auto_dtype_beam_search(self, enable_cuda_graph, enable_padding,
                                    disable_overlap_scheduler):
        max_beam_width = 2
        sampling_params = SamplingParams(n=max_beam_width,
                                         best_of=max_beam_width,
                                         use_beam_search=True)

        if enable_cuda_graph:
            # enable_padding only matters when CUDA Graph is enabled
            if enable_padding:
                batch_sizes = [
                    1, 8
                ]  # Need batch_size != max_batch_size to enable padding
            else:
                batch_sizes = [1, 2, 4, 8]
            cuda_graph_config = CudaGraphConfig(batch_sizes=batch_sizes,
                                                enable_padding=enable_padding)
        else:
            cuda_graph_config = None

        with LLM(
                model=self.MODEL_PATH,
                kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.5),
                max_batch_size=max_beam_width,
                max_seq_len=2048,
                max_beam_width=max_beam_width,
                disable_overlap_scheduler=disable_overlap_scheduler,
                cuda_graph_config=cuda_graph_config,
        ) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm,
                          sampling_params=sampling_params,
                          extra_acc_spec="beam_width=2")

    @skip_pre_hopper
    @parametrize_with_ids("disable_overlap_scheduler", [False, True])
    @parametrize_with_ids(
        "enable_cuda_graph,enable_padding",
        [
            (False, False),  # No CUDA Graph (padding irrelevant)
            (True, False),  # CUDA Graph without padding
            (True, True),  # CUDA Graph with padding
        ])
    def test_fp8_beam_search(self, enable_cuda_graph, enable_padding,
                             disable_overlap_scheduler):
        model_path = f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct-FP8"
        max_beam_width = 2
        sampling_params = SamplingParams(n=max_beam_width,
                                         best_of=max_beam_width,
                                         use_beam_search=True)
        if enable_cuda_graph:
            # enable_padding only matters when CUDA Graph is enabled
            if enable_padding:
                batch_sizes = [
                    1, 8
                ]  # Need batch_size != max_batch_size to enable padding
            else:
                batch_sizes = [1, 2, 4, 8]
            cuda_graph_config = CudaGraphConfig(batch_sizes=batch_sizes,
                                                enable_padding=enable_padding)
        else:
            cuda_graph_config = None

        llm = LLM(
            model=model_path,
            kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.5),
            max_batch_size=max_beam_width,
            max_seq_len=2048,
            max_beam_width=max_beam_width,
            disable_overlap_scheduler=disable_overlap_scheduler,
            cuda_graph_config=cuda_graph_config,
        )

        with llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm,
                          sampling_params=sampling_params,
                          extra_acc_spec="beam_width=2")


class TestLlama3_2_1B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.2-1B"
    MODEL_PATH = f"{llm_models_root()}/llama-3.2-models/Llama-3.2-1B"
    EXAMPLE_FOLDER = "models/core/llama"

    def test_auto_dtype(self):
        with LLM(self.MODEL_PATH) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_ada
    def test_fp8_prequantized(self):
        model_path = f"{llm_models_root()}/llama-3.2-models/Llama-3.2-1B-FP8"
        with LLM(model_path) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_hopper
    @pytest.mark.skip_less_device(4)
    @pytest.mark.parametrize("disable_overlap_scheduler", [True, False])
    @pytest.mark.parametrize("pp_size", [2, 4], ids=["pp2", "pp4"])
    def test_return_logits_pp(self, pp_size, disable_overlap_scheduler):
        prompts = ["A B C"]

        llm = LLM(model=self.MODEL_PATH,
                  pipeline_parallel_size=pp_size,
                  disable_overlap_scheduler=disable_overlap_scheduler)

        sampling_params = SamplingParams(max_tokens=8,
                                         return_context_logits=True,
                                         return_generation_logits=True,
                                         logprobs=True)

        with llm:
            for output in llm.generate(prompts,
                                       sampling_params=sampling_params):
                assert output.context_logits is not None
                # NOTE: prompt_token_ids of "A B C" becomes [1, 319, 350, 315]
                expected_len = len(prompts[0].split()) + 1
                assert expected_len == output.context_logits.shape[0]

                gen_logits = output.outputs[0].generation_logits
                assert gen_logits is not None
                assert gen_logits.ndim == 2
                assert gen_logits.shape[0] == sampling_params.max_tokens
                assert torch.argmax(
                    gen_logits, dim=1).tolist() == output.outputs[0].token_ids

                assert len(
                    output.outputs[0].logprobs) == sampling_params.max_tokens


class TestLlama3_2_3B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.2-3B"
    MODEL_PATH = f"{llm_models_root()}/llama-3.2-models/Llama-3.2-3B"
    EXAMPLE_FOLDER = "models/core/llama"

    def test_auto_dtype(self):
        with LLM(self.MODEL_PATH) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_hopper
    def test_fp8_prequantized(self):
        model_path = f"{llm_models_root()}/llama-3.2-models/Llama-3.2-3B-Instruct-FP8"
        with LLM(model_path) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)


@pytest.mark.timeout(7200)
@pytest.mark.skip_less_host_memory(1000000)
@pytest.mark.skip_less_device_memory(80000)
# 1TB is basic requirement for large model tests. CG4 120G only has 800G host memory, and 480G is shared with GPUs. the test will cause the system crash.
class TestLlama3_3_70BInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

    @pytest.mark.skip_less_mpi_world_size(8)
    def test_auto_dtype_tp8(self):
        model_path = f"{llm_models_root()}/llama-3.3-models/Llama-3.3-70B-Instruct"
        with LLM(model_path, tensor_parallel_size=8) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            task = GPQADiamond(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=dict(apply_chat_template=True))

    @skip_pre_hopper
    @pytest.mark.skip_less_mpi_world_size(8)
    @parametrize_with_ids("eagle3_one_model", [True, False])
    def test_fp8_eagle3_tp8(self, eagle3_one_model):
        model_path = f"{llm_models_root()}/modelopt-hf-model-hub/Llama-3.3-70B-Instruct-fp8"
        eagle_model_dir = f"{llm_models_root()}/EAGLE3-LLaMA3.3-Instruct-70B"
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.6)
        spec_config = EagleDecodingConfig(max_draft_len=4,
                                          speculative_model_dir=eagle_model_dir,
                                          eagle3_one_model=eagle3_one_model)
        pytorch_config = dict(
            disable_overlap_scheduler=True,
            cuda_graph_config=CudaGraphConfig(max_batch_size=1))
        with LLM(model_path,
                 max_batch_size=16,
                 tensor_parallel_size=8,
                 speculative_config=spec_config,
                 kv_cache_config=kv_cache_config,
                 **pytorch_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(4)
    @skip_pre_hopper
    def test_fp8_tp4(self):
        model_path = f"{llm_models_root()}/llama-3.3-models/Llama-3.3-70B-Instruct-FP8"
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.5)
        with LLM(model_path,
                 tensor_parallel_size=4,
                 max_seq_len=8192,
                 max_batch_size=32,
                 kv_cache_config=kv_cache_config) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8
            sampling_params = SamplingParams(
                max_tokens=256,
                temperature=0.0,
                add_special_tokens=False,
            )
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=sampling_params)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=sampling_params)
            task = GPQADiamond(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=dict(apply_chat_template=True))

    @pytest.mark.skip_less_device(4)
    @skip_pre_blackwell
    def test_nvfp4_tp4(self):
        model_path = f"{llm_models_root()}/llama-3.3-models/Llama-3.3-70B-Instruct-FP4"
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.5)
        with LLM(model_path,
                 tensor_parallel_size=4,
                 max_batch_size=32,
                 kv_cache_config=kv_cache_config) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4
            sampling_params = SamplingParams(
                max_tokens=256,
                temperature=0.0,
                add_special_tokens=False,
            )
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=sampling_params)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=sampling_params)
            task = GPQADiamond(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=dict(apply_chat_template=True))

    @pytest.mark.skip_less_device(4)
    @skip_pre_blackwell
    def test_fp8_tp2pp2(self):
        model_path = f"{llm_models_root()}/llama-3.3-models/Llama-3.3-70B-Instruct-FP8"
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.5)
        with LLM(model_path,
                 tensor_parallel_size=2,
                 pipeline_parallel_size=2,
                 max_batch_size=32,
                 kv_cache_config=kv_cache_config) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8
            sampling_params = SamplingParams(
                max_tokens=256,
                temperature=0.0,
                add_special_tokens=False,
            )
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=sampling_params)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=sampling_params)
            task = GPQADiamond(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=dict(apply_chat_template=True))


@pytest.mark.timeout(14400)
class TestLlama4MaverickInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
    MODEL_PATH = f"{llm_models_root()}/llama4-models/Llama-4-Maverick-17B-128E-Instruct"

    @skip_pre_blackwell
    @pytest.mark.skip_less_device_memory(140000)
    @parametrize_with_ids("cuda_graph", [False, True])
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size", [(8, 1, 1), (8, 1, 4), (8, 1, 8), (4, 1, 1),
                                    (4, 1, 2), (4, 1, 4)],
        ids=["tp8", "tp8ep4", "tp8ep8", "tp4", "tp4ep2", "tp4ep4"])
    def test_auto_dtype(self, cuda_graph, tp_size, pp_size, ep_size):
        if get_device_count() != tp_size * pp_size:
            pytest.skip("Device count mismatch with world size")
        if get_device_memory() < 240000 and get_device_count() < 8:
            pytest.skip("Not enough memory for this test")

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.8)
        with LLM(
                self.MODEL_PATH,
                tensor_parallel_size=tp_size,
                # Keep this low to avoid warmup OOM in CI
                max_seq_len=8192,
                pipeline_parallel_size=pp_size,
                moe_expert_parallel_size=ep_size,
                kv_cache_config=kv_cache_config,
                cuda_graph_config=CudaGraphConfig()
                if cuda_graph else None) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @pytest.mark.skip_less_device(8)
    @pytest.mark.skip_less_device_memory(140000)
    @parametrize_with_ids("attn_backend", ["TRTLLM", "FLASHINFER"])
    def test_chunked_prefill(self, attn_backend):
        pytorch_config = dict(attn_backend=attn_backend,
                              disable_overlap_scheduler=True)
        with LLM(self.MODEL_PATH,
                 tensor_parallel_size=8,
                 pipeline_parallel_size=1,
                 moe_expert_parallel_size=1,
                 max_seq_len=8192,
                 enable_chunked_prefill=True,
                 max_num_tokens=256,
                 **pytorch_config) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_hopper
    @pytest.mark.skip_less_device_memory(80000)
    @parametrize_with_ids("cuda_graph", [False, True])
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size", [(8, 1, 1), (8, 1, 4), (8, 1, 8), (4, 1, 1),
                                    (4, 1, 2), (4, 1, 4), (4, 2, 1)],
        ids=["tp8", "tp8ep4", "tp8ep8", "tp4", "tp4ep2", "tp4ep4", "tp4pp2"])
    def test_fp8(self, cuda_graph, tp_size, pp_size, ep_size):
        if get_device_memory() < 140000 and get_device_count() < 8:
            pytest.skip("Not enough memory for this test")
        if get_device_count() != tp_size * pp_size:
            pytest.skip("Device count mismatch with world size")

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75)
        with LLM(
                f"{llm_models_root()}/llama4-models/nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8",
                tensor_parallel_size=tp_size,
                # Keep this low to avoid warmup OOM in CI
                max_seq_len=8192,
                pipeline_parallel_size=pp_size,
                moe_expert_parallel_size=ep_size,
                kv_cache_config=kv_cache_config,
                cuda_graph_config=CudaGraphConfig()
                if cuda_graph else None) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8
            assert llm.args.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_hopper
    @pytest.mark.skip_less_mpi_world_size(8)
    @parametrize_with_ids("cuda_graph", [False, True])
    @pytest.mark.parametrize("tp_size,pp_size,ep_size", [(8, 1, 8)],
                             ids=["tp8ep8"])
    def test_fp8_chunked_prefill(self, cuda_graph, tp_size, pp_size, ep_size):
        with LLM(
                f"{llm_models_root()}/llama4-models/nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8",
                tensor_parallel_size=tp_size,
                # Keep this low to avoid warmup OOM in CI
                max_seq_len=8192,
                pipeline_parallel_size=pp_size,
                moe_expert_parallel_size=ep_size,
                enable_chunked_prefill=True,
                max_num_tokens=256,
                cuda_graph_config=CudaGraphConfig()
                if cuda_graph else None) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8
            assert llm.args.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_hopper
    @pytest.mark.skip_less_mpi_world_size(8)
    @parametrize_with_ids("torch_compile", [True, False])
    @pytest.mark.parametrize("tp_size,pp_size,ep_size", [(8, 1, 1)],
                             ids=["tp8"])
    def test_fp8_eagle3(self, tp_size, pp_size, ep_size, torch_compile):
        model_path = f"{llm_models_root()}/llama4-models/nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8"
        eagle_model_dir = f"{llm_models_root()}/Llama-4-Maverick-17B-128E-Eagle3"
        spec_config = EagleDecodingConfig(max_draft_len=3,
                                          speculative_model_dir=eagle_model_dir)
        kv_cache_config = KvCacheConfig(enable_block_reuse=False,
                                        free_gpu_memory_fraction=0.75)
        torch_compile_config = TorchCompileConfig(
            enable_fullgraph=True,
            enable_piecewise_cuda_graph=True,
            capture_num_tokens=[2048, 8192],
            max_num_streams=3) if torch_compile else None
        pytorch_config = dict(
            cuda_graph_config=CudaGraphConfig(max_batch_size=8),
            enable_attention_dp=False,
            torch_compile_config=torch_compile_config)
        with LLM(model_path,
                 kv_cache_config=kv_cache_config,
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 moe_expert_parallel_size=ep_size,
                 **pytorch_config,
                 speculative_config=spec_config) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)


@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.skip_less_host_memory(100000)
class TestLlama4ScoutInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

    @skip_pre_hopper
    @parametrize_with_ids("cuda_graph", [False, True])
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size", [(8, 1, 1), (8, 1, 4), (8, 1, 8), (4, 1, 1),
                                    (4, 1, 2), (4, 1, 4)],
        ids=["tp8", "tp8ep4", "tp8ep8", "tp4", "tp4ep2", "tp4ep4"])
    def test_auto_dtype(self, cuda_graph, tp_size, pp_size, ep_size):
        if get_device_count() != tp_size * pp_size:
            pytest.skip("Device count mismatch with world size")

        model_path = f"{llm_models_root()}/llama4-models/Llama-4-Scout-17B-16E-Instruct"
        with LLM(
                model_path,
                tensor_parallel_size=tp_size,
                # Keep this low to avoid warmup OOM in CI
                max_seq_len=8192,
                pipeline_parallel_size=pp_size,
                moe_expert_parallel_size=ep_size,
                cuda_graph_config=CudaGraphConfig()
                if cuda_graph else None) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_hopper
    @parametrize_with_ids("cuda_graph", [True])
    @pytest.mark.parametrize("tp_size,pp_size,ep_size", [(8, 1, 8), (4, 1, 1)],
                             ids=["tp8ep8", "tp4"])
    def test_fp8(self, cuda_graph, tp_size, pp_size, ep_size):
        if get_device_count() != tp_size * pp_size:
            pytest.skip("Device count mismatch with world size")

        model_path = f"{llm_models_root()}/llama4-models/Llama-4-Scout-17B-16E-Instruct-FP8"
        with LLM(
                model_path,
                tensor_parallel_size=tp_size,
                # Keep this low to avoid warmup OOM in CI
                max_seq_len=8192,
                pipeline_parallel_size=pp_size,
                moe_expert_parallel_size=ep_size,
                kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.8),
                cuda_graph_config=CudaGraphConfig()
                if cuda_graph else None) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @parametrize_with_ids("cuda_graph", [True])
    @pytest.mark.parametrize("tp_size,pp_size,ep_size", [(8, 1, 8), (4, 1, 1)],
                             ids=["tp8ep8", "tp4"])
    def test_fp4(self, cuda_graph, tp_size, pp_size, ep_size):
        if get_device_count() != tp_size * pp_size:
            pytest.skip("Device count mismatch with world size")

        model_path = f"{llm_models_root()}/llama4-models/Llama-4-Scout-17B-16E-Instruct-FP4"
        with LLM(
                model_path,
                tensor_parallel_size=tp_size,
                # Keep this low to avoid warmup OOM in CI
                max_seq_len=8192,
                pipeline_parallel_size=pp_size,
                moe_expert_parallel_size=ep_size,
                cuda_graph_config=CudaGraphConfig()
                if cuda_graph else None) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4
            assert llm.args.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_hopper
    @pytest.mark.skip_less_mpi_world_size(4)
    @parametrize_with_ids("cuda_graph", [True])
    @pytest.mark.parametrize("tp_size,pp_size,ep_size", [(4, 1, 4)],
                             ids=["tp4ep4"])
    def test_fp8_chunked_prefill(self, cuda_graph, tp_size, pp_size, ep_size):
        with LLM(
                f"{llm_models_root()}/llama4-models/Llama-4-Scout-17B-16E-Instruct-FP8",
                tensor_parallel_size=tp_size,
                max_seq_len=22000,
                pipeline_parallel_size=pp_size,
                moe_expert_parallel_size=ep_size,
                enable_chunked_prefill=True,
                max_num_tokens=256,
                cuda_graph_config=CudaGraphConfig()
                if cuda_graph else None) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8
            assert llm.args.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @pytest.mark.skip_less_mpi_world_size(4)
    @parametrize_with_ids("cuda_graph", [True])
    @pytest.mark.parametrize("tp_size,pp_size,ep_size", [(4, 1, 4)],
                             ids=["tp4ep4"])
    def test_fp4_chunked_prefill(self, cuda_graph, tp_size, pp_size, ep_size):
        with LLM(
                f"{llm_models_root()}/llama4-models/Llama-4-Scout-17B-16E-Instruct-FP4",
                tensor_parallel_size=tp_size,
                pipeline_parallel_size=pp_size,
                moe_expert_parallel_size=ep_size,
                max_seq_len=22000,
                enable_chunked_prefill=True,
                max_num_tokens=256,
                cuda_graph_config=CudaGraphConfig()
                if cuda_graph else None) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4
            assert llm.args.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


class TestMistral7B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "mistralai/Mistral-7B-v0.1"
    MODEL_PATH = f"{llm_models_root()}/mistral-7b-v0.1"

    def test_auto_dtype(self):
        with LLM(self.MODEL_PATH) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)


class TestMistralSmall24B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    MODEL_PATH = f"{llm_models_root()}/Mistral-Small-3.1-24B-Instruct-2503"

    @pytest.mark.skip_less_device_memory(80000)
    def test_auto_dtype(self):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75)
        with LLM(self.MODEL_PATH, kv_cache_config=kv_cache_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_ada
    @pytest.mark.skip_less_device_memory(80000)
    def test_fp8(self):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75)
        model_path = f"{llm_models_root()}/Mistral-Small-3.1-24B-Instruct-2503-fp8"
        with LLM(model_path, kv_cache_config=kv_cache_config) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


class TestMinistral8BInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "mistralai/Ministral-8B-Instruct-2410"
    MODEL_PATH = f"{llm_models_root()}/Ministral-8B-Instruct-2410"

    def test_auto_dtype(self):
        with LLM(self.MODEL_PATH) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_ada
    def test_fp8(self):
        # Test with FP8 quantization if pre-quantized model is available
        model_path = f"{llm_models_root()}/Ministral-8B-Instruct-2410-FP8"
        try:
            with LLM(model_path) as llm:
                assert llm.args.quant_config.quant_algo == QuantAlgo.FP8
                task = GSM8K(self.MODEL_NAME)
                task.evaluate(llm)
                task = MMLU(self.MODEL_NAME)
                task.evaluate(llm)
        except (FileNotFoundError, OSError):
            pytest.skip("FP8 pre-quantized Ministral-8B model not available")


@skip_post_blackwell
@skip_pre_hopper
class TestGemma3_27BInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "google/gemma-3-27b-it"
    MODEL_PATH = f"{llm_models_root()}/gemma/gemma-3-27b-it/"

    def test_auto_dtype(self):
        # Disabling kv cache reuse as a WAR to deal with gaps in kernel support for Gemma3's non-inclusive sliding window size.
        kv_cache_config = KvCacheConfig(
            enable_block_reuse=False,
            enable_partial_reuse=False,
            free_gpu_memory_fraction=0.5,
        )
        # We use FlashInfer as the attention backend for Gemma3 VLM to support custom mask for images.
        # So, testing with it here.
        with LLM(self.MODEL_PATH,
                 kv_cache_config=kv_cache_config,
                 attn_backend="FLASHINFER",
                 cuda_graph_config=None,
                 max_batch_size=128,
                 max_seq_len=4096) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    def test_fp8_prequantized(self):
        # Disabling kv cache reuse as a WAR to deal with gaps in kernel support for Gemma3's non-inclusive sliding window size.
        kv_cache_config = KvCacheConfig(enable_block_reuse=False,
                                        enable_partial_reuse=False,
                                        dtype="fp8")
        # Note: This has only the LLM part quantized. Vision part is in bfloat16.
        prequantized_model_path = f"{llm_models_root()}/gemma/gemma-3-27b-it-fp8/"
        with LLM(prequantized_model_path,
                 kv_cache_config=kv_cache_config,
                 attn_backend="FLASHINFER",
                 cuda_graph_config=None) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)


class TestGemma3_1BInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "google/gemma-3-1b-it"
    MODEL_PATH = f"{llm_models_root()}/gemma/gemma-3-1b-it/"

    # NOTE: Disable block reuse for SWA window model.
    kv_cache_config = KvCacheConfig(enable_block_reuse=True)

    def test_auto_dtype(self):
        # Disabling kv cache reuse as a WAR to deal with gaps in kernel support for Gemma3's non-inclusive sliding window size.
        kv_cache_config = KvCacheConfig(
            enable_block_reuse=False,
            enable_partial_reuse=False,
        )
        with LLM(self.MODEL_PATH, kv_cache_config=kv_cache_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    def test_fp8_prequantized(self):
        # Disabling kv cache reuse as a WAR to deal with gaps in kernel support for Gemma3's non-inclusive sliding window size.
        kv_cache_config = KvCacheConfig(enable_block_reuse=False,
                                        enable_partial_reuse=False,
                                        dtype="fp8")
        prequantized_model_path = f"{llm_models_root()}/gemma/gemma-3-1b-it-fp8/"
        with LLM(prequantized_model_path,
                 kv_cache_config=kv_cache_config) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    def test_auto_dtype_vswa_without_reuse(self):
        # NOTE: Test with VSWA kv cache config.
        kv_cache_config = KvCacheConfig(
            enable_block_reuse=False,
            enable_partial_reuse=False,
            max_attention_window=[512, 512, 512, 512, 512, 32768],
        )

        with LLM(self.MODEL_PATH, kv_cache_config=kv_cache_config) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    def test_auto_dtype_vswa_reuse(self):
        # NOTE: Test with VSWA kv cache config.
        kv_cache_config = KvCacheConfig(
            enable_block_reuse=True,
            max_attention_window=[512, 512, 512, 512, 512, 32768],
        )

        with LLM(self.MODEL_PATH, kv_cache_config=kv_cache_config) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    def test_auto_dtype_vswa_chunked_prefill_without_reuse(self):
        # NOTE: Test with VSWA kv cache config.
        kv_cache_config = KvCacheConfig(
            enable_block_reuse=False,
            enable_partial_reuse=False,
            max_attention_window=[512, 512, 512, 512, 512, 32768],
        )

        # chunked prefill case or more features
        extra_llm_config = dict(
            enable_chunked_prefill=True,
            max_num_tokens=1024,
        )
        with LLM(self.MODEL_PATH,
                 kv_cache_config=kv_cache_config,
                 **extra_llm_config) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    def test_auto_dtype_vswa_chunked_prefill_reuse(self):
        # NOTE: Test with VSWA kv cache config.
        kv_cache_config = KvCacheConfig(
            enable_block_reuse=True,
            max_attention_window=[512, 512, 512, 512, 512, 32768],
        )

        # chunked prefill case or more features
        extra_llm_config = dict(
            enable_chunked_prefill=True,
            max_num_tokens=1024,
        )
        with LLM(self.MODEL_PATH,
                 kv_cache_config=kv_cache_config,
                 **extra_llm_config) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)


class TestMixtral8x7B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "mistralai/Mixtral-8x7B-v0.1"
    MODEL_PATH = f"{llm_models_root()}/Mixtral-8x7B-v0.1"

    @pytest.mark.skip_less_device(2)
    def test_tp2(self):
        with LLM(self.MODEL_PATH, tensor_parallel_size=2) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(2)
    @pytest.mark.skip_device_not_contain(["H100", "H200", "B200"])
    def test_fp8_tp2(self):
        model_path = f"{llm_models_root()}/modelopt-hf-model-hub/Mixtral-8x7B-Instruct-v0.1-fp8"
        with LLM(model_path, tensor_parallel_size=2) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(2)
    @pytest.mark.skip_device_not_contain(["B200"])
    def test_nvfp4_tp2(self):
        model_path = f"{llm_models_root()}/modelopt-hf-model-hub/Mixtral-8x7B-Instruct-v0.1-fp4"
        with LLM(model_path, tensor_parallel_size=2) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)


# This class has extensively parameterized test methods, which yield totally 200 test cases.
# This is because this model requires high test coverage over the feature combinations.
# Normally we should not parameterize test methods so extensively -- just test on the typical/important feature combinations.
@skip_pre_hopper
class TestDeepSeekV3Lite(LlmapiAccuracyTestHarness):
    MODEL_NAME = "deepseek-ai/DeepSeek-V3-Lite"
    MODEL_PATH = f"{llm_models_root()}/DeepSeek-V3-Lite/bf16"

    @pytest.mark.skip_less_device_memory(60000)
    # Chunked Prefill for MLA can only be enabled on SM100
    @parametrize_with_ids("enable_chunked_prefill", [False, True])
    @parametrize_with_ids("torch_compile", [False, True])
    @parametrize_with_ids("attention_dp,cuda_graph,overlap_scheduler",
                          [(False, False, False), (True, False, False),
                           (False, True, False), (False, False, True),
                           (False, True, True), (True, True, True)])
    # Only Hopper and Blackwell MLA kernel supports MTP
    @parametrize_with_ids("mtp_nextn", [0, 2])
    def test_bfloat16(self, mtp_nextn, attention_dp, cuda_graph,
                      overlap_scheduler, torch_compile, enable_chunked_prefill):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75)
        torch_compile_config = TorchCompileConfig(
            enable_fullgraph=True,
            enable_piecewise_cuda_graph=cuda_graph,
            capture_num_tokens=[2048, 8192],
            max_num_streams=3) if torch_compile else None
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            torch_compile_config=torch_compile_config,
        )
        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)
        with LLM(self.MODEL_PATH,
                 kv_cache_config=kv_cache_config,
                 enable_chunked_prefill=enable_chunked_prefill,
                 max_num_tokens=256 if enable_chunked_prefill else 8192,
                 **pytorch_config,
                 enable_attention_dp=attention_dp,
                 speculative_config=mtp_config) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(4)
    @parametrize_with_ids("torch_compile", [False, True])
    @parametrize_with_ids("attention_dp,cuda_graph,overlap_scheduler",
                          [(False, False, False), (True, False, False),
                           (False, True, False), (False, False, True),
                           (False, True, True), (True, True, True)])
    # Only Hopper and Blackwell MLA kernel supports MTP
    @parametrize_with_ids("mtp_nextn",
                          [0, pytest.param(2, marks=skip_pre_hopper)])
    @pytest.mark.parametrize("tp_size,pp_size,ep_size", [(4, 1, 1), (4, 1, 4),
                                                         (2, 2, 1), (1, 4, 1)],
                             ids=["tp4", "ep4", "tp2pp2", "pp4"])
    def test_bfloat16_4gpus(self, tp_size, pp_size, ep_size, mtp_nextn,
                            attention_dp, cuda_graph, overlap_scheduler,
                            torch_compile):
        if torch_compile and pp_size > 1:
            pytest.skip("PP with torch.compile is not supported yet.")
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75)
        torch_compile_config = TorchCompileConfig(
            enable_fullgraph=True,
            enable_piecewise_cuda_graph=cuda_graph,
            capture_num_tokens=[2048, 8192],
            max_num_streams=3) if torch_compile else None
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            torch_compile_config=torch_compile_config,
        )
        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)
        with LLM(self.MODEL_PATH,
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 moe_expert_parallel_size=ep_size,
                 kv_cache_config=kv_cache_config,
                 **pytorch_config,
                 enable_attention_dp=attention_dp,
                 speculative_config=mtp_config) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_hopper
    @parametrize_with_ids("torch_compile", [False, True])
    @parametrize_with_ids("fp8kv,attention_dp,cuda_graph,overlap_scheduler",
                          [(False, False, False, False),
                           (True, False, False, False),
                           (False, True, False, False),
                           (False, False, True, False),
                           (False, False, False, True),
                           (True, False, True, True), (True, True, True, True)])
    @parametrize_with_ids("mtp", ["disable", "eagle", "vanilla"])
    def test_fp8_block_scales(self, mtp, fp8kv, attention_dp, cuda_graph,
                              overlap_scheduler, torch_compile):
        if torch_compile and mtp != "disable":
            pytest.skip("https://nvbugs/5252313")
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75)
        torch_compile_config = TorchCompileConfig(
            enable_fullgraph=True,
            enable_piecewise_cuda_graph=cuda_graph,
            capture_num_tokens=[2048, 8192],
            max_num_streams=3) if torch_compile else None
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            torch_compile_config=torch_compile_config,
            moe_config=MoeConfig(
                backend="DEEPGEMM" if get_sm_version() >= 100 else "CUTLASS"),
        )

        if fp8kv:
            kv_cache_config.dtype = "fp8"

        mtp_config = None
        mtp_nextn = 2
        if mtp == "eagle":
            mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)
        elif mtp == "vanilla":
            mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn,
                                           use_mtp_vanilla=True)

        with LLM(f"{llm_models_root()}/DeepSeek-V3-Lite/fp8",
                 kv_cache_config=kv_cache_config,
                 **pytorch_config,
                 enable_attention_dp=attention_dp,
                 speculative_config=mtp_config) as llm:

            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES

            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @parametrize_with_ids("torch_compile", [False])
    @parametrize_with_ids(
        "fp8kv,attention_dp,cuda_graph,overlap_scheduler",
        [(False, False, False, False)],
    )
    @parametrize_with_ids("mtp_nextn", [0])
    def test_cute_dsl_fp8_block_scales(
        self,
        mtp_nextn,
        fp8kv,
        attention_dp,
        cuda_graph,
        overlap_scheduler,
        torch_compile,
    ):
        if torch_compile and attention_dp:
            pytest.skip("https://nvbugs/5252559")
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.9)
        torch_compile_config = (TorchCompileConfig(
            enable_fullgraph=True,
            enable_piecewise_cuda_graph=cuda_graph,
            capture_num_tokens=[2048, 8192],
            max_num_streams=3) if torch_compile else None)
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            torch_compile_config=torch_compile_config,
            moe_config=MoeConfig(backend="CUTEDSL"),
        )

        if fp8kv:
            kv_cache_config.dtype = "fp8"

        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)

        with LLM(
                f"{llm_models_root()}/DeepSeek-V3-Lite/fp8",
                kv_cache_config=kv_cache_config,
                **pytorch_config,
                enable_attention_dp=attention_dp,
                speculative_config=mtp_config,
        ) as llm:

            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES

            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_hopper
    @parametrize_with_ids("mtp_nextn", [0, 2])
    def test_fp8_block_scales_cuda_graph_padding(self, mtp_nextn):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75)
        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)
        pytorch_config = dict(
            disable_overlap_scheduler=False,
            cuda_graph_config=CudaGraphConfig(
                max_batch_size=512,
                enable_padding=True,
            ),
            moe_config=MoeConfig(
                backend="DEEPGEMM" if get_sm_version() >= 100 else "CUTLASS"),
        )
        with LLM(f"{llm_models_root()}/DeepSeek-V3-Lite/fp8",
                 kv_cache_config=kv_cache_config,
                 **pytorch_config,
                 speculative_config=mtp_config) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(4)
    @skip_pre_hopper
    @parametrize_with_ids("mtp_nextn", [0, 2])
    @parametrize_with_ids("attention_dp", [False, True])
    def test_fp8_block_scales_cuda_graph_padding_4gpus(self, mtp_nextn,
                                                       attention_dp):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75)
        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)
        pytorch_config = dict(
            disable_overlap_scheduler=False,
            cuda_graph_config=CudaGraphConfig(enable_padding=True),
            moe_config=MoeConfig(
                backend="DEEPGEMM" if get_sm_version() >= 100 else "CUTLASS"),
        )

        with LLM(f"{llm_models_root()}/DeepSeek-V3-Lite/fp8",
                 tensor_parallel_size=4,
                 kv_cache_config=kv_cache_config,
                 **pytorch_config,
                 enable_attention_dp=attention_dp,
                 speculative_config=mtp_config) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(4)
    @skip_pre_hopper
    @skip_ray
    @parametrize_with_ids("torch_compile", [False, True])
    @parametrize_with_ids("fp8kv,attention_dp,cuda_graph,overlap_scheduler",
                          [(False, False, False, False),
                           (True, False, False, False),
                           (False, True, False, False),
                           (False, False, True, False),
                           (False, False, False, True),
                           (False, True, True, True), (True, False, True, True),
                           (True, True, True, True)])
    @parametrize_with_ids("mtp_nextn", [0, 2])
    @pytest.mark.parametrize("tp_size,pp_size,ep_size", [(4, 1, 1), (4, 1, 4),
                                                         (2, 2, 1), (1, 4, 1)],
                             ids=["tp4", "ep4", "tp2pp2", "pp4"])
    def test_fp8_block_scales_4gpus(self, tp_size, pp_size, ep_size, mtp_nextn,
                                    fp8kv, attention_dp, cuda_graph,
                                    overlap_scheduler, torch_compile):
        if torch_compile and pp_size > 1:
            pytest.skip("PP with torch.compile is not supported yet.")
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75)
        torch_compile_config = TorchCompileConfig(
            enable_fullgraph=True,
            enable_piecewise_cuda_graph=cuda_graph,
            capture_num_tokens=[2048, 8192],
            max_num_streams=3) if torch_compile else None
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            torch_compile_config=torch_compile_config,
            moe_config=MoeConfig(
                backend="DEEPGEMM" if get_sm_version() >= 100 else "CUTLASS"),
        )

        if fp8kv:
            kv_cache_config.dtype = "fp8"

        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)

        with LLM(f"{llm_models_root()}/DeepSeek-V3-Lite/fp8",
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 moe_expert_parallel_size=ep_size,
                 kv_cache_config=kv_cache_config,
                 **pytorch_config,
                 enable_attention_dp=attention_dp,
                 speculative_config=mtp_config) as llm:

            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES

            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(4)
    @skip_pre_blackwell
    @parametrize_with_ids("torch_compile", [False])
    @parametrize_with_ids(
        "fp8kv,attention_dp,cuda_graph,overlap_scheduler",
        [(False, False, False, False)],
    )
    @parametrize_with_ids("mtp_nextn", [0])
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size",
        [(4, 1, 1), (4, 1, 4), (2, 2, 1), (1, 4, 1)],
        ids=["tp4", "ep4", "tp2pp2", "pp4"],
    )
    def test_cute_dsl_fp8_block_scales_4gpus(
        self,
        tp_size,
        pp_size,
        ep_size,
        mtp_nextn,
        fp8kv,
        attention_dp,
        cuda_graph,
        overlap_scheduler,
        torch_compile,
    ):
        if torch_compile and pp_size > 1:
            pytest.skip("PP with torch.compile is not supported yet.")
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.9)
        torch_compile_config = (TorchCompileConfig(
            enable_fullgraph=True,
            enable_piecewise_cuda_graph=cuda_graph,
            capture_num_tokens=[2048, 8192],
            max_num_streams=3) if torch_compile else None)
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            torch_compile_config=torch_compile_config,
            moe_config=MoeConfig(backend="CUTEDSL"),
        )

        if fp8kv:
            kv_cache_config.dtype = "fp8"

        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)

        with LLM(
                f"{llm_models_root()}/DeepSeek-V3-Lite/fp8",
                tensor_parallel_size=tp_size,
                pipeline_parallel_size=pp_size,
                moe_expert_parallel_size=ep_size,
                kv_cache_config=kv_cache_config,
                **pytorch_config,
                enable_attention_dp=attention_dp,
                speculative_config=mtp_config,
        ) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES

            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(4)
    @skip_pre_hopper
    def test_fp8_block_scales_4gpus_static_eplb(self):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75)

        num_experts = 72
        num_slots = 80
        first_k_dense_replace = 1
        num_hidden_layers = 30
        initial_global_assignments = {}
        for i in range(first_k_dense_replace, num_hidden_layers):
            initial_global_assignments[i] = [(i + j) % num_experts
                                             for j in range(num_slots)]
        eplb_config = MoeLoadBalancerConfig(
            num_slots=num_slots,
            initial_global_assignments=initial_global_assignments,
            layer_updates_per_iter=0)
        pytorch_backend_options = dict(cuda_graph_config=CudaGraphConfig(),
                                       moe_config=MoeConfig(
                                           backend="WIDEEP",
                                           load_balancer=eplb_config))
        with LLM(f"{llm_models_root()}/DeepSeek-V3-Lite/fp8",
                 tensor_parallel_size=4,
                 moe_expert_parallel_size=4,
                 kv_cache_config=kv_cache_config,
                 **pytorch_backend_options,
                 enable_attention_dp=True) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_device_not_contain(["GB200"])
    @parametrize_with_ids("mtp_nextn", [0, 2])
    def test_bfloat16_4gpus_online_eplb(self, mtp_nextn):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.7)
        num_slots = 80
        eplb_config = MoeLoadBalancerConfig(num_slots=num_slots,
                                            layer_updates_per_iter=2)
        pytorch_config = dict(cuda_graph_config=CudaGraphConfig(),
                              moe_config=MoeConfig(backend="WIDEEP",
                                                   load_balancer=eplb_config))
        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)
        with LLM(self.MODEL_PATH,
                 tensor_parallel_size=4,
                 moe_expert_parallel_size=4,
                 kv_cache_config=kv_cache_config,
                 enable_attention_dp=True,
                 **pytorch_config,
                 speculative_config=mtp_config) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_device_not_contain(["GB200"])
    @parametrize_with_ids("fp8kv", [True, False])
    def test_nvfp4_4gpus_online_eplb(self, fp8kv):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.7)
        num_slots = 80
        eplb_config = MoeLoadBalancerConfig(num_slots=num_slots,
                                            layer_updates_per_iter=2)
        pytorch_config = dict(cuda_graph_config=CudaGraphConfig(),
                              moe_config=MoeConfig(backend="WIDEEP",
                                                   load_balancer=eplb_config))
        if fp8kv:
            kv_cache_config.dtype = "fp8"

        with LLM(
                f"{llm_models_root()}/DeepSeek-V3-Lite/nvfp4_moe_only",
                tensor_parallel_size=4,
                moe_expert_parallel_size=4,
                kv_cache_config=kv_cache_config,
                **pytorch_config,
                enable_attention_dp=True,
        ) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @parametrize_with_ids("torch_compile", [False, True])
    @parametrize_with_ids("fp8kv,attention_dp,cuda_graph,overlap_scheduler",
                          [(False, False, False, False),
                           (True, False, False, False),
                           (False, True, False, False),
                           (False, False, True, False),
                           (False, False, False, True),
                           (True, False, True, True), (True, True, True, True)])
    @parametrize_with_ids("mtp_nextn", [0, 2])
    @parametrize_with_ids("moe_backend", ["CUTLASS", "TRTLLM"])
    def test_nvfp4(self, fp8kv, attention_dp, cuda_graph, overlap_scheduler,
                   torch_compile, mtp_nextn, moe_backend):
        if moe_backend == "TRTLLM" and (get_sm_version() == 120
                                        or get_sm_version() == 121):
            pytest.skip(
                "MOE TRTLLM backend does not support SM version 120 or 121")

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75)
        torch_compile_config = TorchCompileConfig(
            enable_fullgraph=True,
            enable_piecewise_cuda_graph=cuda_graph,
            capture_num_tokens=[2048, 8192],
            max_num_streams=3) if torch_compile else None
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            torch_compile_config=torch_compile_config,
            moe_config=MoeConfig(backend=moe_backend))
        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)

        if fp8kv:
            kv_cache_config.dtype = "fp8"

        with LLM(f"{llm_models_root()}/DeepSeek-V3-Lite/nvfp4_moe_only_mtp",
                 kv_cache_config=kv_cache_config,
                 **pytorch_config,
                 enable_attention_dp=attention_dp,
                 speculative_config=mtp_config) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4

            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @parametrize_with_ids("torch_compile", [False, True])
    @parametrize_with_ids("fp8kv,cuda_graph,overlap_scheduler",
                          [(False, False, False), (True, True, True)])
    @parametrize_with_ids("mtp_nextn", [0, 2])
    @parametrize_with_ids(
        "batch_wait_timeout_iters,batch_wait_max_tokens_ratio", [(0, 0),
                                                                 (10, 0.75),
                                                                 (10, 0),
                                                                 (0, 0.75)])
    def test_nvfp4_batch_waiting(self, torch_compile, fp8kv, cuda_graph,
                                 overlap_scheduler, mtp_nextn,
                                 batch_wait_timeout_iters,
                                 batch_wait_max_tokens_ratio):
        moe_backend = "CUTLASS"
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75)
        torch_compile_config = TorchCompileConfig(
            enable_fullgraph=True,
            enable_piecewise_cuda_graph=cuda_graph,
            capture_num_tokens=[2048, 8192],
            max_num_streams=3) if torch_compile else None
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            torch_compile_config=torch_compile_config,
            batch_wait_timeout_iters=batch_wait_timeout_iters,
            batch_wait_max_tokens_ratio=batch_wait_max_tokens_ratio,
            moe_config=MoeConfig(backend=moe_backend))
        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)
        if fp8kv:
            kv_cache_config.dtype = "fp8"
        with LLM(f"{llm_models_root()}/DeepSeek-V3-Lite/nvfp4_moe_only_mtp",
                 kv_cache_config=kv_cache_config,
                 **pytorch_config,
                 enable_attention_dp=False,
                 speculative_config=mtp_config) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4

            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(4)
    @skip_pre_blackwell
    @parametrize_with_ids("torch_compile", [False, True])
    @parametrize_with_ids("fp8kv,attention_dp,cuda_graph,overlap_scheduler",
                          [(False, False, False, False),
                           (True, False, False, False),
                           (False, True, False, False),
                           (False, False, True, False),
                           (False, False, False, True),
                           (True, False, True, True), (True, True, True, True)])
    @pytest.mark.parametrize("tp_size,pp_size,ep_size", [(4, 1, 1), (4, 1, 4),
                                                         (2, 2, 1), (1, 4, 1)],
                             ids=["tp4", "ep4", "tp2pp2", "pp4"])
    @parametrize_with_ids("mtp_nextn", [0, 2])
    @parametrize_with_ids("moe_backend", ["CUTLASS", "TRTLLM"])
    def test_nvfp4_4gpus(self, fp8kv, attention_dp, cuda_graph,
                         overlap_scheduler, tp_size, pp_size, ep_size,
                         torch_compile, mtp_nextn, moe_backend):
        if torch_compile and pp_size > 1:
            pytest.skip("PP with torch.compile is not supported yet.")
        if moe_backend == "TRTLLM" and (get_sm_version() == 120
                                        or get_sm_version() == 121):
            pytest.skip(
                "MOE TRTLLM backend does not support SM version 120 or 121")
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75)
        # Picewise Cuda Graph cannot be enabled for nvfp4 attention dp.
        torch_compile_config = TorchCompileConfig(
            enable_fullgraph=True,
            enable_piecewise_cuda_graph=cuda_graph,
            capture_num_tokens=[2048, 8192],
            max_num_streams=3) if torch_compile else None
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            torch_compile_config=torch_compile_config,
            moe_config=MoeConfig(backend=moe_backend),
        )

        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)

        if fp8kv:
            kv_cache_config.dtype = "fp8"

        with LLM(f"{llm_models_root()}/DeepSeek-V3-Lite/nvfp4_moe_only_mtp",
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 moe_expert_parallel_size=ep_size,
                 kv_cache_config=kv_cache_config,
                 **pytorch_config,
                 enable_attention_dp=attention_dp,
                 speculative_config=mtp_config) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4

            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @parametrize_with_ids(
        "fp8kv,attention_dp,cuda_graph,overlap_scheduler",
        [(False, False, False, False),
         pytest.param(True, False, False, False, marks=skip_no_hopper),
         (False, True, False, False), (False, False, True, False),
         (False, False, False, True), (False, True, True, True),
         pytest.param(True, True, True, True, marks=skip_no_hopper)])
    @parametrize_with_ids("mtp_nextn", [0, 2])
    @parametrize_with_ids("quant_dtype", [
        pytest.param("none", marks=skip_pre_hopper),
        pytest.param("fp8", marks=skip_no_hopper),
        pytest.param("nvfp4", marks=skip_pre_blackwell)
    ])
    def test_no_kv_cache_reuse(self, quant_dtype, mtp_nextn, fp8kv,
                               attention_dp, cuda_graph, overlap_scheduler):
        if quant_dtype == "nvfp4" and mtp_nextn > 0:
            pytest.skip("MTP is not supported for NVFP4")

        model_path = self.MODEL_PATH
        if quant_dtype == "fp8":
            model_path = f"{llm_models_root()}/DeepSeek-V3-Lite/fp8"
        elif quant_dtype == "nvfp4":
            model_path = f"{llm_models_root()}/DeepSeek-V3-Lite/nvfp4_moe_only"

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75,
                                        enable_block_reuse=False)
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
        )
        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)

        if quant_dtype == "none":
            assert not fp8kv
        else:
            if fp8kv:
                kv_cache_config.dtype = "fp8"

        with LLM(model_path,
                 kv_cache_config=kv_cache_config,
                 **pytorch_config,
                 enable_attention_dp=attention_dp,
                 speculative_config=mtp_config) as llm:
            if quant_dtype == "fp8":
                assert llm.args.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
            elif quant_dtype == "nvfp4":
                assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4

            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @parametrize_with_ids("fp8kv,overlap_scheduler", [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ])
    @parametrize_with_ids("kv_cache_reuse", [True, False])
    @parametrize_with_ids("quant_dtype", [
        pytest.param("none", marks=skip_pre_hopper),
        pytest.param("fp8", marks=skip_pre_hopper),
        pytest.param("nvfp4", marks=skip_pre_blackwell)
    ])
    # currently, chunked prefill is not supported for fp8 and nvfp4
    def test_chunked_prefill(self, quant_dtype, kv_cache_reuse, fp8kv,
                             overlap_scheduler):
        model_path = self.MODEL_PATH
        if quant_dtype == "fp8":
            model_path = f"{llm_models_root()}/DeepSeek-V3-Lite/fp8"
        elif quant_dtype == "nvfp4":
            model_path = f"{llm_models_root()}/DeepSeek-V3-Lite/nvfp4_moe_only"

        if quant_dtype == "none" and fp8kv:
            pytest.skip("only fp8 and nvfp4 support fp8 kv cache")

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.6,
                                        enable_block_reuse=kv_cache_reuse)
        pytorch_config = dict(disable_overlap_scheduler=not overlap_scheduler, )
        mtp_config = None

        if quant_dtype == "none":
            assert not fp8kv
        else:
            if fp8kv:
                kv_cache_config.dtype = "fp8"

        with LLM(model_path,
                 kv_cache_config=kv_cache_config,
                 enable_chunked_prefill=True,
                 max_num_tokens=512,
                 **pytorch_config,
                 enable_attention_dp=True,
                 speculative_config=mtp_config) as llm:

            if quant_dtype == "fp8":
                assert llm.args.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
            elif quant_dtype == "nvfp4":
                assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4

            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @parametrize_with_ids("mtp_nextn",
                          [0, pytest.param(2, marks=skip_pre_hopper)])
    @pytest.mark.parametrize("backend", ["xgrammar", "llguidance"])
    def test_guided_decoding(self, backend: str, mtp_nextn: int, mocker):
        mocker.patch.dict(os.environ, {"TRTLLM_XGUIDANCE_LENIENT": "1"})
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75)
        cuda_graph_config = CudaGraphConfig(enable_padding=True)
        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)
        llm = LLM(self.MODEL_PATH,
                  guided_decoding_backend=backend,
                  kv_cache_config=kv_cache_config,
                  cuda_graph_config=cuda_graph_config,
                  enable_chunked_prefill=True,
                  max_num_tokens=256,
                  speculative_config=mtp_config)
        with llm:
            task = JsonModeEval(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(4)
    @parametrize_with_ids("mtp_nextn",
                          [0, pytest.param(2, marks=skip_pre_hopper)])
    @pytest.mark.parametrize("backend", ["xgrammar", "llguidance"])
    def test_guided_decoding_4gpus(self, backend: str, mtp_nextn: int, mocker):
        mocker.patch.dict(os.environ, {"TRTLLM_XGUIDANCE_LENIENT": "1"})
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75)
        cuda_graph_config = CudaGraphConfig(enable_padding=True)
        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)
        llm = LLM(self.MODEL_PATH,
                  tensor_parallel_size=4,
                  moe_expert_parallel_size=4,
                  enable_attention_dp=True,
                  guided_decoding_backend=backend,
                  kv_cache_config=kv_cache_config,
                  cuda_graph_config=cuda_graph_config,
                  enable_chunked_prefill=True,
                  max_num_tokens=256,
                  speculative_config=mtp_config)
        with llm:
            task = JsonModeEval(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_hopper
    def test_dummy_load_format(self):
        llm = LLM(
            f"{llm_models_root()}/DeepSeek-V3-Lite/fp8",
            load_format="dummy",
            moe_config=MoeConfig(
                backend="DEEPGEMM" if get_sm_version() >= 100 else "CUTLASS"),
        )
        with llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm, is_integration_test=True)


@pytest.mark.timeout(14400)
@pytest.mark.skip_less_device_memory(80000)
class TestDeepSeekR1(LlmapiAccuracyTestHarness):
    MODEL_NAME = "deepseek-ai/DeepSeek-R1"
    MODEL_PATH = f"{llm_models_root()}/DeepSeek-R1/DeepSeek-R1"

    @skip_pre_blackwell
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,mtp_nextn,fp8kv,attention_dp,enable_lm_head_tp_in_adp,cuda_graph,overlap_scheduler,max_batch_size,moe_backend",
        [
            #  Use a larger batch_size to speed up the tests
            pytest.param(8,
                         1,
                         4,
                         3,
                         False,
                         False,
                         False,
                         True,
                         True,
                         32,
                         "CUTLASS",
                         marks=pytest.mark.skip_less_mpi_world_size(8)),
            pytest.param(8,
                         1,
                         4,
                         3,
                         False,
                         False,
                         False,
                         True,
                         True,
                         32,
                         "TRTLLM",
                         marks=pytest.mark.skip_less_mpi_world_size(8)),
            pytest.param(8,
                         1,
                         4,
                         3,
                         False,
                         True,
                         True,
                         True,
                         True,
                         32,
                         "CUTLASS",
                         marks=pytest.mark.skip_less_mpi_world_size(8)),
            pytest.param(8,
                         1,
                         4,
                         3,
                         False,
                         True,
                         True,
                         True,
                         True,
                         32,
                         "TRTLLM",
                         marks=pytest.mark.skip_less_mpi_world_size(8)),
            pytest.param(8,
                         1,
                         8,
                         0,
                         True,
                         True,
                         False,
                         True,
                         True,
                         32,
                         "CUTLASS",
                         marks=pytest.mark.skip_less_mpi_world_size(8)),
            pytest.param(8,
                         1,
                         1,
                         0,
                         True,
                         True,
                         False,
                         True,
                         True,
                         32,
                         "CUTLASS",
                         marks=pytest.mark.skip_less_mpi_world_size(8)),
            pytest.param(4,
                         1,
                         1,
                         0,
                         True,
                         True,
                         False,
                         True,
                         True,
                         16,
                         "CUTLASS",
                         marks=pytest.mark.skip_less_mpi_world_size(4)),
            pytest.param(8,
                         1,
                         8,
                         1,
                         True,
                         True,
                         False,
                         True,
                         True,
                         32,
                         "CUTLASS",
                         marks=pytest.mark.skip_less_mpi_world_size(8)),
            pytest.param(8,
                         1,
                         8,
                         1,
                         True,
                         True,
                         False,
                         True,
                         True,
                         8,
                         "CUTLASS",
                         marks=pytest.mark.skip_less_mpi_world_size(8)),
        ],
        ids=[
            "latency", "latency_trtllmgen", "latency_adp_lmtp",
            "latency_trtllmgen_adp_lmtp", "throughput", "throughput_tp8",
            "throughput_tp4", "throughput_mtp", "throughput_bs8_mtp"
        ])
    def test_nvfp4_multi_gpus(self, tp_size, pp_size, ep_size, mtp_nextn, fp8kv,
                              attention_dp, enable_lm_head_tp_in_adp,
                              cuda_graph, overlap_scheduler, max_batch_size,
                              moe_backend):
        if moe_backend == "TRTLLM" and (get_sm_version() == 120
                                        or get_sm_version() == 121):
            pytest.skip(
                "MOE TRTLLM backend does not support SM version 120 or 121")

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.70)
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            moe_config=MoeConfig(backend=moe_backend))

        if fp8kv:
            kv_cache_config.dtype = "fp8"

        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)
        with LLM(f"{llm_models_root()}/DeepSeek-R1/DeepSeek-R1-FP4",
                 max_batch_size=max_batch_size,
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 moe_expert_parallel_size=ep_size,
                 kv_cache_config=kv_cache_config,
                 **pytorch_config,
                 enable_attention_dp=attention_dp,
                 enable_lm_head_tp_in_adp=enable_lm_head_tp_in_adp,
                 speculative_config=mtp_config) as llm:

            assert llm.args.moe_config.backend == moe_backend
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4

            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            # This covers the case with relatively large seqlen in the generation phase.
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            # Commented out because GPQA takes too long to run
            # task = GPQADiamond(self.MODEL_NAME)
            # task.evaluate(llm,
            #               extra_evaluator_kwargs=dict(apply_chat_template=True))

    @skip_pre_blackwell
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,mtp_nextn,fp8kv,attention_dp,cuda_graph,overlap_scheduler,max_batch_size,moe_backend",
        [
            #  Use a larger batch_size to speed up the tests
            pytest.param(8,
                         1,
                         4,
                         3,
                         False,
                         False,
                         True,
                         True,
                         32,
                         "CUTLASS",
                         marks=pytest.mark.skip_less_mpi_world_size(8)),
            pytest.param(4,
                         1,
                         1,
                         0,
                         True,
                         True,
                         True,
                         True,
                         16,
                         "CUTLASS",
                         marks=pytest.mark.skip_less_mpi_world_size(4)),
        ],
        ids=["latency", "throughput_tp4"])
    def test_nvfp4_multi_gpus_chunked_prefill(self, tp_size, pp_size, ep_size,
                                              mtp_nextn, fp8kv, attention_dp,
                                              cuda_graph, overlap_scheduler,
                                              max_batch_size, moe_backend):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.70)
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            moe_config=MoeConfig(backend=moe_backend))

        if fp8kv:
            kv_cache_config.dtype = "fp8"

        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)
        with LLM(f"{llm_models_root()}/DeepSeek-R1/DeepSeek-R1-FP4",
                 max_batch_size=max_batch_size,
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 moe_expert_parallel_size=ep_size,
                 kv_cache_config=kv_cache_config,
                 **pytorch_config,
                 enable_attention_dp=attention_dp,
                 speculative_config=mtp_config,
                 enable_chunked_prefill=True,
                 max_num_tokens=512) as llm:

            assert llm.args.moe_config.backend == moe_backend
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4

            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @pytest.mark.skip_less_device(8)
    def test_nvfp4_multi_gpus_corner_case(self):
        """
        This test is used to test the corner case of the NVFP4 model.
        When using the same value for max_seq_len and max_num_tokens, there will be no
        enough kv block for the dummy requests in CUDA graph warmup when creating
        the py_executor before estimating kv cache. Then CUDA graph capture will be
        triggered when estimating kv cache. This may cause some errors.
        More info in https://nvbugs/5485325.
        """
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.80,
                                        dtype="fp8",
                                        enable_block_reuse=False)
        pytorch_config = dict(disable_overlap_scheduler=False,
                              cuda_graph_config=CudaGraphConfig(
                                  enable_padding=True, max_batch_size=1024),
                              moe_config=MoeConfig(backend="TRTLLM"))

        mtp_config = MTPDecodingConfig(num_nextn_predict_layers=1)
        with LLM(f"{llm_models_root()}/DeepSeek-R1/DeepSeek-R1-FP4",
                 tensor_parallel_size=8,
                 pipeline_parallel_size=1,
                 moe_expert_parallel_size=8,
                 kv_cache_config=kv_cache_config,
                 **pytorch_config,
                 enable_attention_dp=False,
                 speculative_config=mtp_config,
                 max_seq_len=5120,
                 max_num_tokens=5120) as llm:

            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4

            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_mpi_world_size(8)
    @skip_pre_hopper
    @pytest.mark.skipif(get_sm_version() >= 100,
                        reason="https://nvbugs/5547584 WNF")
    @pytest.mark.skip_less_device_memory(140000)
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,mtp_nextn,fp8kv,attention_dp,cuda_graph,overlap_scheduler,max_batch_size,moe_backend",
        [(8, 1, 4, 3, False, False, True, True, 1, "_DEFAULT"),
         (8, 1, 8, 0, True, True, True, True, 24, "_DEFAULT"),
         (8, 1, 8, 1, True, True, True, True, 24, "_DEFAULT"),
         (8, 1, 8, 1, True, True, True, True, 24, "TRTLLM")],
        ids=[
            "latency", "throughput", "throughput_mtp", "throughput_mtp_trtllm"
        ])
    def test_fp8_blockscale(self, tp_size, pp_size, ep_size, mtp_nextn, fp8kv,
                            attention_dp, cuda_graph, overlap_scheduler,
                            max_batch_size, moe_backend):
        if get_sm_version() == 100 or get_sm_version() == 103:
            moe_backend = "DEEPGEMM" if moe_backend == "_DEFAULT" else moe_backend
            moe_config = MoeConfig(backend=moe_backend, max_num_tokens=16384)
            kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.6)
        else:
            if moe_backend != "_DEFAULT":
                pytest.skip("Not supported MoE backend!")
            moe_config = MoeConfig()
            kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.9)

        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            moe_config=moe_config,
        )

        if fp8kv:
            kv_cache_config.dtype = "fp8"

        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)
        with LLM(f"{llm_models_root()}/DeepSeek-R1/DeepSeek-R1",
                 max_batch_size=max_batch_size,
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 moe_expert_parallel_size=ep_size,
                 kv_cache_config=kv_cache_config,
                 **pytorch_config,
                 enable_attention_dp=attention_dp,
                 speculative_config=mtp_config) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES

            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_mpi_world_size(8)
    @skip_pre_hopper
    @pytest.mark.skip_less_device_memory(140000)
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,mtp_nextn,fp8kv,attention_dp,cuda_graph,overlap_scheduler,max_batch_size",
        [(8, 1, 4, 3, False, False, True, True, 1),
         (8, 1, 8, 0, True, True, True, True, 24)],
        ids=["latency", "throughput"])
    def test_fp8_blockscale_chunked_prefill(self, tp_size, pp_size, ep_size,
                                            mtp_nextn, fp8kv, attention_dp,
                                            cuda_graph, overlap_scheduler,
                                            max_batch_size):
        if get_sm_version() == 100:
            moe_config = MoeConfig(backend="DEEPGEMM", max_num_tokens=16384)
            kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.6)
        else:
            moe_config = MoeConfig()
            kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.9)

        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            moe_config=moe_config,
        )

        if fp8kv:
            kv_cache_config.dtype = "fp8"

        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)
        with LLM(f"{llm_models_root()}/DeepSeek-R1/DeepSeek-R1",
                 max_batch_size=max_batch_size,
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 moe_expert_parallel_size=ep_size,
                 kv_cache_config=kv_cache_config,
                 **pytorch_config,
                 enable_attention_dp=attention_dp,
                 speculative_config=mtp_config,
                 enable_chunked_prefill=True,
                 max_num_tokens=512) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES

            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


@pytest.mark.timeout(14400)
@pytest.mark.skip_less_device_memory(80000)
class TestDeepSeekV32(LlmapiAccuracyTestHarness):
    MODEL_NAME = "deepseek-ai/DeepSeek-V3.2-Exp"
    # TODO: This is the native HF ckpt w/ FP8 weights.
    # Once other weights are available, we should update this model path.
    MODEL_PATH = f"{llm_models_root()}/DeepSeek-V3.2-Exp-hf"

    @pytest.mark.skip_less_mpi_world_size(8)
    @skip_pre_blackwell  # TODO: Hopper untested
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,mtp_nextn,fp8kv,attention_dp,cuda_graph,overlap_scheduler,max_batch_size,moe_backend",
        [
            (8, 1, 8, 0, False, True, True, True, 24, "_DEFAULT"),
            (8, 1, 8, 1, False, True, True, True, 24, "_DEFAULT"),
        ],
        ids=["baseline", "baseline_mtp1"])
    def test_fp8_blockscale(self, tp_size, pp_size, ep_size, mtp_nextn, fp8kv,
                            attention_dp, cuda_graph, overlap_scheduler,
                            max_batch_size, moe_backend):
        if get_sm_version() == 100 or get_sm_version() == 103:
            moe_backend = "DEEPGEMM" if moe_backend == "_DEFAULT" else moe_backend
            moe_config = MoeConfig(backend=moe_backend, max_num_tokens=16384)
            # TODO: Support block reuse for DeepSeek-V3.2
            kv_cache_config = KvCacheConfig(enable_block_reuse=False,
                                            free_gpu_memory_fraction=0.6)
        else:
            if moe_backend != "_DEFAULT":
                pytest.skip("Not supported MoE backend!")
            moe_config = MoeConfig()
            # TODO: Support block reuse for DeepSeek-V3.2
            kv_cache_config = KvCacheConfig(enable_block_reuse=False,
                                            free_gpu_memory_fraction=0.7)

        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            moe_config=moe_config,
        )

        if fp8kv:
            kv_cache_config.dtype = "fp8"

        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)
        with LLM(f"{llm_models_root()}/DeepSeek-V3.2-Exp-hf",
                 max_batch_size=max_batch_size,
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 moe_expert_parallel_size=ep_size,
                 kv_cache_config=kv_cache_config,
                 **pytorch_config,
                 enable_attention_dp=attention_dp,
                 speculative_config=mtp_config) as llm:

            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            # GPQA Diamond takes too long to run
            # task = GPQADiamond(self.MODEL_NAME)
            # task.evaluate(llm,
            #               extra_evaluator_kwargs=dict(apply_chat_template=True, chat_template_kwargs=dict(thinking=True)))


@pytest.mark.timeout(7200)
@pytest.mark.skip_less_device_memory(100000)
class TestKimiK2(LlmapiAccuracyTestHarness):
    MODEL_NAME = "moonshotai/Kimi-K2-Instruct"
    MODEL_PATH = f"{llm_models_root()}/Kimi-K2-Instruct"

    @pytest.mark.skip_less_mpi_world_size(8)
    @skip_post_blackwell
    @skip_pre_hopper
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,fp8kv,attention_dp,cuda_graph,overlap_scheduler,max_batch_size",
        [(8, 1, 8, False, False, True, True, 16)],
        ids=["latency"])
    def test_fp8_blockscale(self, tp_size, pp_size, ep_size, fp8kv,
                            attention_dp, cuda_graph, overlap_scheduler,
                            max_batch_size):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.9)
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
        )

        if fp8kv:
            kv_cache_config.dtype = "fp8"

        mtp_config = None
        with LLM(f"{llm_models_root()}/Kimi-K2-Instruct",
                 max_batch_size=max_batch_size,
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 moe_expert_parallel_size=ep_size,
                 trust_remote_code=True,
                 kv_cache_config=kv_cache_config,
                 **pytorch_config,
                 enable_attention_dp=attention_dp,
                 speculative_config=mtp_config) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES

            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


class TestMinitron4BBaseInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "nvidia/Nemotron-Mini-4B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/nemotron/nemotron-mini-4b-instruct_vfp8-fp8-bf16-export"

    @skip_pre_ada
    def test_fp8_prequantized(self):
        with LLM(self.MODEL_PATH) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)


class TestNemotronNas(LlmapiAccuracyTestHarness):
    MODEL_NAME = "nemotron-nas/Llama-3_1-Nemotron-51B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/nemotron-nas/Llama-3_1-Nemotron-51B-Instruct"

    @pytest.mark.skip_less_device_memory(80000)
    @pytest.mark.skip_less_device(8)
    def test_auto_dtype_tp8(self):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.8)
        pytorch_config = dict()

        with LLM(self.MODEL_PATH,
                 tensor_parallel_size=8,
                 kv_cache_config=kv_cache_config,
                 **pytorch_config) as llm:

            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)


class TestMistralNemo12B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "mistralai/Mistral-Nemo-12b-Base"
    MODEL_PATH = f"{llm_models_root()}/Mistral-Nemo-Base-2407"

    @pytest.mark.skip_less_device_memory(80000)
    def test_auto_dtype(self):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.9)

        with LLM(self.MODEL_PATH,
                 kv_cache_config=kv_cache_config,
                 max_batch_size=8) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(2)
    def test_auto_dtype_tp2(self):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.9)

        with LLM(self.MODEL_PATH,
                 kv_cache_config=kv_cache_config,
                 tensor_parallel_size=2,
                 max_batch_size=8) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)


@pytest.mark.timeout(5400)
@pytest.mark.skip_less_device_memory(80000)
class TestLlama3_3NemotronSuper49Bv1(LlmapiAccuracyTestHarness):
    MODEL_NAME = "nvidia/Llama-3_3-Nemotron-Super-49B-v1"
    MODEL_PATH = f"{llm_models_root()}/nemotron-nas/Llama-3_3-Nemotron-Super-49B-v1"

    @pytest.mark.skip_less_device(2)
    @pytest.mark.skip_less_device_memory(80000)
    def test_auto_dtype_tp2(self):
        with LLM(self.MODEL_PATH,
                 tensor_parallel_size=2,
                 max_seq_len=8192,
                 max_batch_size=64) as llm:
            # Run only one eval as maximal BS is not large
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_hopper
    @pytest.mark.skip_less_device(2)
    @pytest.mark.skip_device_not_contain(["H100", "B200"])
    def test_fp8_prequantized_tp2(self):
        model_path = f"{llm_models_root()}/nemotron-nas/Llama-3_3-Nemotron-Super-49B-v1-FP8"
        with LLM(model_path,
                 tensor_parallel_size=2,
                 max_seq_len=8192,
                 max_batch_size=64) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8

            # Run only one eval as maximal BS is not large
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)


class TestLlama3_1NemotronNano8Bv1(LlmapiAccuracyTestHarness):
    MODEL_NAME = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
    MODEL_PATH = f"{llm_models_root()}/Llama-3.1-Nemotron-Nano-8B-v1"

    def test_auto_dtype(self):
        with LLM(self.MODEL_PATH) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            # skip gpqa test due to time consuming
            # task = GPQADiamond(self.MODEL_NAME)
            # task.evaluate(llm,
            #               extra_evaluator_kwargs=dict(apply_chat_template=True))

    @skip_pre_hopper
    @pytest.mark.skip_device_not_contain(["H100", "B200"])
    def test_fp8_prequantized(self):
        model_path = f"{llm_models_root()}/Llama-3.1-Nemotron-Nano-8B-v1-FP8"
        with LLM(model_path) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            # skip gpqa test due to time consuming
            # task = GPQADiamond(self.MODEL_NAME)
            # task.evaluate(llm,
            #               extra_evaluator_kwargs=dict(apply_chat_template=True))


class TestNemotronUltra(LlmapiAccuracyTestHarness):
    MODEL_NAME = "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1"
    MODEL_PATH = f"{llm_models_root()}/nemotron-nas/Llama-3_1-Nemotron-Ultra-253B-v1"

    @pytest.mark.skip_less_device(8)
    @pytest.mark.skip_less_device_memory(140000)
    @parametrize_with_ids("cuda_graph", [False, True])
    @pytest.mark.parametrize("tp_size,pp_size,ep_size", [(8, 1, 1), (8, 1, 4),
                                                         (8, 1, 8)],
                             ids=["tp8", "tp8ep4", "tp8ep8"])
    def test_auto_dtype(self, cuda_graph, tp_size, pp_size, ep_size):
        with LLM(self.MODEL_PATH,
                 max_batch_size=32,
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 moe_expert_parallel_size=ep_size,
                 cuda_graph_config=CudaGraphConfig()
                 if cuda_graph else None) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            # task = GPQADiamond(self.MODEL_NAME)
            # task.evaluate(llm,
            #                 extra_evaluator_kwargs=dict(apply_chat_template=True))

    @skip_pre_hopper
    @pytest.mark.skip_less_device(8)
    @pytest.mark.skip_device_not_contain(["H100", "B200"])
    @parametrize_with_ids("cuda_graph", [False, True])
    @pytest.mark.parametrize("tp_size,pp_size,ep_size", [(8, 1, 1), (8, 1, 4),
                                                         (8, 1, 8)],
                             ids=["tp8", "tp8ep4", "tp8ep8"])
    def test_fp8_prequantized(self, cuda_graph, tp_size, pp_size, ep_size):
        model_path = f"{llm_models_root()}/nemotron-nas/Llama-3_1-Nemotron-Ultra-253B-v1-FP8"
        with LLM(model_path,
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 moe_expert_parallel_size=ep_size,
                 cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
                 kv_cache_config=KvCacheConfig(
                     free_gpu_memory_fraction=0.85)) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            task = GPQADiamond(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=dict(apply_chat_template=True))


class TestNemotronH(LlmapiAccuracyTestHarness):
    MODEL_NAME = "nvidia/Nemotron-H-8B-Base-8K"
    MODEL_PATH = f"{llm_models_root()}/Nemotron-H-8B-Base-8K"

    @parametrize_with_ids("cuda_graph", [False, True])
    def test_auto_dtype(self, cuda_graph):
        # TODO: remove max_batch_size after mamba cache manager is supported
        # Once removed max_batch_size, the test will OOM
        kv_cache_config = KvCacheConfig(enable_block_reuse=False)
        with LLM(self.MODEL_PATH,
                 kv_cache_config=kv_cache_config,
                 max_batch_size=128,
                 cuda_graph_config=CudaGraphConfig()
                 if cuda_graph else None) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_ada
    @parametrize_with_ids("cuda_graph", [False, True])
    def test_reasoning_fp8_prequantized(self, cuda_graph):
        kv_cache_config = KvCacheConfig(enable_block_reuse=False)
        with LLM(f"{llm_models_root()}/Nemotron-H-8B-Reasoning-128K-FP8",
                 kv_cache_config=kv_cache_config,
                 max_batch_size=256,
                 cuda_graph_config=CudaGraphConfig()
                 if cuda_graph else None) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


@pytest.mark.skip_less_device(8)
@pytest.mark.skip_less_device_memory(80000)
class TestNemotronH_47B_Base(LlmapiAccuracyTestHarness):
    MODEL_NAME = "nvidia/Nemotron-H-47B-Base-8K"
    MODEL_PATH = f"{llm_models_root()}/Nemotron-H-47B-Base-8K"

    @parametrize_with_ids("cuda_graph", [False, True])
    @pytest.mark.parametrize("tp_size,pp_size,ep_size", [(8, 1, 1), (8, 1, 4),
                                                         (8, 1, 8)],
                             ids=["tp8", "tp8ep4", "tp8ep8"])
    def test_auto_dtype(self, cuda_graph, tp_size, pp_size, ep_size):
        kv_cache_config = KvCacheConfig(enable_block_reuse=False,
                                        free_gpu_memory_fraction=0.6)
        with LLM(self.MODEL_PATH,
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 moe_expert_parallel_size=ep_size,
                 kv_cache_config=kv_cache_config,
                 max_batch_size=256,
                 cuda_graph_config=CudaGraphConfig()
                 if cuda_graph else None) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_ada
    @parametrize_with_ids("cuda_graph", [False, True])
    @pytest.mark.parametrize("tp_size,pp_size,ep_size", [(8, 1, 1), (8, 1, 4),
                                                         (8, 1, 8)],
                             ids=["tp8", "tp8ep4", "tp8ep8"])
    def test_reasoning_fp8_prequantized(self, cuda_graph, tp_size, pp_size,
                                        ep_size):
        kv_cache_config = KvCacheConfig(enable_block_reuse=False,
                                        free_gpu_memory_fraction=0.6)
        with LLM(f"{llm_models_root()}/Nemotron-H-47B-Reasoning-128K-FP8",
                 kv_cache_config=kv_cache_config,
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 moe_expert_parallel_size=ep_size,
                 max_batch_size=256,
                 cuda_graph_config=CudaGraphConfig()
                 if cuda_graph else None) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


@pytest.mark.skip_less_device(8)
@pytest.mark.skip_less_device_memory(80000)
class TestNemotronH_56B_Base(LlmapiAccuracyTestHarness):
    MODEL_NAME = "nvidia/Nemotron-H-56B-Base-8K"
    MODEL_PATH = f"{llm_models_root()}/Nemotron-H-56B-Base-8K"

    @parametrize_with_ids("cuda_graph", [False, True])
    @pytest.mark.parametrize("tp_size,pp_size,ep_size", [(8, 1, 1), (8, 1, 4),
                                                         (8, 1, 8)],
                             ids=["tp8", "tp8ep4", "tp8ep8"])
    def test_auto_dtype(self, cuda_graph, tp_size, pp_size, ep_size):
        kv_cache_config = KvCacheConfig(enable_block_reuse=False,
                                        free_gpu_memory_fraction=0.6)
        with LLM(self.MODEL_PATH,
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 moe_expert_parallel_size=ep_size,
                 kv_cache_config=kv_cache_config,
                 max_batch_size=256,
                 cuda_graph_config=CudaGraphConfig()
                 if cuda_graph else None) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


class TestQwen2_7BInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/Qwen2-7B-Instruct"
    EXTRA_EVALUATOR_KWARGS = dict(
        apply_chat_template=True,
        system_prompt=
        "You are a helpful assistant, please summarize the article entered by the user with one or two sentences."
    )

    def test_auto_dtype(self):
        with LLM(self.MODEL_PATH) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)


class TestQwen3_8B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "Qwen3/Qwen3-8B"

    @skip_pre_hopper
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,attention_dp,cuda_graph,overlap_scheduler",
        [(1, 1, 1, False, True, True)],
        ids=["latency"])
    def test_fp8_block_scales(self, tp_size, pp_size, ep_size, attention_dp,
                              cuda_graph, overlap_scheduler):
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None)

        with LLM(f"{llm_models_root()}/Qwen3/Qwen3-8B-FP8",
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 moe_expert_parallel_size=ep_size,
                 **pytorch_config,
                 enable_attention_dp=attention_dp,
                 max_batch_size=64) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_hopper
    def test_dummy_load_format(self):
        llm = LLM(
            f"{llm_models_root()}/Qwen3/Qwen3-8B-FP8",
            load_format="dummy",
        )
        with llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm, is_integration_test=True)

    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,attention_dp,cuda_graph,overlap_scheduler,is_cached",
        [(1, 1, 1, False, True, True, True),
         pytest.param(8,
                      1,
                      1,
                      False,
                      True,
                      True,
                      False,
                      marks=pytest.mark.skip_less_mpi_world_size(8))],
        ids=["latency", "multi_gpus_no_cache"])
    def test_bf16(self, tp_size, pp_size, ep_size, attention_dp, cuda_graph,
                  overlap_scheduler, is_cached):
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None)

        with LLM(f"{llm_models_root()}/Qwen3/Qwen3-8B"
                 if is_cached else "Qwen/Qwen3-8B",
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 moe_expert_parallel_size=ep_size,
                 **pytorch_config,
                 enable_attention_dp=attention_dp) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @parametrize_with_ids("eagle3_one_model", [True, False])
    @parametrize_with_ids("enable_chunked_prefill", [False, True])
    def test_eagle3(self, enable_chunked_prefill, eagle3_one_model):
        pytorch_config = dict(
            disable_overlap_scheduler=True,
            cuda_graph_config=CudaGraphConfig(),
        )
        kv_cache_config = KvCacheConfig(
            enable_block_reuse=False,
            free_gpu_memory_fraction=0.6,
        )

        eagle_model_dir = f"{llm_models_root()}/Qwen3/qwen3_8b_eagle3"
        target_model_dir = f"{llm_models_root()}/Qwen3/Qwen3-8B"

        draft_len = 4
        spec_config = EagleDecodingConfig(max_draft_len=draft_len,
                                          speculative_model_dir=eagle_model_dir,
                                          eagle3_one_model=eagle3_one_model)

        llm = LLM(model=target_model_dir,
                  **pytorch_config,
                  kv_cache_config=kv_cache_config,
                  enable_chunked_prefill=enable_chunked_prefill,
                  max_num_tokens=256 if enable_chunked_prefill else 8192,
                  speculative_config=spec_config,
                  build_config=None)

        with llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,attention_dp,cuda_graph,overlap_scheduler",
        [(1, 1, 1, False, True, True)],
        ids=["latency"])
    @pytest.mark.parametrize("activation_dtype", ["fp8", "mxfp8"])
    def test_w4a8_mxfp4(self, tp_size, pp_size, ep_size, attention_dp,
                        cuda_graph, overlap_scheduler, activation_dtype):
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None)

        llm = LLM(
            f"{llm_models_root()}/mxfp4-qwen3/saved_models_Qwen3-8B_w4a8_mxfp4_{activation_dtype}_kv_none_hf",
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            moe_expert_parallel_size=ep_size,
            **pytorch_config,
            enable_attention_dp=attention_dp)
        with llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)


class TestQwen3_30B_A3B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "Qwen3/Qwen3-30B-A3B"

    @skip_pre_hopper
    @skip_post_blackwell
    @parametrize_with_ids("torch_compile", [False, True])
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,attention_dp,cuda_graph,overlap_scheduler",
        [(1, 1, 1, False, False, True)],
        ids=["latency"])
    def test_fp8_block_scales(self, tp_size, pp_size, ep_size, attention_dp,
                              cuda_graph, overlap_scheduler, torch_compile):
        torch_compile_config = TorchCompileConfig(
            enable_fullgraph=True,
            enable_piecewise_cuda_graph=cuda_graph,
            capture_num_tokens=[2048, 8192],
            max_num_streams=3) if torch_compile else None

        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            torch_compile_config=torch_compile_config)

        with LLM(f"{llm_models_root()}/Qwen3/Qwen3-30B-A3B-FP8",
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 moe_expert_parallel_size=ep_size,
                 **pytorch_config,
                 enable_attention_dp=attention_dp) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_hopper
    def test_dummy_load_format(self):
        llm = LLM(
            f"{llm_models_root()}/Qwen3/Qwen3-30B-A3B-FP8",
            load_format="dummy",
        )
        with llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm, is_integration_test=True)

    @skip_pre_hopper
    @parametrize_with_ids("torch_compile", [False, True])
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,attention_dp,cuda_graph,overlap_scheduler",
        [(1, 1, 1, True, True, True)],
        ids=["latency"])
    def test_fp8(self, tp_size, pp_size, ep_size, attention_dp, cuda_graph,
                 overlap_scheduler, torch_compile):
        torch_compile_config = TorchCompileConfig(
            enable_fullgraph=True,
            enable_piecewise_cuda_graph=cuda_graph,
            capture_num_tokens=[2048, 8192],
            max_num_streams=3) if torch_compile else None

        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            torch_compile_config=torch_compile_config)

        with LLM(f"{llm_models_root()}/Qwen3/saved_models_Qwen3-30B-A3B_fp8_hf",
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 moe_expert_parallel_size=ep_size,
                 **pytorch_config,
                 enable_attention_dp=attention_dp) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @parametrize_with_ids("torch_compile", [False, True])
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,attention_dp,cuda_graph,overlap_scheduler,moe_backend",
        [
            (1, 1, 1, False, True, True, "CUTLASS"),
            (1, 1, 1, False, True, True, "TRTLLM"),
            (4, 1, 4, True, True, True, "CUTLASS"),
            (4, 1, 4, True, True, True, "TRTLLM"),
            (4, 1, 4, False, True, True, "CUTLASS"),
            (4, 1, 4, False, True, True, "TRTLLM"),
        ],
        ids=[
            "latency_moe_cutlass",
            "latency_moe_trtllm",
            "dep4_latency_moe_cutlass",
            "dep4_latency_moe_trtllm",
            "tep4_latency_moe_cutlass",
            "tep4_latency_moe_trtllm",
        ],
    )
    def test_nvfp4(
        self,
        tp_size,
        pp_size,
        ep_size,
        attention_dp,
        cuda_graph,
        overlap_scheduler,
        moe_backend,
        torch_compile,
    ):

        if moe_backend == "TRTLLM" and (get_sm_version() == 120
                                        or get_sm_version() == 121):
            pytest.skip(
                "MOE TRTLLM backend does not support SM version 120 or 121")

        torch_compile_config = TorchCompileConfig(
            enable_fullgraph=True,
            enable_piecewise_cuda_graph=cuda_graph,
            capture_num_tokens=[2048, 8192],
            max_num_streams=3) if torch_compile else None

        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            moe_config=MoeConfig(backend=moe_backend),
            torch_compile_config=torch_compile_config,
        )

        with LLM(
                f"{llm_models_root()}/Qwen3/saved_models_Qwen3-30B-A3B_nvfp4_hf",
                tensor_parallel_size=tp_size,
                pipeline_parallel_size=pp_size,
                moe_expert_parallel_size=ep_size,
                **pytorch_config,
                enable_attention_dp=attention_dp,
                max_batch_size=32) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    def test_eagle3(self):
        pytorch_config = dict(
            disable_overlap_scheduler=True,
            cuda_graph_config=CudaGraphConfig(batch_sizes=[1, 2, 3, 4, 8]),
        )
        kv_cache_config = KvCacheConfig(enable_block_reuse=False)

        eagle_model_dir = f"{llm_models_root()}/Qwen3/Qwen3-30B-eagle3"
        target_model_dir = f"{llm_models_root()}/Qwen3/Qwen3-30B-A3B"

        draft_len = 1
        spec_config = EagleDecodingConfig(max_draft_len=draft_len,
                                          speculative_model_dir=eagle_model_dir,
                                          eagle3_one_model=True)

        llm = LLM(model=target_model_dir,
                  **pytorch_config,
                  kv_cache_config=kv_cache_config,
                  speculative_config=spec_config,
                  max_seq_len=8192)

        with llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.parametrize("moe_backend", ["CUTLASS", "TRITON", "TRTLLM"])
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,attention_dp,cuda_graph,overlap_scheduler", [
            (1, 1, 1, False, True, True),
            (2, 1, 2, False, True, True),
            (4, 1, 4, False, True, True),
        ],
        ids=["latency", "ep2", "ep4"])
    @pytest.mark.parametrize("activation_dtype", ["static_fp8", "mxfp8"],
                             ids=["fp8", "mxfp8"])
    def test_w4a8_mxfp4(self, moe_backend, tp_size, pp_size, ep_size,
                        attention_dp, cuda_graph, overlap_scheduler,
                        activation_dtype):
        if moe_backend == "TRITON":
            if not IS_TRITON_KERNELS_AVAILABLE:
                pytest.skip("TRITON moe backend is not available.")
            if get_sm_version() < 90:
                pytest.skip("TRITON moe backend requires Hopper or newer.")
        if moe_backend in ["CUTLASS", "TRTLLM"] and get_sm_version() < 100:
            pytest.skip(
                "CUTLASS or TRTLLM moe backend requires Blackwell or newer.")
        if activation_dtype == "mxfp8" and moe_backend not in [
                "TRTLLM", "CUTLASS"
        ]:
            pytest.skip(
                "Mxfp8 is only supported for TRTLLM or CUTLASS moe backend.")

        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None)

        llm = LLM(
            f"{llm_models_root()}/mxfp4-qwen3/saved_models_Qwen3-30B-A3B_w4a8_mxfp4_{activation_dtype}_kv_none_hf_moeonly",
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            moe_expert_parallel_size=ep_size,
            **pytorch_config,
            enable_attention_dp=attention_dp,
            moe_config=MoeConfig(backend=moe_backend))
        with llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,attention_dp,cuda_graph,overlap_scheduler,moe_backend",
        [(1, 1, 1, False, True, True, "TRTLLM")],
        ids=["latency-TRTLLM"])
    def test_w4a16_mxfp4(self, tp_size, pp_size, ep_size, attention_dp,
                         cuda_graph, overlap_scheduler, moe_backend):
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            moe_config=MoeConfig(backend=moe_backend))

        llm = LLM(
            f"{llm_models_root()}/mxfp4-qwen3/saved_models_Qwen3-30B-A3B_w4a16_mxfp4_kv_none_hf_moeonly",
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            moe_expert_parallel_size=ep_size,
            enable_attention_dp=attention_dp,
            **pytorch_config)
        with llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)


class TestQwen3_32B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "Qwen3/Qwen3-32B"

    @skip_pre_hopper
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,attention_dp,cuda_graph,overlap_scheduler",
        [(1, 1, 1, False, False, True)],
        ids=["latency"])
    def test_fp8_block_scales(self, tp_size, pp_size, ep_size, attention_dp,
                              cuda_graph, overlap_scheduler):
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None)

        with LLM(f"{llm_models_root()}/Qwen3/Qwen3-32B-FP8",
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 moe_expert_parallel_size=ep_size,
                 **pytorch_config,
                 enable_attention_dp=attention_dp) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)


class TestQwen3_235B_A22B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "Qwen3/Qwen3-235B-A22B"

    @skip_pre_hopper
    @pytest.mark.skip_less_device(8)
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,attention_dp,cuda_graph,overlap_scheduler",
        [(8, 1, 8, False, True, True), (8, 1, 8, True, True, True)],
        ids=["latency", "throughput_latency"])
    def test_fp8(self, tp_size, pp_size, ep_size, attention_dp, cuda_graph,
                 overlap_scheduler):
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None)

        free_gpu_memory_fraction = 0.6 if attention_dp else 0.3
        kv_cache_config = KvCacheConfig(
            free_gpu_memory_fraction=free_gpu_memory_fraction)
        with LLM(
                f"{llm_models_root()}/Qwen3/saved_models_Qwen3-235B-A22B_fp8_hf",
                tensor_parallel_size=tp_size,
                pipeline_parallel_size=pp_size,
                moe_expert_parallel_size=ep_size,
                **pytorch_config,
                enable_attention_dp=attention_dp,
                kv_cache_config=kv_cache_config) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_hopper
    @pytest.mark.skip_less_device(8)
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,attention_dp,cuda_graph,overlap_scheduler,moe_backend",
        [(8, 1, 8, True, True, True, "DEEPGEMM"),
         (8, 1, 8, False, True, True, "DEEPGEMM"),
         (8, 1, 8, True, True, True, "TRTLLM"),
         (8, 1, 8, False, True, True, "TRTLLM")],
        ids=[
            "latency_deepgemm", "throughput_latency_deepgemm", "latency_trtllm",
            "throughput_latency_trtllm"
        ])
    def test_fp8_block_scales(self, tp_size, pp_size, ep_size, attention_dp,
                              cuda_graph, overlap_scheduler, moe_backend):
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            moe_config=MoeConfig(backend=moe_backend))

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.6)
        with LLM(f"{llm_models_root()}/Qwen3/Qwen3-235B-A22B-FP8",
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 moe_expert_parallel_size=ep_size,
                 **pytorch_config,
                 enable_attention_dp=attention_dp,
                 kv_cache_config=kv_cache_config) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @pytest.mark.skip_less_mpi_world_size(8)
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,attention_dp,cuda_graph,overlap_scheduler,moe_backend,eagle3",
        [
            (8, 1, 8, True, True, True, "CUTLASS", False),
            (8, 1, 8, False, True, True, "TRTLLM", False),
            (8, 1, 8, True, True, True, "TRTLLM", False),
            (8, 1, 8, False, True, True, "TRTLLM", True),
            (8, 1, 8, True, True, True, "TRTLLM", True),
        ],
        ids=[
            "latency_moe_cutlass",
            "latency_moe_trtllm",
            "latency_moe_trtllm_attention_dp",
            "latency_moe_trtllm_eagle3",
            "latency_moe_trtllm_eagle3_attention_dp",
        ],
    )
    def test_nvfp4(self, tp_size, pp_size, ep_size, attention_dp, cuda_graph,
                   overlap_scheduler, moe_backend, eagle3):

        if moe_backend == "TRTLLM" and (get_sm_version() == 120
                                        or get_sm_version() == 121):
            pytest.skip(
                "MOE TRTLLM backend does not support SM version 120 or 121")

        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            moe_config=MoeConfig(backend=moe_backend))

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4,
                                        enable_block_reuse=not eagle3)
        spec_config = None
        if eagle3:
            spec_config = EagleDecodingConfig(
                max_draft_len=2,
                speculative_model_dir=
                f"{llm_models_root()}/Qwen3/qwen3-235B-eagle3/",
                eagle3_one_model=True)
        with LLM(
                f"{llm_models_root()}/Qwen3/saved_models_Qwen3-235B-A22B_nvfp4_hf",
                tensor_parallel_size=tp_size,
                pipeline_parallel_size=pp_size,
                moe_expert_parallel_size=ep_size,
                **pytorch_config,
                enable_attention_dp=attention_dp,
                kv_cache_config=kv_cache_config,
                speculative_config=spec_config) as llm:

            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @pytest.mark.skip_less_mpi_world_size(4)
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,attention_dp,cuda_graph,overlap_scheduler,moe_backend,eagle3",
        [
            (4, 1, 4, False, False, False, "TRTLLM",
             True),  # TP8 has bug when we use TRTLLM moe backend and eagle3
            (4, 1, 4, False, False, False, "CUTLASS", False),
            (4, 1, 4, False, False, False, "CUTLASS", True),
        ],
        ids=[
            "latency_moe_trtllm_eagle3",
            "latency_moe_cutlass",
            "latency_moe_cutlass_eagle3",
        ],
    )
    def test_nvfp4_4gpus(self, tp_size, pp_size, ep_size, attention_dp,
                         cuda_graph, overlap_scheduler, moe_backend, eagle3):

        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            moe_config=MoeConfig(backend=moe_backend))

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4,
                                        enable_block_reuse=not eagle3)
        spec_config = None
        if eagle3:
            spec_config = EagleDecodingConfig(
                max_draft_len=2,
                speculative_model_dir=
                f"{llm_models_root()}/Qwen3/qwen3-235B-eagle3/",
                eagle3_one_model=True)
        with LLM(
                f"{llm_models_root()}/Qwen3/saved_models_Qwen3-235B-A22B_nvfp4_hf",
                tensor_parallel_size=tp_size,
                pipeline_parallel_size=pp_size,
                moe_expert_parallel_size=ep_size,
                **pytorch_config,
                enable_attention_dp=attention_dp,
                kv_cache_config=kv_cache_config,
                speculative_config=spec_config) as llm:

            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


class TestPhi4MiniInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "microsoft/Phi-4-mini-instruct"
    MODEL_PATH = f"{llm_models_root()}/Phi-4-mini-instruct"

    def test_auto_dtype(self):
        with LLM(self.MODEL_PATH, max_seq_len=4096) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


class TestCodestral_22B_V01(LlmapiAccuracyTestHarness):
    MODEL_NAME = "mistralai/Codestral-22B-v0.1"
    MODEL_PATH = f"{llm_models_root()}/Codestral-22B-v0.1"

    @pytest.mark.skip_less_device_memory(80000)
    def test_auto_dtype(self):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.6)
        with LLM(self.MODEL_PATH, kv_cache_config=kv_cache_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


class TestKanana_Instruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "kanana-1.5-2.1b-instruct-2505"
    MODEL_PATH = f"{llm_models_root()}/kanana-1.5-2.1b-instruct-2505"

    @pytest.mark.skip_device_not_contain(["H20", "H100"])
    def test_auto_dtype(self):
        "RCCA: https://nvbugspro.nvidia.com/bug/5310520"
        pytorch_config = dict(cuda_graph_config=CudaGraphConfig(
            enable_padding=True, max_batch_size=384))
        with LLM(self.MODEL_PATH, **pytorch_config,
                 enable_attention_dp=True) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


class TestBielik11BInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "speakleash/Bielik-11B-v2.2-Instruct"

    def test_auto_dtype(self):
        with LLM(f"{llm_models_root()}/Bielik-11B-v2.2-Instruct") as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_hopper
    def test_fp8(self):
        with LLM(f"{llm_models_root()}/Bielik-11B-v2.2-Instruct-FP8") as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


class TestPhi4(LlmapiAccuracyTestHarness):
    MODEL_NAME = "microsoft/phi-4"

    def test_auto_dtype(self):
        with LLM(f"{llm_models_root()}/Phi-4") as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_hopper
    def test_fp8(self):
        with LLM(f"{llm_models_root()}/Phi-4-FP8") as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


class TestPhi4MM(LlmapiAccuracyTestHarness):
    # phi4-mm can also support text input.
    MODEL_NAME = "microsoft/Phi-4-multimodal-instruct"
    MODEL_PATH = f"{llm_models_root()}/multimodals/Phi-4-multimodal-instruct"

    def test_auto_dtype(self):
        # Set max_seq_len to 4096 to use short rope factor.
        model_name = "microsoft/Phi-4-multimodal-instruct"
        with LLM(self.MODEL_PATH, max_seq_len=4096) as llm:
            task = MMLU(model_name)
            task.evaluate(llm)
            task = GSM8K(model_name)
            task.evaluate(llm)

    def test_auto_dtype_long_rope(self):
        # Set max_seq_len larger than 4096 to use long rope factor.
        model_name = "microsoft/Phi-4-multimodal-instruct-long-rope"
        with LLM(self.MODEL_PATH, max_seq_len=8192) as llm:
            task = MMLU(model_name)
            task.evaluate(llm)
            task = GSM8K(model_name)
            task.evaluate(llm)

    @skip_pre_blackwell
    def test_fp4(self):
        model_path = f"{self.MODEL_PATH}-FP4"
        with LLM(model_path, max_seq_len=4096) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_hopper
    def test_fp8(self):
        model_path = f"{self.MODEL_PATH}-FP8"
        with LLM(model_path, max_seq_len=4096) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


@skip_pre_hopper
@pytest.mark.skip_less_device_memory(80000)
class TestGPTOSS(LlmapiAccuracyTestHarness):
    extra_evaluator_kwargs = {
        "fewshot_as_multiturn": True,
        "apply_chat_template": True,
    }

    MODEL_PATH = f"{llm_models_root()}/gpt_oss/gpt-oss-120b"

    @pytest.mark.skip(reason="https://nvbugs/5596343")
    @pytest.mark.parametrize(
        "kv_cache_dtype",
        ["auto", pytest.param("fp8", marks=skip_pre_blackwell)])
    @pytest.mark.parametrize(
        "moe_backend",
        ["CUTLASS",
         pytest.param("TRTLLM", marks=skip_pre_blackwell), "TRITON"],
        ids=["cutlass", "trtllm", "triton"])
    @pytest.mark.parametrize("cuda_graph,overlap_scheduler", [
        (True, True),
    ])
    def test_w4_1gpu(self, kv_cache_dtype, moe_backend, cuda_graph,
                     overlap_scheduler, mocker):
        MODEL_PATH = f"{llm_models_root()}/gpt_oss/gpt-oss-20b"
        mocker.patch.object(GSM8K, "MAX_OUTPUT_LEN", 8192)
        mocker.patch.dict(GSM8K.EVALUATE_KWARGS,
                          {"scores_filter": "exact_match,flexible-extract"})
        if moe_backend == "TRITON" and not IS_TRITON_KERNELS_AVAILABLE:
            pytest.skip("Triton kernels are not available")

        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None)

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.5,
                                        dtype=kv_cache_dtype)

        llm = LLM(MODEL_PATH,
                  tensor_parallel_size=1,
                  pipeline_parallel_size=1,
                  moe_expert_parallel_size=1,
                  kv_cache_config=kv_cache_config,
                  **pytorch_config,
                  moe_config=MoeConfig(backend=moe_backend))

        with llm:
            model_name = "GPT-OSS/MXFP4"
            task = GSM8K(model_name)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.extra_evaluator_kwargs)

    def test_dummy_load_format(self):
        llm = LLM(
            f"{llm_models_root()}/gpt_oss/gpt-oss-20b",
            load_format="dummy",
        )
        with llm:
            model_name = "GPT-OSS/MXFP4"
            task = GSM8K(model_name)
            task.evaluate(llm, is_integration_test=True)

    @pytest.mark.skip(reason="https://nvbugs/5596343")
    @pytest.mark.skip_less_device(4)
    @pytest.mark.parametrize(
        "kv_cache_dtype",
        ["auto", pytest.param("fp8", marks=skip_pre_blackwell)])
    @pytest.mark.parametrize(
        "moe_backend",
        ["CUTLASS",
         pytest.param("TRTLLM", marks=skip_pre_blackwell), "TRITON"],
        ids=["cutlass", "trtllm", "triton"])
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,attention_dp,cuda_graph,overlap_scheduler", [
            (4, 1, 1, False, True, True),
            (4, 1, 4, False, True, True),
            (4, 1, 4, True, True, True),
        ],
        ids=["tp4", "ep4", "dp4"])
    def test_w4_4gpus(self, kv_cache_dtype, moe_backend, tp_size, pp_size,
                      ep_size, attention_dp, cuda_graph, overlap_scheduler,
                      mocker):
        mocker.patch.object(GSM8K, "MAX_OUTPUT_LEN", 8192)
        mocker.patch.dict(GSM8K.EVALUATE_KWARGS,
                          {"scores_filter": "exact_match,flexible-extract"})
        if moe_backend == "TRITON":
            if not IS_TRITON_KERNELS_AVAILABLE:
                pytest.skip("Triton kernels are not available")

        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None)

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.5,
                                        dtype=kv_cache_dtype)

        llm = LLM(self.MODEL_PATH,
                  tensor_parallel_size=tp_size,
                  pipeline_parallel_size=pp_size,
                  moe_expert_parallel_size=ep_size,
                  kv_cache_config=kv_cache_config,
                  **pytorch_config,
                  enable_attention_dp=attention_dp,
                  moe_config=MoeConfig(backend=moe_backend))

        with llm:
            model_name = "GPT-OSS/MXFP4"
            task = GSM8K(model_name)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.extra_evaluator_kwargs)

    @pytest.mark.skip_less_device(4)
    @pytest.mark.parametrize(
        "kv_cache_dtype",
        ["auto", pytest.param("fp8", marks=skip_pre_blackwell)])
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,attention_dp,cuda_graph,overlap_scheduler", [
            (4, 1, 4, True, True, True),
        ],
        ids=["dp4"])
    def test_w4a16(self, kv_cache_dtype, tp_size, pp_size, ep_size,
                   attention_dp, cuda_graph, overlap_scheduler, monkeypatch,
                   mocker):
        mocker.patch.object(GSM8K, "MAX_OUTPUT_LEN", 8192)
        mocker.patch.dict(GSM8K.EVALUATE_KWARGS,
                          {"scores_filter": "exact_match,flexible-extract"})
        if not IS_TRITON_KERNELS_AVAILABLE:
            pytest.skip("Triton kernels are not available")
        monkeypatch.setenv("OVERRIDE_QUANT_ALGO", "W4A16_MXFP4")

        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None)

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.5,
                                        dtype=kv_cache_dtype)

        llm = LLM(self.MODEL_PATH,
                  tensor_parallel_size=tp_size,
                  pipeline_parallel_size=pp_size,
                  moe_expert_parallel_size=ep_size,
                  kv_cache_config=kv_cache_config,
                  **pytorch_config,
                  enable_attention_dp=attention_dp,
                  moe_config=MoeConfig(backend="TRITON"))
        with llm:
            model_name = "GPT-OSS/BF16"
            task = GSM8K(model_name)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.extra_evaluator_kwargs)

    @pytest.mark.skip_less_device(2)
    @pytest.mark.parametrize(
        "kv_cache_dtype",
        ["auto", pytest.param("fp8", marks=skip_pre_blackwell)])
    @pytest.mark.parametrize(
        "moe_backend",
        ["CUTLASS",
         pytest.param("TRTLLM", marks=skip_pre_blackwell), "TRITON"],
        ids=["cutlass", "trtllm", "triton"])
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,attention_dp,cuda_graph,overlap_scheduler", [
            (2, 1, 1, False, True, True),
            (2, 1, 2, False, True, True),
            (2, 1, 2, True, True, True),
        ],
        ids=["tp2", "ep2", "dp2"])
    def test_w4_2gpus(self, kv_cache_dtype, moe_backend, tp_size, pp_size,
                      ep_size, attention_dp, cuda_graph, overlap_scheduler,
                      mocker):
        MODEL_PATH = f"{llm_models_root()}/gpt_oss/gpt-oss-20b"
        if moe_backend == "TRITON":
            if not IS_TRITON_KERNELS_AVAILABLE:
                pytest.skip("Triton kernels are not available")

        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None)

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.5,
                                        dtype=kv_cache_dtype)

        llm = LLM(MODEL_PATH,
                  tensor_parallel_size=tp_size,
                  pipeline_parallel_size=pp_size,
                  moe_expert_parallel_size=ep_size,
                  kv_cache_config=kv_cache_config,
                  max_seq_len=8192,
                  **pytorch_config,
                  enable_attention_dp=attention_dp,
                  moe_config=MoeConfig(backend=moe_backend))

        with llm:
            model_name = "GPT-OSS/MXFP4"
            task = GSM8K(model_name)
            mocker.patch.object(GSM8K, "MAX_OUTPUT_LEN", 8192)
            mocker.patch.dict(GSM8K.EVALUATE_KWARGS,
                              {"scores_filter": "exact_match,flexible-extract"})
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.extra_evaluator_kwargs)

    @pytest.mark.skip_less_device(4)
    @pytest.mark.parametrize(
        "kv_cache_dtype",
        ["auto", pytest.param("fp8", marks=skip_pre_blackwell)])
    @pytest.mark.parametrize(
        "moe_backend",
        ["CUTLASS",
         pytest.param("TRTLLM", marks=skip_pre_blackwell), "TRITON"],
        ids=["cutlass", "trtllm", "triton"])
    def test_w4_chunked_prefill(self, kv_cache_dtype, moe_backend, mocker):
        if moe_backend == "TRITON":
            if not IS_TRITON_KERNELS_AVAILABLE:
                pytest.skip("Triton kernels are not available")

        pytorch_config = dict(disable_overlap_scheduler=True,
                              cuda_graph_config=CudaGraphConfig())
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.6,
                                        dtype=kv_cache_dtype)

        model_name = "GPT-OSS/MXFP4"
        with LLM(self.MODEL_PATH,
                 tensor_parallel_size=4,
                 pipeline_parallel_size=1,
                 moe_expert_parallel_size=1,
                 kv_cache_config=kv_cache_config,
                 max_seq_len=8192,
                 max_num_tokens=512,
                 enable_chunked_prefill=True,
                 enable_attention_dp=False,
                 moe_config=MoeConfig(backend=moe_backend),
                 **pytorch_config) as llm:
            mocker.patch.object(GSM8K, "MAX_OUTPUT_LEN", 8192)
            mocker.patch.dict(GSM8K.EVALUATE_KWARGS,
                              {"scores_filter": "exact_match,flexible-extract"})
            task = GSM8K(model_name)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.extra_evaluator_kwargs)


@skip_pre_hopper
class TestEXAONE4(LlmapiAccuracyTestHarness):
    MODEL_NAME = "LGAI-EXAONE/EXAONE-4.0-32B"
    kv_cache_config = KvCacheConfig(enable_block_reuse=False,
                                    enable_partial_reuse=False)

    def test_auto_dtype(self):
        model_path = f"{llm_models_root()}/EXAONE-4.0-32B"
        with LLM(model_path, kv_cache_config=self.kv_cache_config) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


class TestQwen2_VL_7B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/Qwen2-VL-7B-Instruct"

    # NOTE: MMMU adds <|endoftext|> to the stop token.
    sampling_params = SamplingParams(max_tokens=MMMU.MAX_OUTPUT_LEN,
                                     truncate_prompt_tokens=MMMU.MAX_INPUT_LEN,
                                     stop="<|endoftext|>")

    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.6)

    @pytest.mark.skip(reason="https://nvbugs/5601909")
    def test_auto_dtype(self):
        with LLM(self.MODEL_PATH,
                 max_num_tokens=16384,
                 kv_cache_config=self.kv_cache_config) as llm:
            task = MMMU(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=self.sampling_params)


class TestQwQ_32B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "Qwen/QwQ-32B"
    MODEL_PATH = f"{llm_models_root()}/QwQ-32B"

    @pytest.mark.skip_less_device_memory(80000)
    @pytest.mark.skip_less_device(4)
    def test_auto_dtype_tp4(self):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.5)

        with LLM(self.MODEL_PATH,
                 max_num_tokens=16384,
                 kv_cache_config=kv_cache_config,
                 tensor_parallel_size=4,
                 max_batch_size=8) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)


@pytest.mark.skip_less_device_memory(80000)
class TestQwen3NextThinking(LlmapiAccuracyTestHarness):
    MODEL_NAME = "Qwen3/Qwen3-Next-80B-A3B-Thinking"
    MODEL_PATH = f"{llm_models_root()}/{MODEL_NAME}"

    @skip_pre_hopper
    @pytest.mark.skip_less_device(4)
    @pytest.mark.parametrize("tp_size,pp_size,ep_size", [(4, 1, 4)],
                             ids=["tp4ep4"])
    def test_auto_dtype(self, tp_size, pp_size, ep_size):
        if get_device_count() != tp_size * pp_size:
            pytest.skip("Device count mismatch with world size")

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.6,
                                        enable_block_reuse=False)
        cuda_graph_config = CudaGraphConfig(enable_padding=True,
                                            max_batch_size=720)

        with LLM(self.MODEL_PATH,
                 max_num_tokens=4096,
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 moe_expert_parallel_size=ep_size,
                 kv_cache_config=kv_cache_config,
                 cuda_graph_config=cuda_graph_config) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


class TestNano_V2_VLM(LlmapiAccuracyTestHarness):
    MODEL_NAME = "nvidia/Nano-v2-VLM"
    MODEL_PATH = f"{llm_models_root()}/Nano-v2-VLM"
    MAX_NUM_TOKENS = 25600

    # NOTE: MMMU adds <|endoftext|> to the stop token.
    sampling_params = SamplingParams(max_tokens=MAX_NUM_TOKENS,
                                     truncate_prompt_tokens=MMMU.MAX_INPUT_LEN,
                                     stop="<|endoftext|>")

    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.8,
                                    enable_block_reuse=False)

    @pytest.mark.skip(reason="Nano V2 VLM ckpt is not released yet.")
    def test_auto_dtype(self):
        with LLM(self.MODEL_PATH,
                 max_batch_size=128,
                 max_num_tokens=self.MAX_NUM_TOKENS,
                 kv_cache_config=self.kv_cache_config) as llm:
            task = MMMU(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=self.sampling_params)


class TestSeedOss_36B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "ByteDance-Seed/Seed-OSS-36B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/Seed-OSS/Seed-OSS-36B-Instruct"

    gsm8k_sampling_params = SamplingParams(temperature=1.1,
                                           top_p=0.95,
                                           max_tokens=16384)

    @skip_pre_hopper
    @pytest.mark.skip_less_device_memory(140000)
    def test_auto_dtype(self):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.8)
        chat_template_kwargs = dict(thinking_budget=-1)

        with LLM(self.MODEL_PATH, kv_cache_config=kv_cache_config) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm,
                          sampling_params=self.gsm8k_sampling_params,
                          extra_evaluator_kwargs=dict(
                              apply_chat_template=True,
                              chat_template_kwargs=chat_template_kwargs))
