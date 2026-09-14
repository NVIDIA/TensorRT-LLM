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

import gc

import pytest
import torch

from tensorrt_llm import LLM
from tensorrt_llm.evaluate.post_processing import strip_thinking_and_extract_mmmu_answer
from tensorrt_llm.llmapi import (
    CudaGraphConfig,
    KvCacheConfig,
    MambaStateConfig,
    MTPDecodingConfig,
    SamplingParams,
)
from tensorrt_llm.quantization import QuantAlgo

from ..conftest import llm_models_root, parametrize_with_ids, skip_pre_blackwell
from .accuracy_core import (
    GSM8K,
    MMMU,
    ForceTokenLogitsProcessor,
    LlmapiAccuracyTestHarness,
    assert_acceptance_length_for_llm,
)


class TestGLM53FlashFP8(LlmapiAccuracyTestHarness):
    """GLM-5.3-Flash FP8 accuracy and runtime tests on B200.

    Block reuse is enabled only in the periodic-snapshot test.
    """

    MODEL_NAME = "zai-org/GLM-5.3-Flash"
    MODEL_PATH = f"{llm_models_root()}/GLM-5.3-Flash"

    @staticmethod
    def _llm_kwargs(tp_size: int, ep_size: int) -> dict:
        return dict(
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=1,
            moe_expert_parallel_size=ep_size,
            kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.5, enable_block_reuse=False),
            max_batch_size=64,
            max_num_tokens=16384,
            max_seq_len=8192,
            cuda_graph_config=CudaGraphConfig(max_batch_size=64, enable_padding=True),
            disable_overlap_scheduler=False,
        )

    @staticmethod
    def _assert_glm5_next_stack(llm: LLM) -> None:
        assert llm.args.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
        assert llm.args.kv_cache_config.enable_block_reuse is False
        assert llm.args.cuda_graph_config is not None
        assert llm.args.cuda_graph_config.enable_padding is True

    @skip_pre_blackwell
    @pytest.mark.skip_less_mpi_world_size(4)
    @parametrize_with_ids("tp_size,ep_size", [(4, 4)])
    def test_tep(self, tp_size, ep_size):
        with LLM(self.MODEL_PATH, **self._llm_kwargs(tp_size, ep_size)) as llm:
            self._assert_glm5_next_stack(llm)
            assert llm.args.speculative_config is None
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @pytest.mark.skip_less_mpi_world_size(4)
    @parametrize_with_ids("tp_size,ep_size", [(4, 4)])
    def test_attention_dp(self, tp_size, ep_size):
        """Attention data parallelism.

        Replicated KDA / sparse-MLA / dense weights per rank, the fused MoE
        combining across ranks.
        """
        kwargs = self._llm_kwargs(tp_size, ep_size)
        kwargs["enable_attention_dp"] = True
        # Every rank prefills its own batch and the fused MoE gathers all
        # ranks' tokens, so the per-rank token budget is a quarter of the
        # tensor-parallel one for the same activation peak.
        kwargs["max_num_tokens"] = 4096
        with LLM(self.MODEL_PATH, **kwargs) as llm:
            self._assert_glm5_next_stack(llm)
            assert llm.args.enable_attention_dp is True
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @pytest.mark.skip_less_mpi_world_size(4)
    @parametrize_with_ids("tp_size,ep_size", [(4, 4)])
    def test_block_reuse(self, tp_size, ep_size):
        """KV block reuse with periodic Mamba state snapshots.

        The hybrid V2 manager silently disables reuse unless a snapshot
        policy is set; with one, a prefix hit restores the KDA recurrent
        state from the snapshot and reuses the latent / indexer pages. The
        5-shot GSM8K prompts share a long prefix, so nearly every request
        exercises the path.
        """
        kwargs = self._llm_kwargs(tp_size, ep_size)
        kwargs["kv_cache_config"] = KvCacheConfig(
            free_gpu_memory_fraction=0.5,
            enable_block_reuse=True,
            mamba_state_config=MambaStateConfig(periodic_snapshot_interval=256),
        )
        with LLM(self.MODEL_PATH, return_perf_metrics=True, **kwargs) as llm:
            assert llm.args.kv_cache_config.enable_block_reuse is True
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
            self._assert_kv_cache_reuse(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @staticmethod
    def _assert_kv_cache_reuse(llm: LLM) -> None:
        # Long enough to cross a 256-token snapshot point; the warm request
        # must report reused blocks and reproduce the cold request's tokens.
        prompt = llm.tokenizer.encode(
            "The capital of France is Paris. The capital of Germany is Berlin. " * 40
        )
        sampling_params = SamplingParams(
            max_tokens=8, temperature=0, end_id=-1, return_perf_metrics=True
        )
        cold = llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)[0].outputs[0]
        warm = llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)[0].outputs[0]
        assert cold.request_perf_metrics.kv_cache_metrics.num_reused_blocks == 0
        assert warm.request_perf_metrics.kv_cache_metrics.num_reused_blocks > 0
        assert warm.token_ids == cold.token_ids

    # MMMU through the multimodal wrapper (image inputs). The thinking model
    # answers inside <think>, so the K2.5 strip-thinking extractor scores it;
    # reasoning_effort is the checkpoint's own chat-template knob. Output budget
    # follows the native-HF reference protocol (4096 generated tokens).
    MMMU_EXTRA_EVALUATOR_KWARGS = dict(
        chat_template_kwargs={"reasoning_effort": "max"},
        post_process_fn=strip_thinking_and_extract_mmmu_answer,
        preserve_caller_max_tokens=True,
    )

    @skip_pre_blackwell
    @pytest.mark.skip_less_mpi_world_size(4)
    @pytest.mark.timeout(7200)
    @parametrize_with_ids("tp_size,ep_size", [(4, 4)])
    def test_mmmu(self, tp_size, ep_size):
        kwargs = self._llm_kwargs(tp_size, ep_size)
        # MMMU prompts fit in 8K (MAX_INPUT_LEN); a smaller token budget and
        # batch keep the run cheap. Profiling peaks are the same as the text
        # tests' (about 92 GiB per GPU at 16K / batch 64 on B200).
        kwargs["max_num_tokens"] = MMMU.MAX_INPUT_LEN
        kwargs["max_seq_len"] = MMMU.MAX_INPUT_LEN + 4096
        kwargs["max_batch_size"] = 32
        kwargs["cuda_graph_config"] = CudaGraphConfig(max_batch_size=32, enable_padding=True)
        with LLM(self.MODEL_PATH, **kwargs) as llm:
            self._assert_glm5_next_stack(llm)
            task = MMMU(self.MODEL_NAME)
            task.evaluate(
                llm,
                sampling_params=SamplingParams(
                    max_tokens=4096, truncate_prompt_tokens=MMMU.MAX_INPUT_LEN
                ),
                extra_evaluator_kwargs=self.MMMU_EXTRA_EVALUATOR_KWARGS,
            )

    @skip_pre_blackwell
    @pytest.mark.skip_less_device(4)
    @pytest.mark.timeout(1800)
    def test_video_url(self, tmp_path):
        """Exercise GLM video ingestion through the OpenAI serving endpoint."""
        import json
        import sys
        from pathlib import Path

        import yaml
        from openai import OpenAI

        from ..common import get_free_port_in_ci
        from ..examples.serve.test_serve import _wait_for_server_ready
        from ..trt_test_alternative import popen

        pytest.importorskip("cv2", reason="video decoding requires OpenCV")
        pytest.importorskip("transformers.models.glm5_next.processing_glm5_next")
        video = Path(llm_models_root()) / "multimodals/test_data/OAI-sora-tokyo-walk.mp4"
        if not video.is_file():
            pytest.skip(f"video fixture is unavailable: {video}")
        config_path = tmp_path / "video.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "kv_cache_config": {
                        "free_gpu_memory_fraction": 0.5,
                        "enable_block_reuse": False,
                    },
                    "cuda_graph_config": {"max_batch_size": 4, "enable_padding": True},
                    "enable_chunked_prefill": True,
                }
            )
        )
        port = get_free_port_in_ci()
        command = [
            sys.executable,
            "-m",
            "tensorrt_llm.commands.serve",
            self.MODEL_PATH,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--served_model_name",
            self.MODEL_NAME,
            "--tp_size",
            "4",
            "--ep_size",
            "4",
            "--max_batch_size",
            "4",
            "--max_num_tokens",
            "8192",
            "--max_seq_len",
            "8192",
            "--config",
            str(config_path),
            "--media_io_kwargs",
            json.dumps({"video": {"num_frames": 4, "fps": 2}}),
        ]
        with popen(command) as process:
            _wait_for_server_ready(process, http_port=port, timeout=1200)
            with OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="tensorrt_llm") as client:
                response = client.chat.completions.create(
                    model=self.MODEL_NAME,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "video_url", "video_url": {"url": str(video)}},
                                {
                                    "type": "text",
                                    "text": "Describe what happens in this video briefly.",
                                },
                            ],
                        }
                    ],
                    temperature=0,
                    max_completion_tokens=256,
                    extra_body={"chat_template_kwargs": {"reasoning_effort": "low"}},
                )
                assert len(response.choices) == 1
                message = response.choices[0].message
                # Thinking output may fill the short budget before the final answer.
                assert (message.content or getattr(message, "reasoning_content", "") or "").strip()
                assert response.usage.completion_tokens > 0

    @skip_pre_blackwell
    @pytest.mark.skip_less_mpi_world_size(4)
    @parametrize_with_ids("tp_size,ep_size", [(4, 4)])
    def test_mtp(self, tp_size, ep_size):
        # The checkpoint's single MTP layer is chained for three drafts per
        # step; KDA verification runs on the fused replay kernel.
        with LLM(
            self.MODEL_PATH,
            speculative_config=MTPDecodingConfig(max_draft_len=3),
            max_stats_len=-1,
            enable_iter_perf_stats=True,
            **self._llm_kwargs(tp_size, ep_size),
        ) as llm:
            self._assert_glm5_next_stack(llm)
            assert llm.args.speculative_config.max_draft_len == 3
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            assert_acceptance_length_for_llm("TestGLM53FlashFP8::test_mtp", llm)

    @skip_pre_blackwell
    @pytest.mark.timeout(900)
    @pytest.mark.skip_less_mpi_world_size(4)
    @pytest.mark.threadleak(enabled=False)
    def test_runtime(self) -> None:
        """One model load, several runtime contracts on short prompts."""
        num_ranks = 4
        max_num_tokens = 256
        with LLM(
            self.MODEL_PATH,
            tensor_parallel_size=num_ranks,
            pipeline_parallel_size=1,
            moe_expert_parallel_size=num_ranks,
            kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.5, enable_block_reuse=False),
            max_batch_size=num_ranks,
            max_num_tokens=max_num_tokens,
            max_seq_len=2048,
            enable_chunked_prefill=True,
            cuda_graph_config=None,
            disable_overlap_scheduler=True,
            enable_autotuner=False,
            return_perf_metrics=True,
        ) as llm:
            assert llm.args.enable_chunked_prefill is True
            assert llm.args.max_num_tokens == max_num_tokens
            self._assert_chunked_prefill(llm)
            self._assert_logits_processor(llm)
            self._assert_repeated_request_is_deterministic(llm)

    @skip_pre_blackwell
    @pytest.mark.skip_less_mpi_world_size(4)
    @pytest.mark.timeout(1800)
    def test_chunked_prefill_parity(self) -> None:
        """Compare the first-token choice and probability across chunk boundaries."""
        logits = []
        prompt_ids = None
        for chunked in (False, True):
            kwargs = self._llm_kwargs(4, 4)
            kwargs.update(
                max_batch_size=1,
                max_seq_len=2048,
                max_num_tokens=256 if chunked else 1024,
                enable_chunked_prefill=chunked,
                cuda_graph_config=None,
                disable_overlap_scheduler=True,
                enable_autotuner=False,
                return_perf_metrics=True,
                disable_mm_encoder=True,
            )
            with LLM(self.MODEL_PATH, **kwargs) as llm:
                if prompt_ids is None:
                    text = "The capital of France is Paris. The capital of Germany is Berlin. "
                    prompt_ids = llm.tokenizer.encode(text * 128)[:768]
                    assert len(prompt_ids) == 768
                result = llm.generate(
                    [prompt_ids],
                    sampling_params=SamplingParams(
                        max_tokens=1,
                        temperature=0,
                        end_id=-1,
                        return_generation_logits=True,
                        return_perf_metrics=True,
                    ),
                    use_tqdm=False,
                )[0]
                logits.append(result.outputs[0].generation_logits[0].float().cpu().clone())
                chunks = result.time_breakdown_metrics["ctx_chunk_metrics"]
                if chunked:
                    assert len(chunks) >= 3
                else:
                    assert len(chunks) == 1
            del result, llm
            gc.collect()
            torch.cuda.empty_cache()
        # Follow the Nemotron MoE BCG/eager comparison: FP8 GEMM changes can
        # change routing, so strict whole-logit equality is not a stable gate.
        # Require the chunked choice to remain in the reference top-2 and
        # bound the reference choice's log-probability change (2.30 nats).
        assert torch.isfinite(torch.stack(logits)).all()
        reference_top2 = logits[0].topk(2).indices
        assert logits[1].argmax() in reference_top2
        logprobs = torch.log_softmax(torch.stack(logits), dim=-1)
        difference = (logprobs[0, reference_top2[0]] - logprobs[1, reference_top2[0]]).abs()
        assert difference < 2.30, f"first-token log-probability changed by {difference.item()} nats"

    @staticmethod
    def _assert_chunked_prefill(llm: LLM) -> None:
        prompt_length = 768
        output_length = 16
        prompt_token_ids = [1] + [44] * (prompt_length - 2) + [45]
        outputs = llm.generate(
            [prompt_token_ids],
            sampling_params=SamplingParams(max_tokens=output_length, temperature=0, end_id=-1),
            use_tqdm=False,
        )
        assert isinstance(outputs, list)
        assert len(outputs) == 1
        assert len(outputs[0].outputs[0].token_ids) == output_length
        time_breakdown = outputs[0].time_breakdown_metrics
        assert time_breakdown is not None
        context_chunks = time_breakdown.get("ctx_chunk_metrics")
        assert isinstance(context_chunks, list)
        # 768 prompt tokens at max_num_tokens=256: three context chunks, each
        # continuing the KDA recurrent state and the sparse pools.
        assert len(context_chunks) >= 3

    @staticmethod
    def _assert_logits_processor(llm: LLM) -> None:
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

    @staticmethod
    def _assert_repeated_request_is_deterministic(llm: LLM) -> None:
        """The same greedy natural-language request decoded twice yields identical tokens.

        Same instance, same batch composition: the hybrid cache must hand a
        fresh request a clean KDA / sparse slot, so leftover state from the
        previous (chunked, 768-token) occupant would show up as a token flip
        here. Natural text keeps the logits peaked; on flat, garbage-token
        prompts the decode kernels' run-to-run numeric jitter alone flips
        greedy choices, which is not what this checks.
        """
        output_length = 8
        prompt = llm.tokenizer.encode("The capital of France is Paris. The capital of Germany is")
        sampling_params = SamplingParams(max_tokens=output_length, temperature=0, end_id=-1)
        first = llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)[0]
        second = llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)[0]
        assert len(first.outputs[0].token_ids) == output_length
        assert second.outputs[0].token_ids == first.outputs[0].token_ids
