# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VideoMME accuracy over llmapi encode / prefill-decode (E/PD) disaggregation.

Separated from test_disaggregated_serving.py: the EPD-multimodal path uses an
in-process MultimodalEncoder plus a combined prefill/decode LLM, which is a
different mechanism from the trtllm-serve subprocess disaggregation exercised by
the other tests in that file.
"""

# NOTE:
# The encoder and PD are resident on the same physical GPU in the current test
# harness. Placing them on different physical GPUs silently corrupts the
# embeddings (garbage output, no error raised) in TRT-LLM's current state because
# the consumer (PD worker) rebuilds the encoder's embedding from a CUDA-IPC handle
# that currently never copies the tensor onto the PD's own compute device.
# Real cross-GPU E/PD therefore requires a real cross-device transfer
# (CPU staging or NIXL/RDMA) that is currently not natively supported in TRT-LLM.

import contextlib
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Optional, Protocol
from unittest import mock

import pytest

from tensorrt_llm import LLM, MultimodalEncoder
from tensorrt_llm.llmapi import KvCacheConfig, RequestOutput, SamplingParams
from tensorrt_llm.quantization import QuantAlgo

from ..conftest import llm_models_root, skip_pre_blackwell, skip_pre_hopper
from .accuracy_core import LlmapiAccuracyTestHarness, VideoMME
from .test_disaggregated_serving import DEFAULT_TEST_TIMEOUT, MyThreadPoolExecutor


class VideoMMECompatibleLLM(Protocol):
    """LLM surface consumed by the VideoMME evaluator."""

    args: Any
    model: str
    _hf_model_dir: str
    tokenizer: Any
    input_processor: Any

    def generate_async(
        self,
        inputs: Dict[str, Any],
        sampling_params: Optional[SamplingParams] = None,
        streaming: bool = False,
    ) -> Any: ...


class _MultimodalEncoderPDAdapter:
    """Adapter that runs VideoMME dict inputs through llmapi E/PD."""

    def __init__(
        self, encoder: MultimodalEncoder, pd_llm: LLM, thread_pool: MyThreadPoolExecutor
    ) -> None:
        self._encoder = encoder
        self._pd_llm = pd_llm
        self._thread_pool = thread_pool
        self.args = pd_llm.args
        self.model = pd_llm._hf_model_dir
        self._hf_model_dir = pd_llm._hf_model_dir
        self.tokenizer = pd_llm.tokenizer
        self.input_processor = pd_llm.input_processor

    def _generate(
        self, inputs: Dict[str, Any], sampling_params: Optional[SamplingParams], streaming: bool
    ) -> RequestOutput:
        if not isinstance(inputs, dict):
            raise TypeError(f"Unsupported E/PD request input type: {type(inputs)}")

        encoder_output = self._encoder.generate_async(inputs).result()
        disaggregated_params = encoder_output.disaggregated_params
        if disaggregated_params is None:
            raise RuntimeError("Multimodal encoder did not return disaggregated params.")
        if disaggregated_params.multimodal_embedding_handles is None:
            raise RuntimeError("Multimodal encoder did not return embedding handles.")

        disaggregated_params.request_type = "context_and_generation"
        return self._pd_llm.generate_async(
            inputs,
            sampling_params=sampling_params,
            streaming=streaming,
            disaggregated_params=disaggregated_params,
        ).result()

    def generate_async(
        self,
        inputs: Dict[str, Any],
        sampling_params: Optional[SamplingParams] = None,
        streaming: bool = False,
    ):
        future = self._thread_pool.submit(self._generate, inputs, sampling_params, streaming)
        self._thread_pool.futures.append(future)
        return future


@contextlib.contextmanager
def launch_multimodal_encoder_pd_llm(
    encoder_llm_config: Dict[str, Any],
    pd_llm_config: Dict[str, Any],
    model_name: str,
    max_workers: int = 16,
) -> Iterator[VideoMMECompatibleLLM]:
    """Launch separate encoder and combined prefill/decode llmapi instances."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.dict(os.environ, {"TLLM_MULTIMODAL_DISAGGREGATED": "1"}))
        thread_pool = stack.enter_context(MyThreadPoolExecutor(max_workers=max_workers))
        encoder = MultimodalEncoder(model=model_name, **encoder_llm_config)
        pd_llm = LLM(model=model_name, **pd_llm_config)
        with encoder, pd_llm:
            yield _MultimodalEncoderPDAdapter(encoder, pd_llm, thread_pool)


@dataclass(frozen=True)
class EPDVariant:
    """Immutable per-variant config for a VideoMME E/PD run."""

    model_name: str
    model_path: str
    encoder_config: Mapping[str, Any]
    pd_config: Mapping[str, Any]
    expected_quant_algo: Optional[QuantAlgo]
    max_workers: int

    @classmethod
    def _build(
        cls,
        *,
        model_name: str,
        model_path: str,
        kv_cache_config: KvCacheConfig,
        max_batch_size: int,
        expected_quant_algo: Optional[QuantAlgo],
        max_num_tokens: int = 512,
        attn_backend: Optional[str] = None,
        max_workers: Optional[int] = None,
    ) -> "EPDVariant":
        """Fill shared encoder/PD defaults for one variant.

        Optional overrides are applied before construction so the frozen
        instance never needs post-hoc mutation.
        """
        # Optional attn_backend override, applied to both configs via a spread
        # so the frozen instance never needs post-hoc mutation.
        attn_override = {"attn_backend": attn_backend} if attn_backend is not None else {}
        encoder_config = {
            "trust_remote_code": True,
            "max_batch_size": max_batch_size,
            "cuda_graph_config": None,
            **attn_override,
        }
        pd_config = {
            "backend": "pytorch",
            "disable_overlap_scheduler": True,
            "trust_remote_code": True,
            "kv_cache_config": kv_cache_config,
            "enable_chunked_prefill": True,
            "max_num_tokens": max_num_tokens,
            "max_batch_size": max_batch_size,
            "cuda_graph_config": None,
            **attn_override,
        }

        return cls(
            model_name=model_name,
            model_path=model_path,
            encoder_config=encoder_config,
            pd_config=pd_config,
            expected_quant_algo=expected_quant_algo,
            max_workers=max_workers if max_workers is not None else VideoMME.MAX_BATCH_SIZE,
        )

    @classmethod
    def qwen3vl_2b(cls) -> "EPDVariant":
        return cls._build(
            model_name="Qwen/Qwen3-VL-2B-Instruct",
            model_path=f"{llm_models_root()}/Qwen3/Qwen3-VL-2B-Instruct",
            kv_cache_config=KvCacheConfig(
                free_gpu_memory_fraction=0.8,
                enable_block_reuse=False,
                dtype="auto",
            ),
            max_batch_size=16,
            expected_quant_algo=None,
            max_workers=16,
            attn_backend="VANILLA",
            # Qwen3-VL VideoMME prompts can exceed 1024 tokens after visual
            # expansion; avoid splitting a single context across vanilla
            # SDPA chunks in the E/P handoff path.
            max_num_tokens=2048,
        )

    @classmethod
    def nano_omni_fp8(cls) -> "EPDVariant":
        return cls._build(
            model_name="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8",
            model_path=f"{llm_models_root()}/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8",
            kv_cache_config=KvCacheConfig(
                free_gpu_memory_fraction=0.8,
                mamba_ssm_cache_dtype="float32",
                enable_block_reuse=False,
                dtype="fp8",
            ),
            max_batch_size=64,
            expected_quant_algo=QuantAlgo.FP8,
        )

    @classmethod
    def nano_omni_nvfp4(cls) -> "EPDVariant":
        return cls._build(
            model_name="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4",
            model_path=f"{llm_models_root()}/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4",
            kv_cache_config=KvCacheConfig(
                free_gpu_memory_fraction=0.8,
                mamba_ssm_cache_dtype="float32",
                enable_block_reuse=False,
                dtype="fp8",
            ),
            max_batch_size=128,
            expected_quant_algo=QuantAlgo.MIXED_PRECISION,
        )


class TestVideoMMEEPD(LlmapiAccuracyTestHarness):
    """VideoMME accuracy over llmapi encode / prefill-decode (E/PD) disaggregation."""

    SAMPLING_PARAMS = SamplingParams(
        max_tokens=VideoMME.MAX_OUTPUT_LEN,
        truncate_prompt_tokens=VideoMME.MAX_INPUT_LEN,
        temperature=0.0,
        top_k=1,
    )

    # Identical across all variants today; lifted to a class constant to mirror
    # agg no_thinking_evaluator_kwargs.
    NO_THINKING_EVALUATOR_KWARGS = {
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
    }

    def _launch_epd(self, variant: EPDVariant):
        """Context manager: encoder + combined PD llmapi."""
        return launch_multimodal_encoder_pd_llm(
            variant.encoder_config,
            variant.pd_config,
            variant.model_path,
            max_workers=variant.max_workers,
        )

    def _run_videomme(self, llm, variant: EPDVariant) -> None:
        actual_quant_algo = (
            llm.args.quant_config.quant_algo if llm.args.quant_config is not None else None
        )
        assert actual_quant_algo == variant.expected_quant_algo
        VideoMME(variant.model_name).evaluate(
            llm,
            sampling_params=self.SAMPLING_PARAMS,
            extra_evaluator_kwargs=self.NO_THINKING_EVALUATOR_KWARGS,
        )

    @pytest.mark.timeout(DEFAULT_TEST_TIMEOUT)
    @skip_pre_hopper
    @pytest.mark.skip_less_device_memory(80000)
    @pytest.mark.parametrize(
        "variant",
        [
            pytest.param(
                EPDVariant.qwen3vl_2b(), marks=skip_pre_blackwell, id="qwen3vl_2b_instruct"
            ),
            pytest.param(
                EPDVariant.nano_omni_fp8(), marks=skip_pre_hopper, id="nemotron_nano_v3_omni_fp8"
            ),
            pytest.param(
                EPDVariant.nano_omni_nvfp4(),
                marks=skip_pre_blackwell,
                id="nemotron_nano_v3_omni_nvfp4",
            ),
        ],
    )
    # `torch.compile` uses a thread pool to compile and it's used in audio pre-processing.
    @pytest.mark.threadleak(enabled=False)
    def test_disaggregated_videomme(self, variant: EPDVariant) -> None:
        """Run VideoMME shard through a model-specific llmapi E/PD config."""
        with self._launch_epd(variant) as llm:
            self._run_videomme(llm, variant)
