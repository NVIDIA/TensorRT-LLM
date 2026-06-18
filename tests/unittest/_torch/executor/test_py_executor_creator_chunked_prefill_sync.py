# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Regression test for the chunked-prefill fallback desync.

When ``create_py_executor`` decides to disable chunked prefill via the
FlashInfer-Star gate or the MLA SM gate, the user-facing
``llm_args.enable_chunked_prefill`` flag must also be flipped to ``False`` so
that ``LLM._check_arguments`` (llmapi/llm.py:1017) reads the effective state
and runs the per-request prompt-length validation.
"""

from types import SimpleNamespace

from tensorrt_llm._torch.pyexecutor import py_executor_creator
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm.quantization import QuantAlgo


class _DummyKvCacheCreator:
    def __init__(self, **kwargs):
        self._max_seq_len = kwargs["max_seq_len"]
        self._kv_cache_config = kwargs["kv_cache_config"]
        self._execution_stream = kwargs["execution_stream"]

    def try_prepare_estimation(self):
        return False

    def build_managers(self, resources, estimating_kv_cache):
        del estimating_kv_cache
        resources[ResourceManagerType.KV_CACHE_MANAGER] = SimpleNamespace(
            enable_block_reuse=self._kv_cache_config.enable_block_reuse,
            _stream=self._execution_stream,
        )


class _DummyModelEngine:
    def __init__(self, *, attn_runtime_features, kv_cache_quant_algo):
        self.attn_runtime_features = attn_runtime_features
        self.max_seq_len = 128
        self.max_num_tokens = 128
        self.sparse_attention_config = None
        self.attn_metadata = None
        self.model = SimpleNamespace(
            model_config=SimpleNamespace(
                enable_flash_mla=False,
                is_generation=True,
                pretrained_config=SimpleNamespace(),
                quant_config=SimpleNamespace(kv_cache_quant_algo=kv_cache_quant_algo),
            ),
            vocab_size_padded=32000,
        )


def _make_llm_args(attn_backend="TRTLLM", enable_chunked_prefill=False):
    kv_cache_config = SimpleNamespace(
        enable_block_reuse=True,
        enable_partial_reuse=False,
        tokens_per_block=32,
        max_attention_window=None,
        mamba_state_cache_interval=1,
    )
    scheduler_config = SimpleNamespace(
        context_chunking_policy=None,
        capacity_scheduler_policy=None,
    )
    return SimpleNamespace(
        garbage_collection_gen0_threshold=0,
        lora_config=None,
        kv_connector_config=None,
        scheduler_config=scheduler_config,
        peft_cache_config=SimpleNamespace(),
        kv_cache_config=kv_cache_config,
        decoding_config=None,
        guided_decoding_backend=None,
        custom_tokenizer=None,
        trust_remote_code=False,
        mm_encoder_only=False,
        enable_chunked_prefill=enable_chunked_prefill,
        attn_backend=attn_backend,
        speculative_config=None,
        disable_overlap_scheduler=True,
        sleep_config=None,
        cache_transceiver_config=None,
        dwdp_config=None,
        layer_wise_benchmarks_config=SimpleNamespace(
            calibration_mode=None,
            calibration_file_path=None,
            calibration_layer_indices=None,
        ),
        sampler_type=None,
        disable_flashinfer_sampling=False,
        cuda_graph_config=None,
        parallel_config=SimpleNamespace(to_mapping=lambda: SimpleNamespace()),
        get_runtime_sizes=lambda: (1, 128, 128, 4),
    )


def _run_create_py_executor(monkeypatch, *, sm_version, attn_backend, enable_chunked_prefill):
    """Run ``create_py_executor`` with mocked dependencies and return the
    post-call value of ``llm_args.enable_chunked_prefill``.
    """
    llm_args = _make_llm_args(
        attn_backend=attn_backend,
        enable_chunked_prefill=enable_chunked_prefill,
    )
    fake_mapping = SimpleNamespace(
        rank=0,
        tp_size=1,
        enable_attention_dp=False,
        is_last_pp_rank=lambda: True,
    )

    monkeypatch.setattr(
        py_executor_creator,
        "_load_config_and_create_checkpoint_loader",
        lambda llm_args, checkpoint_dir: (llm_args, None),
    )
    monkeypatch.setattr(py_executor_creator, "_get_mapping", lambda _: fake_mapping)
    monkeypatch.setattr(
        py_executor_creator.Distributed,
        "get",
        staticmethod(lambda mapping: SimpleNamespace()),
    )
    monkeypatch.setattr(
        py_executor_creator,
        "validate_feature_combination",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        py_executor_creator,
        "get_calibrator",
        lambda: SimpleNamespace(init=lambda *a, **k: None, maybe_wrap_model=lambda m: m),
    )
    monkeypatch.setattr(
        py_executor_creator,
        "instantiate_sampler",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        py_executor_creator,
        "get_spec_resource_manager",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        py_executor_creator,
        "get_spec_drafter",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        py_executor_creator,
        "_adjust_torch_mem_fraction",
        lambda: None,
    )
    monkeypatch.setattr(
        py_executor_creator,
        "log_memory_usage",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(py_executor_creator, "is_mla", lambda _: True)
    monkeypatch.setattr(py_executor_creator, "is_hybrid_linear", lambda _: False)
    monkeypatch.setattr(py_executor_creator, "get_sm_version", lambda: sm_version)
    monkeypatch.setattr(py_executor_creator, "KvCacheCreator", _DummyKvCacheCreator)

    monkeypatch.setattr(
        py_executor_creator.torch.cuda,
        "mem_get_info",
        lambda: (2 << 30, 4 << 30),
    )
    monkeypatch.setattr(py_executor_creator.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        py_executor_creator.torch.cuda,
        "reset_peak_memory_stats",
        lambda: None,
    )
    monkeypatch.setattr(
        py_executor_creator.torch.cuda,
        "memory_stats",
        lambda: {"allocated_bytes.all.current": 0},
    )
    monkeypatch.setattr(
        py_executor_creator.torch.cuda,
        "Stream",
        lambda: SimpleNamespace(cuda_stream=123),
    )

    def _create_model_engine(**kwargs):
        return _DummyModelEngine(
            attn_runtime_features=kwargs["attn_runtime_features"],
            kv_cache_quant_algo=QuantAlgo.NO_QUANT,
        )

    monkeypatch.setattr(py_executor_creator, "PyTorchModelEngine", _create_model_engine)

    def _create_py_executor_instance(**kwargs):
        return SimpleNamespace(
            resource_manager=SimpleNamespace(
                get_resource_manager=lambda rt: SimpleNamespace(
                    enable_block_reuse=llm_args.kv_cache_config.enable_block_reuse,
                    _stream=kwargs["execution_stream"],
                ),
            ),
            model_engine=kwargs["model_engine"],
            peft_cache_config=kwargs["peft_cache_config"],
            execution_stream=kwargs["execution_stream"],
            started=False,
            start_worker=lambda: None,
        )

    monkeypatch.setattr(
        py_executor_creator,
        "create_py_executor_instance",
        _create_py_executor_instance,
    )

    py_executor_creator.create_py_executor(llm_args=llm_args, checkpoint_dir=None)

    return llm_args.enable_chunked_prefill


def test_mla_unsupported_sm_fallback_syncs_llm_args_chunked_prefill(monkeypatch):
    """When the MLA SM gate fires, llm_args.enable_chunked_prefill must
    be flipped to False so the user-facing LLM._check_arguments runs the
    per-request prompt-length validation."""
    llm_args_chunked_prefill = _run_create_py_executor(
        monkeypatch,
        sm_version=89,
        attn_backend="TRTLLM",
        enable_chunked_prefill=True,
    )
    assert llm_args_chunked_prefill is False


def test_flashinfer_star_fallback_syncs_llm_args_chunked_prefill(monkeypatch):
    """When the FlashInfer-Star gate fires, llm_args.enable_chunked_prefill
    must be flipped to False so the user-facing LLM._check_arguments runs
    the per-request prompt-length validation."""
    llm_args_chunked_prefill = _run_create_py_executor(
        monkeypatch,
        sm_version=90,
        attn_backend="FLASHINFER_STAR_ATTENTION",
        enable_chunked_prefill=True,
    )
    assert llm_args_chunked_prefill is False


def test_supported_path_preserves_llm_args_chunked_prefill(monkeypatch):
    """When neither gate fires, llm_args.enable_chunked_prefill must stay
    as the user requested (chunked prefill on for a supported SM with
    TRTLLM backend)."""
    llm_args_chunked_prefill = _run_create_py_executor(
        monkeypatch,
        sm_version=90,
        attn_backend="TRTLLM",
        enable_chunked_prefill=True,
    )
    assert llm_args_chunked_prefill is True
