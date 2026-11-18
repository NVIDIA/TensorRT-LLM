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
"""Tests for the configurations and requests that enable_wide_ep_fault_tolerance rejects."""

import re
import subprocess  # nosec B404
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from tensorrt_llm import _wide_ep_fault_tolerance
from tensorrt_llm._wide_ep_fault_tolerance import (
    FAULT_TOLERANCE_RANK_ENV,
    TURN_FAULT_TOLERANCE_OFF,
    check_node_supervisor_launch,
)
from tensorrt_llm.executor.utils import RequestError
from tensorrt_llm.llmapi import LLM, SamplingParams
from tensorrt_llm.llmapi.llm_args import (
    AttentionDpConfig,
    AutoDecodingConfig,
    CacheTransceiverConfig,
    ColdPageQuantizationCompressionConfig,
    DFlashDecodingConfig,
    DraftTargetDecodingConfig,
    DSparkDecodingConfig,
    Eagle3DecodingConfig,
    KvCacheConfig,
    KvCacheConnectorConfig,
    KVEventsConfig,
    MTPDecodingConfig,
    NGramDecodingConfig,
    PARDDecodingConfig,
    SADecodingConfig,
    SAEnhancerConfig,
    SleepConfig,
    TorchCompileConfig,
    TorchLlmArgs,
    UserProvidedDecodingConfig,
)
from tensorrt_llm.logger import logger

pytestmark = pytest.mark.cpu_only


def _args(**fields) -> TorchLlmArgs:
    """Return a supported configuration updated with ``fields``."""
    return TorchLlmArgs(
        **{
            "model": "dummy",
            "enable_wide_ep_fault_tolerance": True,
            "enable_attention_dp": True,
            "max_seq_len": 4096,
            "moe_config": {"load_balancer": {"num_slots": 8}},
            **fields,
        }
    )


@pytest.mark.parametrize(
    "fields",
    [
        {},
        # The balance counters follow per-step gathers over the serving ranks only.
        {
            "attention_dp_config": AttentionDpConfig(
                enable_balance=True, batching_wait_iters=0, timeout_iters=60
            )
        },
        # DeepSeek's single MTP layer runs MTP-Eagle one-model, the default MTP path.
        {"speculative_config": MTPDecodingConfig(num_nextn_predict_layers=1)},
        {"speculative_config": MTPDecodingConfig(max_draft_len=1, use_mtp_vanilla=True)},
        # A window without a threshold turns no acceptance-rate gate on.
        {"speculative_config": MTPDecodingConfig(max_draft_len=1, acceptance_rate_window_size=16)},
        {"max_seq_len": 8192},
        # An unset max_num_tokens is the default budget of 8192 tokens.
        {"max_num_tokens": None},
        {"max_seq_len": None, "enable_chunked_prefill": True},
        # The attached frontends read the membership from the rendezvous store too.
        {"num_serve_frontends": 8},
        # Whoever starts the executor checks the orchestrator, not the arguments.
        {"orchestrator_type": "rpc"},
        {"orchestrator_type": "ray"},
        # Without fault tolerance, nothing is checked.
        {
            "enable_wide_ep_fault_tolerance": False,
            "disable_overlap_scheduler": True,
            "num_serve_frontends": 2,
            "sleep_config": SleepConfig(),
        },
    ],
)
def test_supported_configuration_is_accepted(fields):
    assert _args(**fields).enable_wide_ep_fault_tolerance == fields.get(
        "enable_wide_ep_fault_tolerance", True
    )


@pytest.mark.parametrize(
    "fields, fix",
    [
        ({"enable_attention_dp": False}, "enable_attention_dp=True"),
        ({"pipeline_parallel_size": 2}, "pipeline_parallel_size=1, not 2"),
        ({"max_beam_width": 2}, "max_beam_width=1, not 2"),
        ({"disable_overlap_scheduler": True}, "disable_overlap_scheduler=False"),
        ({"moe_config": {}}, "moe_config.load_balancer.num_slots, for example the number"),
        # Per-layer dynamic EPLB is EPLB too, but without spare slots.
        (
            {"moe_config": {"load_balancer": {"mode": "per_layer_dynamic", "num_slots": 16}}},
            "moe_config.load_balancer.mode='standard', not 'per_layer_dynamic'",
        ),
        (
            {
                "moe_config": {
                    "load_balancer": {
                        "num_slots": 8,
                        "warmup_placement": {"start_step": 0, "interval_steps": 1},
                    }
                }
            },
            "no warmup placement (moe_config.load_balancer.warmup_placement=None)",
        ),
        *(
            (
                {"attention_dp_config": AttentionDpConfig(**{flag: True})},
                "no enable_kv_balanced_admission, enable_kv_cache_aware_routing or "
                f"kv_cache_routing_conversation_affinity in attention_dp_config, which sets {flag})",
            )
            for flag in (
                "enable_kv_cache_aware_routing",
                "enable_kv_balanced_admission",
                "kv_cache_routing_conversation_affinity",
            )
        ),
        (
            {"kv_cache_config": KvCacheConfig(event_buffer_max_size=1024)},
            "kv_cache_config.event_buffer_max_size=0, not 1024",
        ),
        (
            {
                "kv_cache_config": KvCacheConfig(
                    kv_events_config=KVEventsConfig(enable_kv_cache_events=True)
                )
            },
            "no streaming KV cache events (kv_cache_config.kv_events_config.enable_kv_cache_events=False, not True)",
        ),
        (
            {"kv_cache_config": KvCacheConfig(host_cache_bind_to_gpu_numa_node=True)},
            "no NUMA binding of the host cache tier "
            "(kv_cache_config.host_cache_bind_to_gpu_numa_node=False, not True)",
        ),
        (
            {"cache_transceiver_config": CacheTransceiverConfig(backend="DEFAULT")},
            "no disaggregated serving (no cache_transceiver_config.backend, not 'DEFAULT')",
        ),
        (
            {
                "kv_connector_config": KvCacheConnectorConfig(
                    connector_module="connector",
                    connector_scheduler_class="Scheduler",
                    connector_worker_class="Worker",
                )
            },
            "kv_connector_config=None), because a connector saves the KV cache",
        ),
        (
            {"kv_cache_compression_config": ColdPageQuantizationCompressionConfig()},
            "kv_cache_compression_config=None",
        ),
        *(
            (
                {"speculative_config": config},
                "speculative_config: none, or decoding_type 'MTP' without sa_config, "
                f"not {current})",
            )
            for config, current in (
                (NGramDecodingConfig(max_draft_len=1), "'NGram'"),
                (UserProvidedDecodingConfig(max_draft_len=1, drafter=object()), "'User_Provided'"),
                (DSparkDecodingConfig(max_draft_len=1, speculative_model="draft"), "'DSpark'"),
                (Eagle3DecodingConfig(max_draft_len=3, speculative_model="draft"), "'Eagle3'"),
                (SADecodingConfig(max_draft_len=4), "'SA'"),
                # AUTO resolves to SA when the executor starts.
                (AutoDecodingConfig(max_draft_len=4), "'AUTO'"),
                (PARDDecodingConfig(max_draft_len=2, speculative_model="draft"), "'PARD'"),
                (DFlashDecodingConfig(max_draft_len=2, use_rejection_sampling=False), "'DFlash'"),
                (
                    DraftTargetDecodingConfig(max_draft_len=4, speculative_model="draft"),
                    "'Draft_Target'",
                ),
                (
                    MTPDecodingConfig(max_draft_len=1, sa_config=SAEnhancerConfig()),
                    "'MTP' with sa_config",
                ),
            )
        ),
        *(
            (
                {"speculative_config": MTPDecodingConfig(max_draft_len=1, **options)},
                "a fixed draft length (no speculative_config.draft_len_schedule or max_concurrency, "
                "and no acceptance-rate gate, which is acceptance_rate_window_size with "
                f"acceptance_rate_threshold; not {current})",
            )
            for options, current in (
                ({"draft_len_schedule": {4: 1}}, "draft_len_schedule={4: 1}"),
                ({"max_concurrency": 8}, "max_concurrency=8"),
                (
                    {"acceptance_rate_window_size": 16, "acceptance_rate_threshold": 0.3},
                    "acceptance_rate_window_size=16 with acceptance_rate_threshold=0.3",
                ),
            )
        ),
        # A recomputed request's context can reach max_seq_len, the model's length by default.
        (
            {"max_seq_len": None},
            "not an unset max_seq_len (the model's maximum length) with max_num_tokens=8192",
        ),
        ({"max_seq_len": 8193}, "not max_seq_len=8193 with max_num_tokens=8192"),
        (
            {"max_seq_len": 8193},
            "without chunked prefill the scheduler never schedules a context longer than "
            "max_num_tokens",
        ),
        ({"enable_lm_head_tp_in_adp": True}, "enable_lm_head_tp_in_adp=False"),
        *(
            (fields, "no torch.compile (torch_compile_config=None")
            for fields in (
                {"torch_compile_config": TorchCompileConfig(enable_userbuffers=False)},
                # A piecewise prefill CUDA graph runs torch.compile with the default configuration.
                {"prefill_cuda_graph_backend": "piecewise"},
                {
                    "prefill_cuda_graph_backend": "piecewise",
                    "torch_compile_config": TorchCompileConfig(enable_userbuffers=False),
                },
            )
        ),
        ({"sleep_config": SleepConfig()}, "sleep_config=None), because sleep releases"),
        ({"modeling_v2": "require"}, "modeling_v2='auto' or 'off', not 'require'"),
        (
            {"enable_return_routed_experts": True},
            "no routed-expert capture (enable_return_routed_experts=False), because a step",
        ),
    ],
)
def test_unsupported_configuration_names_field(fields, fix):
    with pytest.raises(ValueError, match=re.escape(fix)) as error:
        _args(**fields)
    assert TURN_FAULT_TOLERANCE_OFF in str(error.value)


@pytest.mark.parametrize("fields", [{}, {"modeling_v2": "auto"}], ids=["default", "explicit"])
def test_modeling_v2_auto_uses_builtin_model(monkeypatch, fields):
    """Check that fault tolerance turns modeling_v2 'auto' off with one warning."""
    warning = Mock()
    monkeypatch.setattr(logger, "warning", warning)
    assert _args(**fields).modeling_v2 == "off"
    messages = [call.args[0] for call in warning.call_args_list]
    assert sum("using the built-in implementation" in message for message in messages) == 1
    assert _args(enable_wide_ep_fault_tolerance=False, **fields).modeling_v2 == "auto"


def test_check_leaves_args_unchanged():
    """Check that checking the arguments for fault tolerance does not change them."""
    args = _args(enable_wide_ep_fault_tolerance=False)
    _wide_ep_fault_tolerance.check_llm_args(args)
    assert args.modeling_v2 == "auto"


@pytest.mark.parametrize("mode, rejected", [("standard", False), ("per_layer_dynamic", True)])
def test_load_balancer_from_yaml_checked(tmp_path, mode, rejected):
    """Check that a load balancer given as a YAML path is checked once loaded."""
    path = tmp_path / "load_balancer.yaml"
    path.write_text(yaml.safe_dump({"mode": mode, "num_slots": 16}))
    fields = {"moe_config": {"load_balancer": str(path)}}
    if rejected:
        with pytest.raises(ValueError, match=re.escape("load_balancer.mode='standard'")):
            _args(**fields)
    else:
        assert _args(**fields).moe_config.load_balancer.num_slots == 16


class _ArgsOnlyLLM:
    """Drive LLM's request checks without a checkpoint or a GPU."""

    _check_arguments = LLM._check_arguments
    _check_one_model_speculative_sampling = LLM._check_one_model_speculative_sampling

    def __init__(self, enabled):
        self.args = SimpleNamespace(
            backend="pytorch",
            enable_chunked_prefill=True,
            max_num_tokens=None,
            speculative_config=None,
            enable_wide_ep_fault_tolerance=enabled,
        )


@pytest.mark.parametrize(
    "params, rejected",
    [
        (SamplingParams(), False),
        (SamplingParams(n=2, temperature=0.8), True),
        (SamplingParams(best_of=2, temperature=0.8), True),
        (SamplingParams(use_beam_search=True), True),
    ],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_multi_sequence_requests_rejected(params, rejected, enabled):
    check = _ArgsOnlyLLM(enabled)._check_arguments
    if rejected and enabled:
        with pytest.raises(RequestError, match="does not support requests with n > 1, best_of > 1"):
            check(8, params, is_gen_only=False)
    else:
        check(8, params, is_gen_only=False)


@pytest.mark.parametrize("replacement", [False, True], ids=["original", "replacement"])
def test_rank_identity_from_node_supervisor(monkeypatch, replacement):
    """Check that the node supervisor gives the rank, the world size and the replacement flag."""
    monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, f"2,4,4,{int(replacement)}")
    identity = _wide_ep_fault_tolerance.wide_ep_fault_tolerance_rank()
    assert identity == (2, 4, 4, replacement)
    assert identity.env_value() == f"2,4,4,{int(replacement)}"
    monkeypatch.delenv(FAULT_TOLERANCE_RANK_ENV)
    assert _wide_ep_fault_tolerance.wide_ep_fault_tolerance_rank() is None


def test_supervised_rank_requires_flag(monkeypatch):
    monkeypatch.setenv(FAULT_TOLERANCE_RANK_ENV, "0,4,4,0")
    with pytest.raises(ValueError, match="enable_wide_ep_fault_tolerance is false"):
        check_node_supervisor_launch(False)


def test_llmapi_import_skips_fault_tolerance():
    """Check that importing the LLM API loads neither the membership nor the node supervisor."""
    prefixes = (
        "tensorrt_llm._torch.distributed.membership",
        "tensorrt_llm.llmapi._llmapi_node_supervisor",
        "tensorrt_llm.llmapi._mpirun_job",
        "tensorrt_llm.llmapi._node_supervised_mpi_session",
    )
    code = (
        "import sys, tensorrt_llm.llmapi; "
        f"print(sorted(m for m in sys.modules if m.startswith({prefixes!r})))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip().splitlines()[-1] == "[]", result.stdout
