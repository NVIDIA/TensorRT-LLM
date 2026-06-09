# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional, Union

from tensorrt_llm.llmapi.llm_args import (
    CudaGraphConfig,
    Field,
    KvCacheConfig,
    TorchCompileConfig,
    TorchLlmArgs,
)
from tensorrt_llm.llmapi.utils import StrictBaseModel
from tensorrt_llm.usage import usage_lib
from tensorrt_llm.usage.llmapi_config import collect_llm_api_config_payloads


class _NestedConfig(StrictBaseModel):
    marked: int = Field(default=7, telemetry={"kind": "value"})
    unmarked: int = Field(default=11)


class _ExampleConfig(StrictBaseModel):
    safe_marked: int = Field(default=3, telemetry={"kind": "value"})
    safe_unmarked: int = Field(default=5)
    private_path: str = Field(default="/customer/private/model", telemetry={"kind": "value"})
    mode: Literal["auto", "slow"] = Field(default="auto", telemetry={"kind": "categorical"})
    nested: _NestedConfig = Field(default_factory=_NestedConfig)
    unsafe_union: Optional[Union[str, Path]] = Field(
        default="/customer/tokenizer", telemetry={"kind": "categorical"}
    )


def _loads_payloads(args) -> tuple[dict, dict]:
    config_json, meta_json = collect_llm_api_config_payloads(args)
    return json.loads(config_json), json.loads(meta_json)


def test_collect_llm_api_config_uses_type_driven_autoenroll_and_safety_vetoes():
    # Renamed from ..._uses_strict_opt_in_...: under auto-enroll, unmarked
    # type-safe ints (safe_unmarked, nested.unmarked) are now captured; bare
    # str / Union[str,Path] without an approved allowlist remain uncapturable.
    config, meta = _loads_payloads(_ExampleConfig())

    assert config == {
        "mode": "auto",
        "nested.marked": 7,
        "nested.unmarked": 11,
        "safe_marked": 3,
        "safe_unmarked": 5,
    }
    assert "private_path" not in config  # bare str, no allowlist -> not capturable
    assert "unsafe_union" not in config  # Union[str,Path], no allowlist -> not capturable
    assert meta["source"] == "effective_validated_llm_args"
    assert meta["args_class"] == "_ExampleConfig"
    assert meta["capturable_field_count"] == 5
    assert meta["captured_field_count"] == 5
    assert meta["excluded_field_count"] == 0
    assert meta["unsafe_excluded"] is False
    assert meta["payload_truncated"] is False


def test_collect_llm_api_config_allows_approved_string_converters_only():
    # The union_backend / union_path fixtures below are Union[str, Path] solely
    # to exercise the value-fail-closed allowlist seam: union_path defaults to a
    # Path, which is dropped because it is not an allowlisted scalar, while
    # union_backend's allowlisted str is captured. No production telemetry field
    # is Union[str, Path]; the only real Union allowlist fields are
    # Union[str, Enum] (sampler_type, load_format). See CR-E (declined).
    class _StringConfig(StrictBaseModel):
        backend: Optional[str] = Field(
            default="pytorch",
            telemetry={
                "kind": "categorical",
                "converter": "allowlist",
                "allowed_values": ["pytorch", "tensorrt"],
            },
        )
        unsafe_backend: Optional[str] = Field(
            default="file:///customer/private",
            telemetry={
                "kind": "categorical",
                "converter": "allowlist",
                "allowed_values": ["pytorch", "tensorrt"],
            },
        )
        unconverted: Optional[str] = Field(
            default="arbitrary-user-string", telemetry={"kind": "categorical"}
        )
        union_backend: Union[str, Path] = Field(
            default="tensorrt",
            telemetry={
                "kind": "categorical",
                "converter": "allowlist",
                "allowed_values": ["pytorch", "tensorrt"],
            },
        )
        union_path: Union[str, Path] = Field(
            default=Path("/customer/private"),
            telemetry={
                "kind": "categorical",
                "converter": "allowlist",
                "allowed_values": ["pytorch", "tensorrt"],
            },
        )

    config, meta = _loads_payloads(_StringConfig())

    assert config == {"backend": "pytorch", "union_backend": "tensorrt"}
    assert meta["captured_field_count"] == 2
    # unconverted (kind=categorical, no converter) is not capturable -> not in
    # manifest; unsafe_backend + union_path resolve but fail the allowlist sanitizer.
    assert meta["capturable_field_count"] == 4
    assert meta["excluded_field_count"] == 2
    assert meta["unsafe_excluded"] is True


def test_sanitize_allowlist_is_value_fail_closed_for_non_scalars():
    """The allowlist only emits scalars, even if a non-scalar is allowlisted.

    Documents the CR-E decision (declined): the sanitizer is value-fail-closed,
    so a Path or arbitrary object cannot leak through the allowlist converter
    even if it were placed in allowed_values. _sanitize_allowlist returns a
    candidate only when it is BOTH in allowed_values AND a scalar
    (bool/int/float/str) or None. Excluding Union-with-Any/Path from allowlist
    eligibility at the type level is therefore unnecessary for safety, and a
    coarse rule would also break legitimate Union[str, Enum] allowlist fields
    such as sampler_type and load_format (verified captured elsewhere).
    """
    from tensorrt_llm.usage import llmapi_config

    secret = Path("/customer/secret")
    metadata = {"converter": "allowlist", "allowed_values": [secret]}
    # A Path placed in the allowlist is still rejected: not a scalar.
    assert llmapi_config._sanitize_allowlist(secret, metadata) == (False, None)
    # An allowlisted scalar is captured.
    str_metadata = {"converter": "allowlist", "allowed_values": ["pytorch"]}
    assert llmapi_config._sanitize_allowlist("pytorch", str_metadata) == (True, "pytorch")


def test_collect_llm_api_config_walks_only_declared_pydantic_fields():
    class _DeclaredFieldsOnlyConfig(StrictBaseModel):
        safe_value: int = Field(default=3, telemetry={"kind": "value"})

        @property
        def leaked_value(self):
            raise AssertionError("collector must not inspect arbitrary attributes")

    config, meta = _loads_payloads(_DeclaredFieldsOnlyConfig())

    assert config == {"safe_value": 3}
    assert meta["capturable_field_count"] == 1
    assert meta["excluded_field_count"] == 0


def test_collect_llm_api_config_rejects_unsafe_annotations_even_for_safe_values():
    class _UnsafeAnnotationConfig(StrictBaseModel):
        safe_value: int = Field(default=3, telemetry={"kind": "value"})
        raw_any: Any = Field(default=11, telemetry={"kind": "value"})
        object_like: object = Field(default=True, telemetry={"kind": "value"})
        raw_dict: dict[str, Any] = Field(default_factory=dict, telemetry={"kind": "value"})
        converted_any: Any = Field(
            default="known",
            telemetry={
                "kind": "categorical",
                "converter": "allowlist",
                "allowed_values": ["known"],
            },
        )

    config, meta = _loads_payloads(_UnsafeAnnotationConfig())

    assert config == {"converted_any": "known", "safe_value": 3}
    # raw_any/object_like/raw_dict have unsafe annotations -> excluded at the
    # MANIFEST level (never selected), so the sanitizer never sees them.
    assert "raw_any" not in config
    assert "object_like" not in config
    assert "raw_dict" not in config
    assert meta["capturable_field_count"] == 2
    assert meta["captured_field_count"] == 2
    assert meta["excluded_field_count"] == 0
    assert meta["unsafe_excluded"] is False


def test_collect_llm_api_config_is_deterministic_for_effective_torch_args():
    args = TorchLlmArgs(
        model="/customer/private/Llama",
        tokenizer="/customer/private/tokenizer",
        skip_tokenizer_init=True,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        context_parallel_size=1,
        dtype="float16",
        load_format="dummy",
        enable_chunked_prefill=True,
        max_num_tokens=4096,
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False,
            free_gpu_memory_fraction=0.5,
            dtype="bfloat16",
            tokens_per_block=64,
        ),
        cuda_graph_config=CudaGraphConfig(batch_sizes=[4, 1], max_batch_size=4),
        torch_compile_config=TorchCompileConfig(
            enable_inductor=True,
            enable_piecewise_cuda_graph=True,
            capture_num_tokens=[128, 64],
        ),
    )

    first_config_json, first_meta_json = collect_llm_api_config_payloads(args)
    second_config_json, second_meta_json = collect_llm_api_config_payloads(args)
    config = json.loads(first_config_json)
    meta = json.loads(first_meta_json)

    assert first_config_json == second_config_json
    assert first_meta_json == second_meta_json
    assert config["tensor_parallel_size"] == 2
    assert config["dtype"] == "float16"
    assert config["load_format"] == "dummy"
    assert config["enable_chunked_prefill"] is True
    assert config["max_num_tokens"] == 4096
    assert config["kv_cache_config.enable_block_reuse"] is False
    assert config["kv_cache_config.free_gpu_memory_fraction"] == 0.5
    assert config["kv_cache_config.dtype"] == "bfloat16"
    assert config["kv_cache_config.tokens_per_block"] == 64
    assert config["cuda_graph_config.batch_sizes"] == [1, 4]
    assert config["cuda_graph_config.max_batch_size"] == 4
    assert config["torch_compile_config.enable_inductor"] is True
    assert config["torch_compile_config.capture_num_tokens"] == [128, 64]
    assert "model" not in config
    assert "tokenizer" not in config
    assert meta["capture_succeeded"] is True
    assert meta["args_class"] == "TorchLlmArgs"
    assert meta["schema_digest"]
    assert meta["capture_manifest_digest"]


def test_collect_llm_api_config_captures_nvfp4_kv_cache_dtype():
    args = TorchLlmArgs(
        model="/customer/private/Llama",
        skip_tokenizer_init=True,
        kv_cache_config=KvCacheConfig(dtype="nvfp4"),
    )

    config, meta = _loads_payloads(args)

    assert config["kv_cache_config.dtype"] == "nvfp4"
    assert meta["capture_succeeded"] is True


def test_collect_llm_api_config_captures_int_enum_name():
    class _LoadMode(Enum):
        AUTO = 0

    class _PrecisionMode(Enum):
        FP8 = "fp8"

    class _EnumConfig(StrictBaseModel):
        mode: _LoadMode = Field(default=_LoadMode.AUTO, telemetry={"kind": "categorical"})
        precision: _PrecisionMode = Field(
            default=_PrecisionMode.FP8, telemetry={"kind": "categorical"}
        )

    config, meta = _loads_payloads(_EnumConfig())

    assert config == {"mode": "AUTO", "precision": "fp8"}
    assert meta["unsafe_excluded"] is False


def test_collect_llm_api_config_keeps_bool_values_boolean():
    class _BoolConfig(StrictBaseModel):
        enabled: bool = Field(default=True, telemetry={"kind": "value"})

    config, _ = _loads_payloads(_BoolConfig())

    assert config["enabled"] is True
    assert type(config["enabled"]) is bool


def test_collect_llm_api_config_rejects_non_finite_floats():
    """Non-finite floats (nan/inf) are excluded; finite floats are captured.

    json.dumps emits the bare NaN/Infinity tokens for non-finite floats, which
    are invalid JSON and break downstream parsing and digest stability. Guard
    the float branch with math.isfinite so a marked float set to inf/nan is
    dropped (unsafe_excluded) while a finite float is still captured.
    """

    class _FloatConfig(StrictBaseModel):
        finite: float = Field(default=0.5, telemetry={"kind": "value"})
        infinite: float = Field(default=float("inf"), telemetry={"kind": "value"})
        not_a_number: float = Field(default=float("nan"), telemetry={"kind": "value"})

    config, meta = _loads_payloads(_FloatConfig())

    assert config == {"finite": 0.5}
    assert "infinite" not in config
    assert "not_a_number" not in config
    assert meta["excluded_field_count"] == 2
    assert meta["unsafe_excluded"] is True


def test_collect_llm_api_config_rejects_non_finite_floats_in_sequence():
    """A single non-finite float poisons the whole marked sequence.

    The sequence sanitizer fails closed on one bad item, so a list containing
    inf/nan is dropped entirely rather than emitting invalid JSON tokens.
    """

    class _FloatSeqConfig(StrictBaseModel):
        finite_buckets: list[float] = Field(
            default_factory=lambda: [0.1, 0.5, 1.0], telemetry={"kind": "value"}
        )
        poisoned_buckets: list[float] = Field(
            default_factory=lambda: [0.1, float("inf"), 1.0], telemetry={"kind": "value"}
        )

    config, meta = _loads_payloads(_FloatSeqConfig())

    assert config == {"finite_buckets": [0.1, 0.5, 1.0]}
    assert "poisoned_buckets" not in config
    assert meta["unsafe_excluded"] is True


def test_collect_llm_api_config_caps_long_sequences_and_flags_truncation():
    """A marked sequence longer than MAX_SEQ_ITEMS is clipped and flagged.

    llmApiConfigJson is unbounded on the wire and the reporter is fail-silent,
    so a pathological user-sized list could silently drop the whole payload.
    Cap captured sequences to MAX_SEQ_ITEMS and record a single honest
    sequence_truncated boolean in the metadata.
    """
    from tensorrt_llm.usage import llmapi_config

    cap = llmapi_config.MAX_SEQ_ITEMS

    class _LongSeqConfig(StrictBaseModel):
        values: list[int] = Field(
            default_factory=lambda: list(range(cap + 50)), telemetry={"kind": "value"}
        )

    config, meta = _loads_payloads(_LongSeqConfig())

    assert len(config["values"]) == cap
    assert config["values"] == list(range(cap))
    assert meta["sequence_truncated"] is True


def test_collect_llm_api_config_caps_nested_inner_sequences():
    """Each inner list of a nested List[List[int]] is capped independently."""
    from tensorrt_llm.usage import llmapi_config

    cap = llmapi_config.MAX_SEQ_ITEMS

    class _NestedSeqConfig(StrictBaseModel):
        rows: list[list[int]] = Field(
            default_factory=lambda: [list(range(cap + 10)), list(range(cap + 20))],
            telemetry={"kind": "value"},
        )

    config, meta = _loads_payloads(_NestedSeqConfig())

    assert len(config["rows"]) == 2
    assert all(len(inner) == cap for inner in config["rows"])
    assert config["rows"][0] == list(range(cap))
    assert meta["sequence_truncated"] is True


def test_collect_llm_api_config_caps_outer_nested_sequence():
    """The outer list of a nested List[List[int]] is also capped."""
    from tensorrt_llm.usage import llmapi_config

    cap = llmapi_config.MAX_SEQ_ITEMS

    class _WideNestedSeqConfig(StrictBaseModel):
        rows: list[list[int]] = Field(
            default_factory=lambda: [[0, 1] for _ in range(cap + 30)],
            telemetry={"kind": "value"},
        )

    config, meta = _loads_payloads(_WideNestedSeqConfig())

    assert len(config["rows"]) == cap
    assert meta["sequence_truncated"] is True


def test_collect_llm_api_config_small_sequence_not_truncated():
    """A sequence within the cap is captured whole and the flag stays false."""

    class _SmallSeqConfig(StrictBaseModel):
        values: list[int] = Field(default_factory=lambda: [1, 2, 3], telemetry={"kind": "value"})

    config, meta = _loads_payloads(_SmallSeqConfig())

    assert config["values"] == [1, 2, 3]
    assert meta["sequence_truncated"] is False


def test_collect_llm_api_config_failure_meta_has_truncation_key():
    """Failure metadata carries sequence_truncated for shape parity."""
    from tensorrt_llm.usage import llmapi_config

    meta = llmapi_config._failure_meta(args_class="Foo")
    assert meta["sequence_truncated"] is False


def test_failure_meta_uses_new_contract_keys_and_versions():
    from tensorrt_llm.usage import llmapi_config as rc

    meta = rc._failure_meta(args_class="X")
    assert meta["capture_version"] == "2"
    assert meta["api_contract_version"] == "0.2.0"
    assert meta["field_policy_version"] == "2"
    assert meta["excluded_field_count"] == 0  # renamed from the old marked-count key
    assert meta["payload_truncated"] is False
    # The pre-migration keys must be gone from the new contract; assert by literal
    # so a regression that reintroduces them fails loudly.
    assert "excluded_marked_field_count" not in meta
    assert "included_field_count" not in meta


def test_collect_llm_api_config_rejects_heterogeneous_tuples():
    class _TupleConfig(StrictBaseModel):
        pair: tuple[int, Literal["safe"]] = Field(default=(1, "safe"), telemetry={"kind": "value"})

    config, meta = _loads_payloads(_TupleConfig())

    assert config == {}
    assert meta["excluded_field_count"] == 1
    assert meta["unsafe_excluded"] is True


def test_collect_llm_api_config_derives_manifest_kind_from_annotation():
    """Manifest 'kind' is derived per D1, not taken from the registered value.

    Categorical iff (Optional-unwrapped) annotation is Literal/Enum OR an
    allowlist is present; otherwise 'value'. The registered kind is ignored.
    """

    class _Mode(Enum):
        AUTO = "auto"

    class _KindConfig(StrictBaseModel):
        # Literal but deliberately registered with the wrong kind -> categorical.
        literal_field: Literal["a", "b"] = Field(default="a", telemetry={"kind": "value"})
        # Enum -> categorical.
        enum_field: _Mode = Field(default=_Mode.AUTO, telemetry=True)
        # Bare str + allowlist -> categorical.
        allowlist_field: str = Field(
            default="x",
            telemetry={
                "kind": "value",
                "converter": "allowlist",
                "allowed_values": ["x", "y"],
            },
        )
        # Plain int -> value.
        int_field: int = Field(default=3, telemetry={"kind": "categorical"})

    from tensorrt_llm.usage.llmapi_config import build_capture_manifest

    by_path = {e.path: e for e in build_capture_manifest(_KindConfig)}
    assert by_path["literal_field"].kind == "categorical"
    assert by_path["enum_field"].kind == "categorical"
    assert by_path["allowlist_field"].kind == "categorical"
    assert by_path["int_field"].kind == "value"


def test_collect_llm_api_config_captures_star_attention_backend():
    """attn_backend allowlist recognizes the real FLASHINFER_STAR_ATTENTION value.

    The recognized set mirrors get_attention_backend dispatch; the previously
    listed FLASH_ATTENTION is not a real backend and is removed.
    """
    args = TorchLlmArgs(
        model="/customer/private/Llama",
        skip_tokenizer_init=True,
        attn_backend="FLASHINFER_STAR_ATTENTION",
    )

    config, meta = _loads_payloads(args)

    assert config["attn_backend"] == "FLASHINFER_STAR_ATTENTION"
    assert meta["capture_succeeded"] is True


def test_collect_llm_api_config_swallows_expected_capture_errors(monkeypatch):
    """The inner net stays fail-silent for the expected sanitizer error family."""
    from tensorrt_llm.usage import llmapi_config

    def raise_value_error(*_args, **_kwargs):
        raise ValueError("synthetic sanitizer error")

    monkeypatch.setattr(llmapi_config, "build_capture_manifest", raise_value_error)

    config, meta = _loads_payloads(_ExampleConfig())

    assert config == {}
    assert meta["capture_succeeded"] is False
    assert meta["args_class"] == "_ExampleConfig"


def test_collect_llm_api_config_propagates_unexpected_errors(monkeypatch):
    """Unexpected errors must propagate, not get swallowed by the inner net."""
    import pytest

    from tensorrt_llm.usage import llmapi_config

    def raise_runtime_error(*_args, **_kwargs):
        raise RuntimeError("unexpected collector bug")

    monkeypatch.setattr(llmapi_config, "build_capture_manifest", raise_runtime_error)

    with pytest.raises(RuntimeError, match="unexpected collector bug"):
        collect_llm_api_config_payloads(_ExampleConfig())


def test_collect_llm_api_config_captures_expanded_value_fields():
    """Representative newly-marked value fields are captured on TorchLlmArgs."""
    args = TorchLlmArgs(
        model="/customer/private/Llama",
        skip_tokenizer_init=True,
        moe_expert_parallel_size=2,
        moe_tensor_parallel_size=1,
        moe_cluster_parallel_size=1,
        num_postprocess_workers=3,
        stream_interval=4,
        trust_remote_code=True,
    )

    config, meta = _loads_payloads(args)

    assert config["moe_expert_parallel_size"] == 2
    assert config["moe_tensor_parallel_size"] == 1
    assert config["moe_cluster_parallel_size"] == 1
    assert config["num_postprocess_workers"] == 3
    assert config["stream_interval"] == 4
    assert config["trust_remote_code"] is True
    # backend on TorchLlmArgs is the Literal["pytorch"] override -> value capture.
    assert config["backend"] == "pytorch"
    assert meta["capture_succeeded"] is True


def test_collect_llm_api_config_captures_nested_config_value_fields():
    """Newly-marked nested-config value fields are captured via recursion."""
    from tensorrt_llm.llmapi.llm_args import MoeConfig

    args = TorchLlmArgs(
        model="/customer/private/Llama",
        skip_tokenizer_init=True,
        moe_config=MoeConfig(max_num_tokens=8192, disable_finalize_fusion=True),
    )

    config, _ = _loads_payloads(args)

    assert config["moe_config.max_num_tokens"] == 8192
    assert config["moe_config.disable_finalize_fusion"] is True
    # MoeConfig.backend is a Literal -> derived categorical, value captured.
    assert config["moe_config.backend"] == "AUTO"


def test_collect_llm_api_config_captures_quant_algo_cross_module():
    """quant_config.quant_algo (QuantConfig in modeling_utils.py) is captured.

    QuantConfig is defined outside llm_args.py but the collector recurses into
    any nested pydantic model, so marking the field at its definition site is
    sufficient. quant_config is a real model field only on TrtLlmArgs.
    """
    from tensorrt_llm.llmapi.llm_args import TrtLlmArgs
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    args = TrtLlmArgs(
        model="/customer/private/Llama",
        skip_tokenizer_init=True,
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8),
    )

    config, meta = _loads_payloads(args)

    assert config["quant_config.quant_algo"] == "FP8"
    assert meta["capture_succeeded"] is True


def test_field_wrapper_preserves_callable_json_schema_extra_with_metadata():
    """Preserve callable json_schema_extra when adding status/telemetry metadata.

    The schema callable still runs for Pydantic JSON schema generation, while
    the collector can read the attached TRT-LLM metadata without executing it.
    """

    def mark_schema(schema: dict[str, object]) -> dict[str, object]:
        schema["original"] = True
        return {"returned": True}

    field = Field(
        default=1,
        status="beta",
        telemetry=True,
        json_schema_extra=mark_schema,
    )
    schema: dict[str, object] = {}
    assert callable(field.json_schema_extra)
    field.json_schema_extra(schema)

    assert schema == {
        "original": True,
        "returned": True,
        "status": "beta",
        "telemetry": {"kind": "value"},
    }

    class _CallableExtraConfig(StrictBaseModel):
        value: int = Field(
            default=1,
            telemetry=True,
            json_schema_extra=mark_schema,
        )

    config, meta = _loads_payloads(_CallableExtraConfig())
    assert config == {"value": 1}
    assert meta["capture_succeeded"] is True


def test_collect_llm_api_config_captures_none_on_optional_allowlist_field():
    """None on an Optional allowlist field is captured as null, not excluded.

    Regression: the None check must precede the allowlist branch, else a None
    default (e.g. reasoning_parser, TrtLlmArgs.backend) fails the allowlist and
    permanently flips unsafe_excluded on default configs.
    """

    class _C(StrictBaseModel):
        backend: Optional[str] = Field(
            default=None,
            telemetry={
                "kind": "categorical",
                "converter": "allowlist",
                "allowed_values": ["pytorch", "tensorrt"],
            },
        )

    config, meta = _loads_payloads(_C())
    assert config == {"backend": None}
    assert meta["captured_field_count"] == 1
    assert meta["excluded_field_count"] == 0
    assert meta["unsafe_excluded"] is False


def test_collect_llm_api_config_captures_quant_algo_none():
    """quant_config.quant_algo defaults to None and is captured as null."""
    from tensorrt_llm.llmapi.llm_args import TrtLlmArgs

    args = TrtLlmArgs(model="/customer/private/Llama", skip_tokenizer_init=True)

    config, _ = _loads_payloads(args)

    assert config["quant_config.quant_algo"] is None


def test_collect_llm_api_config_captures_sampler_type_categorical():
    """sampler_type is a bounded Union[str, SamplerType] categorical allowlist."""
    args = TorchLlmArgs(
        model="/customer/private/Llama",
        skip_tokenizer_init=True,
        sampler_type="TorchSampler",
    )

    config, meta = _loads_payloads(args)

    assert config["sampler_type"] == "TorchSampler"
    assert meta["capture_succeeded"] is True


def test_collect_llm_api_config_redacts_out_of_allowlist_categorical_str():
    """An out-of-allowlist value on a categorical bare-str field is dropped.

    reasoning_parser is captured via TelemetryField.categorical mirroring the
    ReasoningParserFactory registry; any value outside that recognized domain
    (e.g. injected free-form text) must be excluded, not captured.
    """
    args = TorchLlmArgs(
        model="/customer/private/Llama",
        skip_tokenizer_init=True,
        reasoning_parser="not-a-real-parser-/customer/secret",
    )

    config, meta = _loads_payloads(args)

    assert "reasoning_parser" not in config
    assert meta["unsafe_excluded"] is True
    # An in-allowlist value is captured.
    args_ok = TorchLlmArgs(
        model="/customer/private/Llama",
        skip_tokenizer_init=True,
        reasoning_parser="deepseek-r1",
    )
    config_ok, _ = _loads_payloads(args_ok)
    assert config_ok["reasoning_parser"] == "deepseek-r1"


def test_collect_llm_api_config_captures_gms_load_format():
    """load_format=GMS is captured as 'gms' (was dropped before the allowlist fix).

    LoadFormat.GMS is a real, accepted value (convert_load_format maps the
    string 'gms' to the enum), but it was missing from the load_format telemetry
    allowlist, so GMS deployments were silently excluded from llmApiConfigJson.
    """
    args = TorchLlmArgs(
        model="/customer/private/Llama",
        skip_tokenizer_init=True,
        load_format="gms",
    )

    config, meta = _loads_payloads(args)

    assert config["load_format"] == "gms"
    assert meta["capture_succeeded"] is True


def test_collect_llm_api_config_captures_backend_allowlist_on_trt_args():
    """TrtLlmArgs inherits backend as Optional[str] -> bounded categorical allowlist."""
    from tensorrt_llm.llmapi.llm_args import TrtLlmArgs

    args = TrtLlmArgs(
        model="/customer/private/Llama",
        skip_tokenizer_init=True,
        backend="tensorrt",
    )

    config, meta = _loads_payloads(args)

    assert config["backend"] == "tensorrt"
    assert meta["capture_succeeded"] is True


def _walk_captured_keys(model) -> set[str]:
    """Capture a single nested config model and return its captured keys."""
    config, _ = _loads_payloads(model)
    return set(config)


def test_collect_llm_api_config_captures_decoding_type_for_every_arm():
    """The speculative discriminator decoding_type is captured for every arm.

    decoding_type is the single most valuable categorical (it identifies which
    speculative mode is active). The runtime collector walks the concrete active
    arm's model_fields, so marking it on only one arm drops it for the others.
    Assert representative non-UserProvided arms capture it.
    """
    from tensorrt_llm.llmapi.llm_args import (
        AutoDecodingConfig,
        MedusaDecodingConfig,
        MTPDecodingConfig,
        NGramDecodingConfig,
    )

    mtp = MTPDecodingConfig(num_nextn_predict_layers=1)
    assert _walk_captured_keys(mtp) >= {"decoding_type"}

    medusa = MedusaDecodingConfig(max_draft_len=1, medusa_choices=[[0]])
    assert _walk_captured_keys(medusa) >= {"decoding_type"}

    ngram = NGramDecodingConfig(max_draft_len=1, max_matching_ngram_size=2)
    assert _walk_captured_keys(ngram) >= {"decoding_type"}

    auto = AutoDecodingConfig()
    assert _walk_captured_keys(auto) >= {"decoding_type"}


def test_collect_llm_api_config_captures_max_total_draft_tokens_for_every_arm():
    """max_total_draft_tokens is a safe value field on the shared base.

    Marking it only on a single override (SaveHiddenStates) drops it for every
    other arm at runtime even though the doc-gen union-collapse advertises it.
    Mark it on DecodingBaseConfig so all arms capture it.
    """
    from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

    mtp = MTPDecodingConfig(num_nextn_predict_layers=1)
    assert "max_total_draft_tokens" in _walk_captured_keys(mtp)


def test_collect_llm_api_config_captures_sparse_algorithm_for_every_arm():
    """The sparse-attention discriminator algorithm is captured for every arm.

    algorithm identifies the active sparse algorithm (rocket, dsa, skip_softmax).
    Marking it on only one arm drops it for the others at runtime even though the
    doc-gen union-collapse advertises sparse_attention_config.algorithm.
    """
    from tensorrt_llm.llmapi.llm_args import (
        DeepSeekSparseAttentionConfig,
        RocketSparseAttentionConfig,
        SkipSoftmaxAttentionConfig,
    )

    assert _walk_captured_keys(RocketSparseAttentionConfig()) >= {"algorithm"}
    assert _walk_captured_keys(DeepSeekSparseAttentionConfig()) >= {"algorithm"}
    assert _walk_captured_keys(SkipSoftmaxAttentionConfig()) >= {"algorithm"}


def test_background_reporter_keeps_initial_report_when_config_capture_fails(monkeypatch):
    sent_payloads = []

    monkeypatch.setattr(usage_lib, "_MAX_HEARTBEATS", 0)
    monkeypatch.setattr(usage_lib, "_get_trtllm_version", lambda: "0.0.0-test")
    monkeypatch.setattr(
        usage_lib,
        "_collect_system_info",
        lambda: {
            "platform": "linux",
            "python_version": "3",
            "cpu_architecture": "x86",
            "cpu_count": 1,
        },
    )
    monkeypatch.setattr(
        usage_lib,
        "_collect_gpu_info",
        lambda: {"gpu_count": 0, "gpu_name": "", "gpu_memory_mb": 0, "cuda_version": ""},
    )
    monkeypatch.setattr(usage_lib, "_send_to_gxt", sent_payloads.append)

    def raise_capture_error(_):
        raise RuntimeError("capture failed")

    monkeypatch.setattr(usage_lib, "_collect_llm_api_config_payloads", raise_capture_error)

    usage_lib._background_reporter(
        llm_args=object(), pretrained_config=None, usage_context="llm_class"
    )

    params = sent_payloads[0]["events"][0]["parameters"]
    assert json.loads(params["llmApiConfigJson"]) == {}
    meta = json.loads(params["llmApiConfigMetaJson"])
    assert meta["capture_succeeded"] is False
    assert meta["args_class"] == "object"
    assert params["featuresJson"]


def test_field_wrapper_records_explicit_exclude_marker():
    from tensorrt_llm.usage.llmapi_config import _get_telemetry_metadata

    class _C(StrictBaseModel):
        a: int = Field(default=1, telemetry=False)

    meta = _get_telemetry_metadata(_C.model_fields["a"])
    assert meta == {"exclude": True}


def test_collect_llm_api_config_honors_explicit_exclude_sentinel():
    class _ExcludeConfig(StrictBaseModel):
        kept: int = Field(default=1)
        secret_seed: int = Field(default=42, telemetry=False)

    config, meta = _loads_payloads(_ExcludeConfig())
    assert config == {"kept": 1}
    assert "secret_seed" not in config
    assert meta["capturable_field_count"] == 1


def test_collect_llm_api_config_honors_raw_json_schema_extra_exclude():
    # Cross-module models use bare pydantic Field with json_schema_extra={"telemetry": ...}.
    # A raw {"telemetry": False} must be honored as an exclude, like the wrapper telemetry=False.
    from pydantic import Field as PydField

    class _RawExcludeConfig(StrictBaseModel):
        kept: int = 1
        secret: int = PydField(default=2, json_schema_extra={"telemetry": False})

    config, meta = _loads_payloads(_RawExcludeConfig())
    assert "secret" not in config
    assert config == {"kept": 1}


def test_runtime_keys_are_subset_of_manifest_for_fixture():
    from tensorrt_llm.usage.llmapi_config import build_capture_manifest

    inst = _ExampleConfig()
    manifest_paths = {e.path for e in build_capture_manifest(_ExampleConfig)}
    config, _ = _loads_payloads(inst)
    assert set(config) <= manifest_paths


def test_manifest_excludes_loosely_typed_model_children():
    # B-1 regression: moe_config.load_balancer is Optional[Union[object, str]];
    # a validator coerces it into a MoeLoadBalancerConfig at runtime, but the
    # annotation names no BaseModel, so its children must NOT be capturable.
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
    from tensorrt_llm.usage.llmapi_config import build_capture_manifest

    paths = {e.path for e in build_capture_manifest(TorchLlmArgs)}
    assert not any(p.startswith("moe_config.load_balancer.") for p in paths)


def test_collect_llm_api_config_caps_total_payload_size(monkeypatch):
    from tensorrt_llm.usage import llmapi_config as rc

    class _BigConfig(StrictBaseModel):
        a: int = 11111111
        b: int = 22222222
        c: int = 33333333

    monkeypatch.setattr(rc, "MAX_CONFIG_BYTES", 20)  # force truncation
    config, meta = _loads_payloads(_BigConfig())
    assert meta["payload_truncated"] is True
    assert len(rc._canonical_json(config).encode("utf-8")) <= 20
