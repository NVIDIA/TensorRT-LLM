# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MX-specific ModelLoader branches."""

from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from transformers import LlamaConfig
from utils.post_transform_qualification import (
    PostTransformQualificationCase,
    assert_post_transform_lifecycle_equivalent,
)

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_llama as modeling_llama_mod
from tensorrt_llm._torch.models.checkpoints.mx.checkpoint_loader import MXCheckpointLoader
from tensorrt_llm._torch.modules import mla as mla_mod
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.mla import MLA
from tensorrt_llm._torch.pyexecutor import model_loader as model_loader_mod
from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader
from tensorrt_llm._torch.weight_sharing import (
    ARTIFACT_IDENTITY_FORMAT_VERSION,
    SOURCE_IDENTITY_FORMAT_VERSION,
    ArtifactIdentity,
    PostTransformFeature,
    PostTransformProfile,
    PostTransformProfileRegistry,
    PostTransformQualificationReason,
    PostTransformTransferScope,
)
from tensorrt_llm.llmapi.llm_args import LoadFormat

_SOURCE_IDENTITY = model_loader_mod.SourceIdentity(
    format_version=SOURCE_IDENTITY_FORMAT_VERSION,
    artifact_identity=ArtifactIdentity(
        format_version=ARTIFACT_IDENTITY_FORMAT_VERSION,
        scheme="checkpoint_manifest_sha256",
        digest="0" * 64,
    ),
    model_fingerprint="model",
    quant_fingerprint="quant",
    backend_fingerprint="backend",
    parallel_fingerprint="parallel",
    rank=0,
    shard_fingerprint="shard",
)


class _LinearStub(nn.Module):
    def __init__(self):
        super().__init__()
        self._weights_transformed = False

    def post_load_weights(self):
        pass


def _make_draft_model_config():
    pretrained_config = SimpleNamespace(
        architectures=["DraftArch"],
        num_attention_heads=1,
        num_key_value_heads=1,
        tie_word_embeddings=False,
        torch_dtype=torch.float16,
    )
    return ModelConfig(pretrained_config=pretrained_config)


class _DraftModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        self.config = model_config.pretrained_config
        self.linear = _LinearStub()


class _TinyModel(nn.Module):
    def __init__(self, events):
        super().__init__()
        self.model_config = SimpleNamespace(
            pretrained_config=SimpleNamespace(architectures=["TinyForCausalLM"], model_type="tiny")
        )
        self._weights_transformed = False
        self.linear = _LinearStub()
        self.draft_config = _make_draft_model_config()
        self.draft_model = _DraftModel(self.draft_config)
        self._events = events

    def _apply(self, fn):
        # The test is about ModelLoader's MX branching, not CUDA allocation.
        return self

    def to(self, *args, **kwargs):
        return self

    def load_weights(self, weights, mapper):
        self._events.append("load_weights")

    def load_draft_weights(self, weights, mapper):
        self._events.append("load_draft_weights")

    def setup_aliases(self):
        self._events.append("setup_aliases")

    def cache_derived_state(self):
        self._events.append("cache_derived_state")

    def post_load_weights(self):
        self._events.append("post_load_weights")


@contextmanager
def _moe_context(config, mapping):
    yield None


def _tiny_llama_model(monkeypatch):
    monkeypatch.setattr(modeling_llama_mod, "get_sm_version", lambda: 90)
    llama_config = LlamaConfig(
        architectures=["LlamaForCausalLM"],
        attention_bias=False,
        hidden_act="silu",
        hidden_size=16,
        intermediate_size=32,
        max_position_embeddings=16,
        mlp_bias=False,
        num_attention_heads=2,
        num_hidden_layers=2,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        torch_dtype=torch.float32,
        vocab_size=32,
    )
    return model_loader_mod.LlamaForCausalLM(
        ModelConfig(
            pretrained_config=llama_config,
            max_num_tokens=16,
            max_seq_len=16,
        )
    )


def _llama_alias_state(model):
    layers = model.model.layers
    return {
        "skip_norm": model.model.skip_norm,
        "layer0_next_norm": layers[0].next_layer_layernorm is layers[1].input_layernorm,
        "layer0_next_attn": layers[0].next_attn is layers[1].self_attn,
        "layer1_skip_input_norm": layers[1].skip_input_layernorm,
        "layer1_next_norm": layers[1].next_layer_layernorm is model.model.norm,
        "layer1_next_attn": layers[1].next_attn is None,
    }


def _tiny_profile_registry(*, speculative_mode: str | None = None) -> PostTransformProfileRegistry:
    return PostTransformProfileRegistry(
        profiles=(
            PostTransformProfile(
                profile_id="tiny-for-causal-lm-target-v1",
                root_model_class=_TinyModel,
                architecture="TinyForCausalLM",
                model_type="tiny",
                speculative_mode=speculative_mode,
                protocol_version=(ModelLoader._MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION),
                transfer_scope=PostTransformTransferScope.TARGET_MODEL,
            ),
        )
    )


def _make_loader(monkeypatch, *, events, spec_config=None):
    llm_args = SimpleNamespace(load_format=LoadFormat.AUTO)
    loader = ModelLoader(
        llm_args=llm_args,
        mapping=MagicMock(name="mapping"),
        spec_config=spec_config,
        sparse_attention_config=None,
        max_num_tokens=128,
        max_seq_len=128,
    )
    loader._call_load_weights = MagicMock(
        side_effect=lambda fn, weights, mapper, **kwargs: fn(weights, mapper)
    )
    loader._load_and_validate_config = MagicMock(
        return_value=SimpleNamespace(name="config", mapping=SimpleNamespace())
    )

    monkeypatch.setattr(model_loader_mod, "timing", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(model_loader_mod, "maybe_create_moe_load_balancer", _moe_context)
    monkeypatch.setattr(model_loader_mod, "MetaInitMode", lambda: nullcontext())
    # These tests stub ModelConfig, while SourceIdentity has dedicated
    # coverage. Keep this file focused on ModelLoader MX branch behavior.

    def _build_artifact_identity(_cls, checkpoint_dir):
        assert checkpoint_dir == "/ckpt"
        return _SOURCE_IDENTITY.artifact_identity

    monkeypatch.setattr(
        model_loader_mod.ArtifactIdentity,
        "from_checkpoint",
        classmethod(_build_artifact_identity),
    )

    def _build_source_identity(_cls, *_args, **kwargs):
        assert kwargs["artifact_identity"] is _SOURCE_IDENTITY.artifact_identity
        return _SOURCE_IDENTITY

    monkeypatch.setattr(
        model_loader_mod.SourceIdentity,
        "from_model_config",
        classmethod(_build_source_identity),
    )
    monkeypatch.setattr(
        model_loader_mod.AutoModelForCausalLM,
        "from_config",
        MagicMock(return_value=_TinyModel(events)),
    )
    monkeypatch.setattr(model_loader_mod, "get_rank_model_storage", lambda _model: 0)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda: SimpleNamespace(synchronize=lambda: None),
    )
    return loader


def test_construct_checkpoint_loader_passes_mx_config():
    mx_config = SimpleNamespace(
        server_url="http://mx:8001",
        server_query_timeout_s=17,
    )

    checkpoint_loader = model_loader_mod._construct_checkpoint_loader(
        "pytorch",
        None,
        "MX",
        mx_config=mx_config,
        mx_model_name="Qwen/Qwen2.5-7B-Instruct",
    )

    assert isinstance(checkpoint_loader, MXCheckpointLoader)
    assert checkpoint_loader.mx_server_url == "http://mx:8001"
    assert checkpoint_loader.query_timeout_s == 17
    assert checkpoint_loader.model_name == "Qwen/Qwen2.5-7B-Instruct"


def test_mx_success_initializes_mapper_skips_weight_mapping_and_reload_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    loader = _make_loader(monkeypatch, events=events)
    monkeypatch.setattr(
        ModelLoader,
        "_POST_TRANSFORM_PROFILE_REGISTRY",
        _tiny_profile_registry(),
    )
    checkpoint_loader = MagicMock(name="checkpoint_loader")
    checkpoint_loader.checkpoint_format = "MX"
    checkpoint_loader.is_weights_preloaded.return_value = True
    checkpoint_loader.load_weights.return_value = {}

    model, _ = loader.load("/ckpt", checkpoint_loader)

    checkpoint_loader.load_weights.assert_called_once()
    _args, kwargs = checkpoint_loader.load_weights.call_args
    assert kwargs["mapping"] is loader.mapping
    assert kwargs["model"] is model
    assert kwargs["source_identity"] is loader._source_identity
    assert kwargs["allow_post_transform_weights"] is True
    assert loader._call_load_weights.call_count == 0
    checkpoint_loader.get_initialized_weight_mapper.assert_called_once()
    assert loader.weight_mapper is checkpoint_loader.get_initialized_weight_mapper.return_value
    checkpoint_loader.post_load_publish.assert_called_once_with(
        model,
        checkpoint_dir="/ckpt",
        weights_preloaded=True,
        source_identity=loader._source_identity,
    )

    # reload() uses self.weight_mapper unconditionally; MX success must
    # initialize it even though the initial load skipped _call_load_weights.
    model._weights_transformed = True
    model.linear._weights_transformed = True
    loader.reload(model, {"reloaded": MagicMock()})
    assert loader._call_load_weights.call_count == 1
    assert model._weights_transformed is False
    assert model.linear._weights_transformed is False
    assert events == ["post_load_weights", "load_weights"]


def test_reload_partial_loading_preserves_weights_transformed_flags(monkeypatch):
    events = []
    loader = _make_loader(monkeypatch, events=events)
    loader.weight_mapper = MagicMock(name="weight_mapper")
    model = _TinyModel(events)
    model._weights_transformed = True
    model.linear._weights_transformed = True

    loader.reload(model, {"reloaded": MagicMock()}, allow_partial_loading=True)

    assert loader._call_load_weights.call_count == 1
    assert loader._call_load_weights.call_args.kwargs["allow_partial_loading"] is True
    assert model._weights_transformed is True
    assert model.linear._weights_transformed is True
    assert events == ["load_weights"]


def test_mx_partial_fallback_merges_returned_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    loader = _make_loader(monkeypatch, events=events)
    monkeypatch.setattr(
        ModelLoader,
        "_POST_TRANSFORM_PROFILE_REGISTRY",
        _tiny_profile_registry(),
    )
    checkpoint_loader = MagicMock(name="checkpoint_loader")
    checkpoint_loader.checkpoint_format = "MX"
    checkpoint_loader.is_weights_preloaded.return_value = True
    fallback_weights = {"mismatched.weight": MagicMock()}
    checkpoint_loader.load_weights.return_value = fallback_weights

    model, _ = loader.load("/ckpt", checkpoint_loader)

    assert loader._call_load_weights.call_count == 1
    load_fn, weights, mapper = loader._call_load_weights.call_args.args
    assert load_fn == model.load_weights
    assert weights is fallback_weights
    assert mapper is loader.weight_mapper
    checkpoint_loader.post_load_publish.assert_called_once_with(
        model,
        checkpoint_dir="/ckpt",
        weights_preloaded=True,
        source_identity=loader._source_identity,
    )


class _PostTransformMxLoader:
    checkpoint_format = "MX"

    def __init__(self, *, post_transform: bool) -> None:
        self._post_transform = post_transform
        self._weights_preloaded = True
        self._disk_weight = MagicMock()
        self.load_weights = MagicMock(side_effect=self._load_weights)
        self.is_weights_preloaded = MagicMock(side_effect=lambda: self._weights_preloaded)
        self.get_initialized_weight_mapper = MagicMock(return_value=MagicMock())
        self.post_load_apply = MagicMock()
        self.post_load_publish = MagicMock()

    def _load_weights(self, *_args: object, **kwargs: object) -> dict[str, object]:
        if self._post_transform and kwargs.get("allow_post_transform_weights") is False:
            self._post_transform = False
            self._weights_preloaded = False
            return {"disk.weight": self._disk_weight}
        if self._post_transform:
            prepare_receiver = cast(
                Callable[[nn.Module], None], kwargs["prepare_post_transform_receiver"]
            )
            prepare_receiver(cast(nn.Module, kwargs["model"]))
        return {}

    def is_post_transform_weights_preloaded(self) -> bool:
        return self._post_transform


class _UnsafePostTransformMxLoader(_PostTransformMxLoader):
    def _load_weights(self, *_args: object, **_kwargs: object) -> dict[str, object]:
        return {}


def test_mx_post_transform_receiver_uses_staged_path_when_qualified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    loader = _make_loader(monkeypatch, events=events)
    monkeypatch.setattr(
        ModelLoader,
        "_POST_TRANSFORM_PROFILE_REGISTRY",
        _tiny_profile_registry(),
    )
    checkpoint_loader = _PostTransformMxLoader(post_transform=True)

    model, _ = loader.load("/ckpt", checkpoint_loader)

    loader._call_load_weights.assert_not_called()
    _args, kwargs = checkpoint_loader.load_weights.call_args
    assert kwargs["allow_post_transform_weights"] is True
    assert callable(kwargs["prepare_post_transform_receiver"])
    checkpoint_loader.post_load_publish.assert_called_once_with(
        model,
        checkpoint_dir="/ckpt",
        weights_preloaded=True,
        source_identity=loader._source_identity,
    )
    # Post-transform receivers skip transform_weights(), but the accepted
    # tensors are already in final layout. Keep the transform guard in sync so
    # future reload/refactor paths do not accidentally treat them as raw bytes.
    assert model._weights_transformed is True
    assert model.linear._weights_transformed is True
    assert model.draft_model.linear._weights_transformed is True
    assert events == ["setup_aliases", "setup_aliases", "cache_derived_state"]


def test_default_profile_qualifies_real_tiny_llama_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = PostTransformQualificationCase(
        profile_id="llama-for-causal-lm-target-v1",
        model_factory=lambda: _tiny_llama_model(monkeypatch),
        qualify_model=lambda model: ModelLoader._qualify_post_transform_profile(
            model,
            speculative_mode=None,
            loads_draft_weights=False,
        ),
        state_probes=(("aliases", _llama_alias_state),),
    )

    assert_post_transform_lifecycle_equivalent(case)


def test_separate_draft_model_is_not_qualified_by_target_only_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ModelLoader,
        "_POST_TRANSFORM_PROFILE_REGISTRY",
        _tiny_profile_registry(speculative_mode="eagle3_one_model"),
    )

    decision = ModelLoader._qualify_post_transform_profile(
        _TinyModel([]),
        speculative_mode="eagle3_one_model",
        loads_draft_weights=True,
    )

    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.FEATURE_NOT_SUPPORTED
    assert decision.unsupported_features == frozenset({PostTransformFeature.SEPARATE_DRAFT_MODEL})


def test_one_engine_speculative_mode_is_not_qualified_by_target_only_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ModelLoader,
        "_POST_TRANSFORM_PROFILE_REGISTRY",
        _tiny_profile_registry(),
    )

    decision = ModelLoader._qualify_post_transform_profile(
        _TinyModel([]),
        speculative_mode="mtp",
        loads_draft_weights=False,
    )

    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.SPECULATIVE_MODE_NOT_REGISTERED
    assert decision.unsupported_features == frozenset()


def test_speculative_mode_name_is_canonical_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warning = MagicMock()
    monkeypatch.setattr(model_loader_mod.logger, "warning", warning)

    assert ModelLoader._speculative_mode_name(None) is None
    warning.assert_not_called()
    assert (
        ModelLoader._speculative_mode_name(
            SimpleNamespace(spec_dec_mode=SimpleNamespace(name="MTP"))
        )
        == "mtp"
    )
    warning.assert_not_called()
    assert (
        ModelLoader._speculative_mode_name(SimpleNamespace(spec_dec_mode=SimpleNamespace()))
        == "unknown"
    )
    warning.assert_called_once_with(
        "Unable to identify the speculative decoding mode from %s; "
        "post-transform sharing is disabled for this load.",
        "SimpleNamespace",
    )


def test_mx_post_transform_receiver_falls_back_for_unqualified_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    loader = _make_loader(monkeypatch, events=events)
    checkpoint_loader = _PostTransformMxLoader(post_transform=True)

    model, _ = loader.load("/ckpt", checkpoint_loader)

    _args, kwargs = checkpoint_loader.load_weights.call_args
    assert kwargs["allow_post_transform_weights"] is False
    assert "prepare_post_transform_receiver" not in kwargs
    assert checkpoint_loader.is_weights_preloaded() is False
    assert loader._call_load_weights.call_count == 1
    load_fn, weights, mapper = loader._call_load_weights.call_args.args
    assert load_fn == model.load_weights
    assert weights == {"disk.weight": checkpoint_loader._disk_weight}
    assert mapper is loader.weight_mapper
    assert events == ["load_weights", "post_load_weights"]
    checkpoint_loader.post_load_publish.assert_not_called()


def test_mx_rejects_post_transform_preload_after_failed_qualification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _make_loader(monkeypatch, events=[])
    checkpoint_loader = _UnsafePostTransformMxLoader(post_transform=True)

    with pytest.raises(
        RuntimeError,
        match="reason=root_model_class_not_registered",
    ):
        loader.load("/ckpt", checkpoint_loader)

    _args, kwargs = checkpoint_loader.load_weights.call_args
    assert kwargs["allow_post_transform_weights"] is False
    checkpoint_loader.post_load_publish.assert_not_called()


def test_mx_fallback_runs_standard_weight_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    loader = _make_loader(monkeypatch, events=events)
    monkeypatch.setattr(
        ModelLoader,
        "_POST_TRANSFORM_PROFILE_REGISTRY",
        _tiny_profile_registry(),
    )
    checkpoint_loader = MagicMock(name="checkpoint_loader")
    checkpoint_loader.checkpoint_format = "MX"
    checkpoint_loader.is_weights_preloaded.return_value = False
    checkpoint_loader.load_weights.return_value = {"weight": MagicMock()}
    checkpoint_loader.get_initialized_weight_mapper.return_value = MagicMock()

    model, _ = loader.load("/ckpt", checkpoint_loader)

    assert loader._call_load_weights.call_count == 1
    assert events[0] == "load_weights"
    assert "post_load_weights" in events
    checkpoint_loader.post_load_publish.assert_called_once_with(
        model,
        checkpoint_dir="/ckpt",
        weights_preloaded=False,
        source_identity=loader._source_identity,
    )


def test_mx_artifact_identity_failure_falls_back_to_disk(monkeypatch):
    events = []
    loader = _make_loader(monkeypatch, events=events)
    monkeypatch.setattr(
        ModelLoader,
        "_POST_TRANSFORM_PROFILE_REGISTRY",
        _tiny_profile_registry(),
    )
    artifact_error = ValueError(
        "Checkpoint manifests do not support nested symlinked directories: /ckpt/shards"
    )
    monkeypatch.setattr(
        model_loader_mod.ArtifactIdentity,
        "from_checkpoint",
        MagicMock(side_effect=artifact_error),
    )
    source_identity_factory = MagicMock()
    monkeypatch.setattr(
        model_loader_mod.SourceIdentity,
        "from_model_config",
        source_identity_factory,
    )
    warning = MagicMock()
    monkeypatch.setattr(model_loader_mod.logger, "warning", warning)

    checkpoint_loader = MagicMock(name="checkpoint_loader")
    checkpoint_loader.checkpoint_format = "MX"
    checkpoint_loader.is_weights_preloaded.return_value = False
    checkpoint_loader.load_weights.return_value = {"weight": MagicMock()}
    checkpoint_loader.get_initialized_weight_mapper.return_value = MagicMock()

    model, _ = loader.load("/ckpt", checkpoint_loader)

    assert loader._source_identity is None
    source_identity_factory.assert_not_called()
    warning.assert_called_once()
    assert "falling back to regular checkpoint loading" in warning.call_args.args[0]
    _args, kwargs = checkpoint_loader.load_weights.call_args
    assert kwargs["source_identity"] is None
    checkpoint_loader.post_load_publish.assert_called_once_with(
        model,
        checkpoint_dir="/ckpt",
        weights_preloaded=False,
        source_identity=None,
    )


class _HookRecorder(nn.Module):
    def __init__(
        self,
        name: str,
        events: list[tuple[str, str]],
        *,
        removed: bool | None = None,
        transformed: bool | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.events = events
        if removed is not None:
            self._weights_removed = removed
        if transformed is not None:
            self._weights_transformed = transformed

    def setup_aliases(self) -> None:
        self.events.append((self.name, "setup_aliases"))

    def transform_weights(self) -> None:
        self.events.append((self.name, "transform_weights"))
        self._weights_transformed = True

    def cache_derived_state(self) -> None:
        self.events.append((self.name, "cache_derived_state"))

    def post_load_weights(self) -> None:
        self.events.append((self.name, "post_load_weights"))


class _HookModel(_HookRecorder):
    def __init__(self, events):
        super().__init__("model", events)
        self.child = _HookRecorder("child", events)
        self.transformed_child = _HookRecorder("transformed_child", events, transformed=True)
        self.removed_child = _HookRecorder("removed_child", events, removed=True)


def test_staged_hook_setup_aliases_walks_skip_removed_modules():
    events = []
    model = _HookModel(events)

    ModelLoader._setup_aliases(model)

    assert events == [
        ("model", "setup_aliases"),
        ("child", "setup_aliases"),
        ("transformed_child", "setup_aliases"),
    ]


def test_staged_hook_walks_skip_removed_and_transformed_modules():
    events = []
    model = _HookModel(events)

    ModelLoader._walk_transform(model)
    ModelLoader._walk_cache_state(model)
    ModelLoader._walk_full_post_load(model)

    assert events == [
        ("model", "transform_weights"),
        ("child", "transform_weights"),
        ("model", "cache_derived_state"),
        ("child", "cache_derived_state"),
        ("transformed_child", "cache_derived_state"),
        ("model", "post_load_weights"),
        ("child", "post_load_weights"),
        ("transformed_child", "post_load_weights"),
    ]


def test_reset_weights_transformed_only_resets_existing_flags():
    events = []
    model = _HookModel(events)
    model._weights_transformed = True
    model.child._weights_transformed = True

    ModelLoader._reset_weights_transformed(model)

    assert model._weights_transformed is False
    assert model.child._weights_transformed is False
    assert model.transformed_child._weights_transformed is False
    assert not hasattr(model.removed_child, "_weights_transformed")


def test_mark_weights_transformed_only_sets_existing_flags():
    events = []
    model = _HookModel(events)
    model._weights_transformed = False
    model.child._weights_transformed = False

    ModelLoader._mark_weights_transformed(model)

    assert model._weights_transformed is True
    assert model.child._weights_transformed is True
    assert model.transformed_child._weights_transformed is True
    assert not hasattr(model.removed_child, "_weights_transformed")


def test_linear_transform_weights_is_idempotent():
    linear = Linear(
        1,
        1,
        bias=False,
        reduce_output=False,
        skip_create_weights_in_init=True,
    )
    linear.quant_method = MagicMock()

    linear.transform_weights()
    linear.post_load_weights()

    linear.quant_method.transform_weights.assert_called_once_with(linear)
    assert linear._weights_transformed is True

    linear._weights_transformed = False
    linear.post_load_weights()
    assert linear.quant_method.transform_weights.call_count == 2

    linear._weights_transformed = False
    linear.cache_derived_state()
    assert linear._weights_transformed is True


def test_mla_transform_weights_is_idempotent(monkeypatch):
    monkeypatch.setattr(mla_mod, "get_sm_version", lambda: 120)
    quant_mode = SimpleNamespace(has_fp8_block_scales=lambda: True)
    mla = MLA.__new__(MLA)
    mla._weights_transformed = False
    mla.kv_b_proj = SimpleNamespace(quant_config=SimpleNamespace(quant_mode=quant_mode))
    mla.k_b_proj_trans = "k_weight"
    mla.k_b_proj_trans_scale = "k_scale"
    mla.v_b_proj = "v_weight"
    mla.v_b_proj_scale = "v_scale"
    calls = []

    def fake_resmooth(weight, scale, recipe):
        calls.append((weight, scale, recipe))
        return f"{weight}_transformed", f"{scale}_transformed"

    mla.resmooth_parameters = fake_resmooth

    MLA.transform_weights(mla)
    MLA.post_load_weights(mla)

    assert calls == [
        ("k_weight", "k_scale", (1, 128, 128)),
        ("v_weight", "v_scale", (1, 128, 128)),
    ]
    assert mla.k_b_proj_trans == "k_weight_transformed"
    assert mla.k_b_proj_trans_scale == "k_scale_transformed"
    assert mla.v_b_proj == "v_weight_transformed"
    assert mla.v_b_proj_scale == "v_scale_transformed"
    assert mla._weights_transformed is True

    mla._weights_transformed = False
    MLA.cache_derived_state(mla)
    assert mla._weights_transformed is True
