# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``MXCheckpointLoader`` (``checkpoint_format='MX'``).

These tests intentionally do NOT exercise the upstream ``modelexpress``
library. Tests for the import-failure fallback path mock
``modelexpress.*`` symbols out of ``sys.modules`` so the assertion is
about *our* fallback behavior, not about the upstream API.
"""

import os
import sys
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm._torch.models.checkpoints.auto_mapper import AutoCheckpointMapper
from tensorrt_llm._torch.models.checkpoints.base_checkpoint_loader import BaseCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import HfCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.hf.qwen3_next_weight_mapper import (
    Qwen3NextHfWeightMapper,
)
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.models.checkpoints.mx.checkpoint_loader import (
    MXCheckpointLoader,
    _normalize_model_identity,
    _resolve_mx_model_name,
)

# ---------------------------------------------------------------------------
# Construction & static properties
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_no_args_constructs(self):
        loader = MXCheckpointLoader()
        assert loader.mx_server_url is None
        assert loader.p2p_succeeded is False

    def test_mx_server_url_stored(self):
        loader = MXCheckpointLoader(mx_server_url="http://mx:8001")
        assert loader.mx_server_url == "http://mx:8001"
        assert loader.p2p_succeeded is False

    def test_query_timeout_stored(self):
        loader = MXCheckpointLoader(mx_server_url="http://mx:8001", query_timeout_s=900)
        assert loader.query_timeout_s == 900

    def test_subclasses_hf_loader(self):
        # Inheriting from HfCheckpointLoader is what gives us free disk
        # fallback. Don't break this.
        loader = MXCheckpointLoader()
        assert isinstance(loader, HfCheckpointLoader)
        assert isinstance(loader, BaseCheckpointLoader)

    def test_checkpoint_format_property(self):
        loader = MXCheckpointLoader()
        assert loader.checkpoint_format == "MX"

    def test_checkpoint_format_backing_attr(self):
        # Several call sites read ``self._checkpoint_format`` directly
        # (instead of going through the property). The constructor must
        # align the backing attribute with the property override.
        loader = MXCheckpointLoader()
        assert loader._checkpoint_format == "MX"

    def test_p2p_succeeded_property_initial(self):
        loader = MXCheckpointLoader()
        assert loader.p2p_succeeded is False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registered_under_mx(self):
        # ``ModelLoader.load`` resolves a checkpoint loader from
        # ``checkpoint_format`` via ``BaseCheckpointLoader.get``. PR #13045
        # registers ``MX`` via ``@register_checkpoint_loader("MX")``.
        # The real call site (in _construct_checkpoint_loader) passes
        # weight_loader=, weight_mapper=, config_loader=, plus optional
        # mx_server_url= — so test with the same call shape.
        loader = BaseCheckpointLoader.get(
            checkpoint_format="MX",
            weight_loader=None,
            weight_mapper=None,
            config_loader=None,
            mx_server_url="http://mx:8001",
            query_timeout_s=900,
        )
        assert isinstance(loader, MXCheckpointLoader)
        assert loader.checkpoint_format == "MX"
        assert loader.mx_server_url == "http://mx:8001"
        assert loader.query_timeout_s == 900


class TestMxMapperFallback:
    def test_arch_specific_mx_mapper_falls_back_to_hf_mapper(self):
        mapper = AutoCheckpointMapper.get("MX", "Qwen3NextForCausalLM")
        assert isinstance(mapper, Qwen3NextHfWeightMapper)

    def test_unknown_arch_uses_default_mx_mapper(self):
        mapper = AutoCheckpointMapper.get("MX", "UnknownArchitecture")
        assert isinstance(mapper, HfWeightMapper)


# ---------------------------------------------------------------------------
# load_weights — disk-fallback paths (no upstream library involved)
# ---------------------------------------------------------------------------


class TestLoadWeightsFallback:
    """Disk-fallback paths that should not touch the upstream MX library.

    All four fallback triggers share the same observable contract:
    ``p2p_succeeded`` stays False, the parent ``HfCheckpointLoader.
    load_weights`` is invoked exactly once, and its return value is
    propagated unchanged. We parameterize the trigger to keep that
    contract in one place.
    """

    # Trigger setup builders.
    @staticmethod
    def _no_url(stack):  # noqa: ARG004 — stack unused for this trigger
        return MXCheckpointLoader(), {"model": MagicMock()}

    @staticmethod
    def _no_model(stack):  # noqa: ARG004
        return MXCheckpointLoader(mx_server_url="http://mx:8001"), {}

    @staticmethod
    def _modelexpress_unavailable(stack):
        stack.enter_context(_block_modelexpress())
        return (MXCheckpointLoader(mx_server_url="http://mx:8001"), {"model": MagicMock()})

    @staticmethod
    def _upstream_raises(stack):
        fake_mx = _build_fake_modelexpress(load_weights_side_effect=RuntimeError("boom"))
        stack.enter_context(_install_fake_modelexpress(fake_mx))
        return (MXCheckpointLoader(mx_server_url="http://mx:8001"), {"model": MagicMock()})

    @pytest.mark.parametrize(
        "trigger_id, setup",
        [
            ("no_mx_server_url", _no_url),
            ("no_model_kwarg", _no_model),
            ("modelexpress_not_installed", _modelexpress_unavailable),
            ("upstream_raises", _upstream_raises),
        ],
    )
    def test_falls_back_to_disk(self, trigger_id, setup):
        sentinel = {"disk-load": "result"}
        with ExitStack() as stack:
            loader, extra_kwargs = setup(stack)
            mock_super_load = stack.enter_context(
                patch.object(HfCheckpointLoader, "load_weights", return_value=sentinel)
            )

            result = loader.load_weights("/nonexistent", mapping=MagicMock(), **extra_kwargs)

        assert result is sentinel, (
            f"trigger={trigger_id}: production code must propagate the "
            "parent loader's return value unchanged"
        )
        assert loader.p2p_succeeded is False, (
            f"trigger={trigger_id}: p2p_succeeded must stay False on any fallback path"
        )
        mock_super_load.assert_called_once()


# ---------------------------------------------------------------------------
# load_weights — MX-success and mixed-success paths (mocked upstream)
# ---------------------------------------------------------------------------


class TestLoadWeightsMxPath:
    def test_p2p_full_success_returns_empty_dict(self):
        # Empty fallback dict means MX delivered all weights into model
        # params; ``ModelLoader`` interprets the empty dict + p2p_succeeded
        # flag as "skip the standard weight-mapping pipeline".
        loader = MXCheckpointLoader(mx_server_url="http://mx:8001")
        fake_mx = _build_fake_modelexpress(load_weights_return={})
        mapping = MagicMock(name="mapping")
        model = MagicMock(name="model")

        with _install_fake_modelexpress(fake_mx):
            result = loader.load_weights("/nonexistent", mapping=mapping, model=model)

        assert result == {}
        assert loader.p2p_succeeded is True

        # Verify the integration contract with the upstream library:
        # 1. Constructed MxLiveWeightLoader with our mx_server_url.
        fake_mx.trtllm_live_transfer.MxLiveWeightLoader.assert_called_once_with(
            mx_server="http://mx:8001"
        )
        # 2. Called load_weights with the right positional/keyword args.
        weight_loader_instance = fake_mx.trtllm_live_transfer.MxLiveWeightLoader.return_value
        weight_loader_instance.load_weights.assert_called_once_with(
            "/nonexistent", mapping=mapping, model=model
        )

    def test_mixed_success_returns_fallback_weights(self):
        # When MX returns a non-empty fallback dict (size-mismatched
        # tensors), keep the P2P transfer and let ModelLoader merge these
        # tensors through the standard disk pipeline.
        loader = MXCheckpointLoader(mx_server_url="http://mx:8001")
        fallback = {"some.weight": MagicMock()}
        fake_mx = _build_fake_modelexpress(load_weights_return=fallback)

        with (
            _install_fake_modelexpress(fake_mx),
            patch.object(HfCheckpointLoader, "load_weights") as mock_super_load,
        ):
            result = loader.load_weights("/nonexistent", mapping=MagicMock(), model=MagicMock())

        assert loader.p2p_succeeded is True
        assert result is fallback
        mock_super_load.assert_not_called()


# ---------------------------------------------------------------------------
# publish_as_source — env-var dance and graceful no-op
# ---------------------------------------------------------------------------


class TestPublishAsSource:
    def test_no_mx_server_url_is_noop(self):
        loader = MXCheckpointLoader()  # mx_server_url is None
        # Any attempt to import modelexpress would raise here, so we
        # don't even need to mock.
        with _block_modelexpress():
            loader.publish_as_source(MagicMock())  # must not raise

    def test_modelexpress_unavailable_is_noop(self):
        loader = MXCheckpointLoader(mx_server_url="http://mx:8001")
        with _block_modelexpress():
            loader.publish_as_source(MagicMock())  # must not raise

    def test_publish_called_with_model(self):
        loader = MXCheckpointLoader(mx_server_url="http://mx:8001")
        fake_mx = _build_fake_modelexpress()
        model = MagicMock(name="model")

        with _install_fake_modelexpress(fake_mx):
            loader.publish_as_source(model)

        fake_mx.trtllm_live_transfer.publish_model_params.assert_called_once_with(model)

    def test_env_var_set_during_publish_then_restored(self):
        loader = MXCheckpointLoader(mx_server_url="http://mx-instance:9999")
        captured_env = {}

        def _capture(model):
            captured_env["MODEL_EXPRESS_URL"] = os.environ.get("MODEL_EXPRESS_URL")

        fake_mx = _build_fake_modelexpress(publish_side_effect=_capture)

        prior = os.environ.pop("MODEL_EXPRESS_URL", None)
        try:
            with _install_fake_modelexpress(fake_mx):
                loader.publish_as_source(MagicMock())
        finally:
            if prior is not None:
                os.environ["MODEL_EXPRESS_URL"] = prior

        # During the call, our per-loader URL was visible.
        assert captured_env["MODEL_EXPRESS_URL"] == "http://mx-instance:9999"
        # After the call, the env var is back to the pre-call state.
        assert "MODEL_EXPRESS_URL" not in os.environ

    def test_env_var_restored_to_prior_value(self):
        loader = MXCheckpointLoader(mx_server_url="http://mx-instance:9999")
        fake_mx = _build_fake_modelexpress()
        prior = os.environ.get("MODEL_EXPRESS_URL")
        os.environ["MODEL_EXPRESS_URL"] = "http://prior-value:1234"
        try:
            with _install_fake_modelexpress(fake_mx):
                loader.publish_as_source(MagicMock())
            assert os.environ["MODEL_EXPRESS_URL"] == "http://prior-value:1234"
        finally:
            if prior is None:
                os.environ.pop("MODEL_EXPRESS_URL", None)
            else:
                os.environ["MODEL_EXPRESS_URL"] = prior

    def test_publish_exception_swallowed(self):
        # publish_as_source is a best-effort hook; an upstream exception
        # must NOT take down model loading.
        loader = MXCheckpointLoader(mx_server_url="http://mx:8001")
        fake_mx = _build_fake_modelexpress(publish_side_effect=RuntimeError("upstream went away"))

        with _install_fake_modelexpress(fake_mx):
            loader.publish_as_source(MagicMock())  # must not raise


# ---------------------------------------------------------------------------
# Helpers — fake modelexpress modules and import blockers
# ---------------------------------------------------------------------------


def _modelexpress_module_names():
    return [
        "modelexpress",
        "modelexpress.trtllm_live_transfer",
    ]


def _block_modelexpress():
    """Context manager that makes ``import modelexpress`` raise ImportError."""
    saved = {name: sys.modules.get(name) for name in _modelexpress_module_names()}

    class _Blocker:
        def __enter__(self):
            for name in _modelexpress_module_names():
                # Setting to None makes ``import name`` raise
                # ImportError per PEP 328 / sys.modules semantics.
                sys.modules[name] = None
            return self

        def __exit__(self, exc_type, exc, tb):
            for name, prior in saved.items():
                if prior is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = prior

    return _Blocker()


def _build_fake_modelexpress(
    *,
    load_weights_return=None,
    load_weights_side_effect=None,
    publish_side_effect=None,
    source_instances=None,
):
    """Build a fake modelexpress module tree mimicking the symbols we use."""
    fake_pkg = MagicMock(name="modelexpress")

    fake_trtllm_live = MagicMock(name="modelexpress.trtllm_live_transfer")

    # MxLiveWeightLoader(mx_server=url).load_weights(ckpt_dir, mapping=, model=)
    weight_loader_instance = MagicMock(name="MxLiveWeightLoader instance")
    if load_weights_side_effect is not None:
        weight_loader_instance.load_weights.side_effect = load_weights_side_effect
    else:
        weight_loader_instance.load_weights.return_value = (
            load_weights_return if load_weights_return is not None else {}
        )
    fake_trtllm_live.MxLiveWeightLoader = MagicMock(return_value=weight_loader_instance)
    client_instance = MagicMock(name="MxClient instance")
    client_instance.list_sources.return_value = MagicMock(instances=source_instances or [])
    fake_trtllm_live.MxClient = MagicMock(return_value=client_instance)
    fake_trtllm_live._build_trtllm_identity = MagicMock(return_value=MagicMock())

    # publish_model_params(model)
    if publish_side_effect is not None:
        fake_trtllm_live.publish_model_params = MagicMock(side_effect=publish_side_effect)
    else:
        fake_trtllm_live.publish_model_params = MagicMock()

    fake_pkg.trtllm_live_transfer = fake_trtllm_live
    return fake_pkg


def _install_fake_modelexpress(fake_pkg):
    """Context manager that installs a fake ``modelexpress`` into sys.modules."""

    class _Installer:
        _saved = {}

        def __enter__(self):
            for name in _modelexpress_module_names():
                self._saved[name] = sys.modules.get(name)
            sys.modules["modelexpress"] = fake_pkg
            sys.modules["modelexpress.trtllm_live_transfer"] = fake_pkg.trtllm_live_transfer
            return self

        def __exit__(self, exc_type, exc, tb):
            for name, prior in self._saved.items():
                if prior is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = prior

    return _Installer()


# ---------------------------------------------------------------------------
# Item 2: defensive MX_SOURCE_QUERY_TIMEOUT default
# ---------------------------------------------------------------------------


class TestMxSourceQueryTimeoutDefault:
    """``load_weights`` caps upstream's source-query timeout on P2P attempts.

    The first replica on a cold cluster should not block for the upstream
    default of 1 hour. ``setdefault`` semantics — never overrides an
    explicit user value.
    """

    @pytest.fixture(autouse=True)
    def _isolated_env(self, monkeypatch):
        monkeypatch.delenv("MX_SOURCE_QUERY_TIMEOUT", raising=False)
        yield

    def test_no_registered_source_gets_short_default_during_load(self):
        def _assert_timeout(*args, **kwargs):
            assert os.environ.get("MX_SOURCE_QUERY_TIMEOUT") == "30"
            return {}

        loader = MXCheckpointLoader(mx_server_url="http://mx:8001")
        fake_mx = _build_fake_modelexpress(load_weights_side_effect=_assert_timeout)
        with _install_fake_modelexpress(fake_mx):
            loader.load_weights("/nonexistent", mapping=MagicMock(), model=MagicMock())
        assert "MX_SOURCE_QUERY_TIMEOUT" not in os.environ

    def test_existing_source_keeps_upstream_default_when_unset(self):
        def _assert_no_timeout(*args, **kwargs):
            assert "MX_SOURCE_QUERY_TIMEOUT" not in os.environ
            return {}

        loader = MXCheckpointLoader(mx_server_url="http://mx:8001")
        fake_mx = _build_fake_modelexpress(
            load_weights_side_effect=_assert_no_timeout,
            source_instances=[MagicMock()],
        )
        with _install_fake_modelexpress(fake_mx):
            loader.load_weights("/nonexistent", mapping=MagicMock(), model=MagicMock())
        assert "MX_SOURCE_QUERY_TIMEOUT" not in os.environ

    def test_env_value_preserved(self, monkeypatch):
        # If the user/orchestrator already set a value, our defensive
        # default must not stomp it.
        monkeypatch.setenv("MX_SOURCE_QUERY_TIMEOUT", "120")

        def _assert_env_timeout(*args, **kwargs):
            assert os.environ.get("MX_SOURCE_QUERY_TIMEOUT") == "120"
            return {}

        loader = MXCheckpointLoader(mx_server_url="http://mx:8001")
        fake_mx = _build_fake_modelexpress(load_weights_side_effect=_assert_env_timeout)
        with _install_fake_modelexpress(fake_mx):
            loader.load_weights("/nonexistent", mapping=MagicMock(), model=MagicMock())
        assert os.environ.get("MX_SOURCE_QUERY_TIMEOUT") == "120"

    def test_configured_timeout_applies_during_load_and_restores_env(self):
        def _assert_config_timeout(*args, **kwargs):
            assert os.environ.get("MX_SOURCE_QUERY_TIMEOUT") == "900"
            return {}

        loader = MXCheckpointLoader(mx_server_url="http://mx:8001", query_timeout_s=900)
        fake_mx = _build_fake_modelexpress(load_weights_side_effect=_assert_config_timeout)
        with _install_fake_modelexpress(fake_mx):
            loader.load_weights("/nonexistent", mapping=MagicMock(), model=MagicMock())
        assert "MX_SOURCE_QUERY_TIMEOUT" not in os.environ

    def test_no_mx_url_does_not_touch_env(self):
        # HF-only loads must not surprise users by setting MX-namespaced
        # env vars they didn't ask for.
        MXCheckpointLoader()  # mx_server_url=None
        assert "MX_SOURCE_QUERY_TIMEOUT" not in os.environ


# ---------------------------------------------------------------------------
# Item 3: model_name plumbing + resolver + publish-side env handoff
# ---------------------------------------------------------------------------


class TestModelNameConstructor:
    def test_default_model_name_none(self):
        loader = MXCheckpointLoader()
        assert loader.model_name is None

    def test_model_name_stored_as_string(self):
        loader = MXCheckpointLoader(model_name="Qwen/Qwen2.5-72B-Instruct")
        assert loader.model_name == "Qwen/Qwen2.5-72B-Instruct"

    def test_model_name_path_coerced_to_string(self, tmp_path):
        # Constructor accepts Path (e.g. llm_args.model resolved as Path)
        # and stores it as a string for downstream env-var publishing.
        path = tmp_path / "my-checkpoint"
        loader = MXCheckpointLoader(model_name=path)
        assert loader.model_name == str(path)
        assert isinstance(loader.model_name, str)


class TestNormalizeModelIdentity:
    """``_normalize_model_identity`` is the path-vs-id heuristic used by
    the resolver. Pinning it down with parametrized cases."""

    @pytest.mark.parametrize(
        "label, value, expected",
        [
            # Hub IDs and bare names pass through unchanged.
            ("bare_name", "llama-3-70b", "llama-3-70b"),
            ("hub_id", "Qwen/Qwen2.5-72B-Instruct", "Qwen/Qwen2.5-72B-Instruct"),
            ("nested_hub_id", "meta-llama/Llama-3-70B", "meta-llama/Llama-3-70B"),
            # Absolute paths get reduced to basenames.
            ("abs_path_simple", "/scratch/local-model", "local-model"),
            ("abs_path_nested", "/cache/foo/bar/baz", "baz"),
            # Relative-looking paths are treated as paths.
            ("dot_relative", "./local-model", "local-model"),
            ("home_expansion", "~/models/my-model", "my-model"),
            # Empty / sentinel.
            ("empty", "", "unknown"),
        ],
    )
    def test_basic_cases(self, label, value, expected):
        assert _normalize_model_identity(value) == expected, f"case={label}"

    def test_hf_snapshot_unmangling(self):
        # HF cache layout: ".../models--<org>--<name>/snapshots/<sha>/"
        # → "<org>/<name>" (not the commit sha).
        snapshot = (
            "/cache/huggingface/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/abc123def456789"
        )
        assert _normalize_model_identity(snapshot) == "Qwen/Qwen2.5-72B-Instruct"

    def test_hf_snapshot_unmangling_nested_org(self):
        # Multi-component HF org names use "--" as the separator.
        snapshot = "/cache/hub/models--meta-llama--Llama-3-70B-Instruct/snapshots/sha"
        assert _normalize_model_identity(snapshot) == "meta-llama/Llama-3-70B-Instruct"


class TestResolveMxModelName:
    """``_resolve_mx_model_name`` is the priority-ordered lookup used by
    ``publish_as_source``. Verifying the ordering."""

    @pytest.fixture(autouse=True)
    def _isolated_env(self, monkeypatch):
        monkeypatch.delenv("MODEL_NAME", raising=False)
        yield

    def test_explicit_arg_wins(self, monkeypatch):
        monkeypatch.setenv("MODEL_NAME", "from-env")
        assert _resolve_mx_model_name("explicit", "/cache/snapshot/abc") == "explicit"

    def test_env_used_when_arg_none(self, monkeypatch):
        monkeypatch.setenv("MODEL_NAME", "from-env")
        assert _resolve_mx_model_name(None, "/cache/snapshot/abc") == "from-env"

    def test_basename_fallback_when_arg_and_env_missing(self):
        assert _resolve_mx_model_name(None, "/scratch/local-model") == "local-model"

    def test_snapshot_fallback_when_arg_and_env_missing(self):
        snapshot = "/cache/huggingface/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/abc123"
        assert _resolve_mx_model_name(None, snapshot) == "Qwen/Qwen2.5-72B-Instruct"

    def test_unknown_when_all_missing(self):
        assert _resolve_mx_model_name(None, None) == "unknown"
        assert _resolve_mx_model_name("", None) == "unknown"

    def test_explicit_arg_normalized_too(self):
        # If the explicit arg looks like a path (e.g. llm_args.model
        # was a Path), it gets normalized too.
        assert _resolve_mx_model_name("/scratch/explicit-path", "/cache/ignored") == "explicit-path"


class TestPublishAsSourceModelName:
    """``publish_as_source`` must set ``MODEL_NAME`` from the resolved
    identity so upstream's ``publish_model_params`` reads it via env,
    and must restore the prior env value afterwards.
    """

    @pytest.fixture(autouse=True)
    def _isolated_env(self, monkeypatch):
        monkeypatch.delenv("MODEL_NAME", raising=False)
        monkeypatch.delenv("MODEL_EXPRESS_URL", raising=False)
        monkeypatch.delenv("MX_SOURCE_QUERY_TIMEOUT", raising=False)
        yield

    def test_uses_explicit_constructor_model_name(self):
        loader = MXCheckpointLoader(
            mx_server_url="http://mx:8001",
            model_name="Qwen/Qwen2.5-72B-Instruct",
        )
        captured = {}

        def _capture(model):
            captured["MODEL_NAME"] = os.environ.get("MODEL_NAME")
            captured["MODEL_EXPRESS_URL"] = os.environ.get("MODEL_EXPRESS_URL")

        fake_mx = _build_fake_modelexpress(publish_side_effect=_capture)
        with _install_fake_modelexpress(fake_mx):
            loader.publish_as_source(MagicMock())

        assert captured["MODEL_NAME"] == "Qwen/Qwen2.5-72B-Instruct"
        assert captured["MODEL_EXPRESS_URL"] == "http://mx:8001"
        # Both env vars restored to the (unset) prior state.
        assert "MODEL_NAME" not in os.environ
        assert "MODEL_EXPRESS_URL" not in os.environ

    def test_falls_back_to_checkpoint_dir_basename(self):
        loader = MXCheckpointLoader(mx_server_url="http://mx:8001")
        # No constructor model_name, no MODEL_NAME env → use basename.
        captured = {}

        def _capture(model):
            captured["MODEL_NAME"] = os.environ.get("MODEL_NAME")

        fake_mx = _build_fake_modelexpress(publish_side_effect=_capture)
        with _install_fake_modelexpress(fake_mx):
            loader.publish_as_source(MagicMock(), checkpoint_dir="/scratch/local-model")

        assert captured["MODEL_NAME"] == "local-model"

    def test_unmangles_hf_snapshot_path(self):
        loader = MXCheckpointLoader(mx_server_url="http://mx:8001")
        snapshot = (
            "/cache/huggingface/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/abc123def456"
        )
        captured = {}

        def _capture(model):
            captured["MODEL_NAME"] = os.environ.get("MODEL_NAME")

        fake_mx = _build_fake_modelexpress(publish_side_effect=_capture)
        with _install_fake_modelexpress(fake_mx):
            loader.publish_as_source(MagicMock(), checkpoint_dir=snapshot)

        # Critical: NOT the commit hash, the human-readable Hub-ID form.
        assert captured["MODEL_NAME"] == "Qwen/Qwen2.5-72B-Instruct"
        assert captured["MODEL_NAME"] != "abc123def456"

    def test_constructor_model_name_takes_priority_over_env(self, monkeypatch):
        # Explicit constructor value > existing MODEL_NAME env.
        monkeypatch.setenv("MODEL_NAME", "from-env")
        loader = MXCheckpointLoader(
            mx_server_url="http://mx:8001",
            model_name="explicit",
        )
        captured = {}

        def _capture(model):
            captured["MODEL_NAME"] = os.environ.get("MODEL_NAME")

        fake_mx = _build_fake_modelexpress(publish_side_effect=_capture)
        with _install_fake_modelexpress(fake_mx):
            loader.publish_as_source(MagicMock())

        assert captured["MODEL_NAME"] == "explicit"
        # Restored to the prior env value, not unset.
        assert os.environ.get("MODEL_NAME") == "from-env"

    def test_env_used_when_no_constructor_value(self, monkeypatch):
        monkeypatch.setenv("MODEL_NAME", "from-env-only")
        loader = MXCheckpointLoader(mx_server_url="http://mx:8001")
        captured = {}

        def _capture(model):
            captured["MODEL_NAME"] = os.environ.get("MODEL_NAME")

        fake_mx = _build_fake_modelexpress(publish_side_effect=_capture)
        with _install_fake_modelexpress(fake_mx):
            loader.publish_as_source(MagicMock())

        assert captured["MODEL_NAME"] == "from-env-only"
        assert os.environ.get("MODEL_NAME") == "from-env-only"

    def test_unknown_when_all_sources_missing(self):
        # No constructor, no env, no checkpoint_dir → upstream sentinel.
        loader = MXCheckpointLoader(mx_server_url="http://mx:8001")
        captured = {}

        def _capture(model):
            captured["MODEL_NAME"] = os.environ.get("MODEL_NAME")

        fake_mx = _build_fake_modelexpress(publish_side_effect=_capture)
        with _install_fake_modelexpress(fake_mx):
            loader.publish_as_source(MagicMock())

        assert captured["MODEL_NAME"] == "unknown"
