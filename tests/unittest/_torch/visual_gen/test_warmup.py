# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for VisualGen warmup configuration, plan resolution, and shape validation."""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from tensorrt_llm._torch.visual_gen.config import CompilationConfig, VisualGenArgs
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline


class TestCompilationConfig:
    """CompilationConfig construction and validation."""

    def test_default_values(self):
        cfg = CompilationConfig()
        assert cfg.resolutions is None
        assert cfg.num_frames is None

    def test_explicit_values(self):
        cfg = CompilationConfig(
            resolutions=[(480, 832), (720, 1280)],
            num_frames=[33, 81],
        )
        assert len(cfg.resolutions) == 2
        assert cfg.num_frames == [33, 81]

    def test_empty_lists(self):
        cfg = CompilationConfig(resolutions=[], num_frames=[])
        assert cfg.resolutions == []
        assert cfg.num_frames == []

    def test_partial_resolutions_only(self):
        cfg = CompilationConfig(resolutions=[(1920, 1080)])
        assert cfg.resolutions == [(1920, 1080)]
        assert cfg.num_frames is None

    def test_partial_num_frames_only(self):
        cfg = CompilationConfig(num_frames=[81])
        assert cfg.resolutions is None
        assert cfg.num_frames == [81]

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            CompilationConfig(resolutions=[(480, 832)], bad_field=True)


class TestVisualGenArgsWarmup:
    """CompilationConfig integration with VisualGenArgs."""

    def test_default_warmup(self):
        args = VisualGenArgs(checkpoint_path="/tmp/model")
        assert isinstance(args.compilation, CompilationConfig)
        assert args.compilation.resolutions is None
        assert args.compilation.num_frames is None

    def test_warmup_from_dict(self):
        args = VisualGenArgs(
            checkpoint_path="/tmp/model",
            compilation={"resolutions": [(480, 832)], "num_frames": [33, 81]},
        )
        assert args.compilation.resolutions == [(480, 832)]
        assert args.compilation.num_frames == [33, 81]

    def test_warmup_from_yaml(self, tmp_path):
        yaml_path = tmp_path / "config.yml"
        yaml_path.write_text(
            "checkpoint_path: /tmp/model\n"
            "compilation:\n"
            "  resolutions:\n"
            "    - [480, 832]\n"
            "    - [720, 1280]\n"
            "  num_frames: [33, 81]\n"
        )
        args = VisualGenArgs.from_yaml(yaml_path)
        assert len(args.compilation.resolutions) == 2
        assert args.compilation.num_frames == [33, 81]

    def test_warmup_pickle_roundtrip(self):
        import pickle

        args = VisualGenArgs(
            checkpoint_path="/tmp/model",
            compilation=CompilationConfig(resolutions=[(480, 832)], num_frames=[33]),
        )
        restored = pickle.loads(pickle.dumps(args))
        assert restored.compilation.resolutions == [(480, 832)]
        assert restored.compilation.num_frames == [33]


# ---------------------------------------------------------------------------
# Stub pipelines for testing resolve_warmup_plan() and validate_resolution()
# without loading a real model.
# ---------------------------------------------------------------------------


class _BaseStubPipeline(BasePipeline):
    """Minimal stub that skips real __init__ (no DiffusionModelConfig needed)."""

    def __init__(self, warmup_cfg):
        self._warmed_up_shapes = set()
        self.model_config = MagicMock()
        self.model_config.compilation = warmup_cfg or CompilationConfig()

    def forward(self, *args, **kwargs):
        pass

    def _init_transformer(self):
        pass

    def infer(self, req):
        pass

    def _run_warmup(self, height, width, num_frames, steps):
        pass


def _make_stub_pipeline(warmup_cfg=None):
    """Wan-like stub: resolution must be multiple of 16, (num_frames-1) % 4 == 0."""

    class WanStub(_BaseStubPipeline):
        @property
        def default_warmup_resolutions(self):
            return [(480, 832), (720, 1280)]

        @property
        def default_warmup_num_frames(self):
            return [33, 81]

        @property
        def resolution_multiple_of(self):
            return (16, 16)

    return WanStub(warmup_cfg)


def _make_flux_stub_pipeline(warmup_cfg=None):
    """Flux-like stub: image only, num_frames must be 1."""

    class FluxStub(_BaseStubPipeline):
        @property
        def default_warmup_resolutions(self):
            return [(1024, 1024)]

        @property
        def default_warmup_num_frames(self):
            return [1]

    return FluxStub(warmup_cfg)


class TestResolveWarmupPlan:
    """resolve_warmup_plan() Cartesian product and priority logic."""

    def test_default_no_user_config(self):
        """No user config → model defaults → Cartesian product."""
        pipe = _make_stub_pipeline()
        shapes, steps = pipe.resolve_warmup_plan()
        assert steps == 2
        assert len(shapes) == 4  # 2 resolutions x 2 frame counts
        assert (480, 832, 33) in shapes
        assert (480, 832, 81) in shapes
        assert (720, 1280, 33) in shapes
        assert (720, 1280, 81) in shapes

    def test_user_resolutions_only(self):
        """User specifies resolutions, num_frames from model default."""
        cfg = CompilationConfig(resolutions=[(1920, 1088)])
        pipe = _make_stub_pipeline(cfg)
        shapes, _ = pipe.resolve_warmup_plan()
        assert len(shapes) == 2  # 1 resolution x 2 default frame counts
        assert (1920, 1088, 33) in shapes
        assert (1920, 1088, 81) in shapes

    def test_user_num_frames_only(self):
        """User specifies num_frames, resolutions from model default."""
        cfg = CompilationConfig(num_frames=[81])
        pipe = _make_stub_pipeline(cfg)
        shapes, _ = pipe.resolve_warmup_plan()
        assert len(shapes) == 2  # 2 default resolutions x 1 frame count
        assert (480, 832, 81) in shapes
        assert (720, 1280, 81) in shapes

    def test_user_both(self):
        """User specifies both → user values Cartesian product."""
        cfg = CompilationConfig(
            resolutions=[(256, 256)],
            num_frames=[5],
        )
        pipe = _make_stub_pipeline(cfg)
        shapes, _ = pipe.resolve_warmup_plan()
        assert shapes == [(256, 256, 5)]

    def test_empty_resolutions_skips_warmup(self):
        """Empty resolutions → no shapes → warmup skipped."""
        cfg = CompilationConfig(resolutions=[], num_frames=[33])
        pipe = _make_stub_pipeline(cfg)
        shapes, _ = pipe.resolve_warmup_plan()
        assert shapes == []

    def test_empty_num_frames_skips_warmup(self):
        """Empty num_frames → no shapes → warmup skipped."""
        cfg = CompilationConfig(resolutions=[(480, 832)], num_frames=[])
        pipe = _make_stub_pipeline(cfg)
        shapes, _ = pipe.resolve_warmup_plan()
        assert shapes == []

    def test_invalid_user_resolution_skipped(self):
        """User resolution not multiple of 16 → warning, shape skipped."""
        cfg = CompilationConfig(resolutions=[(481, 832)])
        pipe = _make_stub_pipeline(cfg)
        shapes, _ = pipe.resolve_warmup_plan()
        assert (481, 832, 33) not in shapes
        assert (481, 832, 81) not in shapes

    def test_any_user_num_frames_accepted(self):
        """Any num_frames accepted in warmup (rounded in forward)."""
        cfg = CompilationConfig(num_frames=[30])
        pipe = _make_stub_pipeline(cfg)
        shapes, _ = pipe.resolve_warmup_plan()
        assert len(shapes) == 2  # 2 resolutions x 1 frame count

    def test_flux_default(self):
        """Flux default: 1 resolution x 1 frame = 1 shape."""
        pipe = _make_flux_stub_pipeline()
        shapes, steps = pipe.resolve_warmup_plan()
        assert shapes == [(1024, 1024, 1)]
        assert steps == 2

    def test_flux_any_num_frames_accepted(self):
        """Flux: any num_frames accepted in warmup (no frame validation)."""
        cfg = CompilationConfig(num_frames=[81])
        pipe = _make_flux_stub_pipeline(cfg)
        shapes, _ = pipe.resolve_warmup_plan()
        assert len(shapes) == 1  # 1 resolution x 1 frame count


class TestValidateShape:
    """validate_resolution() model constraint checks."""

    def test_valid_wan_shape(self):
        pipe = _make_stub_pipeline()
        pipe.validate_resolution(480, 832, 33)
        pipe.validate_resolution(720, 1280, 81)

    def test_invalid_resolution_height(self):
        pipe = _make_stub_pipeline()
        with pytest.raises(ValueError, match="must be multiples of"):
            pipe.validate_resolution(481, 832, 33)

    def test_invalid_resolution_width(self):
        pipe = _make_stub_pipeline()
        with pytest.raises(ValueError, match="must be multiples of"):
            pipe.validate_resolution(480, 833, 33)

    def test_any_frame_count_accepted(self):
        """validate_resolution does not check frame counts (silently rounded in forward)."""
        pipe = _make_stub_pipeline()
        pipe.validate_resolution(480, 832, 30)
        pipe.validate_resolution(480, 832, 1)
        pipe.validate_resolution(480, 832, 8)


class TestWarmupExecution:
    """warmup() integration: shapes recorded in _warmed_up_shapes."""

    def test_warmup_records_shapes(self):
        pipe = _make_stub_pipeline()
        pipe.warmup()
        assert len(pipe._warmed_up_shapes) == 4
        assert (480, 832, 33) in pipe._warmed_up_shapes
        assert (720, 1280, 81) in pipe._warmed_up_shapes

    def test_warmup_empty_shapes_skips(self):
        cfg = CompilationConfig(resolutions=[], num_frames=[])
        pipe = _make_stub_pipeline(cfg)
        pipe.warmup()
        assert pipe._warmed_up_shapes == set()

    def test_warmup_user_shapes_recorded(self):
        cfg = CompilationConfig(resolutions=[(480, 832)], num_frames=[33])
        pipe = _make_stub_pipeline(cfg)
        pipe.warmup()
        assert pipe._warmed_up_shapes == {(480, 832, 33)}


class TestRequestValidation:
    """Request-level validation: validate_resolution() + _warmed_up_shapes check."""

    def test_valid_warmed_shape_no_warning(self, caplog):
        """Request with a warmed-up shape → no warning."""
        pipe = _make_stub_pipeline()
        pipe.warmup()

        req = MagicMock()
        req.height, req.width, req.num_frames = 480, 832, 33

        pipe.validate_resolution(req.height, req.width, req.num_frames)
        shape = (req.height, req.width, req.num_frames)
        assert shape in pipe._warmed_up_shapes

    def test_valid_unwarmed_shape_warns(self):
        """Request with valid but un-warmed shape → pipeline accepts but would warn."""
        cfg = CompilationConfig(resolutions=[(480, 832)], num_frames=[33])
        pipe = _make_stub_pipeline(cfg)
        pipe.warmup()

        pipe.validate_resolution(720, 1280, 81)
        assert (720, 1280, 81) not in pipe._warmed_up_shapes

    def test_invalid_shape_raises_before_warmup_check(self):
        """Invalid shape → raises ValueError, never reaches warmup check."""
        pipe = _make_stub_pipeline()
        pipe.warmup()

        with pytest.raises(ValueError, match="must be multiples of"):
            pipe.validate_resolution(481, 832, 33)
