"""Unit tests for autotuner bounds checking.

These tests verify that _find_nearest_profile and _optimization_profiles
gracefully handle DynamicTensorSpec / ConstraintSpec entries whose
input_idx or dim_idx exceed the actual tensor dimensions, instead of
raising an IndexError.
"""

import logging

import torch

from tensorrt_llm._torch.autotuner import AutoTuner, ConstraintSpec, DynamicTensorSpec, TuningConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_shapes(*sizes):
    """Convert a list of size-tuples into the Tuple[torch.Size, ...] that
    _find_nearest_profile expects."""
    return tuple(torch.Size(s) for s in sizes)


# ---------------------------------------------------------------------------
# Tests for _find_nearest_profile
# ---------------------------------------------------------------------------


class TestFindNearestProfileBounds:
    """Bounds-checking in AutoTuner._find_nearest_profile."""

    def setup_method(self):
        # Clear the lru_cache between tests so each call actually executes.
        AutoTuner._find_nearest_profile.cache_clear()

    # -- DynamicTensorSpec ------------------------------------------------

    def test_dynamic_spec_input_idx_out_of_range(self, caplog):
        """input_idx exceeds the number of inputs => spec is skipped."""
        shapes = _make_shapes([4, 8])  # 1 tensor, 2 dims
        spec = DynamicTensorSpec(input_idx=5, dim_idx=0, gen_tuning_buckets=(1,))

        with caplog.at_level(logging.DEBUG, logger="tensorrt_llm"):
            result = AutoTuner._find_nearest_profile(
                shapes,
                dynamic_tensor_specs=(spec,),
                constraint_specs=(),
            )

        # Profile should be returned unchanged (no crash).
        assert result == ((4, 8),)
        assert any("Skipping DynamicTensorSpec with input_idx=5" in m for m in caplog.messages)

    def test_dynamic_spec_dim_idx_out_of_range(self, caplog):
        """dim_idx exceeds the tensor dimensions => spec is skipped."""
        shapes = _make_shapes([4, 8])
        spec = DynamicTensorSpec(input_idx=0, dim_idx=10, gen_tuning_buckets=(1,))

        with caplog.at_level(logging.DEBUG, logger="tensorrt_llm"):
            result = AutoTuner._find_nearest_profile(
                shapes,
                dynamic_tensor_specs=(spec,),
                constraint_specs=(),
            )

        assert result == ((4, 8),)
        assert any("Skipping DynamicTensorSpec with dim_idx=10" in m for m in caplog.messages)

    # -- ConstraintSpec ---------------------------------------------------

    def test_constraint_spec_input_idx_out_of_range(self, caplog):
        """ConstraintSpec.input_idx exceeds inputs => spec is skipped."""
        shapes = _make_shapes([4, 8])
        spec = ConstraintSpec(input_idx=3, dim_idx=0, infer_shape=lambda shapes: 1)

        with caplog.at_level(logging.DEBUG, logger="tensorrt_llm"):
            result = AutoTuner._find_nearest_profile(
                shapes,
                dynamic_tensor_specs=(),
                constraint_specs=(spec,),
            )

        assert result == ((4, 8),)
        assert any("Skipping ConstraintSpec with input_idx=3" in m for m in caplog.messages)

    def test_constraint_spec_dim_idx_out_of_range(self, caplog):
        """ConstraintSpec.dim_idx exceeds tensor dims => spec is skipped."""
        shapes = _make_shapes([4, 8])
        spec = ConstraintSpec(input_idx=0, dim_idx=7, infer_shape=lambda shapes: 1)

        with caplog.at_level(logging.DEBUG, logger="tensorrt_llm"):
            result = AutoTuner._find_nearest_profile(
                shapes,
                dynamic_tensor_specs=(),
                constraint_specs=(spec,),
            )

        assert result == ((4, 8),)
        assert any("Skipping ConstraintSpec with dim_idx=7" in m for m in caplog.messages)

    # -- Valid specs still work -------------------------------------------

    def test_valid_specs_unaffected(self):
        """Specs within range should still be applied normally."""
        shapes = _make_shapes([4, 8], [2, 3])
        dyn = DynamicTensorSpec(input_idx=0, dim_idx=1)
        con = ConstraintSpec(input_idx=1, dim_idx=0, infer_shape=lambda shapes: 1)

        result = AutoTuner._find_nearest_profile(
            shapes,
            dynamic_tensor_specs=(dyn,),
            constraint_specs=(con,),
        )

        # DynamicTensorSpec applies identity map => dim stays 8.
        # ConstraintSpec sets dim to -1.
        assert result == ((4, 8), (-1, 3))


# ---------------------------------------------------------------------------
# Tests for _optimization_profiles
# ---------------------------------------------------------------------------


class TestOptimizationProfilesBounds:
    """Bounds-checking in AutoTuner._optimization_profiles."""

    def test_dynamic_spec_input_idx_oob_skipped(self, caplog):
        """DynamicTensorSpec referencing a non-existent input is skipped."""
        tuner = AutoTuner()
        x = torch.rand([4, 8])
        config = TuningConfig(
            dynamic_tensor_specs=(
                DynamicTensorSpec(input_idx=5, dim_idx=0, gen_tuning_buckets=(1, 2)),
            ),
        )

        with caplog.at_level(logging.DEBUG, logger="tensorrt_llm"):
            profiles = tuner._optimization_profiles(config, [x])

        # Should produce exactly 1 profile (the base profile) with no crash.
        assert len(profiles) == 1
        assert profiles[0].get_opt_shapes() == ((4, 8),)

    def test_constraint_spec_dim_idx_oob_skipped(self, caplog):
        """ConstraintSpec referencing a non-existent dim is skipped."""
        tuner = AutoTuner()
        x = torch.rand([4, 8])
        config = TuningConfig(
            dynamic_tensor_specs=(
                DynamicTensorSpec(input_idx=0, dim_idx=0, gen_tuning_buckets=(2, 4)),
            ),
            constraint_specs=(
                ConstraintSpec(input_idx=0, dim_idx=9, infer_shape=lambda shapes: 1),
            ),
        )

        with caplog.at_level(logging.DEBUG, logger="tensorrt_llm"):
            profiles = tuner._optimization_profiles(config, [x])

        # Profiles should be generated; constraint is silently skipped.
        assert len(profiles) > 0
        assert any("Skipping ConstraintSpec with dim_idx=9" in m for m in caplog.messages)
