# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Verify full_hidden_states output from a TRT-LLM engine.

Usage:
    python tests/test_hidden_states.py --engine_path /path/to/rank0.engine
"""
import argparse
import numpy as np


def test_engine_has_full_hidden_states(engine_path: str) -> None:
    """Verify the engine exports full_hidden_states tensor."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

    output_names = [
        engine.get_tensor_name(i)
        for i in range(engine.num_io_tensors)
        if engine.get_tensor_mode(engine.get_tensor_name(i))
        == trt.TensorIOMode.OUTPUT
    ]

    if "full_hidden_states" not in output_names:
        raise AssertionError(
            f"full_hidden_states not in outputs: {output_names}. "
            "Apply the mark_output patch first."
        )
    print(f"PASS: full_hidden_states found in {output_names}")


def test_hidden_states_is_3d(engine_path: str) -> None:
    """Verify full_hidden_states has rank >= 2 (3D or packed 2D)."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

    shape = tuple(engine.get_tensor_shape("full_hidden_states"))
    if len(shape) < 2:
        raise AssertionError(
            f"full_hidden_states should be at least 2D, got shape {shape}."
        )
    print(f"PASS: full_hidden_states shape {shape} (rank {len(shape)})")


def test_token_extraction() -> None:
    """Verify token extraction from a simulated full hidden_states tensor."""
    seq_len = 599
    hidden_dim = 4096
    full_hs = np.random.randn(1, seq_len, hidden_dim).astype(np.float16)

    wp_idx = 42
    ego_feature = full_hs[0, wp_idx, :].copy()

    assert ego_feature.shape == (hidden_dim,)
    assert np.array_equal(ego_feature, full_hs[0, wp_idx, :])
    assert not np.array_equal(ego_feature, full_hs[0, wp_idx + 1, :])
    print(f"PASS: token extraction at idx={wp_idx}")


def main() -> None:
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine_path", required=True)
    args = parser.parse_args()

    test_engine_has_full_hidden_states(args.engine_path)
    test_hidden_states_is_3d(args.engine_path)
    test_token_extraction()
    print("\nAll tests passed!")


if __name__ == "__main__":
    import sys
    if "--engine_path" in sys.argv:
        main()
    else:
        test_token_extraction()
        print("\nRun with --engine_path to test engine output")
