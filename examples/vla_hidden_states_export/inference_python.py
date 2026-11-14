# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Minimal example: read full_hidden_states from a TRT-LLM engine at inference time.

This shows the core usage pattern for VLA models:
1. Run LLM engine inference
2. Extract the full hidden_states from engine output
3. Select the hidden state at a specific token position (e.g., waypoint token)
4. Feed to downstream planning head
"""
import argparse
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class HiddenStatesEngine:
    """Minimal TRT engine wrapper that reads full_hidden_states output."""

    def __init__(self, engine_path: str) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

        self.engine = engine
        self.context = engine.create_execution_context()
        self.stream = cuda.Stream()

        self.input_names = []
        self.output_names = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        if "full_hidden_states" not in self.output_names:
            raise ValueError(
                f"full_hidden_states not in engine outputs: {self.output_names}. "
                "Apply the mark_output patch first."
            )
        print(f"Engine loaded: {len(self.input_names)} inputs, {len(self.output_names)} outputs")

    def infer_and_extract_hidden_states(
        self,
        input_ids: np.ndarray,
        waypoint_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run inference and extract hidden_states at the waypoint token position.

        Args:
            input_ids: Input token IDs (numpy int32 array, shape [batch, seq_len])
            waypoint_idx: Position in the sequence to extract hidden state from.

        Returns:
            full_hidden_states: Complete hidden_states tensor.
            ego_feature: Hidden state at waypoint position [hidden_dim].
        """
        buffers = {}

        for name in self.input_names:
            if name == "input_ids":
                data = np.ascontiguousarray(input_ids.astype(np.int32))
                self.context.set_input_shape(name, data.shape)
            else:
                shape = tuple(max(1, s) for s in self.engine.get_tensor_shape(name))
                data = np.zeros(shape, dtype=np.float32)

            d = cuda.mem_alloc(data.nbytes)
            cuda.memcpy_htod(d, data)
            self.context.set_tensor_address(name, int(d))
            buffers[name] = d

        for name in self.output_names:
            engine_shape = tuple(self.engine.get_tensor_shape(name))
            shape = tuple(max(1, s) for s in engine_shape)
            n_elements = int(np.prod(shape))
            d = cuda.mem_alloc(n_elements * 2)  # fp16
            self.context.set_tensor_address(name, int(d))
            buffers[name] = d

        self.context.execute_async_v3(self.stream.handle)
        self.stream.synchronize()

        hs_shape = tuple(
            max(1, s) for s in self.engine.get_tensor_shape("full_hidden_states")
        )
        full_hs = np.empty(int(np.prod(hs_shape)), dtype=np.float16)
        cuda.memcpy_dtoh(full_hs, buffers["full_hidden_states"])
        full_hs = full_hs.reshape(hs_shape)

        if full_hs.ndim == 3:  # [batch, seq_len, hidden_dim]
            wp_idx = min(waypoint_idx, full_hs.shape[1] - 1)
            ego_feature = full_hs[0, wp_idx, :].copy()
        elif full_hs.ndim == 2:  # [num_tokens, hidden_dim] (packed, remove_input_padding)
            wp_idx = min(waypoint_idx, full_hs.shape[0] - 1)
            ego_feature = full_hs[wp_idx, :].copy()
        else:
            ego_feature = full_hs[-1, :].copy()

        for d in buffers.values():
            d.free()

        return full_hs, ego_feature


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read full_hidden_states from TRT-LLM engine"
    )
    parser.add_argument(
        "--engine_path", required=True, help="Path to rank0.engine"
    )
    parser.add_argument(
        "--seq_len", type=int, default=599,
        help="Sequence length for dummy input (default: 599)",
    )
    parser.add_argument(
        "--waypoint_idx", type=int, default=598,
        help="Token position to extract (default: last token)",
    )
    args = parser.parse_args()

    engine = HiddenStatesEngine(args.engine_path)
    input_ids = np.ones((1, args.seq_len), dtype=np.int32)
    full_hs, ego_feature = engine.infer_and_extract_hidden_states(
        input_ids, args.waypoint_idx
    )

    print(f"\nfull_hidden_states shape: {full_hs.shape}")
    print(f"ego_feature (idx={args.waypoint_idx}): {ego_feature.shape}")
    print(f"ego_feature[:6]: {ego_feature[:6]}")


if __name__ == "__main__":
    main()
