import base64
import functools
import json
import zlib
from enum import Enum
from typing import Optional

import nvtx
import torch

from tensorrt_llm.logger import logger


class Mode(Enum):
    NONE = 1
    MARK = 2
    COLLECT = 3
    REPLAY = 4


class Calibrator:
    """Calibrator for layer-wise benchmarks with MoE expert routing data.

    The calibrator operates in one of the following modes:
        NONE:    Disabled, no-op.
        MARK:    Add NVTX markers for correlating E2E and layer-wise benchmarks.
        COLLECT: Collect `token_selected_slots` data and save to a file.
        REPLAY:  Load `token_selected_slots` data from a file for replay.

    Lifecycle:
        init() -> maybe_wrap_model() -> [CUDA Graph capture] -> start() ->
        [pre_step() -> forward() -> post_step()] x N -> stop()

    Design Notes:
        To ensure CUDA Graphs compatibility, `token_selected_slots` tensors are
        copied to a fixed GPU buffer during graph capture/replay. The actual data
        is then transferred to per-iteration storage in `post_step()`.

        Since `model.forward()` does not execute during CUDA Graphs replay, we
        cannot access Python metadata objects directly. Instead, we copy an integer
        index (`metadata_idx`) pointing to a pre-built metadata list.

        For REPLAY mode, metadata verification is deferred to `stop()` to avoid
        GPU synchronization during profiling. Metadata indices are recorded on GPU
        in `post_step()` and compared after `cuda.synchronize()` in `stop()`.

    Tensor Locations (COLLECT mode):
        - `_metadata_idx_gpu`:              GPU scalar, current iteration's index into `_metadata_list`
        - `_metadata_idx_range_gpu`:        GPU [0..MAX_NUM_METADATA), pre-generated indices for GPU copy
        - `_slots_buffer_gpu`:              GPU, fixed buffer for CUDA Graphs compatibility
        - `_collected_metadata_idx`:        GPU [MAX_COLLECT_ITERATIONS], per-iteration metadata indices
        - `_collected_slots_cpu`:           CPU (pinned), per-iteration slots data
        - `_eager_slots_gpu`:               GPU, list of tensors for non-CG (context) phase

    Tensor Locations (REPLAY mode):
        - `_actual_metadata_idx_gpu`:           GPU scalar, current iteration's actual metadata index
        - `_metadata_idx_range_gpu`:            GPU [0..MAX_NUM_METADATA), pre-generated indices
        - `_slots_buffer_gpu`:                  GPU, fixed buffer for CUDA Graphs compatibility
        - `_collected_actual_metadata_idx`:     GPU [MAX_REPLAY_ITERATIONS], actual metadata indices
        - `_collected_iterations`:              Python list, iterations for expected metadata lookup
        - `_replay_eager_slots_gpu`:            GPU, eager slots data for large context phase
    """

    # Maximum number of int32 elements per iteration for CUDA Graphs buffer
    MAX_SLOTS_BUFFER_SIZE = 4 * 1024 * 1024
    # Maximum number of metadata supported
    MAX_NUM_METADATA = 1024 * 1024
    # Maximum iterations during collect phase
    MAX_COLLECT_ITERATIONS = 101
    # Maximum iterations during replay phase
    MAX_REPLAY_ITERATIONS = 1024 * 1024
    # Data type for token_selected_slots
    SLOTS_DTYPE = torch.int32

    def __init__(self):
        self.mode = Mode.NONE
        self._started = False

    def init(
        self,
        mode: str,
        file_path: str,
        layer_indices: list[int],
        *,
        replay_verify_metadata: Optional[bool] = None,
        mapping=None,
        dist=None,
    ):
        """Initialize the calibrator.

        Args:
            mode: One of "NONE", "MARK", "COLLECT", "REPLAY".
            file_path: Path to the calibration data file.
            layer_indices: Optional list of layer indices to filter.
                          COLLECT mode: If None, all layers are collected.
                          REPLAY mode: Cannot be None.
            replay_verify_metadata: Whether to verify actual metadata in REPLAY mode matches calibration data.
            mapping: Tensor parallel mapping containing rank and world_size.
            dist: Distributed communication wrapper.
        """
        if self.mode != Mode.NONE:
            raise ValueError("double init")

        self.mode = Mode[mode]

        if self.mode == Mode.COLLECT:
            self._init_collect_mode(file_path, layer_indices, dist)

        if self.mode == Mode.REPLAY:
            if replay_verify_metadata is None:
                raise ValueError("missing replay_verify_metadata")
            self._init_replay_mode(file_path, layer_indices, replay_verify_metadata, mapping)

    def _init_collect_mode(self, file_path, layer_indices, dist):
        """Initialize buffers for COLLECT mode."""
        self._file_path = file_path
        self._layer_indices = layer_indices
        self._dist = dist

        # Metadata list that `_metadata_idx_gpu` indexes into
        self._metadata_list = []

        # GPU buffers for CUDA Graphs compatibility:
        #   - Copy from `_metadata_idx_range_gpu[idx]` to `_metadata_idx_gpu`
        #   - Copy flattened slots to `_slots_buffer_gpu`
        self._metadata_idx_gpu = torch.empty((), dtype=torch.long, device="cuda")
        self._metadata_idx_range_gpu = torch.arange(
            self.MAX_NUM_METADATA, dtype=torch.long, device="cuda"
        )
        self._slots_buffer_gpu = torch.empty(
            self.MAX_SLOTS_BUFFER_SIZE,
            dtype=self.SLOTS_DTYPE,
            device="cuda",
        )

    def _init_replay_mode(self, file_path, layer_indices, replay_verify_metadata, mapping):
        """Initialize replay database from file."""
        with open(file_path) as f:
            data = json.load(f)

        if data["world_size"] != mapping.world_size:
            raise ValueError(
                f"World size mismatch: file has {data['world_size']}, "
                f"but current world_size is {mapping.world_size}"
            )

        # Verify all ranks have the same iterations (using data from file, not dist)
        all_ranks_iterations = [
            [record["iteration"] for record in rank_records]
            for rank_records in data["all_ranks_records"]
        ]
        if any(iters != all_ranks_iterations[0] for iters in all_ranks_iterations):
            raise ValueError("Iterations mismatch across ranks in calibration file")

        start_layer_idx = layer_indices[0]
        end_layer_idx = layer_indices[-1]
        if list(layer_indices) != list(range(start_layer_idx, end_layer_idx + 1)):
            raise ValueError("Invalid layer_indices")

        self._replay_db = {}
        for record in data["all_ranks_records"][mapping.rank]:
            iteration = record["iteration"]
            metadata = record["metadata"]
            raw_data = torch.frombuffer(
                zlib.decompress(base64.b64decode(record["raw_data"])),
                dtype=self.SLOTS_DTYPE,
            )

            # Filter metadata for target layers
            start_idx = 0
            while metadata[start_idx]["layer_idx"] < start_layer_idx:
                start_idx += 1
            end_idx = start_idx
            while end_idx < len(metadata) and metadata[end_idx]["layer_idx"] <= end_layer_idx:
                end_idx += 1

            # Calculate data offsets for filtered range
            offset = sum(
                torch.Size(m["token_selected_slots_shape"]).numel() for m in metadata[:start_idx]
            )
            length = sum(
                torch.Size(m["token_selected_slots_shape"]).numel()
                for m in metadata[start_idx:end_idx]
            )
            filtered_slots = raw_data[offset : offset + length]

            self._replay_db[iteration] = {
                "metadata": metadata[start_idx:end_idx],
                "slots_data_gpu": filtered_slots.to("cuda"),
            }

        # Placeholder buffer for CUDA Graphs (actual data copied in pre_step)
        self.MAX_TOPK_AND_MIN_SLOTS = 32
        self._slots_buffer_gpu = (
            torch.arange(
                self.MAX_SLOTS_BUFFER_SIZE,
                dtype=self.SLOTS_DTYPE,
                device="cuda",
            )
            % self.MAX_TOPK_AND_MIN_SLOTS
        )
        self._use_eager_mode = False

        # GPU buffers for deferred metadata verification (must be created before CUDA Graph capture)
        # These are used during forward() which may be captured into CUDA Graphs
        self._replay_verify_metadata = replay_verify_metadata
        if self._replay_verify_metadata:
            self._actual_metadata_list = []
            self._actual_metadata_idx_gpu = torch.empty((), dtype=torch.long, device="cuda")
            self._metadata_idx_range_gpu = torch.arange(
                self.MAX_NUM_METADATA, dtype=torch.long, device="cuda"
            )

    def maybe_wrap_model(self, model):
        """Wrap model and layer forward methods for data collection/replay.

        Args:
            model: The model to wrap.

        Returns:
            The wrapped model.
        """
        if self.mode == Mode.COLLECT:
            self._wrap_model_forward_for_collect(model)

        if self.mode == Mode.REPLAY:
            self._wrap_model_forward_for_replay(model)

        # Wrap layer forward methods for all active modes
        if self.mode in [Mode.MARK, Mode.COLLECT, Mode.REPLAY]:
            self._wrap_layer_forward(model)

        return model

    def _wrap_model_forward_for_collect(self, model):
        """Wrap model.forward() for COLLECT mode."""

        def make_forward(forward_orig):
            @functools.wraps(forward_orig)
            def forward(*args, **kwargs):
                # Initialize per-iteration collection buffer
                self._cur_iter_records = []

                output = forward_orig(*args, **kwargs)

                # Build metadata from collected records
                metadata = [
                    {
                        "layer_idx": layer_idx,
                        "num_slots": num_slots,
                        "token_selected_slots_shape": list(slots.shape),
                    }
                    for layer_idx, num_slots, slots in self._cur_iter_records
                ]

                # Store metadata and copy index to GPU (for CUDA Graphs compatibility)
                metadata_idx = len(self._metadata_list)
                self._metadata_list.append(metadata)
                if metadata_idx >= len(self._metadata_idx_range_gpu):
                    raise ValueError(
                        f"Metadata index overflow: {metadata_idx} >= {len(self._metadata_idx_range_gpu)}. "
                        f"Increase MAX_NUM_METADATA if more iterations are needed."
                    )
                self._metadata_idx_gpu.copy_(self._metadata_idx_range_gpu[metadata_idx])

                # Copy slots data to fixed buffer or use eager mode
                total_size = sum(slots.numel() for _, _, slots in self._cur_iter_records)

                if total_size <= self.MAX_SLOTS_BUFFER_SIZE:
                    # Small data: copy to fixed GPU buffer (CUDA Graphs compatible)
                    if self._cur_iter_records:
                        torch.cat(
                            [slots.flatten() for _, _, slots in self._cur_iter_records],
                            out=self._slots_buffer_gpu[:total_size],
                        )
                else:
                    # Large data (context phase): use eager collection
                    # Context phase does not use CUDA Graphs, so this is safe
                    assert not torch.cuda.is_current_stream_capturing()
                    self._use_eager_mode = True
                    self._eager_slots_gpu = [
                        slots.flatten() for _, _, slots in self._cur_iter_records
                    ]

                del self._cur_iter_records
                return output

            return forward

        model.forward = make_forward(model.forward)

    def _wrap_model_forward_for_replay(self, model):
        """Wrap model.forward() for REPLAY mode."""

        def make_forward(forward_orig):
            @functools.wraps(forward_orig)
            def forward(*args, **kwargs):
                if torch.cuda.is_current_stream_capturing():
                    assert not self._use_eager_mode, (
                        "Eager mode is not compatible with CUDA Graphs capturing"
                    )

                self._replay_chunk_idx = 0
                self._replay_offset = 0

                # Build metadata for verification during this forward pass
                self._cur_iter_actual_metadata = []

                output = forward_orig(*args, **kwargs)

                # Store metadata and copy index to GPU (for deferred verification)
                if self._replay_verify_metadata:
                    metadata_idx = len(self._actual_metadata_list)
                    self._actual_metadata_list.append(self._cur_iter_actual_metadata)
                    if metadata_idx >= len(self._metadata_idx_range_gpu):
                        raise ValueError(
                            f"Metadata index overflow: {metadata_idx} >= {len(self._metadata_idx_range_gpu)}. "
                            f"Increase MAX_NUM_METADATA if more iterations are needed."
                        )
                    with nvtx.annotate("layer_wise_benchmarks ignore"):
                        self._actual_metadata_idx_gpu.copy_(
                            self._metadata_idx_range_gpu[metadata_idx]
                        )

                # Verify all replay data was consumed
                if self._started and self._replay_chunk_idx != len(self._cur_iter_metadata):
                    raise ValueError(
                        f"Unused replay data: chunks [{self._replay_chunk_idx}:{len(self._cur_iter_metadata)}] "
                        f"were not consumed during forward pass"
                    )

                if torch.cuda.is_current_stream_capturing():
                    if self._replay_offset > len(self._slots_buffer_gpu):
                        raise ValueError(
                            f"Slots buffer overflow: required {self._replay_offset} elements, "
                            f"but buffer size is {len(self._slots_buffer_gpu)}"
                        )

                return output

            return forward

        model.forward = make_forward(model.forward)

    def _wrap_layer_forward(self, model):
        """Wrap layer forward methods to track layer index and add NVTX markers."""

        def make_forward(layer_idx, forward_orig):
            @functools.wraps(forward_orig)
            def forward(*args, **kwargs):
                self._current_layer_idx = layer_idx

                if self.mode in [Mode.MARK, Mode.COLLECT]:
                    with nvtx.annotate(f"layer_wise_benchmarks layer_idx {layer_idx}"):
                        output = forward_orig(*args, **kwargs)
                else:
                    output = forward_orig(*args, **kwargs)

                del self._current_layer_idx
                return output

            return forward

        for idx, layer in enumerate(model.model.layers):
            layer.forward = make_forward(idx, layer.forward)

    def maybe_collect_or_replay_slots(self, num_slots, token_selected_slots):
        """Collect or replay token_selected_slots data.

        Args:
            num_slots: Number of slots.
            token_selected_slots: Tensor of selected expert slots.

        Returns:
            Original tensor in COLLECT mode, or replayed tensor in REPLAY mode.
        """
        if self.mode == Mode.COLLECT:
            return self._collect_slots(num_slots, token_selected_slots)

        if self.mode == Mode.REPLAY:
            return self._replay_slots(num_slots, token_selected_slots)

        return token_selected_slots

    def _collect_slots(self, num_slots, token_selected_slots):
        """Collect slots data for current iteration."""
        # Skip if model was not wrapped (no layer context available)
        if not hasattr(self, "_current_layer_idx"):
            return token_selected_slots

        if token_selected_slots.dtype != self.SLOTS_DTYPE:
            raise ValueError(
                f"Unexpected dtype for token_selected_slots: expected {self.SLOTS_DTYPE}, "
                f"got {token_selected_slots.dtype}"
            )

        if self._layer_indices is None or self._current_layer_idx in self._layer_indices:
            self._cur_iter_records.append(
                (self._current_layer_idx, num_slots, token_selected_slots)
            )

        return token_selected_slots

    def _replay_slots(self, num_slots, token_selected_slots):
        """Replay slots data from recorded buffer."""
        # Skip if model was not wrapped (no layer context available)
        if not hasattr(self, "_current_layer_idx"):
            return token_selected_slots

        if token_selected_slots.dtype != self.SLOTS_DTYPE:
            raise ValueError(
                f"Unexpected dtype for token_selected_slots: expected {self.SLOTS_DTYPE}, "
                f"got {token_selected_slots.dtype}"
            )

        # Record actual metadata for deferred verification in stop()
        self._cur_iter_actual_metadata.append(
            {
                "layer_idx": self._current_layer_idx,
                "num_slots": num_slots,
                "token_selected_slots_shape": list(token_selected_slots.shape),
            }
        )

        # Immediate validation (does not depend on GPU data, safe during CUDA Graphs)
        if self._started:
            chunk_metadata = self._cur_iter_metadata[self._replay_chunk_idx]
            expected_layer_idx = chunk_metadata["layer_idx"]
            expected_num_slots = chunk_metadata["num_slots"]
            expected_shape = chunk_metadata["token_selected_slots_shape"]

            if self._current_layer_idx != expected_layer_idx:
                raise ValueError(
                    f"Layer index mismatch during replay: expected {expected_layer_idx}, "
                    f"got {self._current_layer_idx}"
                )
            if num_slots != expected_num_slots:
                raise ValueError(
                    f"num_slots mismatch during replay: expected {expected_num_slots}, "
                    f"got {num_slots}"
                )
            if list(token_selected_slots.shape) != expected_shape:
                raise ValueError(
                    f"Shape mismatch during replay: expected {expected_shape}, "
                    f"got {list(token_selected_slots.shape)}"
                )
        else:
            if (
                num_slots < self.MAX_TOPK_AND_MIN_SLOTS
                or token_selected_slots.shape[-1] > self.MAX_TOPK_AND_MIN_SLOTS
            ):
                raise ValueError(
                    "Invalid initial replayed_slots, please adjust `MAX_TOPK_AND_MIN_SLOTS`"
                )

        if self._started or torch.cuda.is_current_stream_capturing():
            n = token_selected_slots.numel()
            buffer = (
                self._replay_eager_slots_gpu if self._use_eager_mode else self._slots_buffer_gpu
            )
            replayed_slots = buffer[self._replay_offset : self._replay_offset + n].view_as(
                token_selected_slots
            )

            self._replay_chunk_idx += 1
            self._replay_offset += n
            return replayed_slots

        return token_selected_slots

    def start(self):
        """Start calibration. Call before the profiling loop."""
        assert not self._started
        self._started = True

        if self.mode != Mode.NONE:
            logger.info(f"Layer-wise benchmarks: Calibrator started in {self.mode.name} mode")

        if self.mode == Mode.COLLECT:
            # Per-iteration storage buffers
            # CUDA Graphs reuse fixed buffers, so we must copy data out after each step
            self._collected_metadata_idx = torch.empty(
                self.MAX_COLLECT_ITERATIONS, dtype=torch.long, device="cuda"
            )
            self._collected_slots_cpu = torch.empty(
                self.MAX_COLLECT_ITERATIONS,
                self.MAX_SLOTS_BUFFER_SIZE,
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            self._collected_records = []

            # Eager mode flag: True when data exceeds MAX_SLOTS_BUFFER_SIZE
            # (context phase, not using CUDA Graphs)
            self._use_eager_mode = False

        if self.mode == Mode.REPLAY and self._replay_verify_metadata:
            # Per-iteration storage buffers for deferred metadata verification
            # Note: _actual_metadata_list, _actual_metadata_idx_gpu, _metadata_idx_range_gpu
            # are created in _init_replay_mode() before CUDA Graph capture
            self._collected_actual_metadata_idx = torch.empty(
                self.MAX_REPLAY_ITERATIONS, dtype=torch.long, device="cuda"
            )
            # post_step() runs outside CUDA Graphs, so we can use Python list for iterations
            self._collected_iterations = []

    def pre_step(self, iteration: int):
        """Prepare for an iteration. Call before model.forward().

        Args:
            iteration: Current iteration number.
        """
        if self.mode == Mode.REPLAY and self._started:
            self._cur_iter_metadata = self._replay_db[iteration]["metadata"]
            slots_gpu = self._replay_db[iteration]["slots_data_gpu"]

            expected_size = sum(
                torch.Size(m["token_selected_slots_shape"]).numel() for m in self._cur_iter_metadata
            )
            if len(slots_gpu) != expected_size:
                raise ValueError(
                    f"Slots data size mismatch for iteration {iteration}: "
                    f"expected {expected_size}, got {len(slots_gpu)}"
                )

            if expected_size <= self.MAX_SLOTS_BUFFER_SIZE:
                self._use_eager_mode = False
                self._slots_buffer_gpu[:expected_size].copy_(slots_gpu, non_blocking=True)
            else:
                self._use_eager_mode = True
                self._replay_eager_slots_gpu = slots_gpu

    def post_step(self, iteration: int):
        """Finalize an iteration. Call after model.forward().

        Args:
            iteration: Current iteration number.
        """
        if self.mode == Mode.COLLECT and self._started:
            record_idx = len(self._collected_records)
            if record_idx >= self.MAX_COLLECT_ITERATIONS:
                raise ValueError(
                    f"Exceeded MAX_COLLECT_ITERATIONS={self.MAX_COLLECT_ITERATIONS}. "
                    "Increase the limit or reduce the profiling iterations."
                )
            self._collected_metadata_idx[record_idx].copy_(self._metadata_idx_gpu)

            if self._use_eager_mode:
                # TODO: Avoid synchronization by using async copy
                slots_cpu = torch.cat(self._eager_slots_gpu).to("cpu")
            else:
                slots_cpu = self._collected_slots_cpu[record_idx]
                # TODO: Copy only required elements instead of entire buffer
                slots_cpu.copy_(self._slots_buffer_gpu, non_blocking=True)

            self._collected_records.append(
                {
                    "iteration": iteration,
                    "slots_cpu": slots_cpu,
                }
            )

            # Reset eager mode for next step
            self._use_eager_mode = False

        if self.mode == Mode.REPLAY and self._started and self._replay_verify_metadata:
            # Record metadata index on GPU (no sync), verification deferred to stop()
            record_idx = len(self._collected_iterations)
            if record_idx >= self.MAX_REPLAY_ITERATIONS:
                raise ValueError(
                    f"Exceeded MAX_REPLAY_ITERATIONS={self.MAX_REPLAY_ITERATIONS}. "
                    "Increase the limit or reduce the profiling iterations."
                )
            self._collected_actual_metadata_idx[record_idx].copy_(
                self._actual_metadata_idx_gpu, non_blocking=True
            )
            # post_step() runs outside CUDA Graphs, so we can use Python list
            self._collected_iterations.append(iteration)

    def stop(self):
        """Stop calibration and save data. Call after the profiling loop."""
        assert self._started

        if self.mode == Mode.COLLECT:
            self._save_collected_data()

        if self.mode == Mode.REPLAY and self._replay_verify_metadata:
            self._verify_replay_metadata()

        self._started = False

    def _save_collected_data(self):
        """Save collected calibration data to file."""
        torch.cuda.synchronize()

        metadata_idx_list = self._collected_metadata_idx.tolist()
        output_records = []

        for record_idx, record in enumerate(self._collected_records):
            metadata = self._metadata_list[metadata_idx_list[record_idx]]
            slots_size = sum(torch.Size(m["token_selected_slots_shape"]).numel() for m in metadata)
            slots_data = record["slots_cpu"][:slots_size]

            output_records.append(
                {
                    "iteration": record["iteration"],
                    "metadata": metadata,
                    "raw_data": base64.b64encode(
                        zlib.compress(slots_data.numpy().tobytes())
                    ).decode(),
                }
            )

        # Gather from all ranks and save on rank 0
        all_records = self._dist.allgather(output_records)

        if self._dist.rank == 0:
            with open(self._file_path, "w") as f:
                json.dump(
                    {
                        "world_size": self._dist.world_size,
                        "all_ranks_records": all_records,
                    },
                    f,
                )

        if self.mode != Mode.NONE:
            logger.info(f"Layer-wise benchmarks: Calibrator saved data to {self._file_path}")

    def _verify_replay_metadata(self):
        """Verify that replayed metadata matches actual execution."""
        torch.cuda.synchronize()

        record_count = len(self._collected_iterations)
        chunk_count = 0
        actual_idx_list = self._collected_actual_metadata_idx[:record_count].tolist()

        for record_idx, (actual_idx, iteration) in enumerate(
            zip(actual_idx_list, self._collected_iterations)
        ):
            actual_metadata = self._actual_metadata_list[actual_idx]
            expected_metadata = self._replay_db[iteration]["metadata"]

            if len(actual_metadata) != len(expected_metadata):
                raise ValueError(
                    f"Metadata length mismatch at record {record_idx}: "
                    f"actual {len(actual_metadata)} chunks, expected {len(expected_metadata)} chunks"
                )

            for chunk_idx, (actual_chunk, expected_chunk) in enumerate(
                zip(actual_metadata, expected_metadata)
            ):
                if actual_chunk["layer_idx"] != expected_chunk["layer_idx"]:
                    raise ValueError(
                        f"Layer index mismatch at record {record_idx}, chunk {chunk_idx}: "
                        f"actual layer_idx={actual_chunk['layer_idx']}, "
                        f"expected layer_idx={expected_chunk['layer_idx']}"
                    )
                if actual_chunk["num_slots"] != expected_chunk["num_slots"]:
                    raise ValueError(
                        f"num_slots mismatch at record {record_idx}, chunk {chunk_idx}: "
                        f"actual num_slots={actual_chunk['num_slots']}, "
                        f"expected num_slots={expected_chunk['num_slots']}"
                    )
                if (
                    actual_chunk["token_selected_slots_shape"]
                    != expected_chunk["token_selected_slots_shape"]
                ):
                    raise ValueError(
                        f"Shape mismatch at record {record_idx}, chunk {chunk_idx}: "
                        f"actual shape={actual_chunk['token_selected_slots_shape']}, "
                        f"expected shape={expected_chunk['token_selected_slots_shape']}"
                    )
                chunk_count += 1

        logger.info(
            f"Layer-wise benchmarks: Replay metadata verification passed for {record_count} iterations"
            f" and {chunk_count} chunks"
        )

    def get_replay_iteration_range(self):
        """Get the valid iteration range for REPLAY mode.

        Returns a tuple (start_iter, stop_iter) representing the range of iterations
        that can be replayed. This method verifies that the iterations form a
        contiguous range. Cross-rank iteration consistency is verified in init().

        Returns:
            tuple: (start_iter, stop_iter) where iterations are in [start_iter, stop_iter]

        Raises:
            ValueError: If mode is not REPLAY or iterations are not a contiguous range.
        """
        if self.mode != Mode.REPLAY:
            raise ValueError(
                f"get_replay_iteration_range() is only valid in REPLAY mode, "
                f"current mode is {self.mode.name}"
            )

        # Verify iterations form a contiguous range
        local_iterations = sorted(self._replay_db.keys())
        start_iter = local_iterations[0]
        stop_iter = local_iterations[-1]
        if local_iterations != list(range(start_iter, stop_iter + 1)):
            raise ValueError("Iterations are not a contiguous range")

        return start_iter, stop_iter


_calibrator = Calibrator()


def get_calibrator():
    """Get the global calibrator instance."""
    return _calibrator
