# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Generic CUDA-graph runner for multimodal vision/audio encoder stacks.

The runner captures the GPU-heavy block loop of a multimodal encoder for a fixed set of
(total_tokens, num_contexts) buckets and replays the captured graphs at inference time. It is
intentionally agnostic to the encoder model: the caller supplies an `encoder_fn` (the block loop),
tensor specs for the inputs/outputs, and small factory/prepare callbacks for the attention metadata
that the captured graph reads from.

Design highlights:

* The default packed adapter uses an isolated dummy padding context, never an extension of the
  last real context. Other encoder layouts supply their own shape, mask, and padding rules.
  Full-attention vision encoders would otherwise let real tokens attend to padding rows.
* When `enable_padding=True`, the runner reserves one extra token beyond each bucket's real-token
  budget so the dummy padding context is always non-empty (the attention backend rejects
  zero-length contexts). Bucket specs from callers therefore describe real tokens only; the
  runner adds the reservation internally and the captured graph buffers are sized to
  `bucket.total_tokens + 1`.
* The runner owns the static input buffers and the metadata instance it hands to `encoder_fn` during
  capture, so replay never reallocates the tensors whose addresses are baked into the graph.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
)

import torch

from ...logger import logger
from ..nccl_window_graph import nccl_window_graph_capture, release_nccl_window_graph_owner
from ..nccl_window_tensor_scope import discard_nccl_window_tensor_outputs
from ..utils import make_weak_ref, torch_compiling

if TYPE_CHECKING:
    from ...llmapi.llm_args import MultimodalEncoderCudaGraphConfig

_MetadataT = TypeVar("_MetadataT")

# Single token kept aside for the dummy padding context when `enable_padding=True`.
# The attention backend rejects zero-length contexts, so the dummy must contain at least one token
# even when the workload exactly fills the bucket's real-token budget. This constant participates
# in both buffer sizing (`_padded_key_for_bucket`) and per-replay length assignment
# (`_padded_seq_lengths`, `_dummy_padded_seq_lengths`).
_PADDING_TOKEN_RESERVE: int = 1


class EncoderGraphKey(NamedTuple):
    """Internal key identifying an encoder graph workload size.

    The fields' meaning depends on how the key is used:

    * As a configured bucket, `total_tokens` is the maximum number of real (post-encode) tokens
      the bucket is meant to cover, and `num_contexts` is the number of real contexts.
      Neither includes any padding reservation; the runner adds that internally.
    * As a captured graph key, both fields describe the adapter's fixed capture capacity.
      The default packed adapter adds one token and one context when padding is enabled.
    """

    total_tokens: int
    num_contexts: int


class EncoderGraphTensorSpec(NamedTuple):
    """Static shape and dtype for a graph-owned input or output tensor.

    `shape` describes the non-token dimensions only. The token-axis size is filled in from the
    bucket and inserted at position `token_dim`. So for a ragged `(total_tokens, hidden_size)`
    input, the spec is `EncoderGraphTensorSpec(shape=(hidden_size,), token_dim=0)`.
    """

    shape: Tuple[int, ...]
    dtype: torch.dtype
    token_dim: int = 0

    def materialize(self, total_tokens: int, device: torch.device) -> torch.Tensor:
        full_shape = list(self.shape)
        full_shape.insert(self.token_dim, total_tokens)
        return torch.zeros(tuple(full_shape), dtype=self.dtype, device=device)


class EncoderGraphIOAdapter(ABC):
    """Define an encoder's captured layout, padding, and tensor transfers.

    The default layout packs independent contexts along one token axis and isolates padding in a
    dummy context. Encoders with batched time axes or masks can override ``graph_key``,
    ``capture_seq_lengths``, and ``padded_seq_lengths`` together. ``can_replay`` can reject a
    workload that fits the token budget but does not fit that layout.

    Input buffers must retain their addresses across replays. Output tensors are borrowed views
    into graph-owned buffers; collection may trim different axes or output lengths without
    assuming that the encoder preserves the input token count.
    """

    @abstractmethod
    def allocate_inputs(
        self, key: EncoderGraphKey, device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Allocate fixed buffers initialized with safe inputs for warmup and capture."""

    @abstractmethod
    def copy_inputs(
        self,
        static_inputs: Mapping[str, torch.Tensor],
        inputs: Mapping[str, torch.Tensor],
        seq_lengths: Sequence[int],
    ) -> None:
        """Copy live inputs and refresh masks/padding without replacing buffers."""

    @abstractmethod
    def collect_outputs(
        self, outputs: Mapping[str, torch.Tensor], seq_lengths: Sequence[int]
    ) -> dict[str, torch.Tensor]:
        """Return output views for the live workload, including any masks."""

    def can_replay(self, bucket: EncoderGraphKey, seq_lengths: Sequence[int]) -> bool:
        """Whether a workload fitting ``bucket`` also fits this encoder's layout."""
        return True

    def graph_key(self, bucket: EncoderGraphKey, *, enable_padding: bool) -> EncoderGraphKey:
        """Map a configured bucket to its fixed capture capacity."""
        if enable_padding:
            return EncoderGraphKey(
                total_tokens=bucket.total_tokens + _PADDING_TOKEN_RESERVE,
                num_contexts=bucket.num_contexts + 1,
            )
        return bucket

    def padded_seq_lengths(
        self, seq_lengths: Sequence[int], *, bucket: EncoderGraphKey, enable_padding: bool
    ) -> list[int]:
        """Describe the contexts passed to the metadata provider before replay."""
        if not enable_padding:
            return list(seq_lengths)
        padding = bucket.total_tokens - sum(seq_lengths) + _PADDING_TOKEN_RESERVE
        return [*seq_lengths, padding]

    def capture_seq_lengths(self, bucket: EncoderGraphKey, key: EncoderGraphKey) -> list[int]:
        """Use worst-case segment skew so capture covers every selectable workload."""
        if key.num_contexts == bucket.num_contexts:
            return [bucket.total_tokens - (bucket.num_contexts - 1)] + [1] * (
                bucket.num_contexts - 1
            )
        real = [1] * bucket.num_contexts
        padding = key.total_tokens - bucket.num_contexts
        if padding <= 0:
            raise ValueError(
                f"Bucket {bucket} cannot host {bucket.num_contexts} real contexts "
                f"plus a dummy padding context in {key.total_tokens} tokens."
            )
        return real + [padding]


class PackedEncoderGraphIOAdapter(EncoderGraphIOAdapter):
    """Use token-axis tensor specs for packed, length-preserving encoder blocks.

    Args:
        input_specs: Non-token dimensions, dtype, and token axis of each graph input.
        output_specs: Token axis of each graph output; outputs preserve the real input token count.
    """

    def __init__(
        self,
        input_specs: Mapping[str, EncoderGraphTensorSpec],
        output_specs: Mapping[str, int],
    ) -> None:
        self._input_specs = dict(input_specs)
        self._output_specs = dict(output_specs)

    def allocate_inputs(
        self, key: EncoderGraphKey, device: torch.device
    ) -> dict[str, torch.Tensor]:
        return {
            name: spec.materialize(key.total_tokens, device)
            for name, spec in self._input_specs.items()
        }

    def copy_inputs(
        self,
        static_inputs: Mapping[str, torch.Tensor],
        inputs: Mapping[str, torch.Tensor],
        seq_lengths: Sequence[int],
    ) -> None:
        real_tokens = sum(seq_lengths)
        for name, spec in self._input_specs.items():
            if name not in inputs:
                raise KeyError(f"Missing graph input '{name}' for multimodal encoder runner.")
            src = inputs[name]
            dst = static_inputs[name]
            for dim, (d_size, s_size) in enumerate(zip(dst.shape, src.shape, strict=True)):
                if dim != spec.token_dim and d_size != s_size:
                    raise ValueError(
                        f"Encoder graph input mismatch at {dim=}: static {d_size} vs runtime {s_size}."
                    )
            src_token_len = src.shape[spec.token_dim]
            if src_token_len < real_tokens:
                raise ValueError(
                    f"Encoder graph input has only {src_token_len} tokens on axis {spec.token_dim}, "
                    f"but {real_tokens} were promised."
                )
            dst.narrow(spec.token_dim, 0, real_tokens).copy_(
                src.narrow(spec.token_dim, 0, real_tokens)
            )

    def collect_outputs(
        self, outputs: Mapping[str, torch.Tensor], seq_lengths: Sequence[int]
    ) -> dict[str, torch.Tensor]:
        real_tokens = sum(seq_lengths)
        return {
            name: tensor.narrow(self._output_specs[name], 0, real_tokens)
            for name, tensor in outputs.items()
        }


@dataclass(frozen=True)
class _CapturedGraph(Generic[_MetadataT]):
    """Bookkeeping for a single captured graph variant."""

    graph: torch.cuda.CUDAGraph
    inputs: Dict[str, torch.Tensor]
    metadata: _MetadataT
    # Each entry is a tensor view aliased to the captured output's memory via `make_weak_ref`; it
    # stays valid as long as the captured graph holds the underlying allocation alive.
    outputs: Dict[str, torch.Tensor]
    # `data_ptr()` of each provider-declared graph-critical tensor on `metadata` after capture. The
    # runner asserts these stay stable across replays; the captured kernels bake these addresses in,
    # so a reallocation would silently read stale memory.
    # See `EncoderMetadataProvider.graph_critical_attrs` for more detail.
    metadata_tensor_ptrs: Mapping[str, int]


class EncoderMetadataProvider(Protocol[_MetadataT]):
    """Builds and refreshes encoder-specific graph-owned metadata.

    The runner takes a single provider object instead of two separate callbacks so the contract
    between `build` and `refresh_in_place` lives in one place:

    * `build` is called once per bucket; the returned metadata instance is held for the lifetime of
      every replay of that bucket. Its tensor buffer addresses are baked into the captured CUDA
      graph at capture time.
    * `refresh_in_place` runs before every capture and every replay. It must update the metadata's
      contents in place - never rebinding any attribute the captured kernels read from. The runner
      snapshots the `data_ptr()` of each attribute named in `graph_critical_attrs` right after
      capture and raises `RuntimeError` from a later replay if any of those addresses moved.
    """

    # Names of `metadata` attributes whose tensor buffer addresses are read by the captured CUDA
    # graph. Implementations should list every tensor that backs a kernel argument; the runner uses
    # this list to anchor and verify `data_ptr()` stability across replays.
    graph_critical_attrs: Sequence[str]

    def build(self, key: EncoderGraphKey) -> _MetadataT:
        """Return a fresh metadata instance sized to `key`."""

    def refresh_in_place(
        self,
        metadata: _MetadataT,
        padded_seq_lengths: Sequence[int],
    ) -> None:
        """Mutate `metadata` in place for the next capture or replay."""


class MultimodalEncoderGraphRunner(Generic[_MetadataT]):
    """Captures and replays CUDA graphs for a multimodal encoder block stack.

    The runner is intentionally narrow: it does not know how to build attention metadata, what
    model it is wrapping, or how to assemble the eager prelude (patch embedding, positional
    embedding, etc.). It only owns the captured block loop, the static buffers that loop reads from,
    and the mapping from real-input shapes to captured graph keys.

    The runner's existence is itself the enable signal; callers should only construct one when CUDA
    graph capture is wanted. Captures are taken at startup for every configured bucket, and the
    captured set is fixed for the runner's lifetime.

    Args:
        encoder_fn: Callable invoked once per warmup step and once during capture. It receives the
            static input mapping and the graph-owned attention metadata and must return a mapping of
            named output tensors.
        metadata_provider: Bundles metadata construction and per-replay refresh into a single
            object. See `EncoderMetadataProvider` for the contract; the runner enforces the
            "no reallocation" half of that contract via a `data_ptr()` check after every replay's
            refresh call.
        input_specs: Static shape/dtype descriptors for packed graph inputs. Used when
            ``io_adapter`` is omitted.
        output_specs: Token axis (`token_dim`) for each graph output, keyed by output name. Outputs
            are not materialized by the runner - they are aliased from `encoder_fn`'s return - so
            only the token axis is needed, to slice each output back to the real token extent.
        config: Configuration object containing buckets and padding policy. See
            `MultimodalEncoderCudaGraphConfig`.
        io_adapter: Encoder-specific layout, padding, input copying, and output collection. Supply
            either this adapter or the packed ``input_specs``/``output_specs`` pair.
    """

    def __init__(
        self,
        *,
        encoder_fn: Callable[[Mapping[str, torch.Tensor], _MetadataT], Mapping[str, torch.Tensor]],
        metadata_provider: EncoderMetadataProvider[_MetadataT],
        config: MultimodalEncoderCudaGraphConfig,
        input_specs: Mapping[str, EncoderGraphTensorSpec] | None = None,
        output_specs: Mapping[str, int] | None = None,
        io_adapter: EncoderGraphIOAdapter | None = None,
    ) -> None:
        self._encoder_fn = encoder_fn
        self._metadata_provider = metadata_provider
        if io_adapter is None:
            if input_specs is None or output_specs is None:
                raise ValueError("Provide io_adapter or both input_specs and output_specs.")
            io_adapter = PackedEncoderGraphIOAdapter(input_specs, output_specs)
        elif input_specs is not None or output_specs is not None:
            raise ValueError("io_adapter cannot be combined with input_specs or output_specs.")
        self._io_adapter = io_adapter
        self._config = config
        # Adopted from the first captured graph so subsequent bucket captures share the pool.
        self._memory_pool = None

        if not config.buckets:
            raise ValueError(
                "MultimodalEncoderCudaGraphConfig.buckets is empty; the runner has nothing to "
                "capture."
            )

        # Field order is (total_tokens, num_contexts), matching the desired bucket ordering for
        # smallest-fit selection in `maybe_run()`.
        self._buckets: List[EncoderGraphKey] = sorted(
            EncoderGraphKey(total_tokens=total_tokens, num_contexts=num_contexts)
            for total_tokens, num_contexts in config.buckets
        )
        self._log_cuda_graph_memory_warning()

        # One captured graph per internal key. Populated by `capture_all` and fixed for the runner's
        # lifetime; `maybe_run` never adds entries.
        self._captured: dict[EncoderGraphKey, _CapturedGraph[_MetadataT]] = {}
        self._warned_no_bucket_match: bool = False
        self._replay_stats_enabled: bool = config.enable_replay_stats

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def capture_all(self, device: torch.device) -> None:
        """Capture every configured bucket up front.

        `enable_padding=True` captures the with-dummy-context variant for each bucket;
        `enable_padding=False` captures the exact-fit variant.
        """
        # Capture larger workloads first. Attention backends may share a graph workspace across
        # metadata variants; warming up a later, larger workload can resize that workspace and
        # invalidate pointers captured by an earlier graph. This matches the ordering used by the
        # LLM CUDA-graph runners and also lets smaller graphs reuse the larger graph's memory pool.
        for bucket in reversed(self._buckets):
            key = self._padded_key_for_bucket(bucket)
            if key in self._captured:
                continue
            padded_seq_lengths = self._dummy_padded_seq_lengths(bucket, key)
            self._capture_key(key, padded_seq_lengths, device)

    def maybe_run(
        self,
        *,
        seq_lengths: Sequence[int],
        inputs: Mapping[str, torch.Tensor],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Replay a captured graph for `inputs` if one matches.

        Returns the per-output tensors sliced back to the real token extent when replay happened;
        returns `None` when no captured graph applies, in which case the caller is expected to fall
        back to eager.

        Returned tensors are borrowed views into graph-owned static output buffers. A later replay
        of the same graph may overwrite those buffers, so callers must consume the tensors before
        the next replay or explicitly clone values they need to keep.

        Misses happen when:
        (a) no configured bucket can host the request shape or
        (b) the selected bucket has no captured graph.
        """
        real_contexts = len(seq_lengths)
        real_tokens = int(sum(seq_lengths))

        bucket = self._select_bucket(real_contexts, real_tokens, seq_lengths)
        if bucket is None:
            if self._replay_stats_enabled:
                logger.info(
                    "Multimodal encoder CUDA graph bucket miss: no bucket for "
                    f"{real_contexts=}, {real_tokens=}."
                )
            return None

        padded_seq_lengths = self._padded_seq_lengths(list(seq_lengths), bucket=bucket)
        key = self._padded_key_for_bucket(bucket)

        # Every selectable bucket is captured by `capture_all` before the runner is installed, so a
        # miss here is not expected; fall back to eager rather than crash if it ever happens.
        captured = self._captured.get(key)
        if captured is None:
            if self._replay_stats_enabled:
                logger.info(
                    "Multimodal encoder CUDA graph bucket miss: selected "
                    f"{bucket=} with {key=}, but no graph has been captured."
                )
            return None
        self._io_adapter.copy_inputs(captured.inputs, inputs, seq_lengths)
        self._metadata_provider.refresh_in_place(captured.metadata, padded_seq_lengths)
        self._assert_metadata_buffers_stable(captured.metadata, captured.metadata_tensor_ptrs)
        captured.graph.replay()
        if self._replay_stats_enabled:
            logger.info(
                "Multimodal encoder CUDA graph bucket hit: "
                f"{bucket=}, {real_contexts=}, {real_tokens=}, "
                f"pad_slack={bucket.total_tokens - real_tokens}."
            )
        return self._io_adapter.collect_outputs(captured.outputs, seq_lengths)

    def clear(self, *, release_nccl_window_owners: bool = True) -> None:
        """Reset captures, releasing their pool owner only at coordinated teardown."""
        for captured in self._captured.values():
            captured.graph.reset()
        self._captured.clear()
        if self._memory_pool is not None:
            if release_nccl_window_owners:
                release_nccl_window_graph_owner(self._memory_pool)
            self._memory_pool = None

    # ------------------------------------------------------------------
    # Bucket selection / padding
    # ------------------------------------------------------------------

    def _select_bucket(
        self,
        real_contexts: int,
        real_tokens: int,
        seq_lengths: Sequence[int] | None = None,
    ) -> Optional[EncoderGraphKey]:
        for bucket in self._buckets:
            if bucket.num_contexts != real_contexts:
                continue
            if real_tokens > bucket.total_tokens:
                continue
            if seq_lengths is not None and not self._io_adapter.can_replay(bucket, seq_lengths):
                continue
            if self._config.enable_padding:
                # Padding path: exact num_contexts match and real_tokens <= bucket budget.
                return bucket
            # No padding: require an exact match on both num_contexts and total_tokens.
            if real_tokens == bucket.total_tokens:
                return bucket
        if not self._warned_no_bucket_match:
            self._warned_no_bucket_match = True
            logger.warning(
                "Multimodal encoder CUDA graph: no configured bucket matches "
                f"({real_contexts=}, {real_tokens=}). "
                f"Falling back to eager. Configured buckets: {self._buckets}. "
            )
        return None

    def _padded_seq_lengths(
        self,
        seq_lengths: List[int],
        *,
        bucket: EncoderGraphKey,
    ) -> List[int]:
        return self._io_adapter.padded_seq_lengths(
            seq_lengths, bucket=bucket, enable_padding=self._config.enable_padding
        )

    def _padded_key_for_bucket(self, bucket: EncoderGraphKey) -> EncoderGraphKey:
        """Graph key for the captured variant of a bucket."""
        return self._io_adapter.graph_key(bucket, enable_padding=self._config.enable_padding)

    def _dummy_padded_seq_lengths(self, bucket: EncoderGraphKey, key: EncoderGraphKey) -> List[int]:
        """Representative padded seq_lengths used during capture only.

        Backends may derive a graph-captured work shape from metadata built with these lengths, so
        the layout must cover every replay shape that can select the same bucket.
        """
        return self._io_adapter.capture_seq_lengths(bucket, key)

    # ------------------------------------------------------------------
    # Capture & replay primitives
    # ------------------------------------------------------------------

    def _capture_key(
        self,
        key: EncoderGraphKey,
        padded_seq_lengths: Sequence[int],
        device: torch.device,
    ) -> None:
        padded_seq_lengths = list(padded_seq_lengths)
        logger.info(f"Capturing multimodal encoder CUDA graph for key={key} on device={device}.")

        static_inputs = self._io_adapter.allocate_inputs(key, device)
        metadata = self._metadata_provider.build(key)
        self._metadata_provider.refresh_in_place(metadata, padded_seq_lengths)

        graph = torch.cuda.CUDAGraph()
        graph_pool = self._memory_pool or torch.cuda.graph_pool_handle()
        # `torch_compiling(False)` for the same reason as `inference_mode`: ambient state the encoder
        # region must not inherit from whoever called us. Capture can be driven from `load_weights`,
        # outside any engine forward, so the process-global compile flag may still carry a previous
        # engine's value. `encoder_fn` is handed its metadata explicitly, so its layers must take
        # the eager path rather than the custom op, which resolves metadata from `extra_attrs` that
        # only an engine forward binds.
        # Capture can run from `load_weights`, outside any engine forward or `ModelCaller` scope, so
        # NCCL-window tensors produced by a TP-sharded tower need a local lease scope here.
        with torch.inference_mode(), torch_compiling(False):
            for _ in range(self._config.warmup_steps):
                with discard_nccl_window_tensor_outputs(static_inputs):
                    self._encoder_fn(static_inputs, metadata)
            torch.cuda.synchronize()

            with nccl_window_graph_capture(graph, graph_pool):
                with discard_nccl_window_tensor_outputs(static_inputs):
                    outputs = self._encoder_fn(static_inputs, metadata)

        captured_outputs = {name: make_weak_ref(tensor) for name, tensor in outputs.items()}
        self._captured[key] = _CapturedGraph(
            graph=graph,
            inputs=static_inputs,
            metadata=metadata,
            outputs=captured_outputs,
            metadata_tensor_ptrs=self._snapshot_metadata_tensor_ptrs(
                metadata, self._metadata_provider.graph_critical_attrs
            ),
        )
        # Adopt the pool from the first captured graph so subsequent captures share it and reuse
        # memory.
        self._memory_pool = graph_pool

    @staticmethod
    def _snapshot_metadata_tensor_ptrs(metadata: object, attrs: Sequence[str]) -> Dict[str, int]:
        """Record the `data_ptr()` of each named attribute on `metadata`.

        Used after capture to anchor the addresses the captured graph reads from. `attrs` comes from
        the provider's `graph_critical_attrs`; the runner does not inspect any attribute the
        provider did not declare.
        """
        snapshot: Dict[str, int] = {}
        for name in attrs:
            value = getattr(metadata, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Encoder graph metadata attribute `{name}` was declared graph-critical but is "
                    f"{type(value).__name__}, not a tensor."
                )
            snapshot[name] = value.data_ptr()
        return snapshot

    @staticmethod
    def _assert_metadata_buffers_stable(metadata: object, expected_ptrs: Mapping[str, int]) -> None:
        """Raise if any captured-time tensor has been rebound or moved.

        Catches the realistic failure mode where `refresh_in_place` accidentally reallocates a
        graph-read tensor (e.g. by forgetting to set `is_cuda_graph=True` so that
        `AttentionMetadata.seq_lens`'s setter falls back to assignment instead of an in-place
        `copy_`).
        """
        for name, expected_ptr in expected_ptrs.items():
            value = getattr(metadata, name)
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(
                    f"Encoder graph metadata attribute `{name}` was a tensor at capture time but "
                    f"is now {type(value).__name__}. The metadata provider's `refresh_in_place` "
                    "must not change the type of graph-read attributes."
                )
            if value.data_ptr() != expected_ptr:
                raise RuntimeError(
                    f"Encoder graph metadata tensor `{name}` was reallocated between capture and "
                    f"replay (data_ptr 0x{expected_ptr:x} -> 0x{value.data_ptr():x}). "
                    "The metadata provider's `refresh_in_place` must update tensor contents in "
                    "place; rebinding the attribute invalidates addresses baked into the captured "
                    "CUDA graph."
                )

    def _log_cuda_graph_memory_warning(self) -> None:
        max_total_tokens = max(bucket.total_tokens for bucket in self._buckets)
        sum_total_tokens = sum(bucket.total_tokens for bucket in self._buckets)
        logger.warning(
            "NOTE: capturing CUDA graphs for an encoder can reserve a large GPU memory pool, "
            "especially at high encoder-token counts or with many buckets. For multimodal encoders,"
            " relative latency gains are expected to be strongest at lower input sizes / lower "
            "token counts; larger buckets may consume substantial GPU memory for limited benefit. "
            f"Configured {len(self._buckets)} bucket(s), {max_total_tokens=}, {sum_total_tokens=}, "
            f"enable_padding={self._config.enable_padding}."
        )
