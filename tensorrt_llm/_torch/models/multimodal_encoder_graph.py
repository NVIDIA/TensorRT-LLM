# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Generic CUDA-graph runner for multimodal vision/audio encoder stacks.

The runner captures the GPU-heavy block loop of a multimodal encoder for a fixed set of
(total_tokens, num_contexts) buckets and replays the captured graphs at inference time. It is
intentionally agnostic to the encoder model: the caller supplies an `encoder_fn` (the block loop),
tensor specs for the inputs/outputs, and small factory/prepare callbacks for the attention metadata
that the captured graph reads from.

Design highlights:

* Padding uses an isolated dummy context, never an extension of the last real context.
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

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

import torch

from ...logger import logger
from ..utils import make_weak_ref

if TYPE_CHECKING:
    from ...llmapi.llm_args import MultimodalEncoderCudaGraphConfig
    from ..attention_backend import AttentionMetadata

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
    * As a captured graph key, both fields reflect the padded state seen by the captured kernels.
      When `enable_padding=True` the key's `total_tokens` is `bucket.total_tokens + 1` and
      `num_contexts` is `bucket.num_contexts + 1`.
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


class _CapturedGraph(NamedTuple):
    """Bookkeeping for a single captured graph variant."""

    graph: torch.cuda.CUDAGraph
    inputs: Dict[str, torch.Tensor]
    metadata: AttentionMetadata
    # Each entry is a tensor view aliased to the captured output's memory via `make_weak_ref`; it
    # stays valid as long as the captured graph holds the underlying allocation alive.
    outputs: Dict[str, torch.Tensor]
    # `data_ptr()` of each provider-declared graph-critical tensor on `metadata` after capture. The
    # runner asserts these stay stable across replays; the captured kernels bake these addresses in,
    # so a reallocation would silently read stale memory.
    # See `EncoderMetadataProvider.graph_critical_attrs` for more detail.
    metadata_tensor_ptrs: Mapping[str, int]


class EncoderMetadataProvider(Protocol):
    """Builds and refreshes the graph-owned `AttentionMetadata`.

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

    def build(self, key: "EncoderGraphKey") -> "AttentionMetadata":
        """Return a fresh metadata instance sized to `key`."""

    def refresh_in_place(
        self,
        metadata: "AttentionMetadata",
        padded_seq_lengths: Sequence[int],
    ) -> None:
        """Mutate `metadata` in place for the next capture or replay."""


class MultimodalEncoderGraphRunner:
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
        input_specs: Static shape/dtype descriptors for the graph inputs.
        output_specs: Token axis (`token_dim`) for each graph output, keyed by output name. Outputs
            are not materialized by the runner - they are aliased from `encoder_fn`'s return - so
            only the token axis is needed, to slice each output back to the real token extent.
        config: Configuration object containing buckets and padding policy. See
            `MultimodalEncoderCudaGraphConfig`.
    """

    def __init__(
        self,
        *,
        encoder_fn: Callable[
            [Mapping[str, torch.Tensor], AttentionMetadata], Mapping[str, torch.Tensor]
        ],
        metadata_provider: EncoderMetadataProvider,
        input_specs: Mapping[str, EncoderGraphTensorSpec],
        output_specs: Mapping[str, int],
        config: MultimodalEncoderCudaGraphConfig,
    ) -> None:
        self._encoder_fn = encoder_fn
        self._metadata_provider = metadata_provider
        self._input_specs: Dict[str, EncoderGraphTensorSpec] = dict(input_specs)
        self._output_specs: Dict[str, int] = dict(output_specs)
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
        self._captured: Dict[EncoderGraphKey, _CapturedGraph] = {}
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
        for bucket in self._buckets:
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

        bucket = self._select_bucket(real_contexts, real_tokens)
        if bucket is None:
            if self._replay_stats_enabled:
                logger.info(
                    "Multimodal encoder CUDA graph bucket miss: no bucket for "
                    f"{real_contexts=}, {real_tokens=}."
                )
            return None

        padded_seq_lengths = self._padded_seq_lengths(
            list(seq_lengths), bucket=bucket, real_tokens=real_tokens
        )
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
        self._copy_inputs_into_static(captured, inputs, real_tokens=real_tokens)
        self._metadata_provider.refresh_in_place(captured.metadata, padded_seq_lengths)
        self._assert_metadata_buffers_stable(captured.metadata, captured.metadata_tensor_ptrs)
        captured.graph.replay()
        if self._replay_stats_enabled:
            logger.info(
                "Multimodal encoder CUDA graph bucket hit: "
                f"{bucket=}, {real_contexts=}, {real_tokens=}, "
                f"pad_slack={bucket.total_tokens - real_tokens}."
            )
        return self._collect_outputs(captured, real_tokens=real_tokens)

    # ------------------------------------------------------------------
    # Bucket selection / padding
    # ------------------------------------------------------------------

    def _select_bucket(self, real_contexts: int, real_tokens: int) -> Optional[EncoderGraphKey]:
        for bucket in self._buckets:
            if bucket.num_contexts != real_contexts:
                continue
            if real_tokens > bucket.total_tokens:
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
        real_tokens: int,
    ) -> List[int]:
        if not self._config.enable_padding:
            # Exact-fit only: workload tokens must equal the bucket's real-token
            # budget. _select_bucket already enforces this.
            return seq_lengths
        # Append one dummy context that absorbs both the unused real-token slack and the reserved
        # padding token. The reservation ensures the dummy is always non-empty even when the
        # workload exactly fills the bucket.
        padding = (bucket.total_tokens - real_tokens) + _PADDING_TOKEN_RESERVE
        return seq_lengths + [padding]

    def _padded_key_for_bucket(self, bucket: EncoderGraphKey) -> EncoderGraphKey:
        """Graph key for the captured variant of a bucket."""
        if self._config.enable_padding:
            return EncoderGraphKey(
                total_tokens=bucket.total_tokens + _PADDING_TOKEN_RESERVE,
                num_contexts=bucket.num_contexts + 1,
            )
        return EncoderGraphKey(
            total_tokens=bucket.total_tokens,
            num_contexts=bucket.num_contexts,
        )

    def _dummy_padded_seq_lengths(self, bucket: EncoderGraphKey, key: EncoderGraphKey) -> List[int]:
        """Representative padded seq_lengths used during capture only.

        Backends may derive a graph-captured work shape from metadata built with these lengths, so
        the layout must cover every replay shape that can select the same bucket.
        """
        if key.num_contexts == bucket.num_contexts:
            # No padding context. Capture the worst-case skew so plan/run attention backends whose
            # launch shape depends on max_token_per_sequence cover every exact-fit replay for this
            # bucket.
            return [bucket.total_tokens - (bucket.num_contexts - 1)] + [1] * (
                bucket.num_contexts - 1
            )
        # One dummy padding context: give each real context a single token and put the rest in the
        # dummy. The captured buffers are sized to `key.total_tokens`
        # (== bucket.total_tokens + _PADDING_TOKEN_RESERVE),
        # so the dummy absorbs everything not assigned to a real context.
        real = [1] * bucket.num_contexts
        padding = key.total_tokens - bucket.num_contexts
        if padding <= 0:
            raise ValueError(
                f"Bucket {bucket} cannot host {bucket.num_contexts} real contexts "
                f"plus a dummy padding context in {key.total_tokens} tokens."
            )
        return real + [padding]

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

        static_inputs = {
            name: spec.materialize(key.total_tokens, device)
            for name, spec in self._input_specs.items()
        }
        metadata = self._metadata_provider.build(key)
        self._metadata_provider.refresh_in_place(metadata, padded_seq_lengths)

        capture_kwargs: Dict[str, Any] = {}
        if self._memory_pool is not None:
            capture_kwargs["pool"] = self._memory_pool

        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            for _ in range(self._config.warmup_steps):
                self._encoder_fn(static_inputs, metadata)
            torch.cuda.synchronize()

            with torch.cuda.graph(graph, **capture_kwargs):
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
        self._memory_pool = graph.pool()

    @staticmethod
    def _snapshot_metadata_tensor_ptrs(
        metadata: AttentionMetadata, attrs: Sequence[str]
    ) -> Dict[str, int]:
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
    def _assert_metadata_buffers_stable(
        metadata: AttentionMetadata, expected_ptrs: Mapping[str, int]
    ) -> None:
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

    def _copy_inputs_into_static(
        self,
        captured: _CapturedGraph,
        inputs: Mapping[str, torch.Tensor],
        *,
        real_tokens: int,
    ) -> None:
        for name, spec in self._input_specs.items():
            if name not in inputs:
                raise KeyError(f"Missing graph input '{name}' for multimodal encoder runner.")
            src = inputs[name]
            dst = captured.inputs[name]
            self._copy_token_axis(dst, src, spec.token_dim, real_tokens)

    def _collect_outputs(
        self, captured: _CapturedGraph, *, real_tokens: int
    ) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = {}
        for name, tensor in captured.outputs.items():
            token_dim = self._output_specs[name]
            result[name] = self._slice_token_axis(tensor, token_dim, real_tokens)
        return result

    @staticmethod
    def _copy_token_axis(
        dst: torch.Tensor,
        src: torch.Tensor,
        token_dim: int,
        real_tokens: int,
    ) -> None:
        # Pre-flight shape checks: every non-token dim must match exactly so the captured graph's
        # loads stay in bounds.
        for dim, (d_size, s_size) in enumerate(zip(dst.shape, src.shape, strict=True)):
            if dim == token_dim:
                continue
            if d_size != s_size:
                raise ValueError(
                    f"Encoder graph input mismatch at {dim=}: static {d_size} vs runtime {s_size}."
                )
        src_token_len = src.shape[token_dim]
        if src_token_len < real_tokens:
            raise ValueError(
                f"Encoder graph input has only {src_token_len} tokens on axis {token_dim}, but "
                f"{real_tokens} were promised."
            )
        idx = [slice(None)] * dst.ndim
        idx[token_dim] = slice(0, real_tokens)
        dst[tuple(idx)].copy_(src.narrow(token_dim, 0, real_tokens))

    @staticmethod
    def _slice_token_axis(tensor: torch.Tensor, token_dim: int, real_tokens: int) -> torch.Tensor:
        if tensor.shape[token_dim] == real_tokens:
            return tensor
        return tensor.narrow(token_dim, 0, real_tokens)

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
