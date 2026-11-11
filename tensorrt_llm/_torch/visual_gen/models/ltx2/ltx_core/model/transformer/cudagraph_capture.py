"""Self-managed CUDA-graph capture of the transformer block loop.
Captures ``_process_transformer_blocks`` as one CUDA graph per (shape, perturbation signature)
and replays it from persistent static buffers. RoPE prepare / output projection stay eager.
Given a max token count per modality, captures across all runners share static input buffers sized
for that count (see :class:`_InputBufferPool`). Their cost is therefore bounded by the largest
shape, rather than growing with the number of captured shapes or runners.
Not thread-safe: :class:`CudaGraphRunner` (including the shared graph mempool) assumes a single
caller; concurrent capture/replay is unsupported.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

import torch
import torch.utils._pytree as pytree

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.guidance.perturbations import BatchedPerturbationConfig
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.transformer.transformer_args import AdaLNTimestepEmbedding
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.types import HostDeviceTensorPair

if TYPE_CHECKING:
    # torch imports this under TYPE_CHECKING too (a runtime top-level import cycles).
    from torch.cuda import _POOL_HANDLE

    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.transformer.transformer_args import BlockPerturbationsProcessor, TransformerArgs

_BlockLoopFn = Callable[
    ["TransformerArgs | None", "TransformerArgs | None", "BatchedPerturbationConfig"],
    "tuple[TransformerArgs | None, TransformerArgs | None]",
]
_ShapeKey = tuple[tuple[int, ...], ...]
# One capture identity: input shapes, the block's perturbation graph signature (what the block
# specialises on), so a perturbed pass keys to -- and recaptures -- its own graph, and the
# enclosing `capture_key_scope`'s key.
_CaptureKey = tuple[_ShapeKey, Hashable, Hashable]

_scope_key: ContextVar[Hashable] = ContextVar("cudagraph_capture_scope_key", default=None)


@contextmanager
def capture_key_scope(key: Hashable) -> Iterator[None]:
    """Add ``key`` to the capture identity of every :class:`CudaGraphRunner` call inside the scope.
    For host state the block loop branches on that its input shapes do not carry, such as the
    sequence-parallel padding layout: calls under different keys capture and replay their own graphs.
    """
    token = _scope_key.set(key)
    try:
        yield
    finally:
        _scope_key.reset(token)


@dataclasses.dataclass
class _Capture:
    graph: torch.cuda.CUDAGraph
    video_pool: _InputBufferPool
    audio_pool: _InputBufferPool
    perturbations_pool: _InputBufferPool
    # Outputs live in the shared CUDA graph mempool, not in the static input buffer pools above.
    static_video_out: TransformerArgs | None
    static_audio_out: TransformerArgs | None


def _shape_key(video: TransformerArgs | None, audio: TransformerArgs | None, block_masks: torch.Tensor) -> _ShapeKey:
    tensors = [*pytree.tree_leaves(video), *pytree.tree_leaves(audio), block_masks]
    return tuple(tuple(t.shape) for t in tensors if isinstance(t, torch.Tensor))


@dataclasses.dataclass(frozen=True)
class _BufferKey:
    """Identity of one allocation in an input buffer pool.
    ``field`` is a ``TransformerArgs`` field such as ``x`` or ``positional_embeddings.0``.
    """

    field: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


class _InputBufferPool:
    """Own static input buffers shared by CUDA-graph captures.
    Each key maps to one flat allocation sized for a field at its token budget. Captures of smaller
    shapes use contiguous prefixes of that allocation. This is safe because every replay first
    overwrites its inputs.
    The budget shape is part of the key, so genuinely different footprints (for example,
    key-only and dense attention masks) receive separate allocations. Allocations never move after
    creation because captured graphs retain their addresses.
    """

    def __init__(self, max_tokens: int | None = None) -> None:
        self._max_tokens = max_tokens
        self._buffers: dict[_BufferKey, torch.Tensor] = {}

    @property
    def max_tokens(self) -> int | None:
        return self._max_tokens

    def _max_shape(self, shape: torch.Size, tokens: int | None, *, spans_sequence: bool) -> tuple[int, ...]:
        """Return the tensor shape at this pool's token limit.
        For ordinary ``TransformerArgs`` fields, dimensions equal to the current token count are
        sequence dimensions and grow to ``max_tokens``. ``self_attention_mask`` is different under
        sequence parallelism: ``x`` contains a local token shard while the mask spans the global
        sequence on every rank, so its dimensions after batch and head are scaled by the token
        ratio instead. ``tokens=None`` marks inputs with no token dimension, such as block masks.
        """
        if tokens is None:
            return tuple(shape)
        max_tokens = self.max_tokens or tokens
        if spans_sequence:
            batch_and_heads = shape[:2]
            sequence = (
                -(-dim * max_tokens // tokens) if dim > 1 else dim  # ceil, so rounded-up shards fit
                for dim in shape[2:]
            )
            return (*batch_and_heads, *sequence)
        return tuple(max_tokens if dim == tokens else dim for dim in shape)

    def _get_buffer(self, field: str, template: torch.Tensor, max_shape: tuple[int, ...]) -> torch.Tensor:
        """Get a view shaped like ``template``, allocating its buffer on first use.
        Reshaping a flat prefix preserves canonical contiguous strides. Slicing a preallocated
        ``(B, T_max, D)`` tensor along its sequence dimension would not.
        """
        key = _BufferKey(field, max_shape, template.dtype, template.device)
        buffer = self._buffers.get(key)
        if buffer is None:
            buffer = torch.empty(math.prod(max_shape), dtype=template.dtype, device=template.device)
            self._buffers[key] = buffer
        if template.numel() > buffer.numel():
            raise ValueError(
                f"CUDA-graph static buffer {field!r} was sized for {max_shape} but this capture "
                f"supplies {tuple(template.shape)}, which does not fit. The buffers are sized by "
                f"the configured token budget and cannot grow without invalidating the graphs "
                f"already captured -- raise CompilationConfig.max_video_tokens / max_audio_tokens "
                f"to the largest count this rank denoises."
            )
        return buffer[: template.numel()].view(template.shape)

    def copy_value_to_buffer(
        self, field: str, value: object, tokens: int | None, *, spans_sequence: bool = False
    ) -> object:
        """Copy a field into pool storage and return its static representation."""
        if isinstance(value, torch.Tensor):
            static = self._get_buffer(field, value, self._max_shape(value.shape, tokens, spans_sequence=spans_sequence))
            static.copy_(value)
            return static
        if isinstance(value, tuple):
            return tuple(self.copy_value_to_buffer(f"{field}.{i}", t, tokens) for i, t in enumerate(value))
        if isinstance(value, AdaLNTimestepEmbedding):
            return dataclasses.replace(
                value,
                **{
                    f.name: self.copy_value_to_buffer(f"{field}.{f.name}", getattr(value, f.name), tokens)
                    for f in dataclasses.fields(value)
                },
            )
        return value

    def copy_args_to_buffers(self, args: TransformerArgs) -> TransformerArgs:
        """Copy ``TransformerArgs`` fields into persistent buffers."""
        tokens = int(args.x.shape[1])
        return dataclasses.replace(
            args,
            **{
                f.name: self.copy_value_to_buffer(
                    f.name,
                    getattr(args, f.name),
                    tokens,
                    spans_sequence=f.name == "self_attention_mask",
                )
                for f in dataclasses.fields(args)
            },
        )

    def copy_perturbations_to_buffers(self, perturbations: BatchedPerturbationConfig) -> BatchedPerturbationConfig:
        static_block_masks = self.copy_value_to_buffer("block_masks", perturbations.block_masks, tokens=None)
        assert isinstance(static_block_masks, torch.Tensor)
        return BatchedPerturbationConfig.from_masks(
            HostDeviceTensorPair(host_tensor=perturbations.block_masks_cpu, device_tensor=static_block_masks)
        )


class CudaGraphRunner:
    """Wrap ``_process_transformer_blocks(video, audio, perturbations) -> (video_out, audio_out)``
    with a per-(shape, perturbation) captured CUDA graph. Holds the video/audio ``TransformerArgs``
    (and the perturbation ``block_masks``) as persistent static buffers: each call's tensors are
    copied into the static inputs, the graph replayed, and the static OUTPUT buffers returned
    directly (standard CUDA-graph idiom -- the caller reads them, via the eager output projection,
    before the next replay overwrites them).
    Perturbation inputs always use a process-wide :class:`_InputBufferPool`. With
    ``max_video_tokens`` / ``max_audio_tokens``, modality inputs also use process-wide pools sized
    for those per-rank limits. Without a modality budget, each capture owns a private pool sized for
    its input shape.
    Not thread-safe. Capture dict mutation, static-buffer copy-in/out, and the class-level shared
    mempool all assume a single-threaded caller; do not share a runner (or concurrent first-time
    captures across runners) across threads.
    """

    # Shared mempool across runners so captures reuse intermediate memory (lazy init; not locked).
    _shared_pool: _POOL_HANDLE | None = None
    # Budgeted static inputs are process-wide, like the graph mempool (lazy init; not locked).
    _shared_video_pool: _InputBufferPool | None = None
    _shared_audio_pool: _InputBufferPool | None = None
    _shared_perturbations_pool = _InputBufferPool()

    def __init__(
        self,
        block_loop_fn: _BlockLoopFn,
        block_input_processor: BlockPerturbationsProcessor,
        warmup_iters: int = 3,
        max_video_tokens: int = 0,
        max_audio_tokens: int = 0,
    ) -> None:
        self._block_loop = block_loop_fn
        # Keys the capture on the block's perturbation graph signature -- the SAME processor the
        # wrapped block-loop uses, so a pass recaptures exactly when the block recompiles and the
        # signature can never diverge from the actual guards.
        self._block_input_processor = block_input_processor
        self._warmup_iters = warmup_iters
        self._captures: dict[_CaptureKey, _Capture] = {}
        self._max_video_tokens = max_video_tokens
        self._max_audio_tokens = max_audio_tokens
        if max_video_tokens:
            CudaGraphRunner._shared_video_pool = self._configure_shared_input_pool(
                CudaGraphRunner._shared_video_pool, max_video_tokens
            )
        if max_audio_tokens:
            CudaGraphRunner._shared_audio_pool = self._configure_shared_input_pool(
                CudaGraphRunner._shared_audio_pool, max_audio_tokens
            )

    def __call__(
        self, video: TransformerArgs | None, audio: TransformerArgs | None, perturbations: BatchedPerturbationConfig
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        key = self._cache_key(video, audio, perturbations)
        cap = self._captures.get(key)
        if cap is None:
            cap = self._capture(video, audio, perturbations)  # static inputs already hold this call's values
            self._captures[key] = cap
        else:
            # Rebind this call to the same pool buffers used during capture.
            if video is not None:
                cap.video_pool.copy_args_to_buffers(video)
            if audio is not None:
                cap.audio_pool.copy_args_to_buffers(audio)
            cap.perturbations_pool.copy_perturbations_to_buffers(perturbations)
        # Always replay (even right after capture); never return the capture-run outputs.
        cap.graph.replay()
        return cap.static_video_out, cap.static_audio_out

    def _cache_key(
        self, video: TransformerArgs | None, audio: TransformerArgs | None, perturbations: BatchedPerturbationConfig
    ) -> _CaptureKey:
        """Capture identity: input shapes + the block's perturbation graph signature + the enclosing
        :func:`capture_key_scope` key. Keying on the signature makes a pass recapture exactly when it
        would recompile (and no more) -- the perturbed pass no longer silently replays the clean
        same-shape graph."""
        shape = _shape_key(video, audio, perturbations.block_masks)
        return shape, self._block_input_processor.graph_signature(perturbations), _scope_key.get()

    @staticmethod
    def _configure_shared_input_pool(pool: _InputBufferPool | None, max_tokens: int) -> _InputBufferPool:
        if pool is None:
            return _InputBufferPool(max_tokens)
        if pool.max_tokens != max_tokens:
            raise ValueError(
                f"Shared input pool is already sized for {pool.max_tokens} tokens, "
                f"but another CUDA-graph runner requested {max_tokens}."
            )
        return pool

    def _capture(
        self, video: TransformerArgs | None, audio: TransformerArgs | None, perturbations: BatchedPerturbationConfig
    ) -> _Capture:
        video_pool = self._shared_video_pool if self._max_video_tokens else _InputBufferPool()
        audio_pool = self._shared_audio_pool if self._max_audio_tokens else _InputBufferPool()
        perturbations_pool = self._shared_perturbations_pool

        static_video = video_pool.copy_args_to_buffers(video) if video is not None else None
        static_audio = audio_pool.copy_args_to_buffers(audio) if audio is not None else None
        static_perturbations = perturbations_pool.copy_perturbations_to_buffers(perturbations)

        # Warmup on a side stream so any (first-shape) Dynamo/Inductor compile, autotune, cuBLAS
        # workspace, and IPC-kernel residency land OUTSIDE the capture. All ranks run this lockstep.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self._warmup_iters):
                self._block_loop(static_video, static_audio, static_perturbations)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        torch._C._cuda_clearCublasWorkspaces()  # avoid double-counting cuBLAS workspace into the pool

        # Restore this call's inputs (warmup may have written into the static buffers) so the capture
        # records with correct values and the post-capture replay yields this call's output.
        static_video = video_pool.copy_args_to_buffers(video) if video is not None else None
        static_audio = audio_pool.copy_args_to_buffers(audio) if audio is not None else None
        static_perturbations = perturbations_pool.copy_perturbations_to_buffers(perturbations)
        if CudaGraphRunner._shared_pool is None:
            CudaGraphRunner._shared_pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=CudaGraphRunner._shared_pool, capture_error_mode="global"):
            video_out, audio_out = self._block_loop(static_video, static_audio, static_perturbations)

        return _Capture(
            graph=graph,
            video_pool=video_pool,
            audio_pool=audio_pool,
            perturbations_pool=perturbations_pool,
            static_video_out=video_out,
            static_audio_out=audio_out,
        )
