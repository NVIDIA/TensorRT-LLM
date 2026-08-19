"""Utility functions for visual generation pipelines."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from tensorrt_llm._torch.modules.linear import Linear, UnquantizedLinearMethod
from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping


@torch.compile
def postprocess_video_tensor(video: torch.Tensor) -> torch.Tensor:
    """Post-process video tensor from VAE decoder output to final format.

    This is a more efficient implementation than using VideoProcessor for single-batch cases,
    as it avoids loop overhead and processes the entire batch with vectorized operations.

    Args:
        video: Video tensor in (B, C, T, H, W) format from VAE decoder

    Returns:
        Post-processed video tensor in (B, T, H, W, C) uint8 format.

    Note:
        Assumes video values are in [-1, 1] range (standard VAE decoder output).
    """
    # Convert to (B, T, H, W, C) format
    video = video.permute(0, 2, 3, 4, 1)  # (B, C, T, H, W) -> (B, T, H, W, C)

    # Normalize to [0, 1] range
    video = (video / 2 + 0.5).clamp(0, 1)

    # Convert to uint8
    video = (video * 255).round().to(torch.uint8)

    return video


def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)


def classify_worker_error(exc: BaseException) -> str | None:
    """Failure class for the response channel: "client", "capacity", or None.

    Keyed off built-in exception types rather than a VisualGen-specific
    hierarchy: ``ValueError`` means the request's content was unusable
    (400), ``MemoryError`` means a valid request did not fit (503), and
    anything else is an unclassified runtime failure (500). Detail travels
    in the message. ``torch.cuda.OutOfMemoryError`` is spelled out because
    it derives from ``RuntimeError``, not ``MemoryError``.
    """
    if isinstance(exc, (MemoryError, torch.cuda.OutOfMemoryError)):
        return "capacity"
    if isinstance(exc, ValueError):
        return "client"
    return None


def synchronize_media_prepare_status(exc: Exception | None) -> None:
    """All-rank convergence point between media prepare and model collectives.

    Every rank decodes/prepares its media independently; a rank that failed
    while others proceed into the transformer's collectives would hang the
    job. All ranks call this with their local outcome; if any failed, the
    lowest failing rank's error class + message is broadcast, the failing
    rank(s) re-raise their own exception, and every healthy rank raises a
    reconstructed equivalent in lockstep. Runs on CPU tensors so
    the hybrid (``cpu:gloo``) process group carries it even when the failure
    was CUDA/NVDEC initialization. Converges *caught* failures only — a fatal
    process or context death is beyond its reach.
    """
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        if exc is not None:
            raise exc
        return

    healthy_sentinel = 2**31 - 1
    rank = dist.get_rank()
    flag = torch.tensor([rank if exc is not None else healthy_sentinel], dtype=torch.int64)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    failing_rank = int(flag.item())
    if failing_rank == healthy_sentinel:
        return

    payload = [None]
    if rank == failing_rank:
        payload = [(classify_worker_error(exc), str(exc))]
    dist.broadcast_object_list(payload, src=failing_rank)

    if exc is not None:
        raise exc
    kind, message = payload[0]
    message = f"[rank {failing_rank}] {message}"
    if kind == "client":
        raise ValueError(message)
    if kind == "capacity":
        raise MemoryError(message)
    raise RuntimeError(message)


class SequenceSharder:
    """Block-shard / all-gather a tensor along its sequence dimension.

    A single ``SequenceSharder`` collapses the per-model
    ``if attn2d / elif ulysses / elif ring / else`` dispatch and the hand-rolled
    shard / gather of hidden states + RoPE into one model-agnostic helper.

    Built from a :class:`VisualGenMapping` via :meth:`from_vgm`, the sharder
    uses ``vgm.seq_size / seq_rank / seq_group`` so the same call sites work
    uniformly for sequence parallelism (CP × Ulysses).

    Models call ``shard(...)`` / ``gather(...)`` / ``shard_rope(...)`` directly;
    when the sharder is inactive (``size == 1``) every
    method is a no-op pass-through so the call sites do not need an
    ``if is_active`` guard.

    The sharder is intentionally model-agnostic: dimensions are passed
    explicitly at every call site and no model-specific shape conventions
    leak in (the sole exception is :meth:`shard_rope`, which infers the
    seq axis from a ``seq_len`` argument).
    """

    def __init__(
        self,
        size: int,
        rank: int,
        group: Optional[ProcessGroup],
        gather_index: Optional[List[int]] = None,
    ):
        self._size = size
        self._rank = rank
        self._group = group
        # Optional shard-order permutation: gather_index[s] = the GROUP rank that
        # holds shard s. Needed when shard indices don't follow group-rank order
        # (dist.new_group sorts its rank list, so a group built from a permuted
        # rank list still numbers members by ascending global rank).
        if gather_index is not None and sorted(gather_index) != list(range(size)):
            raise ValueError(
                f"gather_index must be a permutation of range({size}), got {gather_index}"
            )
        self._gather_index = gather_index

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_vgm(
        cls,
        vgm: Optional[VisualGenMapping],
        *,
        num_attention_heads: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
    ) -> "SequenceSharder":
        """Build a sharder from a :class:`VisualGenMapping`.

        Uses ``(cp_size * ulysses_size, seq_rank, seq_group)`` so the same
        sharder works for ring, attn2d, ulysses, ring + ulysses, and attn2d + ulysses.

        Validates head divisibility only when Ulysses is part of the seq
        group — ring and attn2d shard the sequence axis and have no
        head-count constraint.
        """
        if vgm is None:
            return cls(size=1, rank=0, group=None)

        size = vgm.seq_size
        rank = vgm.seq_rank
        group = vgm.seq_group()

        if size > 1 and group is None:
            raise ValueError(
                "SequenceSharder.from_vgm requires vgm.seq_group to be set when "
                f"vgm.seq_size ({size}) > 1; otherwise gather() would call "
                "dist.all_gather(..., group=None) and use the default process group."
            )

        if size > 1 and vgm.ulysses_size > 1:
            for label, count in (
                ("num_attention_heads", num_attention_heads),
                ("num_kv_heads", num_kv_heads),
            ):
                if count is not None and count % vgm.ulysses_size != 0:
                    raise ValueError(
                        f"{label}={count} must be divisible by ulysses_size={vgm.ulysses_size}"
                    )

        return cls(size=size, rank=rank, group=group)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    @property
    def is_active(self) -> bool:
        return self._size > 1

    @property
    def size(self) -> int:
        return self._size

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def group(self) -> Optional[ProcessGroup]:
        return self._group

    # ------------------------------------------------------------------
    # Shard
    # ------------------------------------------------------------------
    def shard(
        self,
        tensor: Optional[torch.Tensor],
        dim: int = 1,
        *,
        expected_seq_len: Optional[int] = None,
        pad_to_multiple: bool = False,
    ) -> Optional[torch.Tensor]:
        """Contiguous block-shard ``tensor`` along ``dim``.

        Returns ``tensor`` unchanged when:
          * the sharder is inactive (``size == 1``),
          * ``tensor is None``,
          * ``expected_seq_len`` is given and ``tensor.shape[dim]`` doesn't
            match — used by LTX2 to skip dataclass fields whose seq axis
            doesn't line up with the field being sharded.

        When ``pad_to_multiple`` is ``True``, the sequence dim is right-padded
        with zeros to a multiple of ``size`` before sharding.  The matching
        :meth:`gather` call must then pass ``unpad_to`` to slice the padding
        back off.
        """
        if tensor is None or not self.is_active:
            return tensor

        seq_len = tensor.shape[dim]
        if expected_seq_len is not None and seq_len != expected_seq_len:
            return tensor

        if pad_to_multiple and seq_len % self._size != 0:
            pad = self._size - (seq_len % self._size)
            pad_shape = list(tensor.shape)
            pad_shape[dim] = pad
            tensor = torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=dim)
            seq_len = tensor.shape[dim]

        if seq_len % self._size != 0:
            raise ValueError(
                f"Sequence length ({seq_len}) along dim {dim} is not "
                f"divisible by SequenceSharder.size ({self._size}). "
                f"Pass pad_to_multiple=True or adjust input dimensions."
            )

        chunk = seq_len // self._size
        start = self._rank * chunk
        idx = [slice(None)] * tensor.ndim
        idx[dim] = slice(start, start + chunk)
        # A dim>=1 block slice is non-contiguous when a leading dim is >1 (e.g.
        # batched CFG B=2); fused DiT kernels need a dense buffer. No-op at B==1.
        return tensor[tuple(idx)].contiguous()

    def shard_rope(
        self,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]],
        seq_len: int,
        *,
        seq_dim: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Shard a ``(cos, sin)`` RoPE pair along its sequence axis.

        Callers must pass ``seq_dim`` explicitly based on the known RoPE
        layout at the call site.
        """
        if rope is None or not self.is_active:
            return rope

        cos, sin = rope
        d = seq_dim if seq_dim >= 0 else cos.ndim + seq_dim
        if d < 0 or d >= cos.ndim:
            raise ValueError(f"seq_dim ({seq_dim}) is out of range for RoPE ndim ({cos.ndim}).")
        if cos.shape[d] != seq_len:
            raise ValueError(
                f"RoPE seq_dim ({d}) has size {cos.shape[d]}, expected seq_len ({seq_len})."
            )
        return (self.shard(cos, dim=d), self.shard(sin, dim=d))

    # ------------------------------------------------------------------
    # Gather
    # ------------------------------------------------------------------
    def gather(
        self,
        tensor: torch.Tensor,
        dim: int = 1,
        *,
        unpad_to: Optional[int] = None,
    ) -> torch.Tensor:
        """All-gather ``tensor`` along ``dim``.

        No-op when sharder is inactive.  ``unpad_to`` slices the gathered
        tensor's ``dim`` back to the given length; pair with
        ``shard(..., pad_to_multiple=True)`` to round-trip through padding.
        """
        if not self.is_active:
            return tensor

        tensor = tensor.contiguous()
        parts = [torch.empty_like(tensor) for _ in range(self._size)]
        dist.all_gather(parts, tensor, group=self._group)
        if self._gather_index is not None:
            parts = [parts[g] for g in self._gather_index]
        out = torch.cat(parts, dim=dim)

        if unpad_to is not None:
            idx = [slice(None)] * out.ndim
            idx[dim] = slice(0, unpad_to)
            out = out[tuple(idx)]
        return out


# ===========================================================================
# Packed joint-QKV projection (concat elimination for double-stream DiT).
# ===========================================================================
#
# Functional (alias-free) torch.library custom op: allocates the packed joint
# QKV buffer [1, S_txt + S_img, q_dim + 2*kv_dim] and writes the two
# per-stream packed projections straight into its row slices via
# ``torch.addmm(..., out=)``. Registering this as a custom op (the same
# pattern as ``trtllm::fused_dit_qk_norm_rope``) makes inductor treat it as
# an opaque extern call that OWNS its output buffer: when the same two
# ``addmm(out=slice)`` calls are inlined in a compiled region instead,
# functionalization materializes the addmm outputs and inductor does not
# re-inplace extern kernels into slice views, emitting a cat-sized copy-back
# kernel (measured 2.3% of a Qwen-Image denoise step in the qwenimage-fusion
# e2e profile of the inlined-addmm formulation).
#
# The op is deliberately FUNCTIONAL: it mutates no inputs (``mutates_args``
# is empty) and its output is a fresh tensor aliasing no input, which is the
# alias contract torch.compile supports cleanly (no auto_functionalize
# wrapper, no defensive clones). ``register_fake`` gives tracing the
# shape/dtype-only meta implementation (SymInt-safe).
#
# The op reads the merged QKV Linears' native ``weight``/``bias`` parameters
# as plain tensors. It is deliberately NOT routed through
# ``LinearMethod.apply``: a custom op cannot take a module argument, and only
# the plain addmm/mm path can write into slice views via ``out=``. It is
# therefore restricted to unquantized bf16 Linears — callers must gate on
# ``linear_supports_packed_addmm`` below. Quantized recipes extend this
# pattern with sibling functional leaf ops per GEMM recipe (e.g. FlashInfer's
# ``gemm_fp8_nt_groupwise`` / ``mm_fp4`` accept ``out=``), reading the merged
# Linear's quantized weight/scale parameters the same way — they must not
# reuse this op.


@torch.library.custom_op("trtllm_vgoa::packed_qkv_proj", mutates_args=())
def _packed_qkv_proj(
    encoder_hidden_states: torch.Tensor,
    hidden_states: torch.Tensor,
    txt_weight_packed: torch.Tensor,
    txt_bias_packed: torch.Tensor,
    img_weight_packed: torch.Tensor,
    img_bias_packed: torch.Tensor,
) -> torch.Tensor:
    """Concat-free packed joint QKV: project into ``out=`` row slices.

    Rows [0, S_txt) are the text stream, rows [S_txt, S_txt + S_img) the
    image stream, columns [q | k | v] — bit-compatible with the per-stream
    merged projection + seq-dim ``torch.cat`` layout. Both row-slices of the
    freshly allocated buffer are contiguous 2-D matrices (B == 1, enforced
    by the caller's runtime guard), so eager addmm takes the cublasLt
    bias-epilogue path with zero extra elementwise kernels.
    """
    s_txt = encoder_hidden_states.shape[1]
    s_img = hidden_states.shape[1]
    packed_dim = txt_weight_packed.shape[0]
    qkv = hidden_states.new_empty((1, s_txt + s_img, packed_dim))
    qkv_rows = qkv.view(s_txt + s_img, packed_dim)
    torch.addmm(
        txt_bias_packed,
        encoder_hidden_states[0],
        txt_weight_packed.t(),
        out=qkv_rows[:s_txt],
    )
    torch.addmm(
        img_bias_packed,
        hidden_states[0],
        img_weight_packed.t(),
        out=qkv_rows[s_txt:],
    )
    return qkv


@_packed_qkv_proj.register_fake
def _packed_qkv_proj_fake(
    encoder_hidden_states: torch.Tensor,
    hidden_states: torch.Tensor,
    txt_weight_packed: torch.Tensor,
    txt_bias_packed: torch.Tensor,
    img_weight_packed: torch.Tensor,
    img_bias_packed: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype-only fake impl for tracing (dynamo/fake-tensor)."""
    s_txt = encoder_hidden_states.shape[1]
    s_img = hidden_states.shape[1]
    packed_dim = txt_weight_packed.shape[0]
    return hidden_states.new_empty((1, s_txt + s_img, packed_dim))


def linear_supports_packed_addmm(linear: Linear, *, out_features: int, in_features: int) -> bool:
    """Whether ``linear`` may be computed by raw ``torch.addmm`` on its
    native ``weight``/``bias`` (the ``trtllm_vgoa::packed_qkv_proj`` lane).

    The packed op bypasses ``Linear.forward`` / ``quant_method.apply``, so
    this census must reject every Linear feature the bypass would silently
    skip: quantized methods (strict type check — the FP8 linear methods
    subclass ``UnquantizedLinearMethod``), non-bf16 or non-CUDA weights,
    missing bias (the op signature requires one), TP collectives
    (``gather_output``), LoRA, and the non-default GEMM backends. Callers add
    their own model-level gating (TP size, fused-rope preconditions, batch
    shape) on top.
    """
    if type(linear.quant_method) is not UnquantizedLinearMethod:
        return False
    weight = getattr(linear, "weight", None)
    bias = getattr(linear, "bias", None)
    if weight is None or bias is None:
        return False
    if weight.dtype != torch.bfloat16 or bias.dtype != torch.bfloat16:
        return False
    if not weight.is_cuda:
        return False
    if weight.shape[0] != out_features or weight.shape[1] != in_features:
        return False
    if bias.shape[0] != out_features:
        return False
    # The packed addmm bypasses Linear.forward: it must not skip an
    # allgather, a LoRA branch, or a non-default GEMM backend.
    if linear.gather_output or getattr(linear, "lora", None) is not None:
        return False
    if linear.use_custom_cublas_mm or linear.use_cute_dsl_bf16_gemm:
        return False
    return True
