# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PLE n-gram short-convolution side path for Qwen4-Exp models.

PLE hashes eos-delimited n-gram contexts, looks up a row-sharded embedding,
gates it with the Hyper-Connection stream, and applies a dilated causal
depthwise convolution. The caller owns two recurrent pools indexed by
``PLEMetadata.state_indices``: convolution history initialized to zero and
``ngram_size - 1`` token IDs initialized to ``eos_token_id``. ``forward``
updates both pools in place so chunked prefill, IFB decode, and speculative
verification share the same state lifecycle as other recurrent model layers.
"""

import dataclasses
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from tensorrt_llm._torch.distributed import AllReduce, allgather, reducescatter
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._utils import CUASSERT, prefer_pinned
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from .hyper_connection import GroupedRMSNorm
from .ple_kernels import (
    can_use_ple_decode_short_conv,
    can_use_ple_gate_value,
    can_use_ple_ngram_hash,
    can_use_ple_short_conv_state,
    ple_decode_short_conv,
    ple_gate_value,
    ple_ngram_hash,
    ple_short_conv_state,
)

# SplitMix64 constants are part of the checkpoint's PLE hashing contract.
_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007
_PLE_HOST_OFFLOAD_ENV = "TRTLLM_QWEN4_EXP_PLE_HOST_OFFLOAD"


def _uses_ple_host_offload() -> bool:
    """Resolve the environment-only opt-in for pinned-host PLE weights."""
    value = os.environ.get(_PLE_HOST_OFFLOAD_ENV, "0").strip().lower()
    if value in ("", "0", "false", "no", "off"):
        return False
    if value in ("1", "true", "yes", "on"):
        return True
    raise ValueError(f"{_PLE_HOST_OFFLOAD_ENV} must be a boolean value, got {value!r}")


def _uses_scaled_fp8_ngram_table(config: object) -> bool:
    """Return whether the HF config declares the custom scaled-FP8 PLE table."""
    quantization_config = getattr(config, "quantization_config", None)
    if not isinstance(quantization_config, dict):
        return False
    if quantization_config.get("quant_method") != "fp8":
        return False
    excluded = quantization_config.get("modules_to_not_convert") or ()
    if isinstance(excluded, str):
        excluded = (excluded,)
    marker = "ple.ple_embedding.ngram_embedding"
    return not any(marker in module_name for module_name in excluded)


def _first_eos_token_id(value: object) -> int:
    """Normalize the HF scalar-or-list EOS representation used by PLE."""
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("PLE requires at least one EOS token ID")
        value = value[0]
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"PLE received an invalid EOS token ID: {value!r}") from error


def _splitmix64(x: int) -> int:
    """The finalizer mix used to seed the per-layer n-gram hash multipliers."""
    x = (x + _SPLITMIX_GAMMA) & _MASK64
    x = ((x ^ (x >> 30)) * _SPLITMIX_M1) & _MASK64
    x = ((x ^ (x >> 27)) * _SPLITMIX_M2) & _MASK64
    return (x ^ (x >> 31)) & _MASK64


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _next_prime(start: int) -> int:
    """Smallest prime strictly greater than ``start`` (matches sympy.nextprime)."""
    candidate = int(start) + 1
    while not _is_prime(candidate):
        candidate += 1
    return candidate


def _find_nth_prime_after(start: int, n: int) -> int:
    """The ``n``-th prime strictly greater than ``start`` (``n >= 1``).

    Equivalent to ``n`` successive ``sympy.nextprime`` calls from ``start``; a
    dependency-free primality test keeps the runtime module import-light.
    """
    prime = int(start)
    for _ in range(n):
        prime = _next_prime(prime)
    return prime


def _pad_token_rows(x: torch.Tensor, total_tokens: int) -> torch.Tensor:
    """Zero-pad the leading (token) dim up to ``total_tokens`` (CUDA-graph pad)."""
    if x.shape[0] == total_tokens:
        return x
    out = x.new_zeros((total_tokens, *x.shape[1:]))
    out[: x.shape[0]] = x
    return out


@dataclasses.dataclass
class PLEMetadata:
    """Packed-token layout and recurrent-state slots for one PLE forward.

    All index and length tensors live on the same device as ``input_ids``.
    """

    is_decode: bool
    physical_tokens: int
    processed_tokens: int
    lengths: torch.Tensor  # [num_seq] this-chunk token count per sequence
    row_width: int  # max chunk length (== 1 for decode)
    req_indices: torch.Tensor  # [processed_tokens] -> sequence index
    token_offsets: torch.Tensor  # [processed_tokens] -> offset within the chunk
    valid_tokens: torch.Tensor  # [processed_tokens] bool
    # The scheduler assigns one distinct recurrent-state slot per live request.
    state_indices: torch.Tensor  # [num_seq] -> recurrent-state pool slot
    padded_tokens: torch.Tensor  # [num_seq, row_width] eos-padded token ids
    ngram_eos_token_id: int
    all_rank_num_tokens: Optional[list[int]] = None
    num_contexts: int = 0
    context_tokens: int = 0
    use_spec_decoding: bool = False
    is_cuda_graph: bool = False

    @classmethod
    def build(
        cls,
        input_ids: torch.Tensor,
        seq_lens: torch.Tensor,
        state_indices: torch.Tensor,
        *,
        is_decode: bool,
        eos_token_id: int,
        physical_tokens: Optional[int] = None,
        num_contexts: int = 0,
        use_spec_decoding: bool = False,
        uniform_row_width: Optional[int] = None,
        host_seq_lens: Optional[list[int]] = None,
        all_rank_num_tokens: Optional[list[int]] = None,
        is_cuda_graph: bool = False,
    ) -> "PLEMetadata":
        """Construct metadata from packed ``input_ids`` and per-sequence lengths.

        ``input_ids`` is the flattened, sequence-packed token id vector of length
        ``processed_tokens``. For prefill/extend, ``seq_lens[i]`` is sequence
        ``i``'s chunk length and the tokens are packed contiguously per sequence.
        For decode there is exactly one new token per sequence, so
        ``num_seq == processed_tokens`` and ``seq_lens`` is ignored.
        """
        input_ids = input_ids.reshape(-1)
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("PLE input_ids must use integer storage")
        if seq_lens.ndim != 1 or seq_lens.dtype not in (torch.int32, torch.int64):
            raise ValueError("PLE sequence lengths must be a one-dimensional integer tensor")
        if state_indices.ndim != 1 or state_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("PLE state indices must be a one-dimensional integer tensor")
        device = input_ids.device
        processed_tokens = int(input_ids.shape[0])
        if physical_tokens is None:
            physical_tokens = processed_tokens
        if physical_tokens < processed_tokens:
            raise ValueError(
                f"PLE physical token count {physical_tokens} is smaller than "
                f"the {processed_tokens} processed tokens"
            )
        positions = torch.arange(processed_tokens, device=device, dtype=torch.long)

        if is_decode:
            num_seq = processed_tokens
            if state_indices.numel() != num_seq:
                raise ValueError("PLE decode requires one recurrent-state slot per token")
            lengths = torch.ones(num_seq, dtype=torch.long, device=device)
            row_width = 1
            req_indices = positions
            token_offsets = torch.zeros_like(positions)
            context_tokens = 0
        elif uniform_row_width is not None:
            if uniform_row_width <= 0:
                raise ValueError("PLE uniform row width must be positive")
            num_seq = state_indices.shape[0]
            if processed_tokens != num_seq * uniform_row_width:
                raise ValueError(
                    "PLE uniform layout does not match the processed token count: "
                    f"{processed_tokens} != {num_seq} * {uniform_row_width}"
                )
            row_width = uniform_row_width
            lengths = torch.full((num_seq,), row_width, dtype=torch.long, device=device)
            req_indices = torch.div(positions, row_width, rounding_mode="floor")
            token_offsets = positions.remainder(row_width)
            context_tokens = num_contexts * row_width
        else:
            lengths = seq_lens.to(device=device, dtype=torch.long)
            if host_seq_lens is None:
                seq_lens_cpu = lengths.tolist()
            else:
                if len(host_seq_lens) != lengths.shape[0]:
                    raise ValueError(
                        "PLE host sequence lengths do not match the device batch: "
                        f"{len(host_seq_lens)} != {lengths.shape[0]}"
                    )
                # The model runner already owns this host view of the same
                # attention lengths. Reusing it avoids a device-to-host sync
                # solely to derive Python layout scalars.
                seq_lens_cpu = host_seq_lens
            if any(length < 0 for length in seq_lens_cpu):
                raise ValueError("PLE sequence lengths must be non-negative")
            if sum(seq_lens_cpu) != processed_tokens:
                raise ValueError(
                    "PLE sequence lengths do not match the packed token count: "
                    f"{sum(seq_lens_cpu)} != {processed_tokens}"
                )
            row_width = max(seq_lens_cpu) if seq_lens_cpu else processed_tokens
            num_seq = lengths.shape[0]
            context_tokens = sum(seq_lens_cpu[:num_contexts])
            query_start_loc = torch.cat([lengths.new_zeros(1), torch.cumsum(lengths, dim=0)])
            req_indices = torch.searchsorted(query_start_loc, positions, right=True) - 1
            # Keep this tensor-only: ``processed_tokens`` can be symbolic under
            # torch.compile, and the empty-token assignment below is already a
            # no-op. A Python truth-value guard would force a host-side data
            # dependency and prevent Dynamo from compiling the model forward.
            if num_seq:
                req_indices = req_indices.clamp(min=0, max=num_seq - 1)
            elif processed_tokens:
                raise ValueError("PLE received packed tokens without a sequence")
            token_offsets = positions - query_start_loc.index_select(0, req_indices)

        num_seq = lengths.shape[0]
        if state_indices.numel() != num_seq:
            raise ValueError("PLE requires one recurrent-state slot per sequence")
        if not 0 <= num_contexts <= num_seq:
            raise ValueError(f"PLE num_contexts must be in [0, {num_seq}], got {num_contexts}")
        valid_tokens = token_offsets < lengths.index_select(0, req_indices)
        state_indices = state_indices.to(device=device, dtype=torch.long)

        padded = input_ids.new_full((num_seq, row_width), eos_token_id)
        padded[req_indices, token_offsets] = torch.where(
            valid_tokens, input_ids, input_ids.new_full((), eos_token_id)
        )

        return cls(
            is_decode=is_decode,
            physical_tokens=physical_tokens,
            processed_tokens=processed_tokens,
            lengths=lengths,
            row_width=row_width,
            req_indices=req_indices,
            token_offsets=token_offsets,
            valid_tokens=valid_tokens,
            state_indices=state_indices,
            padded_tokens=padded,
            ngram_eos_token_id=eos_token_id,
            all_rank_num_tokens=all_rank_num_tokens,
            num_contexts=num_contexts,
            context_tokens=context_tokens,
            use_spec_decoding=use_spec_decoding,
            is_cuda_graph=is_cuda_graph,
        )


@triton.jit
def _gather_pinned_embedding_kernel(
    weight_ptr,
    ids_ptr,
    output_ptr,
    weight_scale,
    embedding_dim,
    vocab_start,
    vocab_end,
    is_fp8: tl.constexpr,
    apply_weight_scale: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Gather owned rows through UVA and write zero for another rank's rows."""
    row = tl.program_id(0)
    global_id = tl.load(ids_ptr + row)
    is_owned = (global_id >= vocab_start) & (global_id < vocab_end)
    local_id = tl.where(is_owned, global_id - vocab_start, 0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < embedding_dim
    if is_fp8:
        weight_ptr = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
    else:
        weight_ptr = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.bfloat16))
    values = tl.load(
        weight_ptr + local_id * embedding_dim + offsets,
        mask=is_owned & mask,
        other=0.0,
    ).to(tl.bfloat16)
    if apply_weight_scale:
        # Preserve the checkpoint contract: FP8 -> BF16 -> FP32 scale -> BF16.
        values = (values.to(tl.float32) * weight_scale).to(tl.bfloat16)
    tl.store(output_ptr + row * embedding_dim + offsets, values, mask=mask)


class Qwen4ExpPinnedHostEmbedding(nn.Module):
    """A row-sharded PLE table held in pinned host memory and read through UVA."""

    _requires_standard_hf_loading = True

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        dtype: torch.dtype,
        vocab_start_index: int,
        vocab_end_index: int,
    ) -> None:
        super().__init__()
        if dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            raise TypeError(f"PLE host offload requires BF16 or FP8 weights, got {dtype}")
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.vocab_start_index = int(vocab_start_index)
        self.vocab_end_index = int(vocab_end_index)
        self._block_d = triton.next_power_of_2(self.embedding_dim)
        self._mapped_host_ptr: Optional[int] = None
        self._mapped_device_ptrs: dict[int, int] = {}
        self.weight = nn.Parameter(
            torch.empty(
                (self.num_embeddings, self.embedding_dim),
                device="meta",
                dtype=dtype,
            ),
            requires_grad=False,
        )

    def materialize_pinned(self) -> nn.Parameter:
        """Create the stable pinned allocation exactly once."""
        weight = self.weight
        if weight.device.type == "cpu" and weight.is_pinned():
            return weight
        if weight.device.type != "meta":
            raise RuntimeError(
                f"PLE host-offload weight must be meta or pinned CPU memory, got {weight.device}"
            )
        if not prefer_pinned():
            raise RuntimeError("PLE host offload requires the pinned-memory runtime policy")
        pinned = nn.Parameter(
            torch.empty(
                weight.shape,
                device="cpu",
                dtype=weight.dtype,
                pin_memory=prefer_pinned(),
            ),
            requires_grad=False,
        )
        if not pinned.is_pinned():
            raise RuntimeError("PLE host-offload allocation is not pinned")
        self.register_parameter("weight", pinned)
        return pinned

    def _mapped_device_ptr(self, device: torch.device) -> int:
        """Return the CUDA-visible address for the stable host allocation."""
        weight = self.materialize_pinned()
        host_ptr = weight.data_ptr()
        if self._mapped_host_ptr is not None and self._mapped_host_ptr != host_ptr:
            raise RuntimeError("PLE pinned-host allocation changed after pointer mapping")

        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        mapped_ptr = self._mapped_device_ptrs.get(device_index)
        if mapped_ptr is not None:
            return mapped_ptr
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("PLE host pointer must be resolved before CUDA graph capture")

        from cuda.bindings import runtime as cudart

        with torch.cuda.device(device_index):
            (device_ptr,) = CUASSERT(cudart.cudaHostGetDevicePointer(host_ptr, 0))
        mapped_ptr = int(device_ptr)
        if mapped_ptr == 0:
            raise RuntimeError("CUDA returned a null PLE host-mapping pointer")
        self._mapped_host_ptr = host_ptr
        self._mapped_device_ptrs[device_index] = mapped_ptr
        return mapped_ptr

    def _apply(self, fn, recurse: bool = True) -> "Qwen4ExpPinnedHostEmbedding":
        """Apply module transforms without moving or replacing the host table."""
        weight = self._parameters.pop("weight")
        try:
            result = super()._apply(fn, recurse=recurse)
        finally:
            self._parameters["weight"] = weight
        if weight.device.type == "meta":
            self.materialize_pinned()
        return result

    def allocate_output(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Allocate the BF16 device rows written by the UVA gather."""
        return torch.empty(shape, dtype=torch.bfloat16, device=device)

    def gather(
        self,
        input_ids: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        *,
        weight_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Gather global row IDs, returning zero for IDs owned by another rank."""
        if input_ids.device.type != "cuda":
            raise ValueError("PLE pinned-host gather requires CUDA input IDs")
        weight = self.materialize_pinned()
        if weight_scale is not None and weight.dtype != torch.float8_e4m3fn:
            raise ValueError("PLE pinned-host scaling is valid only for FP8 tables")

        expected_shape = (*input_ids.shape, self.embedding_dim)
        output = self.allocate_output(expected_shape, input_ids.device) if out is None else out
        if tuple(output.shape) != expected_shape:
            raise ValueError(f"PLE gather output shape {tuple(output.shape)} != {expected_shape}")
        if output.dtype != torch.bfloat16 or output.device != input_ids.device:
            raise ValueError("PLE gather output must be BF16 on the input-ID device")
        if not output.is_contiguous():
            raise ValueError("PLE gather output must be contiguous")

        flat_ids = input_ids.reshape(-1).long()
        if flat_ids.numel() > 0:
            _gather_pinned_embedding_kernel[(flat_ids.numel(),)](
                self._mapped_device_ptr(input_ids.device),
                flat_ids,
                output,
                1.0 if weight_scale is None else weight_scale,
                embedding_dim=self.embedding_dim,
                vocab_start=self.vocab_start_index,
                vocab_end=self.vocab_end_index,
                is_fp8=weight.dtype == torch.float8_e4m3fn,
                apply_weight_scale=weight_scale is not None,
                BLOCK_D=self._block_d,
            )
        return output


class Qwen4ExpNGramEmbedding(nn.Module):
    """Hashes per-token n-gram contexts to per-head ids and embeds them.

    The checkpoint contract fixes the SplitMix64 multipliers, prime per-head
    vocabulary sizes, EOS-delimited shifts, and XOR mixing. The embedding table
    is row-sharded across the model TP group. Ordinary TP uses masked lookup
    plus AllReduce; attention-DP uses token AllGather, masked local lookup, and
    ReduceScatter.
    """

    def __init__(
        self,
        config: object,
        embedding_dim: int,
        ple_layer_index: int = 0,
        dtype: Optional[torch.dtype] = None,
        mapping: Optional[Mapping] = None,
    ) -> None:
        super().__init__()
        self.ngram_embed_dim = int(embedding_dim)
        self.ngram_size = int(config.ngram_size)
        self.heads_per_ngram = int(config.heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.ple_layer_index = int(ple_layer_index)
        self.unigram_vocab_size = int(config.vocab_size)
        self.eos_token_id = _first_eos_token_id(config.eos_token_id)
        self.seed = int(getattr(config, "seed", 1234))
        self.mapping = mapping
        self.tp_size = mapping.tp_size if mapping is not None else 1
        self.tp_rank = mapping.tp_rank if mapping is not None else 0
        if self.tp_size <= 0 or not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("PLE received an invalid tensor-parallel mapping")
        self.embedding_output_dtype = dtype
        self.host_offload = _uses_ple_host_offload()
        self.use_attention_dp_sharding = bool(
            mapping is not None and mapping.enable_attention_dp and self.tp_size > 1
        )
        if self.use_attention_dp_sharding and mapping is not None and mapping.cp_size > 1:
            raise NotImplementedError(
                "Qwen4-Exp PLE row sharding does not support attention DP "
                "combined with context parallelism"
            )
        self.register_buffer(
            "ngram_embedding_weight_scale",
            None,
            persistent=False,
        )
        self._host_offload_weight_scale: Optional[float] = None

        if self.ngram_embed_dim <= 0:
            raise ValueError("PLE embedding dimension must be positive")
        if self.ple_layer_index < 0:
            raise ValueError("PLE layer index must be non-negative")
        if self.unigram_vocab_size <= 0:
            raise ValueError("PLE unigram vocabulary size must be positive")
        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}")
        if self.heads_per_ngram <= 0:
            raise ValueError(f"heads_per_ngram must be > 0, got {self.heads_per_ngram}")
        if self.ngram_embed_dim % self.ngram_heads != 0:
            raise ValueError(
                "ple_embed_dim must be divisible by total ngram heads: "
                f"{self.ngram_embed_dim} % {self.ngram_heads} != 0"
            )

        self.ngram_vocab_size_base = int(config.ngram_vocab_size_base)
        if self.ngram_vocab_size_base <= 0:
            raise ValueError("ngram_vocab_size_base must be > 0")
        self.make_ngram_vocab_size_divisible_by = int(config.make_ngram_vocab_size_divisible_by)
        if self.make_ngram_vocab_size_divisible_by <= 0:
            raise ValueError("make_ngram_vocab_size_divisible_by must be > 0")
        self.head_dim_per_ngram = self.ngram_embed_dim // self.ngram_heads

        self.register_buffer(
            "layer_multipliers",
            self._build_layer_multipliers(self.ngram_size),
            persistent=True,
        )
        head_vocab_sizes, head_offsets, total_vocab_size = self._build_head_vocab_and_offsets()
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(head_vocab_sizes, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.tensor(head_offsets, dtype=torch.long),
            persistent=True,
        )
        padded_vocab_size = (
            (total_vocab_size + self.make_ngram_vocab_size_divisible_by - 1)
            // self.make_ngram_vocab_size_divisible_by
        ) * self.make_ngram_vocab_size_divisible_by
        self.padded_vocab_size = padded_vocab_size
        weight_dtype = torch.float8_e4m3fn if _uses_scaled_fp8_ngram_table(config) else dtype
        slice_width = math.ceil(padded_vocab_size / self.tp_size)
        self.vocab_start_index = self.tp_rank * slice_width
        self.vocab_end_index = min((self.tp_rank + 1) * slice_width, padded_vocab_size)
        if self.host_offload:
            if dtype != torch.bfloat16:
                raise TypeError(
                    "PLE host offload currently gathers BF16 activations; "
                    f"got model activation dtype {dtype}"
                )
            self.ngram_embedding = Qwen4ExpPinnedHostEmbedding(
                slice_width,
                self.head_dim_per_ngram,
                dtype=weight_dtype,
                vocab_start_index=self.vocab_start_index,
                vocab_end_index=self.vocab_end_index,
            )
            self.embedding_allreduce = (
                AllReduce(mapping=mapping, dtype=torch.bfloat16)
                if self.tp_size > 1 and not self.use_attention_dp_sharding
                else None
            )
            logger.info(
                "PLE n-gram table uses pinned host memory: "
                f"rank={self.tp_rank}/{self.tp_size}, local_rows={slice_width}, "
                f"dtype={weight_dtype}"
            )
        elif self.use_attention_dp_sharding:
            self.ngram_embedding = Embedding(
                slice_width,
                self.head_dim_per_ngram,
                dtype=weight_dtype,
            )
            self.embedding_allreduce = None
        elif self.tp_size > 1:
            self.ngram_embedding = Embedding(
                padded_vocab_size,
                self.head_dim_per_ngram,
                dtype=weight_dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
            )
            self.vocab_start_index = self.ngram_embedding.vocab_start_index
            self.vocab_end_index = self.ngram_embedding.vocab_end_index
            self.embedding_allreduce = None
        else:
            self.vocab_start_index = 0
            self.vocab_end_index = padded_vocab_size
            self.ngram_embedding = Embedding(
                padded_vocab_size,
                self.head_dim_per_ngram,
                dtype=weight_dtype,
            )
            self.embedding_allreduce = None

    def configure_fp8_weight_storage(
        self,
        weight_scale: torch.Tensor,
        weight_dtype: torch.dtype,
    ) -> None:
        """Keep a scaled-FP8 n-gram table quantized until after lookup."""
        if weight_scale.numel() != 1:
            raise ValueError(
                f"PLE n-gram weight scale must be scalar, got shape {tuple(weight_scale.shape)}"
            )
        scale = float(weight_scale.item())
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"PLE n-gram weight scale must be finite and positive, got {scale}")
        if weight_dtype != torch.float8_e4m3fn:
            raise ValueError(f"PLE scaled weight dtype must be float8_e4m3fn, got {weight_dtype}")

        weight = self.ngram_embedding.weight
        if weight.dtype != weight_dtype:
            if self.host_offload:
                raise TypeError(
                    "PLE host-offload checkpoint dtype does not match its "
                    f"preallocated table: {weight_dtype} != {weight.dtype}"
                )
            weight_shape = weight.shape
            weight_device = weight.device
            # Release the old table before allocating its FP8 replacement;
            # retaining both checkpoint-sized Parameters would inflate peak
            # device memory during weight loading.
            self.ngram_embedding.register_parameter("weight", None)
            del weight
            self.ngram_embedding.weight = nn.Parameter(
                torch.empty(weight_shape, device=weight_device, dtype=weight_dtype),
                requires_grad=False,
            )
        configured_scale = (
            weight_scale.detach()
            .reshape(())
            .to(
                device=self.layer_multipliers.device,
                dtype=torch.float32,
            )
        )
        if self.ngram_embedding_weight_scale is None:
            self.ngram_embedding_weight_scale = configured_scale
        else:
            if self.ngram_embedding_weight_scale.device != configured_scale.device:
                raise RuntimeError("PLE n-gram scale device cannot change after configuration")
            self.ngram_embedding_weight_scale.copy_(configured_scale)
        if self.host_offload:
            self._host_offload_weight_scale = scale
        logger.info(
            "Qwen4-Exp PLE n-gram table configured for scaled FP8 storage: "
            f"dtype={self.ngram_embedding.weight.dtype}, "
            f"scale={scale}"
        )

    def _dequantize_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.ngram_embedding_weight_scale is None:
            return embeddings
        output_dtype = self.embedding_output_dtype or torch.get_default_dtype()
        return embeddings.float().mul_(self.ngram_embedding_weight_scale).to(output_dtype)

    def _embed_fp8_tp(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        """Look up a row-sharded FP8 table and communicate BF16 activations."""
        owned = (ngram_ids >= self.vocab_start_index) & (ngram_ids < self.vocab_end_index)
        local_ids = torch.where(
            owned,
            ngram_ids - self.vocab_start_index,
            torch.zeros_like(ngram_ids),
        )
        partial = F.embedding(local_ids, self.ngram_embedding.weight)
        partial = self._dequantize_embeddings(partial)
        partial.masked_fill_(~owned.unsqueeze(-1), 0)
        if self.tp_size > 1:
            partial = self.ngram_embedding.all_reduce(partial)
        return partial

    def _prepare_embedding_lookup(
        self,
        ngram_ids: torch.Tensor,
        physical_tokens: Optional[int],
        all_rank_num_tokens: Optional[list[int]],
    ) -> tuple[torch.Tensor, int]:
        """Prepare global IDs for a local gather, including ADP token exchange."""
        semantic_tokens = ngram_ids.shape[0]
        if not self.use_attention_dp_sharding:
            return ngram_ids, semantic_tokens

        if all_rank_num_tokens is None or len(all_rank_num_tokens) != self.tp_size:
            raise ValueError(
                "PLE row sharding under attention DP requires one token count per TP rank"
            )
        if physical_tokens is None:
            physical_tokens = semantic_tokens
        if semantic_tokens > physical_tokens:
            raise ValueError(
                f"PLE semantic token count {semantic_tokens} exceeds physical "
                f"token count {physical_tokens}"
            )
        if all_rank_num_tokens[self.tp_rank] != physical_tokens:
            raise ValueError(
                "PLE local physical token count does not match attention-DP metadata: "
                f"{physical_tokens} != {all_rank_num_tokens[self.tp_rank]}"
            )

        physical_ids = ngram_ids.new_zeros((physical_tokens, ngram_ids.shape[1]))
        physical_ids[:semantic_tokens] = ngram_ids
        sizes = None if len(set(all_rank_num_tokens)) == 1 else all_rank_num_tokens
        return allgather(physical_ids, self.mapping, dim=0, sizes=sizes), semantic_tokens

    def _finish_embedding_lookup(
        self,
        partial: torch.Tensor,
        semantic_tokens: int,
        physical_tokens: Optional[int],
        all_rank_num_tokens: Optional[list[int]],
    ) -> torch.Tensor:
        """Reduce row-sharded lookup results and restore local token ownership."""
        # The host gather fuses FP8 scaling before communication. Resident
        # tables still dequantize the selected rows here.
        if not self.host_offload:
            partial = self._dequantize_embeddings(partial)
        if self.use_attention_dp_sharding:
            if all_rank_num_tokens is None:
                raise ValueError("PLE attention-DP token counts are missing")
            sizes = None if len(set(all_rank_num_tokens)) == 1 else all_rank_num_tokens
            partial = reducescatter(partial, self.mapping, dim=0, sizes=sizes)
            if physical_tokens is None:
                physical_tokens = partial.shape[0]
            partial = partial[:physical_tokens]
        elif self.tp_size > 1 and self.host_offload:
            if self.embedding_allreduce is None:
                raise RuntimeError("PLE host-offload TP lookup is missing its AllReduce")
            partial = self.embedding_allreduce(partial)

        return partial[:semantic_tokens]

    def _build_layer_multipliers(self, size: int) -> torch.Tensor:
        max_long = (1 << 63) - 1
        m_max = max_long // max(self.unigram_vocab_size, 1)
        half_bound = max(1, m_max // 2)
        values = []
        base_seed = self.seed + _PRIME_1 * self.ple_layer_index
        for idx in range(size):
            x0 = (base_seed + _SPLITMIX_GAMMA * (idx + 1)) & _MASK64
            mixed = _splitmix64(x0)
            values.append(int(2 * (mixed % half_bound) + 1))
        return torch.tensor(values, dtype=torch.long)

    def _build_head_vocab_and_offsets(self) -> tuple[list[int], list[int], int]:
        sizes = []
        offsets = []
        total = 0
        for head_idx in range(self.ngram_heads):
            global_head_idx = self.ple_layer_index * self.ngram_heads + head_idx
            size = _find_nth_prime_after(self.ngram_vocab_size_base - 1, global_head_idx + 1)
            sizes.append(size)
            offsets.append(total)
            total += size
        return sizes, offsets, total

    def _shift_right_ignore_eos(self, tensor: torch.Tensor, n: int) -> torch.Tensor:
        """Right-shift each row by ``n`` within eos-delimited segments.

        Positions that would pull a token across an EOS boundary (or off the
        left edge) are filled with ``eos_token_id``, so an n-gram never mixes
        tokens from two different segments.
        """
        if n == 0:
            return tensor
        batch_size, seq_len = tensor.shape
        idx = torch.arange(seq_len, device=tensor.device, dtype=torch.long)
        eos_mask = tensor == self.eos_token_id
        eos_pos = torch.where(eos_mask, idx, -1)
        prev_eos_inclusive = torch.cummax(eos_pos, dim=1).values
        prev_eos = torch.cat(
            [eos_pos.new_full((batch_size, 1), -1), prev_eos_inclusive[:, :-1]],
            dim=1,
        )
        segment_start = prev_eos + 1
        pos_in_segment = idx.unsqueeze(0) - segment_start
        src_idx = idx - n
        gather_idx = torch.clamp(src_idx, min=0).unsqueeze(0).expand(batch_size, -1)
        shifted = tensor.gather(dim=1, index=gather_idx)
        valid_mask = (pos_in_segment >= n) & (src_idx.unsqueeze(0) >= 0)
        return torch.where(valid_mask, shifted, tensor.new_full((), self.eos_token_id))

    def hash_contexts(
        self,
        contexts: torch.Tensor,
        *,
        use_decode_fusion: bool = False,
    ) -> torch.Tensor:
        """Hash n-gram windows ``[T, ngram_size]`` to head ids ``[T, ngram_heads]``."""
        if contexts.ndim != 2 or contexts.shape[1] != self.ngram_size:
            raise ValueError(
                f"PLE contexts must have shape [tokens, {self.ngram_size}], "
                f"got {tuple(contexts.shape)}"
            )
        if contexts.dtype not in (torch.int32, torch.int64):
            raise ValueError("PLE contexts must use integer storage")
        contexts = contexts.to(torch.long)
        if use_decode_fusion and can_use_ple_ngram_hash(
            contexts,
            self.layer_multipliers,
            self.ngram_heads_vocab_sizes,
            self.ngram_heads_offsets,
        ):
            return ple_ngram_hash(
                contexts,
                self.layer_multipliers,
                self.ngram_heads_vocab_sizes,
                self.ngram_heads_offsets,
                self.eos_token_id,
            )
        shifted_tokens = [contexts]
        for shift in range(1, self.ngram_size):
            shifted_tokens.append(self._shift_right_ignore_eos(contexts, shift))

        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            ngram_idx = ngram - 2
            start_idx = ngram_idx * self.heads_per_ngram
            end_idx = start_idx + self.heads_per_ngram
            mix = shifted_tokens[0] * self.layer_multipliers[0]
            for pos in range(1, ngram):
                mix = torch.bitwise_xor(mix, shifted_tokens[pos] * self.layer_multipliers[pos])
            head_vocab_sizes = self.ngram_heads_vocab_sizes[start_idx:end_idx]
            head_offsets = self.ngram_heads_offsets[start_idx:end_idx]
            ngram_ids = torch.remainder(mix[:, -1:].unsqueeze(-1), head_vocab_sizes.view(1, 1, -1))
            ngram_ids = ngram_ids + head_offsets.view(1, 1, -1)
            blocks.append(ngram_ids[:, 0])
        return torch.cat(blocks, dim=-1)

    def embed(
        self,
        ngram_ids: torch.Tensor,
        physical_tokens: Optional[int] = None,
        all_rank_num_tokens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """Embed head IDs while preserving each ADP rank's token ownership.

        Ordinary TP ranks process identical tokens, so the in-tree column-sharded
        :class:`Embedding` performs a masked local lookup followed by AllReduce.
        Attention-DP ranks process different requests. They therefore all-gather
        the physical token rows, look up only the locally-owned vocabulary rows,
        and reduce-scatter the summed embeddings back to the source rank.
        """
        if ngram_ids.ndim != 2 or ngram_ids.shape[1] != self.ngram_heads:
            raise ValueError(
                f"PLE n-gram IDs must have shape [tokens, {self.ngram_heads}], "
                f"got {tuple(ngram_ids.shape)}"
            )
        if ngram_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("PLE n-gram IDs must use integer storage")
        if (
            self.ngram_embedding.weight.dtype == torch.float8_e4m3fn
            and self.ngram_embedding_weight_scale is None
        ):
            # FP8 table values are checkpoint integers in a scaled storage
            # format; using them before the mapper attaches the scalar would
            # silently change every lookup.
            raise RuntimeError("PLE FP8 n-gram table is missing its weight scale")
        if self.host_offload:
            lookup_ids, semantic_tokens = self._prepare_embedding_lookup(
                ngram_ids,
                physical_tokens,
                all_rank_num_tokens,
            )
            partial = self.ngram_embedding.gather(
                lookup_ids,
                weight_scale=self._host_offload_weight_scale,
            )
            return self._finish_embedding_lookup(
                partial,
                semantic_tokens,
                physical_tokens,
                all_rank_num_tokens,
            )
        if not self.use_attention_dp_sharding and self.ngram_embedding_weight_scale is not None:
            return self._embed_fp8_tp(ngram_ids)
        if not self.use_attention_dp_sharding:
            return self.ngram_embedding(ngram_ids)

        lookup_ids, semantic_tokens = self._prepare_embedding_lookup(
            ngram_ids,
            physical_tokens,
            all_rank_num_tokens,
        )
        owned = (lookup_ids >= self.vocab_start_index) & (lookup_ids < self.vocab_end_index)
        local_ids = torch.where(
            owned,
            lookup_ids - self.vocab_start_index,
            torch.zeros_like(lookup_ids),
        )
        partial = F.embedding(local_ids, self.ngram_embedding.weight)
        partial.masked_fill_(~owned.unsqueeze(-1), 0)
        return self._finish_embedding_lookup(
            partial,
            semantic_tokens,
            physical_tokens,
            all_rank_num_tokens,
        )


class Qwen4ExpPLE(nn.Module):
    """The Qwen4-Exp PLE n-gram short-convolution side path.

    :meth:`forward` consumes the Hyper-Connection hidden bundle
    ``[physical_tokens, hc_count * hidden]`` as the gate query, the per-forward
    :class:`PLEMetadata`, and the caller-owned ``conv_state`` / ``ngram_context``
    recurrent-state pools (updated in place, mamba-style), and returns the PLE
    contribution ``[physical_tokens, hc_count * hidden]`` to be **added** into the
    hidden stream before attention.
    """

    def __init__(
        self,
        config: object,
        dtype: Optional[torch.dtype] = None,
        ple_layer_index: int = 0,
        layer_id: Optional[int] = None,
        mapping: Optional[Mapping] = None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = int(config.hidden_size)
        self.ple_embed_dim = int(config.ple_embed_dim)
        self.conv_kernel_size = int(config.ple_conv_kernel_size)
        self.hc_count = int(config.hc_count)
        self.hc_hidden_size = self.hidden_size * self.hc_count
        if self.hidden_size <= 0 or self.ple_embed_dim <= 0 or self.hc_count <= 0:
            raise ValueError("PLE hidden, embedding, and Hyper-Connection sizes must be positive")
        if self.conv_kernel_size < 2:
            raise ValueError("PLE short-convolution kernel size must be at least two")

        self.ple_embedding = Qwen4ExpNGramEmbedding(
            config,
            self.ple_embed_dim,
            ple_layer_index=ple_layer_index,
            dtype=dtype,
            mapping=mapping,
        )
        self.ngram_size = self.ple_embedding.ngram_size
        self.short_conv_dilation = self.ngram_size
        self.short_conv_state_len = (self.conv_kernel_size - 1) * self.short_conv_dilation
        self.conv_channels = self.hc_hidden_size
        # Recurrent-state pool contract for KVCacheManagerV2.
        self.ngram_context_len = self.ngram_size - 1
        self.conv_state_shape = (self.conv_channels, self.short_conv_state_len)

        self.key_proj = Linear(
            self.ple_embed_dim,
            self.conv_channels,
            bias=False,
            dtype=dtype,
        )
        self.value_proj = Linear(
            self.ple_embed_dim,
            self.hidden_size,
            bias=False,
            dtype=dtype,
        )
        norm_hidden = self.hc_hidden_size
        norm_group = self.hidden_size
        self.norm_key = GroupedRMSNorm(
            norm_hidden, eps=config.rms_norm_eps, group_size=norm_group, dtype=dtype
        )
        self.norm_query = GroupedRMSNorm(
            norm_hidden, eps=config.rms_norm_eps, group_size=norm_group, dtype=dtype
        )
        self.norm_conv = GroupedRMSNorm(
            norm_hidden, eps=config.rms_norm_eps, group_size=norm_group, dtype=dtype
        )
        # Only ``.weight`` (shape [conv_channels, 1, kernel]) is used; the
        # functional conv below prepends the carried state explicitly, so the
        # module's ``padding`` is never exercised (kept for weight-name parity).
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_channels,
            out_channels=self.conv_channels,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_channels,
            padding=(self.conv_kernel_size - 1) * self.short_conv_dilation,
            dilation=self.short_conv_dilation,
            bias=False,
            dtype=dtype,
        )
        # Target verification advances through every proposed token. Retain the
        # per-prefix candidates until rejection sampling reports the accepted
        # length, then restore the accepted prefix state.
        self._pending_conv_states: Optional[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None
        self._pending_ngram_contexts: Optional[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None
        self._prefetch_stream = torch.cuda.Stream() if self.ple_embedding.host_offload else None
        self._graph_prefetch_buffers: dict[tuple[int, int], torch.Tensor] = {}
        self._eager_prefetch_buffer: Optional[torch.Tensor] = None
        self._prefetch_state: Optional[tuple[torch.Tensor, int, int, torch.Tensor]] = None

    def _apply_ple_norm(self, norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Per-stream grouped norm over a ``[..., hc_count, hidden]`` tensor."""
        y = norm(x.flatten(-2, -1))
        return y.unflatten(-1, (self.hc_count, self.hidden_size))

    def _prepare_ngram_lookup(
        self,
        metadata: PLEMetadata,
        ngram_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build recurrent contexts and the hashed per-head global row IDs."""
        history = ngram_context.index_select(0, metadata.state_indices).to(torch.long)
        combined = torch.cat([history, metadata.padded_tokens.to(torch.long)], dim=1)
        windows = combined.unfold(1, self.ngram_size, 1)
        contexts = windows[metadata.req_indices, metadata.token_offsets]
        return combined, self.ple_embedding.hash_contexts(
            contexts,
            use_decode_fusion=metadata.is_decode,
        )

    def _allocate_prefetch_buffer(
        self,
        lookup_tokens: int,
        lookup_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.ple_embedding.ngram_embedding.allocate_output(
            (lookup_tokens, self.ple_embed_dim), lookup_ids.device
        )

    def _get_prefetch_buffer(
        self,
        lookup_tokens: int,
        lookup_ids: torch.Tensor,
        *,
        is_cuda_graph: bool,
    ) -> torch.Tensor:
        if is_cuda_graph:
            device_index = lookup_ids.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            key = (device_index, lookup_tokens)
            buffer = self._graph_prefetch_buffers.get(key)
            if buffer is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("PLE prefetch buffer must be allocated during graph warmup")
                buffer = self._allocate_prefetch_buffer(lookup_tokens, lookup_ids)
                self._graph_prefetch_buffers[key] = buffer
            return buffer

        buffer = self._eager_prefetch_buffer
        if buffer is None or buffer.device != lookup_ids.device or buffer.shape[0] < lookup_tokens:
            buffer = self._allocate_prefetch_buffer(lookup_tokens, lookup_ids)
            self._eager_prefetch_buffer = buffer
        return buffer[:lookup_tokens]

    def start_prefetch(
        self,
        metadata: PLEMetadata,
        ngram_context: torch.Tensor,
    ) -> None:
        """Start the sparse host lookup before execution reaches the PLE layer."""
        prefetch_stream = self._prefetch_stream
        if prefetch_stream is None:
            return
        if self._prefetch_state is not None:
            raise RuntimeError("PLE prefetch state was not consumed before reuse")

        combined, ngram_ids = self._prepare_ngram_lookup(metadata, ngram_context)
        lookup_ids, semantic_tokens = self.ple_embedding._prepare_embedding_lookup(
            ngram_ids,
            metadata.physical_tokens,
            metadata.all_rank_num_tokens,
        )
        lookup_tokens = lookup_ids.shape[0]
        if lookup_tokens == 0:
            return

        prefetched = self._get_prefetch_buffer(
            lookup_tokens,
            lookup_ids,
            is_cuda_graph=metadata.is_cuda_graph,
        )
        output = prefetched.view(
            lookup_tokens,
            self.ple_embedding.ngram_heads,
            self.ple_embedding.head_dim_per_ngram,
        )
        current_stream = torch.cuda.current_stream()
        prefetch_stream.wait_stream(current_stream)
        lookup_ids.record_stream(prefetch_stream)
        with torch.cuda.stream(prefetch_stream):
            self.ple_embedding.ngram_embedding.gather(
                lookup_ids,
                out=output,
                weight_scale=self.ple_embedding._host_offload_weight_scale,
            )
        self._prefetch_state = (
            prefetched,
            semantic_tokens,
            metadata.physical_tokens,
            combined,
        )

    def abort_prefetch(self) -> None:
        """Discard an unconsumed lookup after the enclosing forward fails."""
        self._prefetch_state = None

    def _consume_prefetched_embeddings(
        self,
        metadata: PLEMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._prefetch_state is None or self._prefetch_stream is None:
            raise RuntimeError("PLE prefetch state is missing")
        prefetched, semantic_tokens, physical_tokens, combined = self._prefetch_state
        torch.cuda.current_stream().wait_stream(self._prefetch_stream)
        embeddings = self.ple_embedding._finish_embedding_lookup(
            prefetched.view(
                prefetched.shape[0],
                self.ple_embedding.ngram_heads,
                self.ple_embedding.head_dim_per_ngram,
            ),
            semantic_tokens,
            physical_tokens,
            metadata.all_rank_num_tokens,
        )
        self._prefetch_state = None
        return embeddings.flatten(start_dim=-2), combined

    def _short_conv(
        self, x: torch.Tensor, metadata: PLEMetadata, conv_state: torch.Tensor
    ) -> torch.Tensor:
        """Dilated causal depthwise short conv with in-place conv-state update.

        ``x`` is ``[processed_tokens, conv_channels]``. ``conv_state`` is the
        caller-owned pool ``[num_slots, conv_channels, short_conv_state_len]``;
        the consumed slots are advanced in place. Returns
        ``silu(conv_output)[processed_tokens, conv_channels]``.
        """
        if x.shape[0] == 0:
            return x
        m = metadata
        num_seq = m.lengths.shape[0]

        # In-flight batching places context requests before generation requests.
        # Padding their joint convolution input to the longest context chunk
        # would scale as ``num_seq * row_width * conv_channels``. A single 8K
        # prefill mixed with hundreds of decode rows can therefore allocate tens
        # of GiB even though only ``processed_tokens`` rows carry data. Split the
        # two contiguous groups: context rows retain their variable-width path,
        # while generation rows use width one. Target verification keeps the
        # joint path below because its per-prefix state candidates span the full
        # static verification row.
        if 0 < m.num_contexts < num_seq and not m.use_spec_decoding:
            context_tokens = m.context_tokens
            if not 0 < context_tokens < x.shape[0]:
                raise RuntimeError(
                    "PLE mixed-batch context boundary is inconsistent with the "
                    f"packed token count: {context_tokens} vs {x.shape[0]}"
                )
            context_meta = dataclasses.replace(
                m,
                physical_tokens=context_tokens,
                processed_tokens=context_tokens,
                lengths=m.lengths[: m.num_contexts],
                req_indices=m.req_indices[:context_tokens],
                token_offsets=m.token_offsets[:context_tokens],
                valid_tokens=m.valid_tokens[:context_tokens],
                state_indices=m.state_indices[: m.num_contexts],
                padded_tokens=m.padded_tokens[: m.num_contexts],
                context_tokens=context_tokens,
            )
            generation_tokens = x.shape[0] - context_tokens
            generation_meta = dataclasses.replace(
                m,
                is_decode=True,
                physical_tokens=generation_tokens,
                processed_tokens=generation_tokens,
                lengths=m.lengths[m.num_contexts :],
                row_width=1,
                req_indices=m.req_indices[context_tokens:] - m.num_contexts,
                token_offsets=m.token_offsets[context_tokens:],
                valid_tokens=m.valid_tokens[context_tokens:],
                state_indices=m.state_indices[m.num_contexts :],
                padded_tokens=m.padded_tokens[m.num_contexts :, :1],
                num_contexts=0,
                context_tokens=0,
            )
            context_output = self._short_conv(x[:context_tokens], context_meta, conv_state)
            generation_output = self._short_conv(x[context_tokens:], generation_meta, conv_state)
            return torch.cat((context_output, generation_output))

        conv_weight = self.conv1d.weight.to(dtype=x.dtype)
        if m.is_decode and can_use_ple_decode_short_conv(
            conv_state, m.state_indices, x, conv_weight
        ):
            return ple_decode_short_conv(conv_state, m.state_indices, x, conv_weight)

        if m.is_decode and can_use_ple_short_conv_state(conv_state, m.state_indices, x):
            conv_input = ple_short_conv_state(conv_state, m.state_indices, x)
            conv_output = F.conv1d(
                conv_input,
                conv_weight,
                bias=None,
                dilation=self.short_conv_dilation,
                groups=self.conv_channels,
            ).squeeze(-1)
            return F.silu(conv_output)

        state = conv_state.index_select(0, m.state_indices).to(dtype=x.dtype)

        padded_seq = x.new_zeros((num_seq, m.row_width, self.conv_channels))
        padded_seq[m.req_indices, m.token_offsets] = x
        conv_input = torch.cat([state, padded_seq.transpose(1, 2)], dim=-1)
        conv_output = F.conv1d(
            conv_input,
            conv_weight,
            bias=None,
            dilation=self.short_conv_dilation,
            groups=self.conv_channels,
        ).transpose(1, 2)

        state_cols = torch.arange(self.short_conv_state_len, device=x.device, dtype=torch.long)

        def _gather_at(offsets: torch.Tensor) -> torch.Tensor:
            return conv_input.gather(
                2,
                (offsets.unsqueeze(1) + state_cols.unsqueeze(0))
                .unsqueeze(1)
                .expand(-1, self.conv_channels, -1),
            )

        if m.use_spec_decoding and m.num_contexts < num_seq:
            generation_slots = m.state_indices[m.num_contexts :]
            candidates = conv_input.unfold(2, self.short_conv_state_len, 1)[
                :, :, 1 : m.row_width + 1
            ]
            candidates = candidates.permute(0, 2, 1, 3).contiguous()
            self._pending_conv_states = (
                conv_state,
                generation_slots,
                conv_state[generation_slots].clone(),
                candidates[m.num_contexts :],
            )

        next_state = _gather_at(m.lengths)
        conv_state[m.state_indices] = next_state.to(dtype=conv_state.dtype)
        return F.silu(conv_output[m.req_indices, m.token_offsets])

    def _commit_ngram_context(
        self, combined: torch.Tensor, metadata: PLEMetadata, ngram_context: torch.Tensor
    ) -> None:
        """Slide each sequence's n-gram history forward by its chunk length.

        The new history is the ``ngram_context_len`` columns of ``combined``
        starting at ``lengths``: the last ``ngram_size - 1`` tokens exposed by
        each chunk. It is written back into the caller's pool in place.
        """
        m = metadata
        context_len = combined.shape[1] - m.row_width
        context_cols = torch.arange(context_len, device=combined.device, dtype=torch.long)
        next_context = combined.gather(1, m.lengths.unsqueeze(1) + context_cols.unsqueeze(0))
        if m.use_spec_decoding and m.num_contexts < m.lengths.shape[0]:
            generation_slots = m.state_indices[m.num_contexts :]
            candidates = combined.unfold(1, context_len, 1)[:, 1 : m.row_width + 1].contiguous()
            self._pending_ngram_contexts = (
                ngram_context,
                generation_slots,
                ngram_context[generation_slots].clone(),
                candidates[m.num_contexts :],
            )

        ngram_context[m.state_indices] = next_context.to(dtype=ngram_context.dtype)

    def commit_speculative_states(
        self,
        num_accepted_tokens: torch.Tensor,
        state_indices: torch.Tensor,
        num_contexts: int,
    ) -> None:
        """Keep the PLE state produced by the accepted verification prefix."""
        del state_indices
        # The decoder always accepts at least the target token. Candidate zero
        # therefore represents one accepted token, hence the zero-based -1.
        accepted = num_accepted_tokens[num_contexts:].to(torch.long) - 1
        for entry in (self._pending_conv_states, self._pending_ngram_contexts):
            if entry is None:
                continue
            state_pool, slots, _, candidates = entry
            num_gens = slots.shape[0]
            rows = torch.arange(num_gens, device=slots.device)
            selected = candidates[rows, accepted[:num_gens]]
            state_pool[slots] = selected.to(dtype=state_pool.dtype)
        self._pending_conv_states = None
        self._pending_ngram_contexts = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        metadata: PLEMetadata,
        conv_state: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the PLE contribution and advance both recurrent-state pools.

        Args:
            hidden_states: ``[physical_tokens, hc_count * hidden]`` HC bundle used
                as the gate query.
            metadata: per-forward token layout (:class:`PLEMetadata`).
            conv_state: ``[num_slots, conv_channels, short_conv_state_len]`` pool,
                updated in place.
            ngram_context: ``[num_slots, ngram_size - 1]`` int64 token-history
                pool, updated in place.

        Returns:
            ``[physical_tokens, hc_count * hidden]`` — the value to add into the
            hidden stream before attention (zero for padded / invalid tokens).
        """
        m = metadata
        hc_dim = self.hc_count * self.hidden_size
        if hidden_states.ndim != 2 or hidden_states.shape[-1] != hc_dim:
            raise RuntimeError(
                "PLE hidden size does not match its hyper-connection layout: "
                f"expected [tokens, {hc_dim}], got {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[0] < m.processed_tokens:
            raise RuntimeError("PLE hidden states are shorter than the processed token layout")
        if conv_state.ndim != 3 or tuple(conv_state.shape[1:]) != self.conv_state_shape:
            raise RuntimeError(
                "PLE convolution-state pool has incompatible shape: "
                f"expected [slots, {self.conv_state_shape}], got {tuple(conv_state.shape)}"
            )
        if ngram_context.ndim != 2 or ngram_context.shape[1] != self.ngram_context_len:
            raise RuntimeError(
                "PLE n-gram context pool has incompatible shape: "
                f"expected [slots, {self.ngram_context_len}], got {tuple(ngram_context.shape)}"
            )
        if ngram_context.dtype not in (torch.int32, torch.int64):
            raise RuntimeError("PLE n-gram context pool must use integer storage")
        if (
            hidden_states.device != conv_state.device
            or hidden_states.device != ngram_context.device
        ):
            raise RuntimeError("PLE activations and recurrent-state pools must share a device")
        hidden_states = hidden_states[: m.processed_tokens]
        if m.processed_tokens == 0:
            # No recurrent state advances on an empty logical step. Preserve
            # the graph's physical row count with zero PLE contributions.
            return hidden_states.new_zeros((m.physical_tokens, hc_dim))

        if self._prefetch_state is not None:
            embeddings, combined = self._consume_prefetched_embeddings(m)
        else:
            combined, ngram_ids = self._prepare_ngram_lookup(m, ngram_context)
            embeddings = self.ple_embedding.embed(
                ngram_ids,
                physical_tokens=m.physical_tokens,
                all_rank_num_tokens=m.all_rank_num_tokens,
            ).flatten(start_dim=-2)

        key = self.key_proj(embeddings)
        value = self.value_proj(embeddings)
        token_count = hidden_states.shape[0]
        key = key.reshape(token_count, self.hc_count, self.hidden_size)
        query = hidden_states.reshape(token_count, self.hc_count, self.hidden_size)
        key_normed = self._apply_ple_norm(self.norm_key, key)
        query_normed = self._apply_ple_norm(self.norm_query, query)
        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True)
        gate = gate / math.sqrt(self.hidden_size)
        if m.is_decode and can_use_ple_gate_value(gate, value):
            gated_value = ple_gate_value(gate, value)
        else:
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = torch.sigmoid(gate)
            gated_value = gate * value.unsqueeze(-2)
        gated_value_normed = self._apply_ple_norm(self.norm_conv, gated_value)
        gated_value = gated_value.flatten(-2)
        gated_value_normed = gated_value_normed.flatten(-2)
        conv_output = self._short_conv(gated_value_normed, metadata, conv_state)
        output = gated_value + conv_output
        output = torch.where(m.valid_tokens.unsqueeze(-1), output, torch.zeros_like(output))

        self._commit_ngram_context(combined, metadata, ngram_context)
        return _pad_token_rows(output, m.physical_tokens)
