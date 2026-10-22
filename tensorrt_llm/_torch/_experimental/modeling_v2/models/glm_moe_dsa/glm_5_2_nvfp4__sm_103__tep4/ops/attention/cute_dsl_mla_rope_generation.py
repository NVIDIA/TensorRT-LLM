# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MLA generation RoPE + fp8 query + paged latent append, on the CuTe DSL kernel, at MTP 2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Mapping, Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.attention import (
    self_attn_mla_rope_generation_h16_kv512_rope64_page32_mtp2_bf16_fp8 as _kernel,
)

#: The MTP width the kernel is built for. Every other width takes the stock
#: `mla_rope_generation` at the call site.
MTP_WIDTH = 2
#: The KV page size the kernel is built for.
TOKENS_PER_BLOCK = _kernel.TOKENS_PER_BLOCK
#: One latent row: kv_lora 512 + rope 64.
HEAD_SIZE = _kernel.HEAD_SIZE
#: The kernel's token tile. A row count that is a multiple of it runs the
#: full-tile variant, any other the predicated one.
TOKENS_PER_CTA = _kernel.TOKENS_PER_CTA
#: e4m3's largest finite value; the kernel's conversion saturates to it.
E4M3_MAX = 448.0


@dataclass(frozen=True)
class CompiledMlaRopeGeneration:
    """The kernels `compile` built, keyed by the specialization each serves.

    One target keeps one of these and passes it as `compiled` on every call.
    """

    device_index: int
    kernels: Mapping[_kernel.MlaRopeSpecialization, object]


def compile(  # noqa: A001 -- the kernel module's own name for this step
    *,
    max_blocks_per_seq: int,
    q_pe_head_stride: int,
    q_pe_token_stride: int,
    latent_row_stride: int,
    device_index: int,
    predicted_tokens_per_seq: int = MTP_WIDTH,
    tokens_per_block: int = TOKENS_PER_BLOCK,
) -> CompiledMlaRopeGeneration:
    """Compile the two variants a decode batch can reach; return the handle.

    The kernel bakes into its code the block-table width
    (`kv_cache_block_offsets.shape[-1]`), `q_pe`'s head and token strides, the
    latent's row stride, the MTP width and the page size. Only one more key
    varies at run time: whether the row count is a multiple of 32. Both values
    of it are built here, so no batch size compiles later.

    The block-table width is an engine value, so a target calls this at its
    first forward (which is eager), not at weight load. Costs two CuTe DSL
    compiles; a second call with the same keys returns the cached kernels.
    """
    kernels = {}
    for full_token_tiles in (True, False):
        spec = _kernel.MlaRopeSpecialization(
            predicted_tokens_per_seq=int(predicted_tokens_per_seq),
            max_blocks_per_seq=int(max_blocks_per_seq),
            tokens_per_block=int(tokens_per_block),
            q_pe_ld=int(q_pe_head_stride),
            q_pe_stride=int(q_pe_token_stride),
            latent_cache_stride=int(latent_row_stride),
            emit_q_base=True,
            full_token_tiles=full_token_tiles,
        )
        kernels[spec] = _kernel._compile_for(spec, int(device_index))
    return CompiledMlaRopeGeneration(device_index=int(device_index), kernels=kernels)


def _latent_row_stride(latent_cache: torch.Tensor) -> int:
    """The latent pitch the kernel is keyed on: `HEAD_SIZE` when packed."""
    return int(latent_cache.stride(0)) if latent_cache.dim() == 2 else HEAD_SIZE


class CuteDslMlaRopeGeneration(OpWrapper):
    """One decode step's MLA preprocessing at MTP width 2, generation rows only.

    Takes the stock `mla_rope_generation` positional list without
    `residual_dim` (38 arguments), plus the `compiled` handle. Each sequence
    carries `P = predicted_tokens_per_seq` new tokens; row `(g, t)` of the
    `G * P` per-token tensors sits at absolute position
    `sequence_length[num_contexts + g] - P + t`.

    Over the fp8 latent pool the call writes, and returns None:

    * `quant_q_buffer`: e4m3 `[fused_q[..., :512] | rope(q_pe)]` per row/head;
    * the pool: e4m3 `[latent[:512] | rope(latent[512:])]` at each row's slot;
    * `cu_q_seqlens[:G+1]` (q rows, `16 * P` per sequence),
      `cu_kv_seqlens[:G+1]`, `fmha_scheduler_counter = 0`, and the two decode
      FMHA scales.

    `latent_cache` may be the strided `[T, 576]` tail of the kv_a projection
    output (any row stride divisible by 8). `fused_q`, `q_pe` and the latent
    are read only.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The target runs this only on a pure decode step at MTP width 2, 1 to 32
    # sequences, with `q_pe` the rope half of the q_b output (head stride 256,
    # token stride 16 * 256) and the latent the kv_a output's tail (row
    # stride 2048 + 576 = 2624). The cells cover both token-tile variants and
    # a pool with more than one layer.
    _GLM = dict(
        num_heads=16,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        qk_nope_head_dim=192,
        q_lora_rank=2048,
        tokens_per_block=32,
        predicted_tokens_per_seq=MTP_WIDTH,
        latent_row_stride=2624,
    )

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="3 sequences, 6 rows: the partial-tile variant; the two rows of a "
            "sequence share a page, straddle a page boundary, or start a fresh page",
            spec=dict(num_seqs=3, cached=[30, 31, 32], num_layers=1, layer_idx=0, **_GLM),
        ),
        Cell(
            why="16 sequences, 32 rows: exactly one full token tile (the full-tile variant)",
            spec=dict(
                num_seqs=16,
                cached=[31 + 29 * i for i in range(16)],
                num_layers=1,
                layer_idx=0,
                **_GLM,
            ),
        ),
        Cell(
            why="the full decode batch: 32 sequences x MTP 2, two full tiles",
            spec=dict(
                num_seqs=32,
                cached=[5 + 29 * i for i in range(32)],
                num_layers=1,
                layer_idx=0,
                **_GLM,
            ),
        ),
        Cell(
            why="a three-layer pool written at layer 2: the layer's pool offset and the "
            "layer-strided block offsets",
            spec=dict(
                num_seqs=5,
                cached=[1, 63, 64, 200, 511],
                num_layers=3,
                layer_idx=2,
                **_GLM,
            ),
        ),
    )

    note = """
    The target compiles one handle, at its first forward, for its own
    block-table width, `q_pe` strides and latent row stride; both token-tile
    variants are in it. It calls this op only when every sequence carries
    exactly 2 tokens (`gen_p == 2`, a pure decode step); every other width,
    and the mixed context+generation batch, takes the stock
    `mla_rope_generation` at the call site. An operand layout outside the
    handle raises here; it never compiles.

    The runner is prepared on every call: it binds the addresses of this
    call's allocations, so a CUDA-graph replay reproduces them. Preparing reads
    the host pool tables and validates every tensor on the host; it does not
    sync the device. `quant_q_buffer` is passed as e4m3 and viewed as uint8,
    which is what the kernel validates; same bytes.

    The kernel rejects loudly: a context sequence in the batch, a bf16 pool, a
    secondary pool, non-1 KV scales, `out_scale`, Helix, `q_scaling != 1`,
    beam width above 1, and any head geometry but 16 x (512 + 64) with a
    192-wide nope half. It writes `mla_bmm1_scale = 1/16` from its compiled
    constant, not from `q_scaling`.

    Not certified: a packed (contiguous) latent, though the kernel takes it
    under its own key; a contiguous `q_pe`.
    """

    def __call__(
        self,
        fused_q: torch.Tensor,
        q_pe: torch.Tensor,
        latent_cache: torch.Tensor,
        rotary_cos_sin: torch.Tensor,
        cu_q_seqlens: torch.Tensor,
        cu_kv_seqlens: torch.Tensor,
        fmha_scheduler_counter: torch.Tensor,
        mla_bmm1_scale: torch.Tensor,
        mla_bmm2_scale: torch.Tensor,
        quant_q_buffer: torch.Tensor,
        sequence_length: torch.Tensor,
        host_past_key_value_lengths: torch.Tensor,
        host_context_lengths: torch.Tensor,
        num_contexts: int,
        kv_cache_block_offsets: torch.Tensor,
        host_kv_cache_pool_pointers: torch.Tensor,
        host_kv_cache_pool_mapping: torch.Tensor,
        kv_scale_orig_quant: Optional[torch.Tensor],
        kv_scale_quant_orig: Optional[torch.Tensor],
        out_scale: Optional[torch.Tensor],
        block_ids_per_seq: Optional[torch.Tensor],
        helix_tensor_params: List[Optional[torch.Tensor]],
        predicted_tokens_per_seq: int,
        layer_idx: int,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        tokens_per_block: int,
        attention_window_size: int,
        beam_width: int,
        quant_mode: int,
        q_scaling: float,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        rope_append: bool,
        compiled: CompiledMlaRopeGeneration,
    ) -> None:
        # The specialization the kernel would pick from these operands; it has
        # to be one the handle holds, or preparing the runner would compile.
        stride = _latent_row_stride(latent_cache)
        spec = _kernel.MlaRopeSpecialization(
            predicted_tokens_per_seq=int(predicted_tokens_per_seq),
            max_blocks_per_seq=int(kv_cache_block_offsets.shape[-1]),
            tokens_per_block=int(tokens_per_block),
            q_pe_ld=int(q_pe.stride(-2)),
            q_pe_stride=int(q_pe.stride(0)),
            latent_cache_stride=stride,
            emit_q_base=True,
            full_token_tiles=int(fused_q.shape[0]) % TOKENS_PER_CTA == 0,
        )
        kernel = compiled.kernels.get(spec)
        if kernel is None or fused_q.device.index != compiled.device_index:
            raise ValueError(
                f"no kernel compiled for {spec} on device {fused_q.device}; build the "
                "handle with this call's block-table width, q_pe strides and latent stride"
            )
        runner = _kernel.prepare_runner_from_serving_abi(
            fused_q,
            q_pe,
            latent_cache,
            rotary_cos_sin,
            cu_q_seqlens,
            cu_kv_seqlens,
            fmha_scheduler_counter,
            mla_bmm1_scale,
            mla_bmm2_scale,
            quant_q_buffer.view(torch.uint8),
            sequence_length,
            host_past_key_value_lengths,
            host_context_lengths,
            num_contexts,
            kv_cache_block_offsets,
            host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping,
            kv_scale_orig_quant,
            kv_scale_quant_orig,
            out_scale,
            block_ids_per_seq,
            helix_tensor_params,
            predicted_tokens_per_seq,
            layer_idx,
            num_heads,
            num_kv_heads,
            head_size,
            tokens_per_block,
            attention_window_size,
            beam_width,
            quant_mode,
            q_scaling,
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            rope_append,
            # A packed latent keys as `HEAD_SIZE`; a strided one as its pitch.
            split_source=stride != HEAD_SIZE,
        )
        runner.compiled = kernel
        runner()

    def reference(
        self,
        fused_q: torch.Tensor,
        q_pe: torch.Tensor,
        latent_cache: torch.Tensor,
        rotary_cos_sin: torch.Tensor,
        cu_q_seqlens: torch.Tensor,
        cu_kv_seqlens: torch.Tensor,
        fmha_scheduler_counter: torch.Tensor,
        mla_bmm1_scale: torch.Tensor,
        mla_bmm2_scale: torch.Tensor,
        quant_q_buffer: torch.Tensor,
        sequence_length: torch.Tensor,
        host_past_key_value_lengths: torch.Tensor,
        host_context_lengths: torch.Tensor,
        num_contexts: int,
        kv_cache_block_offsets: torch.Tensor,
        host_kv_cache_pool_pointers: torch.Tensor,
        host_kv_cache_pool_mapping: torch.Tensor,
        kv_scale_orig_quant: Optional[torch.Tensor],
        kv_scale_quant_orig: Optional[torch.Tensor],
        out_scale: Optional[torch.Tensor],
        block_ids_per_seq: Optional[torch.Tensor],
        helix_tensor_params: List[Optional[torch.Tensor]],
        predicted_tokens_per_seq: int,
        layer_idx: int,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        tokens_per_block: int,
        attention_window_size: int,
        beam_width: int,
        quant_mode: int,
        q_scaling: float,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        rope_append: bool,
        compiled: CompiledMlaRopeGeneration,
    ) -> dict[str, torch.Tensor]:
        """Everything the call writes, keyed by the argument it lands in.

        The op returns None and writes buffers, so this returns what each
        buffer should hold afterwards:

        * `quant_q_buffer`: e4m3 `[T, H, kv_lora + rope]`;
        * `latent_cache`: the e4m3 `[T, kv_lora + rope]` row each token appends,
          with `positions` `[T]` the absolute position it lands at. The pool
          addressing arguments only say where that is; the test reads the slot
          back through the block table;
        * `cu_q_seqlens`, `cu_kv_seqlens`: the first `G + 1` entries;
        * `fmha_scheduler_counter`, `mla_bmm1_scale`, `mla_bmm2_scale`.

        Read: `fused_q`, `q_pe`, `latent_cache`, `rotary_cos_sin`,
        `sequence_length`, `host_context_lengths`, `num_contexts`,
        `predicted_tokens_per_seq`, `num_heads`, `q_scaling`, the three MLA
        widths. Everything else, the handle included, is ignored.

        The rotation is GPT-J interleaved (consecutive elements pair), uses
        `rotary_cos_sin[pos, i]` as `(cos, sin)` for pair `i`, and is rounded to
        bf16 before the e4m3 conversion, as the kernel does. It is evaluated in
        fp64 so that bf16 rounding is the only rounding before e4m3.
        """
        p = int(predicted_tokens_per_seq)
        rope = int(qk_rope_head_dim)
        c = int(kv_lora_rank)
        seqs = int(host_context_lengths.shape[0]) - int(num_contexts)
        lens = sequence_length[num_contexts : num_contexts + seqs].to(torch.int64)
        steps = torch.arange(p, dtype=torch.int64, device=lens.device)
        positions = (lens[:, None] - p + steps[None, :]).reshape(-1)
        table = rotary_cos_sin.reshape(-1, rope, 2).double()
        cos = table[positions, : rope // 2, 0]
        sin = table[positions, : rope // 2, 1]

        def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
            pairs = x.double().reshape(*x.shape[:-1], rope // 2, 2)
            out = torch.empty_like(pairs)
            out[..., 0] = pairs[..., 0] * cos - pairs[..., 1] * sin
            out[..., 1] = pairs[..., 1] * cos + pairs[..., 0] * sin
            return out.reshape(x.shape).to(torch.bfloat16)

        def e4m3(x: torch.Tensor) -> torch.Tensor:
            return x.float().clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)

        q_rot = rotate(q_pe, cos[:, None, :], sin[:, None, :])
        k_rot = rotate(latent_cache[:, c : c + rope], cos, sin)
        quant_q = e4m3(torch.cat([fused_q[..., :c], q_rot], dim=-1))
        rows = e4m3(torch.cat([latent_cache[:, :c], k_rot], dim=-1))

        dev = fused_q.device
        cu_q = torch.arange(seqs + 1, dtype=torch.int32, device=dev) * int(num_heads) * p
        cu_kv = torch.zeros(seqs + 1, dtype=torch.int32, device=dev)
        cu_kv[1:] = torch.cumsum(lens, 0).to(torch.int32)
        x = 1.0 / (float(q_scaling) * math.sqrt(int(qk_nope_head_dim) + rope))
        return {
            "quant_q_buffer": quant_q,
            "latent_cache": rows,
            "positions": positions,
            "cu_q_seqlens": cu_q,
            "cu_kv_seqlens": cu_kv,
            "fmha_scheduler_counter": torch.zeros(1, dtype=torch.uint32, device=dev),
            "mla_bmm1_scale": torch.tensor(
                [x, x * math.log2(math.e)], dtype=torch.float32, device=dev
            ),
            "mla_bmm2_scale": torch.ones(1, dtype=torch.float32, device=dev),
        }

    def is_valid(
        self,
        fused_q: torch.Tensor,
        q_pe: torch.Tensor,
        latent_cache: torch.Tensor,
        rotary_cos_sin: torch.Tensor,
        cu_q_seqlens: torch.Tensor,
        cu_kv_seqlens: torch.Tensor,
        fmha_scheduler_counter: torch.Tensor,
        mla_bmm1_scale: torch.Tensor,
        mla_bmm2_scale: torch.Tensor,
        quant_q_buffer: torch.Tensor,
        sequence_length: torch.Tensor,
        host_past_key_value_lengths: torch.Tensor,
        host_context_lengths: torch.Tensor,
        num_contexts: int,
        kv_cache_block_offsets: torch.Tensor,
        host_kv_cache_pool_pointers: torch.Tensor,
        host_kv_cache_pool_mapping: torch.Tensor,
        kv_scale_orig_quant: Optional[torch.Tensor],
        kv_scale_quant_orig: Optional[torch.Tensor],
        out_scale: Optional[torch.Tensor],
        block_ids_per_seq: Optional[torch.Tensor],
        helix_tensor_params: List[Optional[torch.Tensor]],
        predicted_tokens_per_seq: int,
        layer_idx: int,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        tokens_per_block: int,
        attention_window_size: int,
        beam_width: int,
        quant_mode: int,
        q_scaling: float,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        rope_append: bool,
        compiled: CompiledMlaRopeGeneration,
    ) -> None:
        """The inputs the kernel takes and answers wrongly. Reads the lengths
        back to the host, which is why it is never on the serving path."""
        p = int(predicted_tokens_per_seq)
        seqs = int(host_context_lengths.shape[0]) - int(num_contexts)
        lens = sequence_length[num_contexts : num_contexts + seqs].tolist()

        # A sequence shorter than its own new tokens has a negative position;
        # the kernel clamps it to 0 and writes the row into the first slot of
        # the sequence's first page, over a real token.
        assert min(lens) >= p, (
            f"every sequence length must be at least predicted_tokens_per_seq ({p}); "
            f"got {min(lens)}"
        )
        # The rope table is indexed by position with no bound check: a longer
        # sequence reads past the table's end.
        table_rows = rotary_cos_sin.numel() // (2 * int(qk_rope_head_dim))
        assert max(lens) <= table_rows, (
            f"a sequence of {max(lens)} tokens reads past the rotary_cos_sin table "
            f"({table_rows} positions)"
        )
        # A position whose page index reaches the block-table width is skipped:
        # the token's KV row is never appended, and nothing reports it.
        capacity = int(kv_cache_block_offsets.shape[-1]) * int(tokens_per_block)
        assert max(lens) <= capacity, (
            f"a sequence of {max(lens)} tokens has pages past the block table's "
            f"{kv_cache_block_offsets.shape[-1]} columns; the kernel skips that KV write"
        )
        # The kernel has no cyclic window: it appends at the absolute position.
        # A window shorter than the sequence is a cyclic cache the stock op
        # wraps into, and this kernel writes past.
        assert int(attention_window_size) >= max(lens), (
            f"attention_window_size {attention_window_size} is below a sequence length "
            f"({max(lens)}); the kernel ignores the window"
        )
        # q_pe is read eight elements at a time with a 16-byte vector load the
        # host never checks: a misaligned view faults and takes the CUDA
        # context with it.
        assert q_pe.data_ptr() % 16 == 0 and q_pe.stride(-2) % 8 == 0 and q_pe.stride(0) % 8 == 0, (
            f"q_pe must be 16-byte aligned with head/token strides divisible by 8; got "
            f"ptr % 16 = {q_pe.data_ptr() % 16}, strides {tuple(q_pe.stride())}"
        )


cute_dsl_mla_rope_generation = CuteDslMlaRopeGeneration()
