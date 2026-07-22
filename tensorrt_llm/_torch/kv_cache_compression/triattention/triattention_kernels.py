# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU kernels for the TriAttention KV-eviction pipeline.

The production path uses one fixed-shape trig-score launch across all dense
layers, CuTE-DSL TopK selection, and grouped C++ compaction. Scoring runs
EXCLUSIVELY through the SM100 CuTe-DSL fused score pack
(``triattention_cute_score_fused.py``): mean aggregation, BF16 KV pools,
head size 64/128, 32/128-token pages, GQA group 4 or 8, per-request score
window starts. There is deliberately no other score path -- any geometry
outside that contract raises loudly at setup instead of routing to a slower
kernel (the original Triton score kernel, the C++ CUDA score stack, and the
single-shot CuTe score kernel have all been deleted). The per-head modes use
the pack's score-only entry; union eviction runs its fused score+stats+union
pipeline (with ``triattention_cute_selection.py``). The split Triton
row-stats/union-reduce launches were retired with the union fusion, and
their standalone copies live in the fused-pipeline unit test as references
(the row-stats kernel itself remains: the per-head modes still normalize
with it). The unit tests validate the CuTe kernels against independent
PyTorch oracles. Selection and compaction live in their respective runtime
modules.

House rules honored throughout:
  * fp32 math (loads up-cast to fp32, fp32 accumulators, fp32 score output).
  * int64 for every flat buffer offset that can exceed 2^31.
  * mask ragged valid-width tails (and frequency tails) in every load and store.
  * the kernels are vendored in this module (no lazy-load hub).
"""

from __future__ import annotations

from typing import List, Optional

import torch
import triton
import triton.language as tl

# --------------------------------------------------------------------------- #
# Scoring: trig-score every cached token across all dense layers.             #
# --------------------------------------------------------------------------- #


# Positions past this row count are no longer exactly representable in fp32,
# so a larger table would silently degrade every downstream phase.
_MEAN_PHASE_MAX_ROWS = 1 << 24


@triton.jit
def _gather_mean_phase_kernel(
    round_starts,
    table_cos,
    table_sin,
    mean_cos,
    mean_sin,
    table_rows,
    NUM_FREQS: tl.constexpr,
    F_BLOCK: tl.constexpr,
):
    """Copy each request's precomputed phase-table row into the fixed buffers."""
    request = tl.program_id(0)
    frequency = tl.arange(0, F_BLOCK)
    frequency_mask = frequency < NUM_FREQS
    table_row = tl.load(round_starts + request).to(tl.int64)
    # Clamp stale or padded round starts into the table instead of faulting;
    # staged cohorts are host-validated, so live rows are never clamped.
    table_row = tl.minimum(tl.maximum(table_row, 0), table_rows - 1)
    source_offset = table_row * NUM_FREQS + frequency
    output_offset = request * NUM_FREQS + frequency
    row_cos = tl.load(table_cos + source_offset, mask=frequency_mask, other=0.0)
    row_sin = tl.load(table_sin + source_offset, mask=frequency_mask, other=0.0)
    tl.store(mean_cos + output_offset, row_cos, mask=frequency_mask)
    tl.store(mean_sin + output_offset, row_sin, mask=frequency_mask)


class MeanPhaseTable:
    """RoPE-style position table of mean trig phases, gathered per round.

    Row ``p`` holds ``mean_o(trig((p + offset_o) * omega_f))`` over the
    calibration offsets for every frequency, so refreshing a round's
    ``mean_cos``/``mean_sin`` is one pure-gather launch over the staged
    round starts instead of a per-round trig kernel. The gather writes in
    place because the compiled CuTe score launch captured the destination
    buffers' device pointers. Eviction never runs under CUDA graph
    capture, so the table itself may regrow; callers must ``ensure``
    capacity while the round starts are still host integers (the gather
    clamps stale rows into the table rather than faulting). Building the
    table is plain torch and works on any device; gathering launches the
    Triton kernel and is CUDA-only.
    """

    def __init__(self, offsets: torch.Tensor, omega: torch.Tensor, initial_rows: int) -> None:
        if (
            offsets.numel() <= 0
            or omega.numel() <= 0
            or offsets.dtype != torch.float32
            or omega.dtype != torch.float32
            or offsets.device != omega.device
        ):
            raise ValueError("mean-phase tables require same-device FP32 offsets and frequencies")
        self.offsets = offsets.contiguous()
        self.omega = omega.contiguous()
        self._offset_values: List[float] = self.offsets.tolist()
        self._cos: Optional[torch.Tensor] = None
        self._sin: Optional[torch.Tensor] = None
        self._rows = 0
        self.ensure(max(int(initial_rows), 1))

    @property
    def rows(self) -> int:
        return self._rows

    def ensure(self, rows: int) -> None:
        """Cover positions ``[0, rows)``, rebuilding the table if it must grow."""
        rows = int(rows)
        if rows <= self._rows:
            return
        if rows > _MEAN_PHASE_MAX_ROWS:
            raise ValueError(f"a {rows}-row mean-phase table exceeds the exact-FP32 position range")
        target = 1
        while target < rows:
            target *= 2
        target = min(max(target, 2 * self._rows), _MEAN_PHASE_MAX_ROWS)
        positions = torch.arange(target, device=self.omega.device, dtype=torch.float32)
        cos_table = torch.zeros(
            (target, self.omega.numel()), dtype=torch.float32, device=self.omega.device
        )
        sin_table = torch.zeros_like(cos_table)
        # Accumulate offset-by-offset in fp32, mirroring the retired
        # per-round Triton kernel's summation order.
        for offset in self._offset_values:
            phase = torch.outer(positions + offset, self.omega)
            cos_table += torch.cos(phase)
            sin_table += torch.sin(phase)
        scale = 1.0 / len(self._offset_values)
        self._cos = cos_table.mul_(scale)
        self._sin = sin_table.mul_(scale)
        self._rows = target

    def gather(
        self,
        round_starts: torch.Tensor,
        mean_cos: torch.Tensor,
        mean_sin: torch.Tensor,
        request_count: int,
    ) -> None:
        """Refresh the fixed mean buffers in place from staged round starts."""
        request_count = int(request_count)
        if request_count <= 0 or request_count > round_starts.numel():
            raise ValueError("phase gather request count is outside its fixed buffers")
        num_freqs = self.omega.numel()
        if (
            mean_cos.ndim != 2
            or mean_cos.shape[0] < request_count
            or mean_cos.shape[1] != num_freqs
            or mean_sin.shape != mean_cos.shape
            or round_starts.dtype != torch.int32
            or self.omega.device.type != "cuda"
            or any(
                tensor.device != self.omega.device for tensor in (round_starts, mean_cos, mean_sin)
            )
            or any(tensor.dtype != torch.float32 for tensor in (mean_cos, mean_sin))
        ):
            raise ValueError("phase gather tensors do not share one valid FP32 CUDA geometry")
        _gather_mean_phase_kernel[(request_count,)](
            round_starts,
            self._cos,
            self._sin,
            mean_cos,
            mean_sin,
            self._rows,
            NUM_FREQS=num_freqs,
            F_BLOCK=triton.next_power_of_2(num_freqs),
            num_warps=1,
        )


class _FixedScoreGroup:
    """Persistent score metadata/output for one fixed geometry.

    Since the per-layer absolute-address ABI, ONE group can span dense layers
    living in DISTINCT storages with DISTINCT block tables. ``block_offsets``
    uses the native TRT-LLM attention layout and ``page_table_slots`` maps each
    scored layer to its V2 pool slot.

    LIFETIME: the group retains references to every scored layer pool -- the
    SM100 CuTe score kernel encodes immutable TMA descriptors from their raw
    device addresses at compile time, so the pools must stay alive (and stay
    put) for as long as the group launches. In production the V2 KV-cache
    manager owns them for the manager's lifetime.
    """

    def __init__(
        self,
        layer_pools: List[torch.Tensor],
        layer_indices: List[int],
        max_requests: int,
        page_count: int,
        seq_len: int,
        num_q_heads: int,
        block_offsets: torch.Tensor,  # [num_pools, max_requests, 2, copied_blocks] int32
        page_table_slots: List[int],  # per scored layer: pool slot into block_offsets
        q_real_LHF: torch.Tensor,
        q_imag_LHF: torch.Tensor,
        mlr_coef_LHF: torch.Tensor,
        freq_scale_sq: torch.Tensor,
        omega: torch.Tensor,
        offsets: torch.Tensor,
        output_width: int,
    ) -> None:
        if not layer_indices or min(max_requests, page_count, seq_len) <= 0:
            raise ValueError("fixed score group requires non-empty positive geometry")
        if output_width <= 0 or output_width > seq_len:
            raise ValueError("fixed score group requires a decode width within its capacity")
        if len(page_table_slots) != len(layer_indices):
            raise ValueError("page_table_slots must align with layer_indices")
        self.max_requests = max_requests
        # Prompt lengths are per-request kernel inputs; this capacity only
        # sizes the widest possible decode window of the output buffer.
        self.output_width = int(output_width)
        self.num_layers = len(layer_indices)
        p0 = layer_pools[layer_indices[0]]
        if p0.ndim != 5:
            raise ValueError("fixed score group requires HND pools")
        device = p0.device
        q_real_LHF = q_real_LHF.to(device=device, dtype=torch.float32).contiguous()
        q_imag_LHF = q_imag_LHF.to(device=device, dtype=torch.float32).contiguous()
        mlr_coef_LHF = mlr_coef_LHF.to(device=device, dtype=torch.float32).contiguous()
        freq_scale_sq = freq_scale_sq.to(device=device, dtype=torch.float32).contiguous()
        omega = omega.to(device=device, dtype=torch.float32).contiguous()
        offsets = offsets.to(device=device, dtype=torch.float32).contiguous()
        _, kv_factor, num_kv_heads, tokens_per_block, head_dim = p0.shape
        if num_q_heads % num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        self.num_freqs = head_dim // 2
        strides = tuple(int(value) for value in p0.stride())
        self.geometry_args = (
            num_q_heads,
            num_kv_heads,
            self.num_freqs,
            tokens_per_block,
            kv_factor,
        )
        # Per-layer ABSOLUTE base addresses. Layers may live in distinct
        # storages (V2 TensorWrapper-per-layer); only geometry must be uniform.
        element_size = p0.element_size()
        layer_base_addrs = torch.zeros(len(layer_pools), dtype=torch.int64, device=device)
        for layer in layer_indices:
            pool = layer_pools[layer]
            if (
                tuple(pool.shape[1:]) != tuple(p0.shape[1:])
                or tuple(pool.stride()) != strides
                or pool.dtype != p0.dtype
            ):
                raise ValueError("fixed score layers must share one uniform geometry")
            address = int(pool.data_ptr())
            if address % element_size:
                raise ValueError("fixed score layer base is not element-aligned")
            layer_base_addrs[layer] = address
        # Calibration tables span every model layer; segments index them by
        # ABSOLUTE layer id, so the tables cover the full calibrated extent.
        self._num_calibrated_layers = q_real_LHF.numel() // (int(num_q_heads) * self.num_freqs)
        # Segment layer ids index the calibration tables ON DEVICE where they
        # cannot be range-checked; validate the extent once here, loudly.
        if min(layer_indices) < 0 or max(layer_indices) >= self._num_calibrated_layers:
            raise ValueError("scored layer index exceeds the calibrated layer extent")
        seg_req = torch.arange(max_requests, dtype=torch.int32, device=device).repeat_interleave(
            self.num_layers
        )
        seg_layer = torch.tensor(layer_indices, dtype=torch.int32, device=device).repeat(
            max_requests
        )
        # Each segment reads the K plane for one request from the same native
        # block-offset buffer used by TRT-LLM attention metadata preparation.
        if (
            block_offsets.ndim != 4
            or tuple(block_offsets.shape[1:3]) != (max_requests, 2)
            or block_offsets.shape[3] < page_count
            or block_offsets.dtype != torch.int32
            or block_offsets.device != device
        ):
            raise ValueError("block offsets do not match fixed score geometry")
        if not block_offsets.is_contiguous():
            raise ValueError("fixed score block offsets must be contiguous")
        slots_t = torch.tensor(page_table_slots, dtype=torch.int64, device=device)
        if int(slots_t.max()) >= int(block_offsets.shape[0]):
            raise ValueError("page table slot exceeds staged page-id planes")
        req_idx = torch.arange(max_requests, dtype=torch.int64, device=device).repeat_interleave(
            self.num_layers
        )
        slot_idx = slots_t.repeat(max_requests)
        seg_page_off = slot_idx * block_offsets.stride(0) + req_idx * block_offsets.stride(1)
        self.output = torch.empty(
            max_requests,
            self.num_layers,
            num_q_heads,
            self.output_width,
            dtype=torch.float32,
            device=device,
        )
        self.pointer_prefix = (
            p0,
            layer_base_addrs,
            block_offsets.view(-1),
            seg_page_off,
            seg_req,
            seg_layer,
        )
        self.pointer_middle = (
            q_real_LHF.view(-1),
            q_imag_LHF.view(-1),
            mlr_coef_LHF.view(-1),
        )
        self.pointer_tail = (freq_scale_sq, omega, offsets)
        # The SM100 CuTe fused score pack (see
        # triattention_cute_score_fused.py) is THE score implementation; its
        # score-only entry is compiled by the first ``prepare_cute_score``
        # call. The runner encodes TMA descriptors from the actual pool
        # tensors, hence the pool references retained here (see the LIFETIME
        # note in the class docstring).
        self.seq_len = int(seq_len)
        self._cute_score_runner = None
        self._cute_score_attempted = False
        self._cute_layer_pools = list(layer_pools)
        self._cute_layer_indices = [int(layer) for layer in layer_indices]

    def prepare_cute_score(self, mean_cos: torch.Tensor, mean_sin: torch.Tensor) -> None:
        """Compile the fused CuTe runner's score-only entry; raise loudly otherwise.

        Call this outside CUDA graph capture: compilation allocates memory
        and synchronizes. The fused score pack is the ONLY score
        implementation, so an unsupported geometry raises ValueError here
        and a runner construction failure raises RuntimeError -- there is
        deliberately no fallback path.

        Supported contract: SM100 exactly, BF16 pools, 32- or 128-token
        pages, 32 or 64 frequencies (head size 64/128), 4 or 8 query heads
        per KV head, and a bucket capacity (``seq_len``) aligned to the
        historical score tile — this covers the Qwen3 and GPT-OSS production
        geometries as well as the original validation shape.
        """
        if self._cute_score_attempted:
            return
        self._cute_score_attempted = True
        anchor = self.pointer_prefix[0]
        num_q_heads, num_kv_heads, num_freqs, tokens_per_block, kv_factor = self.geometry_args
        max_segments = self.max_requests * self.num_layers
        # The retired single-shot kernel stored full unmasked compute tiles
        # (64 tokens, or one page for 128-token pages), which forced
        # tile-aligned buckets. The fused kernel masks its ragged tail, but
        # the bucket contract is kept unchanged so the kernel swap cannot
        # silently admit new geometry (production pow2 buckets satisfy it).
        score_tile_tokens = max(64, int(tokens_per_block))
        supported = (
            torch.cuda.get_device_capability(anchor.device) == (10, 0)
            and anchor.dtype == torch.bfloat16
            and kv_factor == 2
            and tokens_per_block in (32, 128)
            and num_freqs in (32, 64)
            and num_q_heads % num_kv_heads == 0
            and num_q_heads // num_kv_heads in (4, 8)
            and int(anchor.stride(-1)) == 1
            and self.seq_len % score_tile_tokens == 0
            # The kernel's head-plane base offset is 64-bit; the widest
            # 32-bit product left is one plane (N-1 head columns of one
            # segment stride), which the score bucket keeps far below 2^31.
            # Group-4 geometries pad the head axis to the MMA tile N=8.
            and (8 - 1) * max_segments * self.seq_len < 2**31
        )
        if not supported:
            raise ValueError(
                "TriAttention score requires SM100, bf16 KV pools, head size "
                "64/128, 32/128-token pages, GQA group 4 or 8, and a bucket "
                "capacity aligned to the score compute tile; got "
                f"capability={torch.cuda.get_device_capability(anchor.device)}, "
                f"dtype={anchor.dtype}, kv_factor={kv_factor}, "
                f"tokens_per_block={tokens_per_block}, num_freqs={num_freqs}, "
                f"heads={num_q_heads}q/{num_kv_heads}kv, "
                f"stride={int(anchor.stride(-1))}, "
                f"seq_len={self.seq_len} (tile {score_tile_tokens}), "
                f"offset_audit={num_kv_heads * 8 * max_segments * self.seq_len}"
            )
        device = anchor.device
        try:
            from .triattention_cute_score_fused import TriAttentionCuteScoreRunner

            # The kernel scores each request's window (from its staged
            # per-request start) into its own head-major scratch (row =
            # query head, column = segment * seq_len + token); ``launch``
            # gathers each request's decode window from that scratch into
            # ``self.output``. All buffers below are persistent because the
            # compiled kernel captures their device pointers.
            # The kernel writes one scratch row per padded head column
            # (GQA group below 8 pads up to the MMA tile); the gather in
            # ``launch`` reads only the real heads.
            scratch = torch.empty(
                num_kv_heads * 8 * max_segments * self.seq_len,
                dtype=torch.float32,
                device=device,
            )
            seg_seq_len = torch.zeros(max_segments, dtype=torch.int32, device=device)
            seg_out_offset = (
                torch.arange(max_segments, dtype=torch.int64, device=device) * self.seq_len
            ).to(torch.int32)
            # Per-request score window starts, staged before each launch;
            # the compiled kernels capture this buffer's device pointer.
            # The union fusion runner shares it (and the segment buffers).
            token_starts = torch.zeros(self.max_requests, dtype=torch.int32, device=device)
            gather_columns = torch.arange(self.output_width, dtype=torch.int64, device=device)
            self._cute_score_runner = TriAttentionCuteScoreRunner(
                layer_pools=self._cute_layer_pools,
                layer_indices=self._cute_layer_indices,
                max_requests=self.max_requests,
                num_layers=self.num_layers,
                seq_len=self.seq_len,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                num_freqs=num_freqs,
                tokens_per_block=tokens_per_block,
                page_ids=self.pointer_prefix[2],
                seg_page_off=self.pointer_prefix[3],
                seg_req_id=self.pointer_prefix[4],
                seg_layer_id=self.pointer_prefix[5],
                seg_seq_len=seg_seq_len,
                seg_out_offset=seg_out_offset,
                token_starts=token_starts,
                q_real=self.pointer_middle[0],
                q_imag=self.pointer_middle[1],
                mlr_coef=self.pointer_middle[2],
                mean_cos=mean_cos,
                mean_sin=mean_sin,
                freq_scale_sq=self.pointer_tail[0],
                output=scratch,
                # Score-only mode: the stats and union-finalize kernels are
                # compiled lazily by ``_union_fusion_runner`` when (and only
                # when) union eviction actually launches.
                enable_partial_stats=False,
            )
        except (ImportError, RuntimeError, ValueError, AssertionError) as error:
            raise RuntimeError(
                "TriAttention CuTe score setup failed and no other score path exists"
            ) from error
        self._cute_scratch = scratch
        self._cute_seg_seq_len = seg_seq_len
        self._cute_seg_out_offset = seg_out_offset
        self._cute_token_starts = token_starts
        self._cute_gather_columns = gather_columns.view(1, 1, 1, -1)
        # Fused score+stats+union pipeline (Fanrong Li's two-kernel scheme):
        # THE union path, built lazily on the first union launch. ONE runner
        # serves every cohort: the score window start is per-request runtime
        # metadata, not a compile-time constant.
        self._union_fusion_runner_entry = None
        from tensorrt_llm.logger import logger

        logger.info(
            f"TriAttention CuTe score enabled: {num_q_heads}q/{num_kv_heads}kv heads, "
            f"{num_freqs} freqs, {tokens_per_block}-token pages"
        )

    def launch(
        self,
        request_count: int,
        valid_seq_lens: torch.Tensor,
        valid_widths: torch.Tensor,
        token_starts_device: torch.Tensor,
        mean_cos: torch.Tensor,
        mean_sin: torch.Tensor,
        aggregation: str = "mean",
    ) -> torch.Tensor:
        """Return decode-only scores as ``[request, layer, head, token]``.

        Runs the fused CuTe runner's score-only entry (the only score
        implementation) and writes each request's decode width
        (``valid_seq_len - token_start``) into ``valid_widths``, which the
        selection reduce kernels consume. Each request scores its own window
        from its staged start, so one cohort may mix prompt lengths. Only
        mean aggregation exists; the runner dispatches every request count
        up to the group capacity.
        """
        if aggregation != "mean":
            raise ValueError(
                f"unsupported score aggregation {aggregation!r}: max aggregation "
                "was removed with the C++ score stack; only 'mean' exists"
            )
        if request_count <= 0 or request_count > self.max_requests:
            raise ValueError("request count exceeds fixed score capacity")
        if (
            valid_widths.ndim != 1
            or valid_widths.numel() < request_count
            or valid_widths.dtype != torch.int32
            or valid_widths.device != self.output.device
        ):
            raise ValueError("score output lengths do not fit the keep-set selector")
        num_segments = request_count * self.num_layers
        output = self.output[:request_count]
        # Lazy compile covers groups used without their owning workspace
        # (unit tests); production compiles in the workspace constructor,
        # outside CUDA graph capture.
        self.prepare_cute_score(mean_cos, mean_sin)
        runner = self._cute_score_runner
        if runner is None or not runner.supports(request_count):
            raise RuntimeError(
                f"TriAttention CuTe score has no compiled variant for "
                f"request_count={request_count} (capacity {self.max_requests}) "
                "and no other score path exists"
            )
        # Per-request decode widths for the selection reduce kernels; the
        # deleted C++ score op used to write these (seq_len - token_start).
        torch.sub(
            valid_seq_lens[:request_count],
            token_starts_device[:request_count],
            out=valid_widths[:request_count],
        )
        # Stage per-segment valid lengths (segment = request x layer).
        torch.index_select(
            valid_seq_lens,
            0,
            self.pointer_prefix[4][:num_segments],
            out=self._cute_seg_seq_len[:num_segments],
        )
        # Stage the per-request score window starts: the compiled kernel
        # captured this buffer's pointer and reads one start per request.
        self._cute_token_starts[:request_count].copy_(token_starts_device[:request_count])
        runner.launch(request_count, mean_cos, mean_sin)
        # The kernel wrote each request's window scores (from its pinned
        # prompt length) into its head-major scratch. Gather each request's
        # decode window into the group output, the ``[request, layer, head,
        # token]`` layout the selection kernels read. Columns past a
        # request's valid width carry unscored scratch data; consumers mask
        # by ``valid_widths``.
        num_q_heads = int(self.geometry_args[0])
        num_kv_heads = int(self.geometry_args[1])
        group_size = num_q_heads // num_kv_heads
        # The scratch head axis is padded to the MMA tile N=8 per KV head;
        # slicing the view to the real group size skips the zero padding
        # columns.
        source = (
            self._cute_scratch[: num_kv_heads * 8 * num_segments * self.seq_len]
            .view(num_kv_heads, 8, request_count, self.num_layers, self.seq_len)[:, :group_size]
            .permute(2, 3, 0, 1, 4)
        )
        columns = token_starts_device[:request_count].to(torch.int64).view(
            -1, 1, 1, 1, 1
        ) + self._cute_gather_columns.view(1, 1, 1, 1, -1)
        columns = columns.clamp_(max=self.seq_len - 1).expand(
            request_count,
            self.num_layers,
            num_kv_heads,
            group_size,
            self.output_width,
        )
        torch.gather(
            source,
            4,
            columns,
            out=output.view(
                request_count,
                self.num_layers,
                num_kv_heads,
                group_size,
                self.output_width,
            ),
        )
        return output

    def _union_fusion_runner(self, mean_cos: torch.Tensor, mean_sin: torch.Tensor):
        """Build (or reuse) the ONE fused score/stats/union runner.

        The score window start is per-request runtime metadata staged into a
        persistent device buffer, so a single compiled runner serves every
        cohort. This is THE union path: a construction failure raises loudly
        instead of recording a fallback.
        """
        if self._union_fusion_runner_entry is not None:
            return self._union_fusion_runner_entry
        num_q_heads, num_kv_heads, num_freqs, tokens_per_block, _ = self.geometry_args
        try:
            from .triattention_cute_score_fused import (
                TriAttentionCuteScoreRunner as _FusedUnionScoreRunner,
            )

            device = self.output.device
            # The union output rows are sized by the whole bucket (the widest
            # possible window); consumers mask by the per-request widths.
            union_rows = torch.empty(
                (self.max_requests, self.seq_len),
                dtype=torch.float32,
                device=device,
            )
            # The scratch, segment buffers, and staged per-request window
            # starts are shared with the score-only runner built by
            # ``prepare_cute_score``; the compiled kernels capture their
            # device pointers.
            runner = _FusedUnionScoreRunner(
                layer_pools=self._cute_layer_pools,
                layer_indices=self._cute_layer_indices,
                max_requests=self.max_requests,
                num_layers=self.num_layers,
                seq_len=self.seq_len,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                num_freqs=num_freqs,
                tokens_per_block=tokens_per_block,
                page_ids=self.pointer_prefix[2],
                seg_page_off=self.pointer_prefix[3],
                seg_req_id=self.pointer_prefix[4],
                seg_layer_id=self.pointer_prefix[5],
                seg_seq_len=self._cute_seg_seq_len,
                seg_out_offset=self._cute_seg_out_offset,
                token_starts=self._cute_token_starts,
                q_real=self.pointer_middle[0],
                q_imag=self.pointer_middle[1],
                mlr_coef=self.pointer_middle[2],
                mean_cos=mean_cos,
                mean_sin=mean_sin,
                freq_scale_sq=self.pointer_tail[0],
                output=self._cute_scratch,
                enable_partial_stats=True,
            )
        except (ImportError, RuntimeError, ValueError, AssertionError) as error:
            raise RuntimeError(
                "TriAttention CuTe union fusion setup failed and no other union path exists"
            ) from error
        self._union_fusion_runner_entry = (runner, union_rows, self._cute_token_starts)
        return self._union_fusion_runner_entry

    def launch_cute_union_fusion(
        self,
        request_count: int,
        valid_seq_lens: torch.Tensor,
        valid_widths: torch.Tensor,
        token_starts_device: torch.Tensor,
        mean_cos: torch.Tensor,
        mean_sin: torch.Tensor,
        union_out: torch.Tensor,
    ) -> None:
        """Run the fused score+stats+normalized-union pipeline (THE union path).

        Each request scores its own window (``token_starts_device`` carries
        the per-request pinned prompt lengths), so mixed-prompt cohorts are
        served directly. There is deliberately no fallback: an unsupported
        geometry or request count raises loudly instead of routing to the
        retired split score/row-stats/union launches.
        """
        self.prepare_cute_score(mean_cos, mean_sin)
        if request_count <= 0 or request_count > self.max_requests:
            raise ValueError("request count exceeds fixed score capacity")
        runner, union_rows, staged_token_starts = self._union_fusion_runner(mean_cos, mean_sin)
        if not runner.supports_union_fusion(request_count):
            raise RuntimeError(
                f"TriAttention CuTe union fusion has no compiled variant for "
                f"request_count={request_count} (capacity {self.max_requests}) "
                "and no other union path exists"
            )
        num_segments = request_count * self.num_layers
        torch.sub(
            valid_seq_lens[:request_count],
            token_starts_device[:request_count],
            out=valid_widths[:request_count],
        )
        torch.index_select(
            valid_seq_lens,
            0,
            self.pointer_prefix[4][:num_segments],
            out=self._cute_seg_seq_len[:num_segments],
        )
        staged_token_starts[:request_count].copy_(token_starts_device[:request_count])
        runner.launch_union_fusion(request_count, mean_cos, mean_sin, union_rows[:request_count])
        columns = min(union_rows.shape[1], union_out.shape[1])
        union_out[:request_count, :columns].copy_(union_rows[:request_count, :columns])


# --------------------------------------------------------------------------- #
# Selection: combine scores per mode, then finalize the top-k set.            #
# --------------------------------------------------------------------------- #


@triton.jit
def _score_row_stats_kernel(
    scores,
    valid_widths,
    row_mean,
    row_inv_std,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Compute one valid-prefix mean and inverse standard deviation per score row."""
    flat_row = tl.program_id(0)
    request = flat_row // ROWS
    valid_width = tl.load(valid_widths + request)
    score_row = scores + flat_row * WIDTH
    lane = tl.arange(0, BLOCK)
    score_sum = 0.0
    for start in tl.static_range(0, WIDTH, BLOCK):
        token = start + lane
        valid = token < valid_width
        value = tl.load(score_row + token, mask=valid, other=0.0).to(tl.float32)
        score_sum += tl.sum(value, axis=0)
    mean = score_sum / valid_width
    square_sum = 0.0
    for start in tl.static_range(0, WIDTH, BLOCK):
        token = start + lane
        valid = token < valid_width
        value = tl.load(score_row + token, mask=valid, other=0.0).to(tl.float32)
        centered = tl.where(valid, value - mean, 0.0)
        square_sum += tl.sum(centered * centered, axis=0)
    std = tl.sqrt(square_sum / valid_width)
    tl.store(row_mean + flat_row, mean)
    tl.store(row_inv_std + flat_row, 1.0 / tl.maximum(std, 1e-6))


@triton.jit
def _score_per_head_reduce_kernel(
    scores,
    valid_widths,
    row_mean,
    row_inv_std,
    selection_scores,
    selection_seq_lens,
    NUM_LAYERS: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    SELECTION_ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    PER_LAYER: tl.constexpr,
    NORMALIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Reduce query-head score rows into one selector row per KV-head domain."""
    request = tl.program_id(0)
    selection_row = tl.program_id(1)
    token_block = tl.program_id(2)
    token = token_block * BLOCK + tl.arange(0, BLOCK)
    valid_width = tl.load(valid_widths + request)
    valid_token = token < valid_width

    if token_block == 0:
        tl.store(
            selection_seq_lens + request * SELECTION_ROWS + selection_row,
            valid_width,
        )

    kv_head = selection_row % NUM_KV_HEADS
    if PER_LAYER:
        layer = selection_row // NUM_KV_HEADS
        reduced = tl.full((BLOCK,), -float("inf"), tl.float32)
        for query_in_group in tl.static_range(0, QUERY_GROUP_SIZE):
            query_head = kv_head * QUERY_GROUP_SIZE + query_in_group
            flat_row = (request * NUM_LAYERS + layer) * NUM_QUERY_HEADS + query_head
            value = tl.load(
                scores + flat_row * WIDTH + token,
                mask=valid_token,
                other=-float("inf"),
            ).to(tl.float32)
            if NORMALIZE:
                mean = tl.load(row_mean + flat_row)
                inv_std = tl.load(row_inv_std + flat_row)
                value = tl.where(valid_token, (value - mean) * inv_std, -float("inf"))
            reduced = tl.maximum(reduced, value)
    else:
        reduced = tl.zeros((BLOCK,), tl.float32)
        for layer in tl.static_range(0, NUM_LAYERS):
            layer_max = tl.full((BLOCK,), -float("inf"), tl.float32)
            for query_in_group in tl.static_range(0, QUERY_GROUP_SIZE):
                query_head = kv_head * QUERY_GROUP_SIZE + query_in_group
                flat_row = (request * NUM_LAYERS + layer) * NUM_QUERY_HEADS + query_head
                value = tl.load(
                    scores + flat_row * WIDTH + token,
                    mask=valid_token,
                    other=-float("inf"),
                ).to(tl.float32)
                if NORMALIZE:
                    mean = tl.load(row_mean + flat_row)
                    inv_std = tl.load(row_inv_std + flat_row)
                    value = tl.where(valid_token, (value - mean) * inv_std, -float("inf"))
                layer_max = tl.maximum(layer_max, value)
            reduced += layer_max
        reduced /= NUM_LAYERS

    output = (request * SELECTION_ROWS + selection_row) * WIDTH + token
    tl.store(selection_scores + output, reduced, mask=token < WIDTH)


def prepare_per_head_scores(
    scores: torch.Tensor,
    valid_widths: torch.Tensor,
    row_mean: torch.Tensor,
    row_inv_std: torch.Tensor,
    selection_scores: torch.Tensor,
    selection_seq_lens: torch.Tensor,
    request_count: int,
    *,
    num_kv_heads: int,
    per_layer: bool,
    normalize_scores: bool,
) -> None:
    """Normalize and reduce score rows for either per-head eviction mode."""
    request_count = int(request_count)
    num_kv_heads = int(num_kv_heads)
    if not scores.is_cuda or scores.ndim != 4 or scores.dtype != torch.float32:
        raise ValueError("per-head score preparation requires CUDA FP32 rows")
    if not scores.is_contiguous() or request_count != scores.shape[0]:
        raise ValueError("per-head score preparation request geometry does not match")
    _, num_layers, num_query_heads, width = scores.shape
    if num_kv_heads <= 0 or num_query_heads % num_kv_heads:
        raise ValueError("per-head score preparation requires valid GQA geometry")
    selection_rows = num_layers * num_kv_heads if per_layer else num_kv_heads
    if (
        valid_widths.shape != (request_count,)
        or valid_widths.dtype != torch.int32
        or valid_widths.device != scores.device
        or row_mean.numel() < request_count * num_layers * num_query_heads
        or row_inv_std.shape != row_mean.shape
        or selection_scores.shape != (request_count, selection_rows, width)
        or selection_scores.dtype != torch.float32
        or selection_scores.device != scores.device
        or selection_seq_lens.shape != (request_count, selection_rows)
        or selection_seq_lens.dtype != torch.int32
        or selection_seq_lens.device != scores.device
    ):
        raise ValueError("per-head score preparation buffers do not match")

    stats_block = 256
    rows = num_layers * num_query_heads
    if normalize_scores:
        _score_row_stats_kernel[(request_count * rows,)](
            scores,
            valid_widths,
            row_mean,
            row_inv_std,
            ROWS=rows,
            WIDTH=width,
            BLOCK=stats_block,
            num_warps=4,
        )
    reduction_block = 256
    _score_per_head_reduce_kernel[
        (request_count, selection_rows, triton.cdiv(width, reduction_block))
    ](
        scores,
        valid_widths,
        row_mean,
        row_inv_std,
        selection_scores,
        selection_seq_lens,
        NUM_LAYERS=num_layers,
        NUM_QUERY_HEADS=num_query_heads,
        NUM_KV_HEADS=num_kv_heads,
        QUERY_GROUP_SIZE=num_query_heads // num_kv_heads,
        SELECTION_ROWS=selection_rows,
        WIDTH=width,
        PER_LAYER=per_layer,
        NORMALIZE=normalize_scores,
        BLOCK=reduction_block,
        num_warps=4,
    )


# --------------------------------------------------------------------------- #
# Compaction: pack the kept ordinals into per-request move indices.           #
# --------------------------------------------------------------------------- #


@triton.jit
def _settle_ties_and_pack_compaction_sources_kernel(
    scores,
    seq_lens,
    prompt_offsets,
    provisional_indices,
    output_indices,
    valid_seq_lens,
    dense_offsets,
    dense_indices,
    swa_offsets,
    swa_indices,
    WIDTH: tl.constexpr,
    KEEP_COUNT: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    SELECTION_ROWS: tl.constexpr,
    DENSE_TOTAL: tl.constexpr,
    SWA_TOTAL: tl.constexpr,
    MOVE_CAPACITY: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    SWA_WINDOW: tl.constexpr,
    UNION: tl.constexpr,
    PER_LAYER: tl.constexpr,
    HAS_SWA: tl.constexpr,
    HAS_SETTLE: tl.constexpr,
    HAS_PACK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Settle one selection row's ties, then pack its compaction move sources.

    One program per (request, selection row). The first half settles the
    provisional top-k: recover the top-k threshold
    from the provisional selection, count the strictly greater scores, then
    emit the kept ordinals in increasing order, rebased by the row's pinned
    prompt length. With ``HAS_PACK`` the same program then packs the move
    sources for the packed rows this selection row feeds: the kept ordinals
    it just wrote, followed by the request's protected tail, plus the SWA
    rows (latest window) under the
    same conditions as the retired standalone kernel. Union selection has one row per
    request feeding every KV head's packed row, so that single program writes
    all of them. ``HAS_PACK=False`` compiles the second half away, leaving
    exactly the settle stage; ``HAS_SETTLE=False`` compiles the first half
    away instead, packing pre-settled ordinals read from
    ``output_indices`` -- the draft co-compaction flow, whose keep set is
    the target's and needs no settling. The pre-fusion standalone copies
    live in the fused-kernel unit test as the bit-equality references.
    Fusing the launches was suggested by Fanrong Li (torch-graph review
    2026-07-20).
    """
    request = tl.program_id(0)
    selection_domain = tl.program_id(1)
    row = request * SELECTION_ROWS + selection_domain
    row_scores = scores + row * WIDTH
    row_selected = provisional_indices + row * KEEP_COUNT
    row_output = output_indices + row * OUTPUT_WIDTH
    if HAS_SETTLE:
        # Scores are decode-relative; this row's pinned prompt length rebases
        # the emitted ordinals to absolute positions (per row, so one launch
        # may mix prompt lengths).
        prompt_len = tl.load(prompt_offsets + row)

        threshold = float("inf")
        for start in tl.static_range(0, KEEP_COUNT, BLOCK):
            selected_offset = start + tl.arange(0, BLOCK)
            selected_mask = selected_offset < KEEP_COUNT
            token_index = tl.load(
                row_selected + selected_offset,
                mask=selected_mask,
                other=0,
            )
            selected_score = tl.load(
                row_scores + token_index,
                mask=selected_mask,
                other=float("inf"),
            ).to(tl.float32)
            threshold = tl.minimum(threshold, tl.min(selected_score, axis=0))

        seq_len = tl.load(seq_lens + row)
        greater_count = 0
        for start in tl.static_range(0, WIDTH, BLOCK):
            token_index = start + tl.arange(0, BLOCK)
            valid = (token_index < WIDTH) & (token_index < seq_len)
            score = tl.load(
                row_scores + token_index,
                mask=valid,
                other=float("-inf"),
            ).to(tl.float32)
            greater_count += tl.sum((valid & (score > threshold)).to(tl.int32))

        tie_quota = KEEP_COUNT - greater_count
        output_count = 0
        ties_seen = 0
        for start in tl.static_range(0, WIDTH, BLOCK):
            token_index = start + tl.arange(0, BLOCK)
            valid = (token_index < WIDTH) & (token_index < seq_len)
            score = tl.load(
                row_scores + token_index,
                mask=valid,
                other=float("-inf"),
            ).to(tl.float32)
            greater = valid & (score > threshold)
            tied = valid & (score == threshold)
            tied_i32 = tied.to(tl.int32)
            tie_rank = ties_seen + tl.cumsum(tied_i32, axis=0) - tied_i32
            selected = greater | (tied & (tie_rank < tie_quota))
            selected_i32 = selected.to(tl.int32)
            write_offset = output_count + tl.cumsum(selected_i32, axis=0) - selected_i32
            tl.store(
                row_output + write_offset,
                token_index + prompt_len,
                mask=selected,
            )
            output_count += tl.sum(selected_i32)
            ties_seen += tl.sum(tied_i32)

    if HAS_PACK:
        if HAS_SETTLE:
            # The emission above scatters through other lanes of this
            # program; make those global stores visible to every lane
            # before the pack half reads the row back.
            tl.debug_barrier()
        dense_begin = tl.load(dense_offsets + request)
        dense_end = tl.load(dense_offsets + request + 1)
        dense_count = dense_end - dense_begin
        valid_len = tl.load(valid_seq_lens + request)
        if HAS_SWA:
            swa_begin = tl.load(swa_offsets + request)
            swa_end = tl.load(swa_offsets + request + 1)
            swa_count = swa_end - swa_begin
        for move_start in tl.static_range(0, MOVE_CAPACITY, BLOCK):
            move = move_start + tl.arange(0, BLOCK)
            selected = tl.load(
                row_output + move,
                mask=move < KEEP_COUNT,
                other=0,
            )
            dense_source = tl.where(move < KEEP_COUNT, selected, valid_len + move - KEEP_COUNT)
            if UNION:
                # The one union row per request feeds every KV head's packed
                # row with the same move sources.
                for head in tl.static_range(0, NUM_KV_HEADS):
                    tl.store(
                        dense_indices + head * DENSE_TOTAL + dense_begin.to(tl.int64) + move,
                        dense_source,
                        mask=move < dense_count,
                    )
            else:
                domain = tl.program_id(1)
                dense_output = domain.to(tl.int64) * DENSE_TOTAL + dense_begin.to(tl.int64) + move
                tl.store(dense_indices + dense_output, dense_source, mask=move < dense_count)
            if HAS_SWA:
                swa_source = valid_len - SWA_WINDOW + move
                if UNION:
                    for head in tl.static_range(0, NUM_KV_HEADS):
                        tl.store(
                            swa_indices + head * SWA_TOTAL + swa_begin.to(tl.int64) + move,
                            swa_source,
                            mask=move < swa_count,
                        )
                else:
                    domain = tl.program_id(1)
                    # Per-layer selection has one dense domain per (layer,
                    # head). SWA uses one shared source row per head, so only
                    # the first layer writes it.
                    if PER_LAYER:
                        write_swa = domain < NUM_KV_HEADS
                    else:
                        write_swa = move >= 0
                    head = domain % NUM_KV_HEADS
                    swa_output = head.to(tl.int64) * SWA_TOTAL + swa_begin.to(tl.int64) + move
                    tl.store(
                        swa_indices + swa_output,
                        swa_source,
                        mask=write_swa & (move < swa_count),
                    )
