# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode attention classes ``layers.adopt_model`` re-classes the built-in attention as."""

from __future__ import annotations

import dataclasses
import importlib
from typing import Optional

import torch

from tensorrt_llm._torch import flashinfer_utils as _flashinfer_utils
from tensorrt_llm._torch.attention.backends.fmha import flashinfer_trtllm_gen as _fi_gen
from tensorrt_llm._torch.attention.backends.fmha.interface import FmhaPhase
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionMetadata,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.attention.rotary_embedding import RotaryEmbedding
from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenAttention
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.logger import logger

from .dispatch import LAUNCH_PDL, sm_of
from .ops.attention.paged_decode_attention import HEAD_DIM as _DECODE_HEAD_DIM
from .ops.attention.paged_decode_attention import MAX_ROWS as _DECODE_MAX_ROWS
from .ops.attention.paged_decode_attention import Q_PER_KV as _DECODE_Q_PER_KV
from .ops.attention.paged_decode_attention import paged_decode_attention, window_splits
from .ops.attention.preattn_prep import HEAD_DIM as _PREP_HEAD_DIM
from .ops.attention.preattn_prep import PreattnPrep, is_rope_table

#: Rows a decode step may carry for ``preattn_prep`` to run: 128 requests of a
#: last accepted token and 7 drafts each (a generation worker of ``max_batch_size``
#: 128). Rows are independent in the op (one position per row), so a step of
#: several sequences runs it as one of one.
_PREATTN_PREP_MAX_ROWS = 1024

#: Rows a context step (prefill chunks, with any generation rows beside them) may
#: carry for ``context_prep`` to run the same op: the largest step its bit-identity
#: sample covers (``ops/attention/preattn_prep.py``, 32768 rows). The op has no row
#: limit of its own; every step a ``max_num_tokens`` of 8192 admits is below this.
_CONTEXT_PREP_MAX_ROWS = 32768


def zero_padded_rows(output: Optional[torch.Tensor], attn_metadata: AttentionMetadata) -> bool:
    """Zero the rows of a context step's attention output past the step's real rows.

    A step with context rows that replays a breakable prefill CUDA graph is padded to the
    graph's bucket (``attn_metadata.padded_num_tokens``). The captured segment in front of
    the attention op allocates its output for every row of the bucket, and the attention
    kernels write the step's ``attn_metadata.num_tokens`` real rows only, so the other rows
    hold whatever the graph's memory pool last held there. Those rows go on through the
    output projection, the router and the experts; their results are discarded and no op
    mixes rows, so no real row's value depends on them, but which experts they load does,
    and that changes with any change of the captured body's allocations. Zeroed, they are
    the same rows on every replay of every build.

    Returns whether rows were zeroed: never on a decode step, an unpadded step or an
    output with no row past the real ones.
    """
    if output is None or getattr(attn_metadata, "padded_num_tokens", None) is None:
        return False
    real = attn_metadata.num_tokens
    if attn_metadata.num_contexts == 0 or output.shape[0] <= real:
        return False
    output[real:].zero_()
    return True


def _builtin_prep_kernel_mismatch() -> Optional[str]:
    """Why the built-in prep would not run the kernels ``preattn_prep`` reproduces, or ``None``.

    ``preattn_prep`` reproduces flashinfer's CuTe-DSL RMSNorm (its summation
    order) and flashinfer's cos/sin-cache RoPE. Without flashinfer the built-in
    norm and RoPE take torch fallbacks (the RoPE one with BF16 cos/sin); with
    the norm's CUDA-JIT kernel the sum runs in another order. Either way a
    different rounding would follow. Which norm kernel runs is read from
    flashinfer's ``_use_cuda_norm``; if that is absent the answer is unknown,
    which counts as a mismatch.
    """
    if not _flashinfer_utils.IS_FLASHINFER_AVAILABLE:
        return "flashinfer is not available, so the built-in norm and RoPE run torch fallbacks"
    try:
        flashinfer_norm = importlib.import_module("flashinfer.norm")
    except ImportError as error:
        return f"flashinfer.norm does not import ({error})"
    uses_cuda_norm = getattr(flashinfer_norm, "_use_cuda_norm", None)
    if uses_cuda_norm is None:
        return "flashinfer.norm has no _use_cuda_norm, so its RMSNorm kernel is unknown"
    if uses_cuda_norm(torch.device("cuda", torch.cuda.current_device())):
        return (
            "the built-in RMSNorm runs flashinfer's CUDA-JIT kernel, not the CuTe-DSL one "
            "(FLASHINFER_USE_CUDA_NORM=1 or an arch the CuTe DSL cannot target)"
        )
    return None


class MuseSparkOpenTepAttention(MuseSparkOpenAttention):
    """The built-in attention, whose pre-attention prep a step can swap for one op.

    Never constructed directly: ``adopt`` re-classes a built-in
    ``MuseSparkOpenAttention`` in place, adding methods and one plain attribute.
    A call carrying ``preattn_prep`` -- only ``DecodeTarget`` passes it -- runs
    the prep as that op when the projection fits it (at most
    ``_PREATTN_PREP_MAX_ROWS`` rows). A call carrying ``context_prep`` -- only
    ``PrefillTarget`` passes it, under the ``context_prep`` fast path -- runs the
    same op on a context step's projection: every row of the step, context and
    generation rows alike (at most ``_CONTEXT_PREP_MAX_ROWS``). Every other call
    runs the built-in ``preprocess_qkv``. Both produce the same bits. The op is
    launched with programmatic dependent launch under the switch the built-in
    flashinfer kernels follow (``TRTLLM_ENABLE_PDL``, on by default).

    With ``_zero_padded_rows`` set (``context_prep``), the attention op also zeroes
    the rows of its output that a replayed prefill CUDA graph pads the step with and
    no attention kernel writes (``zero_padded_rows``).
    """

    _preattn_prep: Optional[PreattnPrep]
    #: The op a ``PrefillTarget`` layer call hands on, for that call only.
    _context_prep: Optional[PreattnPrep] = None
    #: Set on the instance by ``layers.adopt_model`` under ``context_prep``: the attention
    #: op zeroes the padded rows of a replayed prefill graph's output (``zero_padded_rows``).
    #: An instance attribute, not a call keyword: a replay runs the attention op from its
    #: captured arguments without running ``forward``.
    _zero_padded_rows: bool = False

    @classmethod
    def adopt(cls, attention: MuseSparkOpenAttention) -> "MuseSparkOpenTepAttention":
        """Re-class one built-in attention module, after ``check``.

        Raises:
            TypeError, ValueError: as ``check``.
        """
        cls.check(attention)
        attention.__class__ = cls
        attention._preattn_prep = None
        return attention

    @classmethod
    def check(cls, attention: MuseSparkOpenAttention) -> None:
        """Whether ``adopt`` may re-class this module; changes nothing.

        Raises:
            TypeError: if ``attention`` is not exactly the built-in ``MuseSparkOpenAttention``.
            ValueError: if the module's built-in prep is not the chain ``preattn_prep``
                reproduces (the fused-preattention switch, a weighted, Gemma,
                cuda-tile or NVFP4 norm, RoPE in the attention op, a non-fused-QKV
                backend, another head width or rotary form), or its RoPE table is
                not one ``preattn_prep`` reads.
        """
        if type(attention) is not MuseSparkOpenAttention:
            raise TypeError(
                f"{cls.__name__} adopts the built-in MuseSparkOpenAttention, "
                f"got {type(attention).__name__}"
            )
        rotary = attention.rotary_emb
        norms = (attention.q_norm, attention.k_norm)
        gptj_rotary = type(rotary) is RotaryEmbedding and not rotary.is_neox and not rotary.inverse
        reasons = [
            reason
            for failed, reason in (
                (
                    attention._fuse_qk_norm_rope_gate,
                    "MUSESPARK_OPEN_FUSED_PREATTN selects a prep with other rounding",
                ),
                (attention.fuse_qk_norm_rope, "the C++ fused qk-norm+RoPE op is selected"),
                (attention.is_qk_norm, "the Q/K norms carry weights"),
                (any(norm.use_gemma for norm in norms), "Gemma-style norm"),
                (
                    any(norm.use_cuda_tile or norm.is_nvfp4 for norm in norms),
                    "the Q/K norms run a cuda-tile or NVFP4 kernel",
                ),
                (
                    attention.q_norm.variance_epsilon != attention.k_norm.variance_epsilon,
                    "the Q and K norms use different epsilons",
                ),
                (attention.rope_fusion, "RoPE runs inside the attention op"),
                (not attention.support_fused_qkv, "the attention backend takes split Q/K/V"),
                (attention.head_dim != _PREP_HEAD_DIM, f"head_dim is not {_PREP_HEAD_DIM}"),
                (
                    not attention.skip_rope and not gptj_rotary,
                    "RoPE is not the plain interleaved (GPT-J) cos/sin-cache rotation",
                ),
                (
                    not attention.skip_rope
                    and gptj_rotary
                    and not is_rope_table(rotary.rotary_cos_sin),
                    "the RoPE table is not a contiguous fp32 [positions, 2, "
                    f"{_PREP_HEAD_DIM // 2}] tensor",
                ),
            )
            if failed
        ]
        if reasons:
            raise ValueError(
                f"layer {attention.layer_idx}: preattn_prep cannot replace this prep: "
                + "; ".join(reasons)
            )

    def forward(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        *args,
        preattn_prep: Optional[PreattnPrep] = None,
        context_prep: Optional[PreattnPrep] = None,
        **kwargs,
    ) -> torch.Tensor:
        self._preattn_prep = preattn_prep
        self._context_prep = context_prep
        try:
            return super().forward(position_ids, hidden_states, attn_metadata, *args, **kwargs)
        finally:
            self._preattn_prep = None
            self._context_prep = None

    def preprocess_qkv(self, qkv: torch.Tensor, position_ids: Optional[torch.Tensor]):
        op = self._preattn_prep
        if op is not None:
            fits = self._fits_preattn_prep(qkv, position_ids)
        else:
            op = self._context_prep
            fits = op is not None and self._fits_context_prep(qkv, position_ids)
        if not fits:
            return super().preprocess_qkv(qkv, position_ids)
        rope = not self.skip_rope
        qkv_out, gate = op(
            qkv,
            self.rotary_emb.rotary_cos_sin if rope else None,
            position_ids if rope else None,
            self.num_heads,
            self.num_key_value_heads,
            self.q_norm.variance_epsilon,
            self.scale_query_by,
            self._qkv_layout_inverse is not None,
            LAUNCH_PDL,
        )
        return qkv_out, None, None, gate

    def _fits_preattn_prep(self, qkv: torch.Tensor, position_ids: Optional[torch.Tensor]) -> bool:
        rows = qkv.shape[0]
        if rows < 1 or rows > _PREATTN_PREP_MAX_ROWS or torch.compiler.is_compiling():
            return False
        fits = (
            qkv.dim() == 2
            and qkv.is_cuda
            and qkv.dtype == torch.bfloat16
            and qkv.stride(-1) == 1
            and qkv.shape[1] == 2 * (self.q_size + self.kv_size)
            and (self.skip_rope or (position_ids is not None and position_ids.numel() == rows))
        )
        if not fits:
            logger.warning_once(
                "[musespark_open] preattn_prep: a decode projection did not fit preattn_prep "
                f"(layer {self.layer_idx}, {qkv.dtype} {tuple(qkv.shape)}); "
                "running the built-in prep",
                key="musespark_open.preattn_prep.decline",
            )
        return fits

    def _attn_impl(self, q, k, v, attn_metadata, *args, **kwargs):
        result = super()._attn_impl(q, k, v, attn_metadata, *args, **kwargs)
        if self._zero_padded_rows:
            # Positionally, ``output`` is the sixth argument after the metadata.
            output = kwargs.get("output", args[5] if len(args) > 5 else None)
            zero_padded_rows(output, attn_metadata)
        return result

    def _fits_context_prep(self, qkv: torch.Tensor, position_ids: Optional[torch.Tensor]) -> bool:
        """Whether a context step's projection fits the op: ``_fits_preattn_prep``'s
        conditions with ``_CONTEXT_PREP_MAX_ROWS`` for the row bound."""
        rows = qkv.shape[0]
        if rows < 1 or rows > _CONTEXT_PREP_MAX_ROWS or torch.compiler.is_compiling():
            return False
        fits = (
            qkv.dim() == 2
            and qkv.is_cuda
            and qkv.dtype == torch.bfloat16
            and qkv.stride(-1) == 1
            and qkv.shape[1] == 2 * (self.q_size + self.kv_size)
            and (self.skip_rope or (position_ids is not None and position_ids.numel() == rows))
        )
        if not fits:
            logger.warning_once(
                "[musespark_open] context_prep: a context projection did not fit preattn_prep "
                f"(layer {self.layer_idx}, {qkv.dtype} {tuple(qkv.shape)}); "
                "running the built-in prep",
                key="musespark_open.context_prep.decline",
            )
        return fits


@dataclasses.dataclass(frozen=True)
class DecodeAttentionPlan:
    """What a decode step's attention layers run under ``swa_paged_decode``.

    Attributes:
        sliding_window: the sliding-window layers run ``paged_decode_attention``.
        q_fp8: that op quantizes Q to E4M3 as the built-in path does.
    """

    sliding_window: bool
    q_fp8: bool


class MuseSparkOpenTepDecodeAttention(MuseSparkOpenAttention):
    """The built-in attention, whose decode-step attention a ``DecodeAttentionPlan`` can change.

    Never constructed directly: ``adopt`` re-classes a built-in
    ``MuseSparkOpenAttention`` in place, adding methods and one plain attribute.
    Only ``DecodeTarget`` passes a plan, so every prefill step runs the built-in
    attention. With a plan:

    * a sliding-window layer runs ``paged_decode_attention`` -- the step's FP8
      K/V cache write (the same bytes at the same slots as the built-in
      ``applyBiasRopeUpdateKVCacheV2``) and the attention, in two launches --
      when the step fits it: one sequence of at most 8 rows, fused BF16 QKV, an
      FP8 KV cache in the TRT-LLM paged layout, BF16 output, no LoRA, sinks,
      mRoPE, helix or spec-dec tree. Any other step runs the built-in attention
      and logs why once.

    Both kernels launch with programmatic dependent launch under the switch the
    built-in flashinfer kernels follow (``TRTLLM_ENABLE_PDL``).
    """

    _decode_attention: Optional[DecodeAttentionPlan]

    @classmethod
    def adopt(cls, attention: MuseSparkOpenAttention) -> "MuseSparkOpenTepDecodeAttention":
        """Re-class one built-in attention module, after ``check``.

        Raises:
            TypeError, ValueError: as ``check``.
        """
        cls.check(attention)
        attention.__class__ = cls
        attention._decode_attention = None
        return attention

    @classmethod
    def check(cls, attention: MuseSparkOpenAttention) -> None:
        """Whether ``adopt`` may re-class this module; changes nothing.

        Raises:
            TypeError: if ``attention`` is not exactly the built-in ``MuseSparkOpenAttention``.
            ValueError: if the module's attention is not the one the plan is written
                for: the TRTLLM backend over fused QKV, head width 64, 8 query heads
                per KV head, and a sliding-window layer's window set.
        """
        if type(attention) is not MuseSparkOpenAttention:
            raise TypeError(
                f"{cls.__name__} adopts the built-in MuseSparkOpenAttention, "
                f"got {type(attention).__name__}"
            )
        reasons = [
            reason
            for failed, reason in (
                (attention.attn_backend != "TRTLLM", "the attention backend is not TRTLLM"),
                (not attention.support_fused_qkv, "the attention backend takes split Q/K/V"),
                (attention.head_dim != _DECODE_HEAD_DIM, f"head_dim is not {_DECODE_HEAD_DIM}"),
                (
                    attention.num_heads != _DECODE_Q_PER_KV * attention.num_key_value_heads,
                    f"{attention.num_heads} query heads over {attention.num_key_value_heads} "
                    f"KV heads, not {_DECODE_Q_PER_KV} per KV head",
                ),
                (
                    attention.is_sliding and not attention.attention_window_size,
                    "a sliding-window layer without a window",
                ),
            )
            if failed
        ]
        if reasons:
            raise ValueError(
                f"layer {attention.layer_idx}: the swa_paged_decode decode attention cannot run here: "
                + "; ".join(reasons)
            )

    def forward(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        *args,
        decode_attention: Optional[DecodeAttentionPlan] = None,
        **kwargs,
    ) -> torch.Tensor:
        self._decode_attention = decode_attention
        try:
            return super().forward(position_ids, hidden_states, attn_metadata, *args, **kwargs)
        finally:
            self._decode_attention = None

    def _attn_impl(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        attention_mask,
        mrope_rotary_cos_sin: Optional[torch.Tensor],
        mrope_position_deltas: Optional[torch.Tensor],
        attention_window_size: Optional[int],
        attention_mask_data: Optional[torch.Tensor],
        output: Optional[torch.Tensor] = None,
        output_sf: Optional[torch.Tensor] = None,
        attention_sinks: Optional[torch.Tensor] = None,
        relative_attention_bias: Optional[torch.Tensor] = None,
        relative_attention_max_distance: int = 0,
        has_lora: bool = False,
    ):
        plan = self._decode_attention
        builtin = (
            q,
            k,
            v,
            attn_metadata,
            attention_mask,
            mrope_rotary_cos_sin,
            mrope_position_deltas,
            attention_window_size,
            attention_mask_data,
        )
        builtin_kwargs = dict(
            output=output,
            output_sf=output_sf,
            attention_sinks=attention_sinks,
            relative_attention_bias=relative_attention_bias,
            relative_attention_max_distance=relative_attention_max_distance,
            has_lora=has_lora,
        )
        if plan is None:
            return super()._attn_impl(*builtin, **builtin_kwargs)
        if not self.is_sliding:
            return super()._attn_impl(*builtin, **builtin_kwargs)
        if not plan.sliding_window:
            return super()._attn_impl(*builtin, **builtin_kwargs)
        decline = self._paged_decode_decline(
            q,
            k,
            v,
            attn_metadata,
            attention_mask,
            mrope_rotary_cos_sin,
            mrope_position_deltas,
            attention_window_size,
            output,
            output_sf,
            attention_sinks,
            relative_attention_bias,
            has_lora,
        )
        if decline is not None:
            logger.warning_once(
                f"[musespark_open] swa_paged_decode: a decode step did not fit paged_decode_attention "
                f"({decline}); "
                "running the built-in attention",
                key=f"musespark_open.paged_decode.decline_{decline}",
            )
            return super()._attn_impl(*builtin, **builtin_kwargs)
        logger.info_once(
            f"[musespark_open] swa_paged_decode: sliding-window decode attention runs "
            "paged_decode_attention "
            f"(q_fp8={plan.q_fp8})",
            key="musespark_open.paged_decode.installed",
        )
        return self._paged_decode(q, attn_metadata, attention_window_size, plan.q_fp8), None

    def _paged_decode_decline(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        md: AttentionMetadata,
        attention_mask,
        mrope_rotary_cos_sin: Optional[torch.Tensor],
        mrope_position_deltas: Optional[torch.Tensor],
        attention_window_size: Optional[int],
        output: Optional[torch.Tensor],
        output_sf: Optional[torch.Tensor],
        attention_sinks: Optional[torch.Tensor],
        relative_attention_bias: Optional[torch.Tensor],
        has_lora: bool,
    ) -> Optional[str]:
        """Why this call cannot run ``paged_decode_attention``, or ``None``.

        Host-side reads only: under a CUDA graph each is a per-capture constant.
        """
        attn = self.attn
        rows = md.num_tokens
        checks = (
            (torch.compiler.is_compiling(), "torch.compile"),
            (output is not None or output_sf is not None, "a preallocated output"),
            (k is not None or v is not None, "split Q/K/V"),
            (attention_mask != PredefinedAttentionMask.CAUSAL, "a non-causal mask"),
            (
                mrope_rotary_cos_sin is not None or mrope_position_deltas is not None,
                "mRoPE",
            ),
            (attention_sinks is not None, "attention sinks"),
            (relative_attention_bias is not None, "relative attention bias"),
            (has_lora, "LoRA"),
            (self.mapping.has_cp_helix(), "helix CP"),
            (self._use_quantize_output(), "a quantized attention output"),
            (not attention_window_size or attention_window_size < 0, "no window"),
            (getattr(attn, "is_mla_enable", False), "MLA"),
            (getattr(attn, "sparse_params", None) is not None, "sparse attention"),
            (getattr(attn, "q_scaling", 1.0) != 1.0, "a softmax scale other than 1/8"),
            (getattr(attn, "attention_chunk_size", None), "chunked attention"),
            (not getattr(attn, "has_fp8_kv_cache", False), "the KV cache is not FP8"),
            (getattr(md, "kv_cache_manager", None) is None, "no KV cache manager"),
            (getattr(md, "kv_cache_block_offsets", None) is None, "no paged KV cache"),
            (getattr(md, "is_cross", False), "cross attention"),
            (getattr(md, "beam_width", 1) != 1, "beam search"),
            (md.num_seqs != 1 or md.num_generations != 1, "not one generation sequence"),
            (not 1 <= rows <= _DECODE_MAX_ROWS, f"{rows} rows"),
            (
                getattr(md, "is_spec_decoding_enabled", False)
                or getattr(md, "is_spec_dec_tree", False)
                or getattr(md, "spec_decoding_bl_tree_mask", None) is not None,
                "spec-dec tree masks",
            ),
            (
                q.dim() != 2
                or q.dtype != torch.bfloat16
                or q.stride(-1) != 1
                or q.shape[0] < rows
                or q.shape[1] != (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
                "the fused QKV operand",
            ),
            (self._total_num_blocks_fmha() is None, "no trtllm-gen FMHA library"),
        )
        for failed, reason in checks:
            if failed:
                return reason
        tokens_per_block = md.tokens_per_block
        if not tokens_per_block or tokens_per_block & (tokens_per_block - 1):
            return f"{tokens_per_block}-token pages"
        return None

    def _total_num_blocks_fmha(self):
        """The FMHA library whose pool bound the cache view takes (the trtllm-gen one), or None."""
        return trtllm_gen_fmha(self.attn)

    def _paged_decode(
        self,
        q: torch.Tensor,
        md: AttentionMetadata,
        window: int,
        q_fp8: bool,
    ) -> torch.Tensor:
        attn = self.attn
        rows = md.num_tokens
        local_layer_idx = attn.get_local_layer_idx(md)
        kv_pool, block_tables, _ = thop.build_trtllm_gen_kv_cache_metadata(
            md.host_kv_cache_pool_pointers,
            md.host_kv_cache_pool_mapping,
            md.kv_cache_block_offsets,
            local_layer_idx,
            self.num_key_value_heads,
            md.tokens_per_block,
            self.head_dim,
            2,
            self._total_num_blocks_fmha()._get_total_num_blocks(md),
            attn.quant_mode,
            0,
            1,
            torch.bfloat16,
        )
        return paged_decode_attention(
            q[:rows],
            kv_pool,
            block_tables[0],
            md.kv_lens_cuda_runtime[:1],
            attn.kv_scale_orig_quant,
            self.num_heads,
            self.num_key_value_heads,
            window,
            window_splits(window),
            q_fp8,
            LAUNCH_PDL,
        )


class MuseSparkOpenTepPrepDecodeAttention(
    MuseSparkOpenTepAttention, MuseSparkOpenTepDecodeAttention
):
    """``preattn_prep`` and ``swa_paged_decode`` on one attention: the prep feeds the decode attention.

    Never constructed directly: ``adopt`` re-classes a built-in
    ``MuseSparkOpenAttention`` in place. ``MuseSparkOpenTepAttention``
    overrides ``preprocess_qkv`` (and takes ``preattn_prep``),
    ``MuseSparkOpenTepDecodeAttention`` overrides ``_attn_impl`` (and takes
    ``decode_attention``); each ``forward`` takes its own keyword and hands the
    rest on. ``MuseSparkOpenTepAttention._attn_impl`` runs the decode class's
    ``_attn_impl`` first and only then zeroes a padded context step's rows, which
    no decode step has. A decode step
    therefore runs ``preattn_prep`` and hands its fused QKV -- the bits the
    built-in prep would have produced -- to ``paged_decode_attention`` on a
    sliding-window layer, or to the built-in attention on a full-attention
    layer. Every other call is what each feature
    does alone.
    """

    @classmethod
    def check(cls, attention: MuseSparkOpenAttention) -> None:
        """Both features' checks; changes nothing.

        Raises:
            TypeError, ValueError: as ``MuseSparkOpenTepAttention.check`` or
                ``MuseSparkOpenTepDecodeAttention.check``.
        """
        MuseSparkOpenTepAttention.check(attention)
        MuseSparkOpenTepDecodeAttention.check(attention)

    @classmethod
    def adopt(cls, attention: MuseSparkOpenAttention) -> "MuseSparkOpenTepPrepDecodeAttention":
        """Re-class one built-in attention module, after ``check``.

        Raises:
            TypeError, ValueError: as ``check``.
        """
        cls.check(attention)
        attention.__class__ = cls
        attention._preattn_prep = None
        attention._decode_attention = None
        return attention


def _attention_class(
    preattn_prep_on: bool, plan: Optional[DecodeAttentionPlan], norm_gate: bool = False
):
    """The class every layer's attention is re-classed as under ``preattn_prep``, a
    ``swa_paged_decode`` plan and the ``boundary_early_release`` PDL norm-gate
    (``pdl_triggers``), or None."""
    if preattn_prep_on and plan is not None:
        base = MuseSparkOpenTepPrepDecodeAttention
    elif preattn_prep_on:
        base = MuseSparkOpenTepAttention
    elif plan is not None:
        base = MuseSparkOpenTepDecodeAttention
    else:
        base = None
    if not norm_gate:
        return base
    from .kernels.boundary_early_release import triggers as pdl_triggers

    return pdl_triggers.NORM_GATE_CLASSES[base]


#: The fast paths whose decode steps need flashinfer's trtllm-gen FMHA library in each target
#: layer's FMHA manager: the sliding-window ones take its KV-pool page bound, and the
#: full-attention ones also intercept its ``run_generation``, so it must be the library the
#: manager selects for those layers' decode steps.
TRTLLM_GEN_DECODE_PATHS = (
    "swa_paged_decode",
    "swa_block",
    "full_attention_prep_append",
    "full_attention_split_kv",
    "full_attention_prep_fold",
    "multi_request_split_kv",
)


def trtllm_gen_fmha(attn) -> Optional[_fi_gen.FlashInferTrtllmGenFmha]:
    """The flashinfer trtllm-gen FMHA library in ``attn``'s FMHA manager, or None."""
    for lib in getattr(getattr(attn, "_fmha_manager", None), "fmha_libs", ()):
        if isinstance(lib, _fi_gen.FlashInferTrtllmGenFmha):
            return lib
    return None


class DecodeTrtllmGenFmha(_fi_gen.FlashInferTrtllmGenFmha):
    """flashinfer's trtllm-gen FMHA, offered for the generation phase only.

    ``select_decode_fmha`` puts one in front of a target layer's FMHA libraries on
    an architecture whose public library list leaves flashinfer's trtllm-gen FMHA
    out. The manager then selects it for a generation-only step that the
    library's own checks accept (``FlashInferTrtllmGenFmha._is_supported`` for the
    generation phase). A step with a context phase -- a prefill, or a batch that
    mixes both phases -- finds no context support here and is served by the public
    libraries, as without this class.
    """

    def _is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: AttentionMetadata,
        forward_args,
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        if phase != FmhaPhase.GENERATION:
            return False
        return super()._is_supported(q, k, v, metadata, forward_args, phase=phase)


def _decode_fmha_refusal(attn) -> Optional[str]:
    """Why flashinfer's trtllm-gen FMHA cannot serve ``attn``'s decode steps, or None.

    The static checks of ``FlashInferTrtllmGenFmha.is_available`` other than its
    SM list; the request checks run per step (``is_supported``).
    """
    if not _flashinfer_utils.IS_FLASHINFER_AVAILABLE:
        return "flashinfer is not installed"
    missing = _fi_gen.FlashInferTrtllmGenFmha._missing_fused_nanobind_ops()
    if missing:
        return f"the binding lacks {', '.join(missing)}"
    num_heads, num_kv_heads = attn.num_heads, attn.num_kv_heads
    checks = (
        (getattr(attn, "skip_correction_threshold", 0.0) > 0.0, "skip-correction"),
        (getattr(attn, "is_mla_enable", False), "MLA"),
        (getattr(attn, "sparse_params", None) is not None, "sparse attention"),
        (
            num_heads <= 0 or num_kv_heads <= 0 or num_heads % num_kv_heads,
            f"{num_heads} query heads over {num_kv_heads} KV heads",
        ),
    )
    for failed, reason in checks:
        if failed:
            return reason
    return None


def select_decode_fmha(layers, compute_capability) -> int:
    """Make flashinfer's trtllm-gen FMHA the library each layer's attention selects for decode.

    Per architecture, read once from ``compute_capability``:

    * sm_103: the public library list already holds it
      (``FlashInferTrtllmGenFmha.is_available``) and the manager selects it;
      nothing changes.
    * sm_107: the public list leaves it out, so every layer's manager gets a
      ``DecodeTrtllmGenFmha`` in front of its libraries. Decode steps then run
      flashinfer's trtllm-gen FMHA where it supports them; prefill and mixed steps
      keep the public selection.
    * no device (None): nothing changes.

    Returns:
        The number of layers whose manager changed.

    Raises:
        ValueError: on sm_107, for a layer whose attention the library cannot
            serve (nothing changes); for any other architecture.
    """
    sm = sm_of(compute_capability)
    if sm is None or sm == 103:
        return 0
    if sm != 107:
        raise ValueError(f"[musespark_open] no decode FMHA selection is written for sm{sm}")
    refusals = []
    for layer in layers:
        reason = _decode_fmha_refusal(layer.self_attn.attn)
        if reason is not None:
            refusals.append(f"layer {layer.self_attn.layer_idx}: {reason}")
    if refusals:
        raise ValueError(
            "[musespark_open] flashinfer's trtllm-gen FMHA cannot serve the decode steps: "
            + "; ".join(refusals)
        )
    changed = 0
    for layer in layers:
        attn = layer.self_attn.attn
        if trtllm_gen_fmha(attn) is None:
            attn._fmha_manager.fmha_libs.insert(0, DecodeTrtllmGenFmha(attn))
            changed += 1
    return changed
