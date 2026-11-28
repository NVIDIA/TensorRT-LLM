# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

# Self-attention over fewer rows runs unquantized.
DECO_MIN_SEQ_LEN = 8192
DECO_HEAD_DIM = 128


def _kernels():
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell import deco_attention

    return deco_attention


def deco_unavailable_reason(qk_dtype: str | None, v_dtype: str | None) -> str | None:
    """Why the current GPU has no DeCo-Attention kernel for the recipe; None when it has one."""
    try:
        formats = tuple(_kernels().pv_formats())
    except (ImportError, OSError) as error:
        return f"The DeCo-Attention kernels do not import: {error}"
    except NotImplementedError:
        return "DeCo-Attention has kernels for SM100, SM103, SM107 and SM120 GPUs only."
    if qk_dtype != "nvfp4" or v_dtype not in formats:
        return (
            f"DeCo-Attention has no kernel for qk_dtype={qk_dtype!r}, v_dtype={v_dtype!r} on this "
            f"GPU, whose kernels take qk_dtype='nvfp4' with v_dtype in {formats}."
        )
    return None


def deco_attention_ineligible_reason(q, k, v) -> str | None:
    """Why the kernels do not take a call of q, k and v ``[B, S, H, D]``; None when they do."""
    if not q.shape == k.shape == v.shape:
        return "q, k and v of different shapes"
    if q.ndim != 4 or q.shape[-1] != DECO_HEAD_DIM:
        return f"head_dim {q.shape[-1]}, while the kernels take {DECO_HEAD_DIM}"
    if not q.is_cuda:
        return "inputs that are not on a CUDA device"
    if q.numel() == 0:
        return "empty inputs"
    if q.shape[1] < DECO_MIN_SEQ_LEN:
        return f"a sequence of {q.shape[1]} rows, under the {DECO_MIN_SEQ_LEN} the kernels take"
    if not q.dtype == k.dtype == v.dtype == torch.bfloat16:
        dtypes = sorted({str(x.dtype) for x in (q, k, v)} - {"torch.bfloat16"})
        return f"{' and '.join(dtypes)} inputs"
    return None


class DeCoAttention(OpWrapper):
    """Self-attention on NVFP4 Q/K that corrects the first-order score quantization error, with V
    smoothed over key blocks and quantized to ``v_dtype``.

    ``key_padding_mask`` ``[B, S]`` is read as each sample's count of leading tokens; padded rows
    get no attention weight and return zero. ``token_tags`` ``[S]`` (0 video, positive text or
    audio, negative padding) draws half of the compensation-basis tokens from text and audio.
    """

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("FP8 V over an unpadded sequence", dict(v_dtype="fp8", tokens=8192, padding=0)),
        Cell(
            "FP8 V with trailing padding and modality tags",
            dict(v_dtype="fp8", tokens=8448, padding=256),
        ),
    )

    @torch.compiler.disable
    def __call__(self, q, k, v, *, v_dtype: str, key_padding_mask=None, token_tags=None):
        kernels = _kernels()
        # The kernels read contiguous tensors and slices of a fused projection in place.
        q, k, v = (
            x
            if (x.is_contiguous() or kernels.token_rows(x))
            and x.data_ptr() % kernels.TENSOR_ALIGNMENT == 0
            else x.clone(memory_format=torch.contiguous_format)
            for x in (q, k, v)
        )
        batch_size, seq_len, num_heads, _ = q.shape
        token_counts, rows = None, None
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(q.device)
            token_counts = key_padding_mask.sum(dim=1, dtype=torch.int32)
        if token_tags is not None:
            rows = kernels.sample_rows(seq_len, q.device, token_tags, tokens=key_padding_mask)
        num_bytes = kernels.workspace_size(batch_size, seq_len, num_heads, q.device, v_dtype)
        with torch.cuda.device(q.device):
            operands = kernels.prepare(
                q,
                k,
                v,
                torch.empty(num_bytes, dtype=torch.uint8, device=q.device),
                v_dtype,
                rows=rows,
                valid_seq_lens=token_counts,
            )
            return kernels.attention(operands, torch.empty_like(q))

    def reference(self, q, k, v, *, v_dtype: str, key_padding_mask=None, token_tags=None):
        tokens = None
        if key_padding_mask is not None:
            counts = key_padding_mask.sum(dim=1)
            tokens = torch.arange(q.shape[1], device=q.device)[None] < counts[:, None]
            # Padded rows have no influence, whatever they hold.
            k = k.masked_fill(~tokens[:, :, None, None], 0)
            v = v.masked_fill(~tokens[:, :, None, None], 0)
        out = F.scaled_dot_product_attention(
            q.float().transpose(1, 2),
            k.float().transpose(1, 2),
            v.float().transpose(1, 2),
            attn_mask=None if tokens is None else tokens[:, None, None, :],
        ).transpose(1, 2)
        if tokens is not None:
            out = out.masked_fill(~tokens[:, :, None, None], 0)
        return out.to(q.dtype)

    def is_valid(self, q, k, v, *, v_dtype: str, key_padding_mask=None, token_tags=None):
        assert deco_attention_ineligible_reason(q, k, v) is None
        assert deco_unavailable_reason("nvfp4", v_dtype) is None
        if key_padding_mask is not None:
            assert key_padding_mask.dtype == torch.bool
            assert key_padding_mask.shape == q.shape[:2]
            counts = key_padding_mask.sum(dim=1)
            positions = torch.arange(q.shape[1], device=key_padding_mask.device)
            # Interior padding would be attended as if the tokens came first.
            assert torch.equal(key_padding_mask, positions[None] < counts[:, None])
        if token_tags is not None:
            assert token_tags.shape == (q.shape[1],)


deco_attention = DeCoAttention()
