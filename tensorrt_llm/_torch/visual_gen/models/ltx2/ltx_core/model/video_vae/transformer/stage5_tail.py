# ruff: noqa: ANN001, N803, N806, PLR0913

"""DiffVAE stage-5 tail in one pass: ``norm_out`` -> ``conv_out`` -> unpatchify.
The eager tail makes four passes over the stage-5 volume (the normalised copy, the
projection, a channels-first copy, the depth-to-space copy) and holds two extra
full-size tensors. Here each block of tokens is normalised in fp32, rounded to the
activation dtype where ``nn.RMSNorm`` rounds, projected on the tensor cores with the
bias in the epilogue, and every output channel is stored straight to its pixel.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]

if triton is not None:
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.transformer.row_rms_norm import rms_rstd

    @triton.jit
    def _stage5_tail_kernel(
        x_ptr,
        norm_w_ptr,
        proj_w_ptr,
        proj_b_ptr,
        out_ptr,
        n_tokens,
        F,
        Hh,
        Ww,
        eps,
        C: tl.constexpr,
        OUT: tl.constexpr,
        OUT_PAD: tl.constexpr,
        PATCH: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ) -> None:
        """``out[b, c, f, h*P+q, w*P+r] = conv_out(norm_out(x))[b, f, h, w, c*P*P + r*P + q]``.
        int64 offsets: the stage-5 volume passes 2**31 elements at 1080p.
        """
        tokens = tl.program_id(0).to(tl.int64) * BLOCK_T + tl.arange(0, BLOCK_T).to(tl.int64)
        token_mask = tokens < n_tokens
        channels = tl.arange(0, C)
        x = tl.load(x_ptr + tokens[:, None] * C + channels[None, :], mask=token_mask[:, None], other=0.0)
        x = x.to(tl.float32)
        rstd = rms_rstd(x, C, eps)
        normed = (x * rstd[:, None] * tl.load(norm_w_ptr + channels).to(tl.float32)[None, :]).to(x_ptr.dtype.element_ty)

        outs = tl.arange(0, OUT_PAD)
        out_mask = outs < OUT
        w = tl.load(proj_w_ptr + outs[None, :] * C + channels[:, None], mask=out_mask[None, :], other=0.0)
        acc = tl.dot(normed, w)
        acc += tl.load(proj_b_ptr + outs, mask=out_mask, other=0.0).to(tl.float32)[None, :]

        # (T, c*P*P + r*P + q) -> (c*P + q, t*P + r): each row is a run of consecutive pixels
        # along W, so the store coalesces.
        C_PAD: tl.constexpr = OUT_PAD // (PATCH * PATCH)
        vals = tl.reshape(acc.to(out_ptr.dtype.element_ty), (BLOCK_T, C_PAD, PATCH, PATCH))
        vals = tl.reshape(tl.permute(vals, (1, 3, 0, 2)), (C_PAD * PATCH, BLOCK_T * PATCH))
        rows = tl.arange(0, C_PAD * PATCH)
        cols = tl.arange(0, BLOCK_T * PATCH)
        c = rows // PATCH
        q = rows % PATCH
        t = tl.program_id(0).to(tl.int64) * BLOCK_T + (cols // PATCH).to(tl.int64)
        r = cols % PATCH
        ww = t % Ww
        hh = (t // Ww) % Hh
        f = (t // Ww // Hh) % F
        b = t // Ww // Hh // F
        H = Hh * PATCH
        W = Ww * PATCH
        offsets = (
            (
                ((b[None, :] * (OUT // (PATCH * PATCH)) + c[:, None]) * F + f[None, :]) * H
                + hh[None, :] * PATCH
                + q[:, None]
            )
            * W
            + ww[None, :] * PATCH
            + r[None, :]
        )
        tl.store(out_ptr + offsets, vals, mask=(c < OUT // (PATCH * PATCH))[:, None] & (t < n_tokens)[None, :])


def stage5_tail_supported(x: torch.Tensor, norm: torch.nn.RMSNorm, proj: torch.nn.Linear, patch_size: int) -> bool:
    """Whether :func:`stage5_tail` takes this ``(B, T, H, W, C)`` channels-last volume.
    Not while ``torch.compile`` traces: the compiled DiffVAE modes leave the tail to Inductor.
    """
    c = x.shape[-1]
    return (
        triton is not None
        and not torch.compiler.is_compiling()
        and x.is_cuda
        and x.dim() == 5
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.is_contiguous()
        and c >= 16
        and c & (c - 1) == 0
        and norm.weight is not None
        and proj.bias is not None
        and proj.weight.dtype == x.dtype
        and proj.out_features % (patch_size * patch_size) == 0
    )


def _block_tokens(device: torch.device) -> int:
    """Tokens per program of the stage-5 tail kernel for ``device``'s architecture."""
    major, _ = torch.cuda.get_device_capability(device)
    if major == 10:
        # Datacenter Blackwell. Swept at a 1080p stage-5 volume on sm_100.
        return 128
    return 32


def stage5_tail(x: torch.Tensor, norm: torch.nn.RMSNorm, proj: torch.nn.Linear, patch_size: int) -> torch.Tensor:
    """``unpatchify(proj(norm(x)).permute(0, 4, 1, 2, 3))`` as one kernel; ``(B, C_out, T, H*P, W*P)``.
    Check :func:`stage5_tail_supported` first.
    """
    batch, frames, hh, ww, c = x.shape
    out_features = proj.out_features
    out = torch.empty(
        (batch, out_features // (patch_size * patch_size), frames, hh * patch_size, ww * patch_size),
        dtype=x.dtype,
        device=x.device,
    )
    n_tokens = batch * frames * hh * ww
    eps = norm.eps if norm.eps is not None else torch.finfo(x.dtype).eps
    block_tokens = _block_tokens(x.device)
    with torch.cuda.device(x.device):
        _stage5_tail_kernel[(triton.cdiv(n_tokens, block_tokens),)](
            x,
            norm.weight,
            proj.weight,
            proj.bias,
            out,
            n_tokens,
            frames,
            hh,
            ww,
            eps,
            C=c,
            OUT=out_features,
            OUT_PAD=max(16, triton.next_power_of_2(out_features)),
            PATCH=patch_size,
            BLOCK_T=block_tokens,
        )
    return out
