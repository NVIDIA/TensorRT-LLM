# Copyright 2026 The VDN Authors.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VDN-H3 frame-wise hybrid attention.

Algorithm and checkpoint names follow OpenVDN/vdn-minimax-h3. Dense projections,
QK normalization, rotary embeddings and attention kernels reuse MiniMax-H3.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.visual_gen.attention_backend.vanilla import VanillaAttention

from .transformer_minimax_h3 import MiniMaxH3Attention, apply_minimax_h3_rotary_emb
from .vdn_kernels import temporal_conv
from .vdn_window_attention import window_attention


def _qk_norm_rope(
    tensor: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    normalized = F.rms_norm(tensor, (tensor.shape[-1],), weight, eps)
    return apply_minimax_h3_rotary_emb(normalized, cos, sin)


# This small pointwise kernel follows the released inference arithmetic: the
# normalization and partial RoPE share FP32 intermediates and one BF16 store.
_compiled_qk_norm_rope = torch.compile(_qk_norm_rope, dynamic=False)


@dataclass(frozen=True)
class VDNLayout:
    """Target video is a contiguous suffix; conditioning rows remain global."""

    video_start: int
    num_frames: int
    height: int
    width: int
    text_indices: torch.Tensor

    @property
    def tokens_per_frame(self) -> int:
        return self.height * self.width

    @property
    def video_end(self) -> int:
        return self.video_start + self.num_frames * self.tokens_per_frame

    @classmethod
    def from_indices(
        cls, positions: torch.Tensor, video_indices: torch.Tensor, text_indices: torch.Tensor
    ) -> "VDNLayout":
        if video_indices.numel() == 0:
            raise ValueError("VDN requires target video rows.")
        breaks = (video_indices[1:] != video_indices[:-1] + 1).nonzero().flatten()
        target = video_indices[int(breaks[-1]) + 1 :] if breaks.numel() else video_indices
        grid = positions[target]
        height = torch.unique(grid[:, 1]).numel()
        width = torch.unique(grid[:, 2]).numel()
        per_frame = height * width
        if target.numel() % per_frame or int(target[-1]) != positions.shape[0] - 1:
            raise ValueError(
                "VDN requires a rectangular target video at the end of the packed sequence."
            )
        frames = target.numel() // per_frame
        volume = grid.reshape(frames, per_frame, 3)
        if not torch.equal(volume[:, :, 1:], volume[:1, :, 1:].expand_as(volume[:, :, 1:])):
            raise ValueError("VDN requires the same ordered spatial grid in every target frame.")
        if not bool((volume[:, :, 0] == volume[:, :1, 0]).all()):
            raise ValueError("VDN target rows must be grouped by frame.")
        return cls(int(target[0]), frames, height, width, text_indices)


class VDNOutputGate(nn.Module):
    def __init__(
        self, hidden: int, heads: int, dim: int | None = None, dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__()
        self.heads, self.dim = heads, dim or 1
        self.down = nn.Linear(hidden, dim, bias=False, dtype=dtype) if dim else None
        self.up = nn.Linear(dim or hidden, heads * self.dim, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.up(x if self.down is None else self.down(x))
            .sigmoid()
            .view(-1, self.heads, self.dim)
        )


class VDNAlpha(nn.Module):
    def __init__(
        self, hidden: int, heads: int, dim: int, dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__()
        self.heads, self.dim = heads, dim
        self.down = nn.Linear(hidden, dim, bias=False, dtype=dtype)
        self.up = nn.Linear(dim, heads * dim, bias=False, dtype=dtype)
        self.A_log = nn.Parameter(torch.zeros(heads, dtype=dtype))
        self.dt_bias = nn.Parameter(torch.zeros(heads * dim, dtype=dtype))

    def forward(self, frame_mean: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=frame_mean.device.type, enabled=False):
            delta = F.linear(
                F.linear(frame_mean.float(), self.down.weight.float()), self.up.weight.float()
            )
            delta = (delta + self.dt_bias.float()).view(-1, self.heads, self.dim)
            return torch.exp(-self.A_log.float().exp()[:, None] * F.softplus(delta))


class VDNShortConv(nn.Module):
    def __init__(
        self, channels: int, targets: tuple[str, ...], dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__()
        self.targets = targets
        for name in targets:
            self.add_module(
                f"{name}_sp",
                nn.Conv2d(
                    channels, channels, 5, padding=2, groups=channels, bias=False, dtype=dtype
                ),
            )
            self.add_module(
                f"{name}_tm",
                nn.Conv1d(
                    channels, channels, 5, padding=2, groups=channels, bias=False, dtype=dtype
                ),
            )

    def forward(
        self,
        x: torch.Tensor,
        name: str,
        frames: int,
        height: int,
        width: int,
        activate: bool = False,
        l2norm: bool = False,
    ) -> torch.Tensor:
        shape = x.shape
        channels = shape[-2] * shape[-1]
        volume = x.reshape(frames, height, width, channels).permute(0, 3, 1, 2)
        volume = self.get_submodule(f"{name}_sp")(volume)
        volume = volume.permute(0, 2, 3, 1).reshape(frames, height * width, channels)
        weights = self.get_submodule(f"{name}_tm").weight.squeeze(1).to(x.dtype)
        if x.is_cuda:
            return temporal_conv(
                volume.contiguous(),
                weights.contiguous(),
                shape[-1],
                activate=activate,
                l2norm=l2norm,
            ).reshape(shape)
        padded = F.pad(volume, (0, 0, 0, 0, 2, 2))
        out = padded[:frames] * weights[:, 0]
        for tap in range(1, 5):
            out = out + padded[tap : tap + frames] * weights[:, tap]
        return out.reshape(shape)


class VDNLinearBranch(nn.Module):
    """Exact joint-solve DeltaNet with two frame scans and an alpha bridge."""

    def __init__(
        self, hidden: int, heads: int, dim: int, config: dict, dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__()
        if (
            config["delta_rule"] != "vdn_solve"
            or config["bridge"] != "alpha"
            or not config["a_fp32"]
        ):
            raise NotImplementedError(
                "VDN supports the released vdn_solve/alpha/FP32-statistics checkpoints."
            )
        if config["linear_head_dim"] != dim:
            raise ValueError(
                "VDN shares H3 QKV and requires matching linear and softmax head dimensions."
            )
        targets = tuple(config["short_conv"]["targets"])
        if len(set(targets)) != len(targets) or set(targets) - {"q", "k", "v"}:
            raise ValueError("short_conv.targets must be a distinct subset of q, k, v.")
        self.heads, self.dim = heads, dim
        self.alpha = VDNAlpha(hidden, heads, dim, dtype=dtype)
        self.beta_proj = nn.Linear(hidden, heads, bias=False, dtype=dtype)
        self.output_gate = VDNOutputGate(hidden, heads, dim, dtype=dtype)
        self.norm = RMSNorm(hidden_size=dim, eps=1e-6, dtype=dtype)
        self.short_conv = VDNShortConv(heads * dim, targets, dtype=dtype) if targets else None

    def _features(
        self, qkv: tuple[torch.Tensor, ...], frames: int, height: int, width: int, convolve: bool
    ) -> tuple[torch.Tensor, ...]:
        result = []
        for name, tensor in zip(("q", "k", "v"), qkv):
            if convolve and self.short_conv is not None and name in self.short_conv.targets:
                if tensor.is_cuda:
                    result.append(
                        self.short_conv(
                            tensor, name, frames, height, width, activate=True, l2norm=name != "v"
                        )
                    )
                    continue
                tensor = self.short_conv(tensor, name, frames, height, width)
            tensor = F.silu(tensor)
            if name != "v":
                tensor = F.normalize(tensor, dim=-1, eps=1e-6).to(tensor.dtype)
            result.append(tensor)
        return tuple(result)

    @staticmethod
    @torch.compiler.disable
    def _statistics(
        key: torch.Tensor, value: torch.Tensor, beta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = key.contiguous()
        key_fp32 = key.float()
        weighted = key_fp32 * beta.float().unsqueeze(-1)
        # Match upstream's inference statistics without enabling TF32 for the
        # FP32 media projections or the state recurrence.
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            if key.is_cuda:
                torch.backends.cuda.matmul.allow_tf32 = True
            a = weighted.transpose(-1, -2) @ key_fp32
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        a = (a + a.transpose(-1, -2)) * 0.5
        weighted_value = (value * beta.to(value.dtype).unsqueeze(-1)).contiguous()
        b = weighted_value.transpose(-1, -2) @ key
        return a, b.float()

    @staticmethod
    def _factor(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        eye = torch.eye(a.shape[-1], device=a.device, dtype=torch.float32).expand_as(a)
        chol = torch.linalg.cholesky(a + eye)
        inverse_l = torch.linalg.solve_triangular(chol, eye, upper=False)
        inverse = inverse_l.transpose(-1, -2) @ inverse_l
        return inverse, b @ inverse

    def forward(
        self,
        x: torch.Tensor,
        qkv: tuple[torch.Tensor, ...],
        layout: VDNLayout,
        bounds: list[tuple[int, int]],
        text_x: torch.Tensor | None,
        text_qkv: tuple[torch.Tensor, ...] | None,
    ) -> torch.Tensor:
        frames, per_frame = layout.num_frames, layout.tokens_per_frame
        # Both anchor frames are dense in both directions and are excluded from
        # the convolution and recurrence, not merely masked at readout.
        if frames <= 2:
            return x.new_zeros(x.shape[0], self.heads * self.dim)
        inner = slice(per_frame, (frames - 1) * per_frame)
        xv = x[inner]
        frames -= 2
        q, k, v = self._features(
            tuple(t[inner] for t in qkv), frames, layout.height, layout.width, True
        )
        shape = (frames, per_frame, self.heads, self.dim)
        k, v = (t.view(shape).permute(0, 2, 1, 3) for t in (k, v))
        beta = self.beta_proj(xv).sigmoid().view(frames, per_frame, self.heads).permute(0, 2, 1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            a, b = self._statistics(k, v, beta)
            inverse, injection = self._factor(a, b)
            alpha = self.alpha(xv.view(frames, per_frame, -1).mean(1, dtype=torch.float32))
            transition = alpha.unsqueeze(-1) * inverse
            start = torch.zeros_like(injection[0])
            if text_x is not None and text_x.shape[0]:
                _, kt, vt = self._features(text_qkv, 1, 1, text_x.shape[0], False)
                bt = self.beta_proj(text_x).sigmoid().transpose(0, 1).unsqueeze(0)
                at, bt = self._statistics(
                    kt.transpose(0, 1).unsqueeze(0), vt.transpose(0, 1).unsqueeze(0), bt
                )
                start = self._factor(at, bt)[1][0] * 0.5
            prefix, suffix = [], []
            state = start
            for frame in range(frames):
                state = torch.baddbmm(injection[frame], state, transition[frame])
                prefix.append(state)
            state = start
            for frame in reversed(range(frames)):
                state = torch.baddbmm(injection[frame], state, transition[frame])
                suffix.append(state)
            suffix.reverse()
            log_prefix = F.pad(alpha.clamp_min(1e-12).log().cumsum(0), (0, 0, 0, 0, 1, 0))
            prefix = torch.cat((start.unsqueeze(0), torch.stack(prefix)))
            suffix = torch.cat((torch.stack(suffix), start.unsqueeze(0)))
            limits = torch.tensor(bounds[1:-1], device=x.device) - 1
            lo = limits[:, 0].clamp(0, frames)
            hi = (limits[:, 1] + 1).clamp(0, frames)
            before = prefix[lo] * (log_prefix[1:] - log_prefix[lo]).exp().unsqueeze(-2)
            after = suffix[hi] * (log_prefix[hi] - log_prefix[:-1]).exp().unsqueeze(-2)
            states = (before + after).to(x.dtype)
        readout = torch.einsum("fhvk,fshk->fshv", states, q.view(shape)).reshape(
            -1, self.heads, self.dim
        )
        # Match the checkpoint's BF16 rounding after rsqrt, before the weight.
        ms = (
            torch.linalg.vector_norm(readout, dim=-1, keepdim=True, dtype=torch.float32).square()
            / self.dim
        )
        readout = (
            readout
            * torch.rsqrt(ms + self.norm.variance_epsilon).to(readout.dtype)
            * self.norm.weight.to(readout.dtype)
        )
        readout = (readout * self.output_gate(xv)).flatten(1)
        return F.pad(readout, (0, 0, per_frame, per_frame))


class VDNHybridAttention(nn.Module):
    """Reuse H3 QKV, norm, RoPE, output projection and dense attention dispatch."""

    def __init__(
        self,
        orig: MiniMaxH3Attention,
        hidden: int,
        config: dict,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if config["anchor_frames"] != "both" or not config["enable_softmax_gate"]:
            raise NotImplementedError(
                "VDN currently supports released checkpoints with both anchor frames and a softmax gate."
            )
        self.orig = orig
        self.heads, self.dim = orig.local_num_attention_heads, orig.head_dim
        self.radius = config["softmax_attention"]["radius"]
        self.chunk = config["softmax_attention"]["chunk"]
        if self.radius < 0 or self.chunk < 0:
            raise ValueError("VDN window radius and chunk must be nonnegative.")
        self.text_state = config["linear_attention"]["enable_text_state"]
        self.linear_attention = VDNLinearBranch(
            hidden, self.heads, self.dim, config["linear_attention"], dtype=dtype
        )
        self.softmax_gate = VDNOutputGate(hidden, self.heads, dtype=dtype)
        self.to_out_linear = nn.Linear(self.heads * self.dim, hidden, bias=False, dtype=dtype)
        self.layout: VDNLayout | None = None
        self.global_attention = VanillaAttention(num_heads=self.heads, head_dim=self.dim)

    def _dense(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, timestep: torch.Tensor | None
    ) -> torch.Tensor:
        return self.orig._attn_impl(
            q.flatten(2), k.flatten(2), v.flatten(2), timestep=timestep
        ).view_as(q)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_padding_mask: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layout = self.layout
        if layout is None:
            raise ValueError("VDN requires a packed video layout before attention.")
        if hidden_states.shape[0] != 1 or key_padding_mask is not None:
            raise ValueError("VDN supports one unpadded packed document per request.")
        raw = tuple(t.view(1, -1, self.heads, self.dim) for t in self.orig.get_qkv(hidden_states))
        if rotary_emb is not None and hidden_states.is_cuda:
            q, k = (
                _compiled_qk_norm_rope(t, norm.weight, norm.variance_epsilon, *rotary_emb)
                for t, norm in zip(raw[:2], (self.orig.norm_q, self.orig.norm_k))
            )
        else:
            q, k = self.orig.apply_qk_norm(raw[0], raw[1])
            if rotary_emb is not None:
                q, k = (apply_minimax_h3_rotary_emb(t, *rotary_emb) for t in (q, k))
        # FA4 consumes packed-row operands; keep the shared V projection
        # contiguous before the dense calls and window gathers.
        v = raw[2].contiguous()
        frames, per_frame, start = layout.num_frames, layout.tokens_per_frame, layout.video_start
        bounds = [
            (f - self.radius, f + self.radius)
            if not self.chunk
            else (
                (f // self.chunk - self.radius) * self.chunk,
                (f // self.chunk + self.radius + 1) * self.chunk - 1,
            )
            for f in range(frames)
        ]
        full_cover = all(lo <= 0 and hi >= frames - 1 for lo, hi in bounds)
        if full_cover:
            out = self._dense(q, k, v, timestep)
        elif self.orig.attn_backend == "FA4":
            out = window_attention(
                q,
                k,
                v,
                start=start,
                frames=frames,
                per_frame=per_frame,
                chunk=self.chunk,
                radius=self.radius,
                dense_attention=self.global_attention,
            )
        else:
            out = torch.empty_like(q)
            # Global conditioning and both anchor frames attend densely.
            out[:, : start + per_frame] = self._dense(q[:, : start + per_frame], k, v, timestep)
            out[:, -per_frame:] = self._dense(q[:, -per_frame:], k, v, timestep)
            # Frames in the same chunk share their entire key/value window.
            # Gather it once and dispatch all their queries together.
            frame = 1
            while frame < frames - 1:
                end_frame = frame + 1
                while end_frame < frames - 1 and bounds[end_frame] == bounds[frame]:
                    end_frame += 1
                lo, hi = max(0, bounds[frame][0]), min(frames - 1, bounds[frame][1])
                slices = [
                    slice(0, start),
                    slice(start + lo * per_frame, start + (hi + 1) * per_frame),
                ]
                if lo > 0:
                    slices.append(slice(start, start + per_frame))
                if hi < frames - 1:
                    slices.append(slice(start + (frames - 1) * per_frame, layout.video_end))
                # Preserve packed-row order, including anchors outside the window,
                # so softmax reductions follow the released decomposition.
                slices.sort(key=lambda span: span.start)
                keys, values = (torch.cat([t[:, s] for s in slices], dim=1) for t in (k, v))
                rows = slice(start + frame * per_frame, start + end_frame * per_frame)
                out[:, rows] = self._dense(q[:, rows], keys, values, timestep)
                frame = end_frame
        x = hidden_states[0]
        out = self.orig.to_out[0]((out * self.softmax_gate(x).unsqueeze(0)).flatten(2))
        if not full_cover:
            text = layout.text_indices
            readout = self.linear_attention(
                x[start:],
                tuple(t[0, start:] for t in raw),
                layout,
                bounds,
                x[text] if self.text_state else None,
                tuple(t[0, text] for t in raw) if self.text_state else None,
            )
            out = out.clone()
            out[:, start:] += self.to_out_linear(readout).unsqueeze(0)
        return out
