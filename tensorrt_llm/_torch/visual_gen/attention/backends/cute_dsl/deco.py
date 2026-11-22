# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
DeCo-Attention CuTe DSL Backend for Visual Generation Models

Runs self-attention on the DeCo-Attention kernels (``cute_dsl_kernels.blackwell.deco_attention``):
NVFP4 Q/K with compensation channels for the quantization error of the scores, and V smoothed
over blocks of keys and quantized to FP8 (``v_dtype="fp8"``) or NVFP4 (``v_dtype="nvfp4"``).
``supported_v_dtypes()`` lists the V formats that have a kernel on the current GPU.

The kernels take one sequence attending itself: bfloat16 q, k and v of one shape
[B, S, H, 128] on a CUDA device, with no mask other than padding. The backend is a complete
attention layer: it runs every other call on the VANILLA backend, so a model builds it like
any other backend.

* Padding: ``key_padding_mask`` ([B, S], True = token), ``valid_seq_len`` (a host integer: the
  rows after this prefix are padding) or both; a row is a token where both say so. Padded rows
  get no attention weight, whatever they hold, and come back zero. A mask whose tokens are a
  prefix in every sample reaches the kernels as one length per sample. For any other mask the
  backend gathers the tokens of every sample to the front, runs the kernels on the lengths and
  scatters the output rows back; Q and K carry their positions, so the order of the rows does
  not change the result.
* Token tags: the kernels estimate the compensation channels from a sample of the tokens. With
  ``token_tags`` (see ``AttentionBackend.forward``) half of the sample comes from the text and
  audio tokens of a packed sequence, which a draw over the whole sequence almost misses.
* Delegated calls: cross-attention, a causal ``attention_mask``, inputs that are not bfloat16
  CUDA tensors, self-attention over fewer than ``MIN_SEQ_LEN`` rows, and every call of a layer
  whose head dimension or K/V head count the kernels do not take.
* One sequence: a call whose q, k and v have one shape is read as one sequence attending
  itself, on the kernels and on VANILLA alike: its padding excludes the rows as keys and
  returns them zero as queries. This includes a cross-attention call with as many query rows
  as key rows and a ``key_padding_mask``.
"""

from dataclasses import dataclass
from typing import Any

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.visual_gen.args import QuantAttentionConfig

from .....attention.backends.interface import PredefinedAttentionMask
from ..interface import AttentionBackend, AttentionTensorLayout
from ..vanilla import VanillaAttention

_deco_import_error: BaseException | None = None
try:
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell import deco_attention
except (ImportError, OSError) as e:
    deco_attention = None
    _deco_import_error = e

# A self-attention call over fewer rows runs on VANILLA: the text sequence of a token refiner.
MIN_SEQ_LEN = 8192

# The component state keeps the plans of this many recent pairs of token tags and padding, and
# every plan whose basis rows a CUDA graph was captured with: the graph reads them on replay.
_PLANS_KEPT = 4


def supported_v_dtypes() -> tuple[str, ...]:
    """The ``v_dtype`` values DeCo-Attention has a kernel for on the current GPU, as the kernel
    package reports them; empty when the package does not import or the GPU has no kernels."""
    if deco_attention is None:
        return ()
    try:
        return tuple(deco_attention.pv_formats())
    except NotImplementedError:
        return ()


@dataclass(eq=False)
class _Plan:
    """The result of ``DeCoAttention._plan`` for one tags tensor, mask tensor (with its version
    counter) and prefix length, and whether a CUDA graph was captured with the rows."""

    tags: torch.Tensor | None
    mask: torch.Tensor | None
    mask_version: int | None
    prefix_len: int | None
    rows: torch.Tensor | None
    valid_seq_lens: torch.Tensor | None
    order: torch.Tensor | None
    captured: bool = False


class DeCoAttention(AttentionBackend):
    """DeCo-Attention backend for diffusion self-attention (see the module docstring).

    The DeCo layers of a model component share one workspace through
    ``attention_metadata_state``: they run one after another on one stream, and the kernels
    keep nothing in the workspace between calls.

    CUDA graphs. The kernels compile, and the preparation builds its inputs for a sequence
    length and draws the basis rows of token tags, on first use: a capture follows an eager
    forward of the same shape and tags tensor, which the CUDA graph runner's warm-up provides,
    and a first use under capture raises a ``RuntimeError``. A captured call with a
    ``key_padding_mask`` computes the lengths and the order of the tokens inside the graph, so a
    replay follows the content of the mask tensor of the capture. ``valid_seq_len`` and the
    basis rows are host values of the capture; a row that the mask of a replay excludes is
    read as the last token of its sample.
    """

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 128,
        num_kv_heads: int | None = None,
        dtype: torch.dtype | None = None,
        quant_attention_config: QuantAttentionConfig | None = None,
        attention_metadata_state: dict[str, Any] | None = None,
        **kwargs,
    ):
        if _deco_import_error is not None:
            raise ImportError(
                f"DeCo-Attention kernels are not available. Import error: {_deco_import_error}"
            ) from _deco_import_error
        if quant_attention_config is None or not quant_attention_config.deco_attention:
            raise ValueError(
                "DeCoAttention needs a quant_attention_config with deco_attention=True."
            )
        num_kv_heads = num_kv_heads or num_heads
        # The V format of the recipe names the PV format of the kernel; Q/K are always NVFP4.
        v_dtypes = supported_v_dtypes()
        if not v_dtypes:
            raise ValueError(
                "DeCo-Attention has kernels for SM100, SM103, SM107 and SM120 GPUs only."
            )
        qk_dtype, v_dtype = quant_attention_config.qk_dtype, quant_attention_config.v_dtype
        if qk_dtype != "nvfp4" or v_dtype not in v_dtypes:
            raise ValueError(
                f"DeCo-Attention has no kernel for qk_dtype={qk_dtype!r}, v_dtype={v_dtype!r} on "
                f"this GPU, whose kernels take qk_dtype='nvfp4' with v_dtype in {v_dtypes}."
            )

        if attention_metadata_state is None:
            raise ValueError(
                "DeCoAttention requires `attention_metadata_state` from DiffusionModelConfig: "
                "the layers of a model component share one workspace buffer."
            )

        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.dtype = dtype
        self.quant_attention_config = quant_attention_config
        self._state = attention_metadata_state
        # A layer of another head dimension or K/V head count runs all its calls on VANILLA.
        kernel_head_dim = deco_attention.constants.HEAD_DIM
        self._kernel_geometry = head_dim == kernel_head_dim and num_kv_heads == num_heads
        if not self._kernel_geometry:
            logger.warning_once(
                f"DeCo-Attention runs the attention layers with head_dim {head_dim}, {num_heads} "
                f"Q heads and {num_kv_heads} K/V heads unquantized on VANILLA (first seen in "
                f"layer {layer_idx}); its kernels take head_dim {kernel_head_dim} and as many "
                "K/V heads as Q heads.",
                key=f"deco_attention_geometry_{head_dim}_{num_heads}_{num_kv_heads}",
            )
        self._noted_unquantized = False
        self._preferred_layout = AttentionTensorLayout.NHD
        self._vanilla = VanillaAttention(
            layer_idx=layer_idx,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            dtype=dtype,
        )

    def _can_serve(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, one_sequence: bool, **kwargs: Any
    ) -> bool:
        """Whether the DeCo kernels apply to this call; every other call is delegated."""
        if not self._kernel_geometry or not q.is_cuda or q.numel() == 0:
            return False
        # Cross-attention: the kernels attend one sequence to itself.
        if not one_sequence or q.shape[1] < MIN_SEQ_LEN:
            return False
        if not q.dtype == k.dtype == v.dtype == torch.bfloat16:
            return False
        mask = kwargs.get("attention_mask", PredefinedAttentionMask.FULL)
        return mask == PredefinedAttentionMask.FULL

    def _delegate(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, one_sequence: bool, **kwargs: Any
    ) -> torch.Tensor:
        """Run a call the kernels do not take on VANILLA, which works in HND ([B, H, S, D]).

        Cross-attention layers take this path on every call, so it logs nothing. VANILLA takes
        no ``valid_seq_len``: a call of one sequence attends its valid prefix alone, and the
        rows that the length or the mask excludes come back zero, as from the kernels. A call
        with other key rows than query rows keeps all its keys, which its ``key_padding_mask``
        describes; ``valid_seq_len`` zeroes its padded query rows.
        """
        valid_seq_len = kwargs.pop("valid_seq_len", None)
        if valid_seq_len is not None and valid_seq_len < q.shape[1]:
            if one_sequence:
                k, v = k[:, :valid_seq_len], v[:, :valid_seq_len]
                key_padding_mask = kwargs.get("key_padding_mask")
                if key_padding_mask is not None:
                    kwargs["key_padding_mask"] = key_padding_mask[:, :valid_seq_len]
            out = torch.zeros_like(q)
            if valid_seq_len > 0:
                out[:, :valid_seq_len] = self._delegate(
                    q[:, :valid_seq_len], k, v, one_sequence, **kwargs
                )
            return out
        out = self._vanilla.forward(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), **kwargs
        ).transpose(1, 2)
        key_padding_mask = kwargs.get("key_padding_mask")
        if key_padding_mask is not None and one_sequence:
            out = out.masked_fill(~key_padding_mask[:, :, None, None], 0)
        return out

    @torch.compile
    def _compact_qkv(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Slices of a fused QKV projection are strided and the kernels read gap-free
        # [B, S, H, 128] tensors; the compiled copy keeps them on a vectorized kernel, while
        # already contiguous inputs pass through without a copy.
        return q.contiguous(), k.contiguous(), v.contiguous()

    def _workspace(self, num_bytes: int, device: torch.device) -> torch.Tensor:
        """The workspace of a call: the component's buffer, grown to its largest call.

        A CUDA graph keeps the addresses it was captured with, so a call under capture takes
        its workspace from the graph's memory pool instead, like its other temporaries.
        """
        if torch.cuda.is_current_stream_capturing():
            return torch.empty(num_bytes, dtype=torch.uint8, device=device)
        workspace = self._state.get("deco_workspace")
        if workspace is None or workspace.numel() < num_bytes or workspace.device != device:
            # Release the smaller buffer before allocating the larger one.
            workspace = self._state["deco_workspace"] = None
            workspace = torch.empty(num_bytes, dtype=torch.uint8, device=device)
            self._state["deco_workspace"] = workspace
        return workspace

    @staticmethod
    def _basis_rows(
        token_tags: torch.Tensor, tokens: torch.Tensor | None, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """The basis rows of token tags: the draw by modality of the kernels' ``sample_rows``.

        ``tokens`` (``[B, S]`` or ``[1, S]``, None without padding) says where a sample holds
        tokens. When the tags mark the padding of every sample themselves, the batch shares one
        draw ``[m]``. Otherwise every row of ``tokens`` gets the draw of its own tokens,
        ``[B, m]``: a draw shorter than the longest repeats its rows, and a sample without
        tokens gets row 0, which the kernels read as padding.
        """
        tags = token_tags.reshape(-1)
        if tokens is None or bool(((tags >= 0) == tokens).all()):
            draws = [deco_attention.sample_rows(seq_len, device, tags)]
        else:
            draws = [
                deco_attention.sample_rows(seq_len, device, torch.where(sample, tags, -1))
                for sample in tokens
            ]
        num_rows = max(1, max(draw.numel() for draw in draws))
        rows = torch.zeros((len(draws), num_rows), dtype=torch.int32, device=device)
        for sample, draw in enumerate(draws):
            if draw.numel():
                repeats = -(-num_rows // draw.numel())
                rows[sample] = draw.repeat(repeats)[:num_rows]
        return rows[0].contiguous() if len(draws) == 1 else rows

    @torch.compiler.disable
    def _plan(
        self,
        key_padding_mask: torch.Tensor | None,
        valid_seq_len: int | None,
        token_tags: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """What the kernels need besides q, k and v: the basis rows of the token tags (int32
        ``[m]`` or ``[B, m]``; None for the kernels' own draw), the token count of every sample
        (int32 ``[B]`` on the device; None without padding) and, when the tokens of a sample
        are not a prefix, the order ``[B, S]`` that puts them first (None otherwise).

        Drawing the rows and telling a prefix from other padding read back to the host, so the
        component state keeps one plan per tags tensor, mask tensor and prefix length, and the
        layers and steps of a request that pass the same tensors reuse it. The mask is kept
        with its version counter; an inference tensor has none, so its content must not change
        in place while it is passed. Under CUDA graph capture nothing is read back: the counts
        and the order are computed inside the graph, and the rows are those of a plan with the
        same tags and prefix length, which is then kept for the replays.
        """
        if valid_seq_len is not None and valid_seq_len >= seq_len:
            valid_seq_len = None
        if token_tags is not None and token_tags.numel() != seq_len:
            raise ValueError(
                f"DeCo-Attention got {token_tags.numel()} token tags for a sequence of "
                f"{seq_len} tokens."
            )
        prefix_lens = None
        if key_padding_mask is None:
            if valid_seq_len is not None:
                prefix_lens = torch.full(
                    (batch_size,), valid_seq_len, dtype=torch.int32, device=device
                )
            if token_tags is None:
                return None, prefix_lens, None
        elif key_padding_mask.shape != (batch_size, seq_len):
            raise ValueError(
                f"key_padding_mask must have shape {(batch_size, seq_len)}, got "
                f"{tuple(key_padding_mask.shape)}."
            )

        capturing = torch.cuda.is_current_stream_capturing()
        version = None
        if key_padding_mask is not None and not key_padding_mask.is_inference():
            version = key_padding_mask._version
        plans: list[_Plan] = self._state.setdefault("deco_plans", [])
        rows = None
        for plan in plans:
            if plan.tags is not token_tags or plan.prefix_len != valid_seq_len:
                continue
            if capturing:
                # The graph reads these rows on every replay.
                plan.captured = plan.rows is not None
                rows = plan.rows
                break
            if plan.mask is key_padding_mask and plan.mask_version == version:
                valid_seq_lens = prefix_lens if plan.mask is None else plan.valid_seq_lens
                return plan.rows, valid_seq_lens, plan.order
        else:
            if capturing and token_tags is not None:
                raise RuntimeError(
                    "DeCo-Attention draws the basis rows of token tags on first use: run the "
                    "forward eagerly once before capturing a CUDA graph."
                )

        # The rows of every sample that hold tokens; None when every row does.
        tokens = None
        if key_padding_mask is not None:
            tokens = key_padding_mask.to(device)
            if valid_seq_len is not None:
                tokens = tokens & (torch.arange(seq_len, device=device) < valid_seq_len)
            if not capturing and bool(tokens.all()):
                tokens = None
        valid_seq_lens, order = prefix_lens, None
        if tokens is not None:
            valid_seq_lens = tokens.sum(dim=1, dtype=torch.int32)
            if capturing or bool((tokens[:, 1:] & ~tokens[:, :-1]).any()):
                # A stable sort keeps the tokens of a sample in their order.
                order = torch.argsort(~tokens, dim=1, stable=True)
        if capturing:
            return rows, valid_seq_lens, order
        if token_tags is not None:
            if tokens is None and valid_seq_len is not None:
                tokens = (torch.arange(seq_len, device=device) < valid_seq_len)[None]
            rows = self._basis_rows(token_tags, tokens, seq_len, device)
        plans.insert(
            0,
            _Plan(
                token_tags,
                key_padding_mask,
                version,
                valid_seq_len,
                rows,
                None if key_padding_mask is None else valid_seq_lens,
                order,
            ),
        )
        # Drop the oldest plans past the kept count; a captured plan is never dropped.
        for plan in [plan for plan in plans if not plan.captured][_PLANS_KEPT:]:
            plans.remove(plan)
        return rows, valid_seq_lens, order

    @torch.compiler.disable
    def _launch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rows: torch.Tensor | None = None,
        valid_seq_lens: torch.Tensor | None = None,
        order: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the preparation and the attention kernel on gap-free q, k, v. ``order``: the
        call is compacted, its rows are gathered in this order and the output rows scattered
        back."""
        batch_size, seq_len, num_heads, _ = q.shape
        pv = self.quant_attention_config.v_dtype
        num_bytes = deco_attention.workspace_size(batch_size, seq_len, num_heads, q.device, pv)
        if order is not None:
            index = order[:, :, None, None].expand_as(q)
            q, k, v = q.gather(1, index), k.gather(1, index), v.gather(1, index)
            if rows is not None:
                # The basis rows follow their tokens: their positions in the order of each
                # sample. The kernels read a row behind the tokens as the last token.
                position = torch.empty_like(order).scatter_(
                    1, order, torch.arange(seq_len, device=q.device).expand_as(order)
                )
                rows = position.gather(1, rows.long().expand(batch_size, -1))
                rows = rows.to(torch.int32).contiguous()
        operands = deco_attention.prepare(
            q,
            k,
            v,
            self._workspace(num_bytes, q.device),
            pv,
            rows=rows,
            valid_seq_lens=valid_seq_lens,
        )
        out = deco_attention.attention(operands, torch.empty_like(q))
        if order is not None:
            # The rows behind the tokens of a sample are zero and land on its padding.
            out = torch.empty_like(out).scatter_(1, index, out)
        return out

    @torch.compiler.disable
    def _note_unquantized(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: Any
    ) -> None:
        """Warn, once per layer, that one of its self-attention calls is delegated."""
        self._noted_unquantized = True
        if kwargs.get("attention_mask", PredefinedAttentionMask.FULL) != (
            PredefinedAttentionMask.FULL
        ):
            reason = "an attention mask"
        elif not q.dtype == k.dtype == v.dtype == torch.bfloat16:
            dtypes = sorted({str(x.dtype) for x in (q, k, v)} - {"torch.bfloat16"})
            reason = f"{' and '.join(dtypes)} inputs"
        elif q.numel() == 0:
            reason = "empty inputs"
        else:
            reason = "inputs that are not on a CUDA device"
        logger.warning_once(
            f"DeCo-Attention runs self-attention unquantized on VANILLA because of {reason} "
            f"(first seen in layer {self.layer_idx}); its kernels take bfloat16 CUDA inputs and "
            "padding, no other mask.",
            key=f"deco_attention_unquantized_{reason}",
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Attention over q, k, v ``[B, S, H, D]``; returns ``[B, S, H, D]`` in q's dtype.

        The module docstring describes the keywords ``key_padding_mask``, ``valid_seq_len``
        and ``token_tags``.
        """
        valid_seq_len = kwargs.get("valid_seq_len")
        if valid_seq_len is not None and (
            type(valid_seq_len) is not int or not 0 <= valid_seq_len <= q.shape[1]
        ):
            raise ValueError("valid_seq_len must be a host integer in [0, sequence length].")
        key_padding_mask = kwargs.get("key_padding_mask")
        if key_padding_mask is not None and key_padding_mask.dtype != torch.bool:
            key_padding_mask = kwargs["key_padding_mask"] = key_padding_mask.bool()
        one_sequence = q.shape == k.shape == v.shape
        if not self._can_serve(q, k, v, one_sequence, **kwargs):
            # A call of the kernels' shape and size that a mask or a dtype keeps from them.
            if (
                self._kernel_geometry
                and one_sequence
                and q.shape[1] >= MIN_SEQ_LEN
                and not self._noted_unquantized
            ):
                self._note_unquantized(q, k, v, **kwargs)
            return self._delegate(q, k, v, one_sequence, **kwargs)
        rows, valid_seq_lens, order = self._plan(
            key_padding_mask,
            valid_seq_len,
            kwargs.get("token_tags"),
            q.shape[0],
            q.shape[1],
            q.device,
        )
        if order is None:
            q, k, v = self._compact_qkv(q, k, v)
        return self._launch(q, k, v, rows, valid_seq_lens, order)

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        return self._preferred_layout

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False

    @classmethod
    def support_key_padding_mask(cls) -> bool:
        return True

    @classmethod
    def support_valid_seq_len(cls) -> bool:
        return True
