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
"""DeCo-Attention CuTe DSL backend for visual generation models.

Self-attention on the DeCo-Attention kernels (``cute_dsl_kernels.blackwell.deco_attention``):
NVFP4 Q/K with compensation channels for the quantization error of the scores, and V smoothed
over blocks of keys and quantized to FP8 or NVFP4 (``v_dtype``; ``supported_v_dtypes()`` lists
the formats of the current GPU). The backend is a complete attention layer: a model builds it
like any other backend.

* Kernel calls: one sequence attending itself, bfloat16 q, k and v of one shape
  ``[B, S, H, 128]`` on a CUDA device, at least ``MIN_SEQ_LEN`` rows, no mask other than padding.
* VANILLA calls (unquantized): every other call, that is cross-attention, a causal
  ``attention_mask``, other dtypes or devices, shorter sequences, and every call of a layer whose
  head dimension or K/V head count the kernels do not take.
* Padding: ``key_padding_mask`` (``[B, S]``, True for a token). Padded rows get no attention
  weight and come back zero. Tokens that are a prefix in every sample reach the kernels as token
  counts. For any other mask the backend gathers the tokens to the front and scatters the output
  rows back; Q and K carry their positions, so the result does not depend on the order of the
  rows.
* One sequence: a call whose q, k and v have one shape, on the kernels and on VANILLA alike. Its
  padded rows are excluded as keys and come back zero as queries.
* Token tags: with ``token_tags``, half of the tokens that the kernels sample for the
  compensation channels are text and audio tokens of a packed sequence.
* CUDA graphs: first use (kernel compilation, the inputs of a sequence length, the rows of token
  tags) runs eagerly, as in the warm-up of a graph runner, and raises ``RuntimeError`` under
  capture. A captured call with a mask computes its token counts and order inside the graph, so
  a replay follows the content of the captured mask tensor. The basis rows are host values of
  the capture; a row that the mask of a replay excludes is read as the last token of its sample.
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

# Self-attention over fewer rows runs on VANILLA, such as the text of a token refiner.
MIN_SEQ_LEN = 8192

# Number of recent plans the component state keeps, besides those of captured CUDA graphs.
_PLANS_KEPT = 4


def supported_v_dtypes() -> tuple[str, ...]:
    """Return the ``v_dtype`` values with a DeCo-Attention kernel on the current GPU; empty when
    the kernel package does not import or the GPU has no kernels."""
    if deco_attention is None:
        return ()
    try:
        return tuple(deco_attention.pv_formats())
    except NotImplementedError:
        return ()


@dataclass(eq=False)
class _Plan:
    """Kernel inputs resolved by ``DeCoAttention._plan``, kept with the arguments they are for.

    Attributes:
        token_tags, key_padding_mask: Arguments. The tensors of the call, compared by identity.
        mask_version: Argument. Version counter of the mask, which an in-place write advances;
            None for an inference tensor, which has no counter.
        rows: Result. Basis rows; None without tags.
        token_counts: Result. Tokens per sample that the mask gives; None without a mask and
            for a mask that excludes no row.
        order: Result. Order of a compacted call; None when the tokens are a prefix.
        captured: A CUDA graph was captured with ``rows`` and reads them on replay, so the plan
            is never dropped.
    """

    token_tags: torch.Tensor | None
    key_padding_mask: torch.Tensor | None
    mask_version: int | None
    rows: torch.Tensor | None
    token_counts: torch.Tensor | None
    order: torch.Tensor | None
    captured: bool = False


class DeCoAttention(AttentionBackend):
    """DeCo-Attention backend for diffusion self-attention (see the module docstring).

    The DeCo layers of a model component share one workspace and their plans through
    ``attention_metadata_state``. They run one after another on one stream, and the kernels
    keep nothing in the workspace between calls.
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
        """Return whether the kernels take this call; every other call is delegated."""
        if not self._kernel_geometry or not q.is_cuda or q.numel() == 0:
            return False
        # The kernels attend one sequence to itself.
        if not one_sequence or q.shape[1] < MIN_SEQ_LEN:
            return False
        if not q.dtype == k.dtype == v.dtype == torch.bfloat16:
            return False
        mask = kwargs.get("attention_mask", PredefinedAttentionMask.FULL)
        return mask == PredefinedAttentionMask.FULL

    def _delegate(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, one_sequence: bool, **kwargs: Any
    ) -> torch.Tensor:
        """Run a call on VANILLA. For one sequence, the rows that ``key_padding_mask`` excludes
        come back zero, as from the kernels."""
        # VANILLA works in HND ([B, H, S, D]).
        out = self._vanilla.forward(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), **kwargs
        ).transpose(1, 2)
        key_padding_mask = kwargs.get("key_padding_mask")
        if key_padding_mask is not None and one_sequence:
            out = out.masked_fill(~key_padding_mask[:, :, None, None], 0)
        return out

    @torch.compile
    def _contiguous_qkv(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return q, k and v as contiguous tensors, which the kernels read.

        Slices of a fused QKV projection are strided; the compiled copy stays on a vectorized
        kernel, and contiguous inputs pass through without a copy.
        """
        return q.contiguous(), k.contiguous(), v.contiguous()

    def _workspace(self, num_bytes: int, device: torch.device) -> torch.Tensor:
        """Return the workspace of a call: the buffer of the component, grown to its largest call.

        A call under CUDA graph capture allocates from the memory pool of the graph instead,
        like its other temporaries: the graph keeps the addresses it was captured with.
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
        """Draw the basis rows of token tags by modality (the kernels' ``sample_rows``).

        Args:
            tokens: Bool ``[B, S]``, True where a sample holds a token; None without padding.

        Returns:
            int32 ``[m]``, one draw for the batch, when the tags mark the padding of every
            sample themselves. Otherwise ``[B, m]``, one draw per row of ``tokens``: a draw
            shorter than the longest repeats its rows, and a sample without tokens gets row 0,
            which the kernels read as padding.
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

    @staticmethod
    def _captured_rows(plans: list[_Plan], token_tags: torch.Tensor) -> torch.Tensor:
        """Return the basis rows for a CUDA graph capture, those of the newest plan with these
        tags, and keep that plan for the replays."""
        for plan in plans:
            if plan.token_tags is token_tags:
                plan.captured = True
                return plan.rows
        raise RuntimeError(
            "DeCo-Attention draws the basis rows of token tags on first use: run the forward "
            "eagerly once before capturing a CUDA graph."
        )

    def _kept_plan(
        self,
        plans: list[_Plan],
        key_padding_mask: torch.Tensor | None,
        token_tags: torch.Tensor | None,
        seq_len: int,
        device: torch.device,
    ) -> _Plan:
        """Return the plan of these arguments; on their first use resolve one, which reads back
        to the host, and keep it in front of ``plans``."""
        mask_version = None
        if key_padding_mask is not None and not key_padding_mask.is_inference():
            mask_version = key_padding_mask._version
        for plan in plans:
            if (
                plan.token_tags is token_tags
                and plan.key_padding_mask is key_padding_mask
                and plan.mask_version == mask_version
            ):
                return plan

        # tokens: where the samples hold tokens; None when every row does.
        tokens, token_counts, order = None, None, None
        if key_padding_mask is not None:
            tokens = key_padding_mask.to(device)
            if bool(tokens.all()):
                tokens = None
            else:
                token_counts = tokens.sum(dim=1, dtype=torch.int32)
                if bool((tokens[:, 1:] & ~tokens[:, :-1]).any()):
                    # Not a prefix: a stable sort puts the tokens first, in their order.
                    order = torch.argsort(~tokens, dim=1, stable=True)
        rows = None
        if token_tags is not None:
            rows = self._basis_rows(token_tags, tokens, seq_len, device)

        plan = _Plan(token_tags, key_padding_mask, mask_version, rows, token_counts, order)
        plans.insert(0, plan)
        # Drop the oldest plans past the kept count; a captured plan stays.
        for old in [kept for kept in plans if not kept.captured][_PLANS_KEPT:]:
            plans.remove(old)
        return plan

    @torch.compiler.disable
    def _plan(
        self,
        key_padding_mask: torch.Tensor | None,
        token_tags: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Resolve the padding and the token tags of a call into the inputs of the kernels.

        Resolving reads back to the host, so the component state keeps the result (``_Plan``)
        for the layers and steps that pass the same tags tensor and mask tensor; the content of
        an inference mask tensor must not change in place meanwhile. Under CUDA graph capture
        nothing is read back.

        Returns:
            A tuple ``(rows, token_counts, order)``:

            * rows: Basis rows, int32 ``[m]`` or ``[B, m]``; None for the kernels' own draw.
            * token_counts: Tokens per sample, int32 ``[B]`` (the kernels'
              ``valid_seq_lens``); None without padding.
            * order: Order ``[B, S]`` that puts the tokens of every sample first; None when
              they are a prefix.
        """
        if token_tags is not None and token_tags.numel() != seq_len:
            raise ValueError(
                f"DeCo-Attention got {token_tags.numel()} token tags for a sequence of "
                f"{seq_len} tokens."
            )
        if key_padding_mask is not None and key_padding_mask.shape != (batch_size, seq_len):
            raise ValueError(
                f"key_padding_mask must have shape {(batch_size, seq_len)}, got "
                f"{tuple(key_padding_mask.shape)}."
            )
        if key_padding_mask is None and token_tags is None:
            return None, None, None

        plans: list[_Plan] = self._state.setdefault("deco_plans", [])
        if torch.cuda.is_current_stream_capturing():
            rows = None
            if token_tags is not None:
                rows = self._captured_rows(plans, token_tags)
            if key_padding_mask is None:
                return rows, None, None
            # A captured call with a mask is always compacted, inside the graph.
            tokens = key_padding_mask.to(device)
            token_counts = tokens.sum(dim=1, dtype=torch.int32)
            return rows, token_counts, torch.argsort(~tokens, dim=1, stable=True)

        plan = self._kept_plan(plans, key_padding_mask, token_tags, seq_len, device)
        return plan.rows, plan.token_counts, plan.order

    @torch.compiler.disable
    def _launch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rows: torch.Tensor | None = None,
        token_counts: torch.Tensor | None = None,
        order: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the preparation and the attention kernel with the rows, token counts and order
        of ``_plan``. With ``order`` the call is compacted: the rows of q, k and v are gathered
        in this order and the output rows scattered back."""
        batch_size, seq_len, num_heads, _ = q.shape
        pv = self.quant_attention_config.v_dtype
        num_bytes = deco_attention.workspace_size(batch_size, seq_len, num_heads, q.device, pv)
        if order is not None:
            index = order[:, :, None, None].expand_as(q)
            q, k, v = q.gather(1, index), k.gather(1, index), v.gather(1, index)
            if rows is not None:
                # The basis rows follow their tokens to their positions in the order. The
                # kernels read a row behind the tokens as the last token.
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
            valid_seq_lens=token_counts,
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

        The module docstring says how the backend reads the keywords ``key_padding_mask`` and
        ``token_tags``.
        """
        key_padding_mask = kwargs.get("key_padding_mask")
        if key_padding_mask is not None and key_padding_mask.dtype != torch.bool:
            key_padding_mask = kwargs["key_padding_mask"] = key_padding_mask.bool()
        one_sequence = q.shape == k.shape == v.shape
        if not self._can_serve(q, k, v, one_sequence, **kwargs):
            # Warn about a call of the kernels' shape and size that a mask or a dtype keeps
            # from them; cross-attention is delegated on every call and logs nothing.
            if (
                self._kernel_geometry
                and one_sequence
                and q.shape[1] >= MIN_SEQ_LEN
                and not self._noted_unquantized
            ):
                self._note_unquantized(q, k, v, **kwargs)
            return self._delegate(q, k, v, one_sequence, **kwargs)
        rows, token_counts, order = self._plan(
            key_padding_mask, kwargs.get("token_tags"), q.shape[0], q.shape[1], q.device
        )
        if order is None:
            q, k, v = self._contiguous_qkv(q, k, v)
        return self._launch(q, k, v, rows, token_counts, order)

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        return self._preferred_layout

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False

    @classmethod
    def support_key_padding_mask(cls) -> bool:
        return True
