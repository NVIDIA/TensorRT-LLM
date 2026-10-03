# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode path's GEMVs, LM head and embedding on the catalog's single-GPU Kimi K3 entries.

* **Per-site GEMVs** (`K3DecodeGemvs.project`): a projection of a decode step runs on the kernel measured fastest
  at its call site's weight shape (`SITES`): at most `MAX_ROWS` rows on `gemm/k3_decode_gemv`,
  `gemm/k3_ctm_gemv_wide` or `gemm/k3_ctm_gemv_long`, and, where the site lists it, more rows (up to `WIDE_ROWS`)
  on `gemm/k3_ctm_gemv_wide`. The sites are the decode kernels' fused projections (MLA's [W_a; W_g] with the gate
  rows through a sigmoid, KDA's [q | k | v | g | f_a | b]), the attention output projection, the built-in MLA path's
  q_a / kv_a, q_b and gate projections, and the MoE decode path's projections (`decode_moe.py`), two of them with
  fp32 outputs.
* **LM head** (`K3LogitsProcessor`): at most `MAX_ROWS` rows of this rank's vocabulary shard on
  `gemm/k3_head_gemv` over the target's `K3HeadGemvWorkspace`, then the shards gathered (`comm/allgather`) as the
  stock head gathers them. It is the shell's logits processor, so the speculative worker's target logits and the
  drafter's logits on the same head take it too.
* **Embedding** (`K3DecodeGemvs.embed_norm`): a decode step's embedding rows written into the attention-residual
  bank's slot 0 (layer 0's first snapshot) and layer 0's input RMSNorm applied, in one `norm/k3_embed_norm` launch.
* **Dense MLP** (`K3DecodeGemvs.dense_mlp`): layer 0's MLP at most `MAX_ROWS` rows, split over the whole TP group:
  gate_up on `gemm/k3_ctm_gemv_long`, `activation/k3_situ_mul`, down on `gemm/k3_ctm_gemv_long`; the caller then
  runs the down projection's all-reduce.

Each returns None where its kernel does not take the call, and the caller then runs the generic path's module.

A kernel compiles on its first call for a shape, which must not happen under CUDA-graph capture.
`K3DecodeGemvs.create`, run once the weights are final, runs every site's kernel and the head once, eagerly. The
embedding kernel compiles per token count, on the eager warm-up step before each capture. Under capture, a call
whose kernel has not run eagerly is refused.

The head's workspace serves every `k3_head_gemv` call of its weight shape, so those calls must be ordered on one
stream: the logits are computed on the model's stream (see the `gemm/k3_head_gemv` contract's State section).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set

import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.k3_situ_mul import k3_situ_mul
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.allgather import allgather
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.k3_ctm_gemv_long import (
    k3_ctm_gemv_long,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.k3_ctm_gemv_wide import (
    k3_ctm_gemv_wide,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.k3_decode_gemv import k3_decode_gemv
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.k3_head_gemv import (
    K3HeadGemvWorkspace,
    k3_head_gemv,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.k3_embed_norm import k3_embed_norm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.torch.concat import concat
from tensorrt_llm._torch._experimental.modeling_v2.catalog.torch.split import split

# The kernels' support predicates: metadata reads only.
from tensorrt_llm._torch.cute_dsl_kernels.k3_ctm_gemv import op as _ctm_op
from tensorrt_llm._torch.cute_dsl_kernels.k3_decode_gemv import op as _decode_op
from tensorrt_llm._torch.cute_dsl_kernels.k3_embed import op as _embed_op
from tensorrt_llm._torch.cute_dsl_kernels.k3_head_gemv import op as _head_op
from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._utils import mpi_disabled

# The row limit of the decode GEMV and head kernels (one token tile), and of k3_ctm_gemv_wide (a decode step of 8
# requests of 8 tokens).
MAX_ROWS = 8
WIDE_ROWS = 64


@dataclass(frozen=True)
class Site:
    """A call site's weight shape (this target's per-rank shapes) and its kernels: ``small`` at 1..MAX_ROWS rows
    ("decode", "wide" or "long"), and k3_ctm_gemv_wide at MAX_ROWS+1..``wide_rows`` rows where ``wide``. Output columns
    from ``sig_col0`` on are stored through a sigmoid; ``out_fp32``: an fp32 output (k3_ctm_gemv_wide only).
    ``split`` / ``ring`` / ``push``: k3_ctm_gemv_long's CTAs per 128-row weight tile, weight-ring stages, and whether
    the partial sums are pushed to each row's owner."""

    n: int
    k: int
    small: str
    wide: bool = False
    sig_col0: int = -1
    split: int = 0
    ring: int = 0
    push: bool = False
    out_fp32: bool = False
    wide_rows: int = WIDE_ROWS


SITES: Dict[str, Site] = {
    # MLA's [W_a; W_g] on a decode step: [q_a 1536 | kv_a 512 | k_pe 64] then the output gate (6 heads x 128),
    # the gate rows through a sigmoid.
    "mla_ag": Site(2880, 7168, "long", wide=True, sig_col0=2112, split=6, ring=6, push=True),
    # KDA's [q | k | v | g | f_a | b] on a decode step (6 heads x 128 each, then 128 and 6, padded to 3208 rows).
    "kda_proj": Site(3208, 7168, "long", wide=True, split=5, ring=6, push=True),
    # The attention output projection (row parallel; 6 heads x 128 in).
    "o_proj": Site(7168, 768, "decode", wide=True),
    # The built-in MLA path's projections: kv_a_proj_with_mqa, q_b_proj and the output gate.
    "kv_a": Site(2112, 7168, "decode"),
    "q_b": Site(1152, 1536, "wide"),
    "g_proj": Site(768, 7168, "wide"),
    # Layer 0's dense MLP split over the 16-way TP group: gate_up [gate 2112 | up 2112] and down.
    "dense_gate_up": Site(4224, 7168, "long", split=4, ring=5),
    "dense_down": Site(7168, 2112, "long", split=2, ring=6),
    # The MoE decode path (decode_moe.py): this rank's head slice [latent down 224 | router 56] with an fp32 output,
    # the shared experts' gate_up, the row-parallel tail [latent up 224 | padding 32 | shared down 384] and the
    # replicated tail's latent up projection with an fp32 output. The wide kernel takes the head slice and the tail up
    # to 32 rows, where it is faster than the stock GEMM.
    "moe_head": Site(280, 7168, "wide", wide=True, out_fp32=True, wide_rows=32),
    "moe_shared_gate_up": Site(768, 7168, "wide", wide=True),
    "moe_tail": Site(7168, 640, "wide", wide=True, wide_rows=32),
    "moe_up": Site(7168, 3584, "wide", out_fp32=True),
}


def _wide_tile(rows: int) -> int:
    from tensorrt_llm._torch.cute_dsl_kernels.k3_ctm_gemv import k3_ctm_gemv_kernel

    return k3_ctm_gemv_kernel.wide_tile(rows)


def _run(
    spec: Site, kernel: str, x2d: torch.Tensor, weight: torch.Tensor
) -> Optional[torch.Tensor]:
    """``spec``'s ``kernel`` on dense rows ``x2d``, or None where it does not take them."""
    if kernel == "decode":
        if spec.sig_col0 >= 0 or spec.out_fp32 or not _decode_op.supports(x2d, weight):
            return None
        return k3_decode_gemv(x2d, weight)
    if kernel == "wide":
        if not _ctm_op.supports_wide(x2d, weight, spec.sig_col0, spec.out_fp32):
            return None
        return k3_ctm_gemv_wide(x2d, weight, sig_col0=spec.sig_col0, out_fp32=spec.out_fp32)
    if spec.out_fp32:
        return None
    # One wave of the GPU's SMs: beyond it the long GEMV loses to the others.
    sms = torch.cuda.get_device_properties(x2d.device).multi_processor_count
    if math.ceil(spec.n / 128) * spec.split > sms or not _ctm_op.supports_long(
        x2d, weight, spec.split, spec.ring
    ):
        return None
    return k3_ctm_gemv_long(
        x2d,
        weight,
        sig_col0=spec.sig_col0,
        split=spec.split,
        ring=spec.ring,
        trigger_early=True,
        push=spec.push,
    )


def _capturing() -> bool:
    return not torch.compiler.is_compiling() and torch.cuda.is_current_stream_capturing()


def _dense_rows(x2d: torch.Tensor) -> torch.Tensor:
    """``x2d`` itself, or a dense copy: the kernels' TMA descriptors need dense, 16-byte-aligned rows."""
    if x2d.stride() != (x2d.shape[1], 1) or x2d.data_ptr() % 16:
        return x2d.clone(memory_format=torch.contiguous_format)
    return x2d


def _head_takes_module(lm_head: nn.Module) -> bool:
    """Whether ``lm_head(rows)`` is a plain vocabulary-parallel GEMM of a bf16 weight whose shards it gathers along
    the vocabulary in rank order, with nothing else applied: the stock head this path reproduces."""
    weight = getattr(lm_head, "weight", None)
    mapping = getattr(lm_head, "mapping", None)
    return (
        isinstance(weight, torch.Tensor)
        and weight.dim() == 2
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
        and getattr(getattr(lm_head, "tp_mode", None), "name", None) == "COLUMN"
        and getattr(lm_head, "gather_output", False)
        and getattr(lm_head, "gather_output_sizes", None) is None
        and getattr(lm_head, "padding_size", None) == 0
        and getattr(lm_head, "bias", None) is None
        and not getattr(lm_head, "has_any_quant", True)
        and mapping is not None
        and not mapping.enable_attention_dp
    )


def _plain_rmsnorm(norm: nn.Module) -> bool:
    """Whether ``norm`` is the stock RMSNorm that runs flashinfer's kernel, which ``k3_embed_norm`` reproduces bit for
    bit. An unknown module fails closed."""
    weight = getattr(norm, "weight", None)
    return (
        IS_FLASHINFER_AVAILABLE
        and isinstance(weight, torch.Tensor)
        and weight.dtype == torch.bfloat16
        and not getattr(norm, "use_gemma", True)
        and not getattr(norm, "is_nvfp4", True)
        and not getattr(norm, "use_cuda_tile", True)
        and not getattr(norm, "return_hp_output", True)
        and getattr(norm, "nvfp4_scale", None) is None
        and hasattr(norm, "variance_epsilon")
    )


class K3DecodeGemvs:
    """The decode GEMVs' state for one target: the LM head's `K3HeadGemvWorkspace`, and the calls whose kernels ran
    eagerly (compiled), which are the only ones a CUDA-graph capture may take. Built by `create` once the weights are
    final; owned by the target."""

    def __init__(self, head_workspace: Optional[K3HeadGemvWorkspace] = None) -> None:
        self.head_workspace = head_workspace
        self._ran: Set[tuple] = set()

    @classmethod
    def create(
        cls,
        lm_head: Optional[nn.Module] = None,
        sites: Iterable[str] = tuple(SITES),
        device: Optional[torch.device] = None,
    ) -> "K3DecodeGemvs":
        """The state for ``lm_head`` and ``sites`` on ``device`` (default: the head's, else the current one). Eager:
        it allocates the head's workspace and runs every kernel of every site once (each wide row class once) on
        zero rows of a zero weight of the site's shape, so they compile here and not under a capture. A site or
        head whose kernel does not take its shape keeps the generic path."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "K3DecodeGemvs.create allocates and compiles: run it before CUDA-graph capture"
            )
        head_weight = getattr(lm_head, "weight", None) if lm_head is not None else None
        if device is None:
            device = (
                head_weight.device
                if isinstance(head_weight, torch.Tensor)
                else torch.device("cuda", torch.cuda.current_device())
            )
        state = cls()
        for site in sites:
            spec = SITES[site]
            weight = torch.zeros(spec.n, spec.k, dtype=torch.bfloat16, device=device)
            for rows in (1, 16, 32, 64) if spec.wide else (1,):
                if rows > spec.wide_rows:
                    continue
                state._project(site, weight.new_zeros(rows, spec.k), weight, warm=True)
            del weight
        if "dense_gate_up" in sites:
            gu = torch.zeros(1, SITES["dense_gate_up"].n, dtype=torch.bfloat16, device=device)
            if _ctm_op.supports_situ_mul(gu):
                for linear_beta in (None, 1.0):
                    k3_situ_mul(gu, 1.0, linear_beta)
                    state._ran.add(("situ_mul", linear_beta is not None))
        if lm_head is not None and _head_takes_module(lm_head):
            x = head_weight.new_zeros(1, head_weight.shape[1])
            if _head_op.supports(x, head_weight):
                workspace = K3HeadGemvWorkspace.create(
                    head_weight.shape[0], head_weight.shape[1], head_weight.device
                )
                k3_head_gemv(x, head_weight, workspace)
                state.head_workspace = workspace
                state._ran.add(("lm_head",))
        torch.cuda.synchronize(device)
        return state

    def project(self, site: str, x: torch.Tensor, weight: torch.Tensor) -> Optional[torch.Tensor]:
        """``x @ weight.T`` (``[..., N]``: bf16, the site's sigmoid columns through the sigmoid; fp32 at an
        ``out_fp32`` site) for ``site``'s weight on its decode kernel, or None where none takes the call: more rows
        than the site's kernels take, another shape or dtype, or, under capture, a kernel that has not run eagerly.
        The caller then runs its GEMM."""
        return self._project(site, x, weight, warm=False)

    def _project(
        self, site: str, x: torch.Tensor, weight: torch.Tensor, warm: bool
    ) -> Optional[torch.Tensor]:
        spec = SITES[site]
        if (
            weight.dtype != torch.bfloat16
            or tuple(weight.shape) != (spec.n, spec.k)
            or not weight.is_contiguous()
            or x.dtype != torch.bfloat16
            or x.dim() < 1
            or x.shape[-1] != spec.k
        ):
            return None
        rows = x.numel() // spec.k
        if 0 < rows <= MAX_ROWS:
            kernel = spec.small
        elif spec.wide and MAX_ROWS < rows <= spec.wide_rows:
            kernel = "wide"
        else:
            return None
        key = (site, kernel, _wide_tile(rows) if kernel == "wide" else 0)
        capturing = _capturing()
        if capturing and not warm and key not in self._ran:
            return None
        y = _run(spec, kernel, _dense_rows(x.reshape(rows, spec.k)), weight)
        if y is None:
            return None
        if not capturing:
            self._ran.add(key)
        return y.view(*x.shape[:-1], spec.n)

    def lm_head_logits(self, rows: torch.Tensor, lm_head: nn.Module) -> Optional[torch.Tensor]:
        """``lm_head(rows)``, the gathered bf16 logits ``[M, vocab]``, with this rank's shard on
        ``gemm/k3_head_gemv``; None where it does not take the call (more than `MAX_ROWS` rows, another head, ...)."""
        workspace = self.head_workspace
        if workspace is None or not _head_takes_module(lm_head):
            return None
        weight = lm_head.weight
        if (
            tuple(weight.shape) != (workspace.n_out, workspace.k_in)
            or weight.device != workspace.partials.device
            or rows.dim() != 2
            or rows.dtype != torch.bfloat16
            or not 0 < rows.shape[0] <= MAX_ROWS
            or rows.shape[1] != workspace.k_in
        ):
            return None
        if _capturing() and ("lm_head",) not in self._ran:
            return None
        group = lm_head.mapping.tp_group
        if len(group) > 1 and mpi_disabled():
            return None
        x = _dense_rows(rows)
        if not _head_op.supports(x, weight):
            return None
        local = k3_head_gemv(x, weight, workspace)
        if len(group) == 1:
            return local
        gathered = allgather(local, None, group)
        return concat(list(split(gathered, rows.shape[0], dim=0)), dim=-1)

    def dense_mlp(
        self,
        x: torch.Tensor,
        gate_up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        beta: float,
        linear_beta: Optional[float],
    ) -> Optional[torch.Tensor]:
        """Layer 0's dense MLP of at most `MAX_ROWS` rows, before its down projection's all-reduce: gate_up on
        ``gemm/k3_ctm_gemv_long``, ``SituAndMul(beta, linear_beta)`` on ``activation/k3_situ_mul``, down on
        ``gemm/k3_ctm_gemv_long``. None, with nothing launched, where a kernel does not take the call; the caller then
        runs the module."""
        gate_up, down = SITES["dense_gate_up"], SITES["dense_down"]
        rows = x.shape[0] if x.dim() == 2 else 0
        if (
            not 0 < rows <= MAX_ROWS
            or linear_beta == 0.0
            or tuple(gate_up_weight.shape) != (gate_up.n, gate_up.k)
            or tuple(down_weight.shape) != (down.n, down.k)
            or gate_up.n != 2 * down.k
        ):
            return None
        if (
            _capturing()
            and not {
                ("dense_gate_up", "long", 0),
                ("dense_down", "long", 0),
                ("situ_mul", linear_beta is not None),
            }
            <= self._ran
        ):
            return None
        gu = self.project("dense_gate_up", x, gate_up_weight)
        if gu is None or not _ctm_op.supports_situ_mul(gu):
            return None
        return self.project("dense_down", k3_situ_mul(gu, beta, linear_beta), down_weight)

    def embed_norm(
        self,
        input_ids: torch.Tensor,
        table: torch.Tensor,
        norm: nn.Module,
        bank: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Layer 0's normed input for the step's tokens, with their embedding rows written into ``bank[0]``, in one
        ``norm/k3_embed_norm`` launch: bit-identical to the embedding followed by ``norm``. None where it does not
        apply (another norm, a token count or table the kernel does not take, or, under capture, a token count whose
        kernel has not run eagerly)."""
        if not _plain_rmsnorm(norm):
            return None
        ids = input_ids.reshape(-1)
        if not _embed_op.supports_norm(ids, table, norm.weight, bank[0]):
            return None
        key = ("embed_norm", ids.numel(), ids.dtype)
        capturing = _capturing()
        if capturing and key not in self._ran:
            return None
        normed = k3_embed_norm(ids, table, norm.weight, norm.variance_epsilon, bank[0])
        if not capturing:
            self._ran.add(key)
        return normed


class _K3Head:
    """``lm_head(rows)``, with the rows on ``k3_head_gemv`` where it takes them."""

    __slots__ = ("_gemvs", "_lm_head")

    def __init__(self, gemvs: K3DecodeGemvs, lm_head: nn.Module) -> None:
        self._gemvs = gemvs
        self._lm_head = lm_head

    def __call__(self, rows: torch.Tensor) -> torch.Tensor:
        logits = self._gemvs.lm_head_logits(rows, self._lm_head)
        return self._lm_head(rows) if logits is None else logits


class K3LogitsProcessor(nn.Module):
    """The shell's logits processor, with its LM head call on ``gemm/k3_head_gemv`` where ``gemvs`` takes the rows.

    It wraps the stock processor instead of repeating it: the row selection and the fp32 conversion stay the stock
    processor's. ``gemvs`` is set once the weights are final; until then every call is the stock one.
    """

    def __init__(self, stock: nn.Module) -> None:
        super().__init__()
        self.stock = stock
        self.gemvs: Optional[K3DecodeGemvs] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        lm_head: nn.Module,
        attn_metadata,
        return_context_logits: bool = False,
    ) -> torch.Tensor:
        head = lm_head if self.gemvs is None else _K3Head(self.gemvs, lm_head)
        return self.stock.forward(hidden_states, head, attn_metadata, return_context_logits)
