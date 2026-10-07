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
"""Token-sharded TP adapters for a model's existing tensor-parallel modules.

:func:`convert_to_token_sharded_tp` converts the blocks of a model built as for plain TP, in
place: a column-parallel projection that reads the token-sharded residual stream all-gathers its
input first (:class:`TokenShardedColumn`), a row-parallel projection that writes it
reduce-scatters instead of all-reducing (:class:`TokenShardedRow`), and an MLP whose
``down_proj`` writes it does both around the whole MLP (:class:`TokenShardedMLP`). The modules
to convert follow from their TP metadata (rules R1-R3 in :func:`classify`); the converter's
``exceptions`` override the rules.

Each adapter is a subclass of the module's own class, swapped in through ``module.__class__``
(as FSDP2's ``fully_shard`` does) and cached per base class: parameters, state-dict names, quant
methods and ``isinstance`` checks are unchanged, the GEMM stays the module's own
(``super().forward``), and all blocks share one compiled graph. Deepcopy and pickle of a
converted model are not supported.
"""

from collections.abc import Mapping
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING

import torch.nn as nn

from tensorrt_llm.logger import logger

from ...distributed.ops import AllReduce
from ...modules.gated_mlp import GatedMLP
from ...modules.linear import Linear, TensorParallelMode
from ...modules.mlp import MLP
from ..modules.attention import Attention, QKVMode
from ..modules.rms_norm import RMSNormTPAware

if TYPE_CHECKING:
    from .token_sharded_tp import TokenShardedTP

__all__ = [
    "COLUMN",
    "KEEP",
    "MLP_KIND",
    "ROW",
    "TokenShardedAdapter",
    "TokenShardedColumn",
    "TokenShardedMLP",
    "TokenShardedRow",
    "classify",
    "convert_to_token_sharded_tp",
]

COLUMN, ROW, MLP_KIND, KEEP = "column", "row", "mlp", "keep"


# =============================================================================
# Adapters
# =============================================================================


class TokenShardedAdapter:
    """Base of the token-sharded adapters; ``_token_sharded_tp`` is set by the converter.

    A module the rules cannot read (e.g. a joint projection with its own all-reduce) converts
    by naming a subclass, whose ``forward`` wraps the module's, in the converter's
    ``exceptions``. The rules still classify its submodules; name the ones its ``forward``
    handles as ``"keep"``.
    """

    _token_sharded_tp: "TokenShardedTP"

    @classmethod
    def prepare(cls, module: nn.Module, tp: "TokenShardedTP", name: str) -> None:
        """Check ``module`` and set it up for this adapter, before its class is swapped.

        Raise ``ValueError`` if the module was not built for this adapter; an adapter whose
        module all-reduces stops it here (its ``forward`` reduce-scatters instead).
        """


class TokenShardedColumn(TokenShardedAdapter):
    """Column-parallel projection reading the token stream: all-gather, then the GEMM.

    Takes this rank's rows (``[n, g, K]`` or ``[m, K]``, bf16 or a static-scale NVFP4
    ``Fp4QuantizedTensor``) and returns ``[B, S, N_local]`` for all tokens, so code after it
    must take batch and sequence lengths from its output, not from the block input
    (``Attention._attn_impl`` re-derives them from ``q`` / ``k``).
    """

    @classmethod
    def prepare(cls, module: nn.Module, tp: "TokenShardedTP", name: str) -> None:
        _check_linear(module, name, TensorParallelMode.COLUMN, tp)
        if module.gather_output:
            raise ValueError(
                f"token-sharded TP: {name} must be a column-parallel Linear without "
                "gather_output to all-gather its input."
            )

    def forward(self, input, *args, **kwargs):
        tp = self._token_sharded_tp
        out = super().forward(tp.gather_input(self, input), *args, **kwargs)
        p = tp.plan
        return out.view(p.batch_size, p.seq_len, -1)


class TokenShardedRow(TokenShardedAdapter):
    """Row-parallel projection writing the token stream: the GEMM, then a reduce-scatter.

    Takes all tokens (``[B, S, K_local]`` or ``[B * S, K_local]``) and returns this rank's
    reduced rows as ``[n, g, N]``. A padded plan zero-pads the GEMM input per sample (cheaper
    than padding the output partial; zero rows leave a dynamic amax unchanged), so the output's
    pad rows hold only rank 0's bias and are dropped at the next gather.
    """

    @classmethod
    def prepare(cls, module: nn.Module, tp: "TokenShardedTP", name: str) -> None:
        _check_linear(module, name, TensorParallelMode.ROW, tp)
        _stop_all_reduce(module, name)

    def forward(self, input, *args, **kwargs):
        tp = self._token_sharded_tp
        partial = super().forward(tp.pad_row_input(input), *args, **kwargs)
        return tp.local_view(tp.reduce_scatter(partial))


class TokenShardedMLP(TokenShardedAdapter):
    """MLP whose ``down_proj`` writes the token stream: all-gather, the MLP, reduce-scatter.

    Converted whole rather than per projection because its fused GELU paths call the
    up-projection's quant method directly. The MLP runs on the ``B * S`` real rows only (pad
    rows are non-zero after a norm and would perturb a dynamic amax); its output partial is
    padded per sample before the reduce-scatter.
    """

    @classmethod
    def prepare(cls, module: nn.Module, tp: "TokenShardedTP", name: str) -> None:
        if not isinstance(module, (MLP, GatedMLP)):
            raise ValueError(f"token-sharded TP: {name} must be an MLP or GatedMLP.")
        up = "up_proj" if hasattr(module, "up_proj") else "gate_up_proj"
        _check_linear(getattr(module, up, None), f"{name}.{up}", TensorParallelMode.COLUMN, tp)
        _check_linear(module.down_proj, f"{name}.down_proj", TensorParallelMode.ROW, tp)
        _stop_all_reduce(module.down_proj, f"{name}.down_proj")

    def forward(self, x, *args, **kwargs):
        tp = self._token_sharded_tp
        consumer = getattr(self, "up_proj", None) or getattr(self, "gate_up_proj", None)
        partial = super().forward(tp.gather_input(consumer, x), *args, **kwargs)
        return tp.local_view(tp.reduce_scatter(partial))


_KIND_ADAPTERS = {COLUMN: TokenShardedColumn, ROW: TokenShardedRow, MLP_KIND: TokenShardedMLP}


def _check_linear(module: nn.Module | None, name: str, mode: TensorParallelMode, tp) -> None:
    kind = "column" if mode == TensorParallelMode.COLUMN else "row"
    if not isinstance(module, Linear) or module.tp_mode != mode:
        raise ValueError(f"token-sharded TP: {name} must be a {kind}-parallel Linear.")
    if module.tp_size != tp.tp_size:
        raise ValueError(
            f"token-sharded TP: {name} is sharded for tp_size={module.tp_size}, but the TP "
            f"group has {tp.tp_size} ranks."
        )


def _stop_all_reduce(linear: Linear, name: str) -> None:
    """Leave ``linear`` in the state ``reduce_output=False`` builds.

    The quant methods' ``apply()`` paths key on ``all_reduce`` too (NCCL-window output
    buffers, bias-in-GEMM).
    """
    if linear.use_fused_gemm_allreduce:
        # create_weights() already chose the quant method and workspace for the fused op.
        raise ValueError(
            f"token-sharded TP: {name} was built for the fused GEMM + all-reduce "
            "(use_fused_gemm_allreduce), which cannot be undone after construction."
        )
    linear.reduce_output = False
    linear.all_reduce = None


_ADAPTED: dict[tuple[type, type], type] = {}


def _adapted_class(adapter: type, base: type) -> type:
    key = (adapter, base)
    cls = _ADAPTED.get(key)
    if cls is None:
        name = f"{adapter.__name__}{base.__name__}"
        cls = type(name, (adapter, base), {"__module__": base.__module__, "__qualname__": name})
        _ADAPTED[key] = cls
    return cls


# =============================================================================
# Rules
# =============================================================================


def _all_reduces(module: nn.Module | None) -> bool:
    return (
        isinstance(module, Linear)
        and module.tp_mode == TensorParallelMode.ROW
        and bool(module.reduce_output)
    )


def _stream_projections(attn: Attention) -> list[str]:
    """Attention's projections that read the hidden states (the token stream)."""
    if attn.qkv_mode == QKVMode.FUSE_QKV:
        return ["qkv_proj"]
    if getattr(attn, "separate_qkv_is_self_attention", False):
        return ["to_q", "to_k", "to_v"]
    return ["to_q"]  # cross-attention: k / v read the replicated encoder states


def _classify_block(block: nn.Module, exceptions: Mapping[str, object]) -> dict[str, object]:
    """``{relative module name: kind or adapter class}`` for one block."""
    kinds: dict[str, object] = {}
    covered: list[str] = []  # modules converted whole; their submodules are skipped

    def inside(name: str) -> bool:
        return any(name.startswith(prefix + ".") for prefix in covered)

    for name, m in block.named_modules():
        if not name or inside(name):
            continue
        if isinstance(m, (MLP, GatedMLP)) and _all_reduces(m.down_proj):
            kinds[name] = MLP_KIND
            covered.append(name)
        elif _all_reduces(m):
            kinds[name] = ROW
        elif isinstance(m, Attention):
            for proj in _stream_projections(m):
                kinds[f"{name}.{proj}"] = COLUMN
    for pattern, kind in exceptions.items():
        if kind not in (COLUMN, ROW, MLP_KIND, KEEP) and not (
            isinstance(kind, type) and issubclass(kind, TokenShardedAdapter)
        ):
            raise ValueError(
                f"token-sharded TP exception {pattern!r}: {kind!r} is not 'column', 'row', "
                "'mlp', 'keep' or a TokenShardedAdapter subclass."
            )
        names = [name for name, _ in block.named_modules() if name and fnmatchcase(name, pattern)]
        if not names:
            raise ValueError(f"token-sharded TP exception {pattern!r} matches no module.")
        for name in names:
            kinds[name] = kind
    return {name: kind for name, kind in kinds.items() if kind != KEEP}


def _check_block(
    block: nn.Module, where: str, kinds: Mapping[str, object], exceptions: Mapping[str, object]
) -> None:
    if not any(kind != COLUMN for kind in kinds.values()):
        raise ValueError(
            f"token-sharded TP: {where} has no all-reducing row-parallel projection or MLP to "
            "convert; is the model built with tp_size > 1?"
        )
    kept = [pattern for pattern, kind in exceptions.items() if kind == KEEP]
    for name, m in block.named_modules():
        if not isinstance(m, AllReduce):
            continue
        owner_name = name.rpartition(".")[0]
        owner = block.get_submodule(owner_name) if owner_name else block
        handled = (
            isinstance(owner, RMSNormTPAware)  # reduces over heads within each token
            or (isinstance(owner, Linear) and owner.tp_mode != TensorParallelMode.ROW)  # dormant
            or any(owner_name == k or owner_name.startswith(k + ".") for k in kinds)
            or any(fnmatchcase(owner_name, p) or fnmatchcase(name, p) for p in kept)
        )
        if not handled:
            raise ValueError(
                f"token-sharded TP: {where}.{name} all-reduces outside the projections the "
                "rules convert; under this layout it would reduce a token shard. Name it in the "
                "converter's exceptions."
            )


def classify(
    root: nn.Module, *, exceptions: Mapping[str, object] | None = None
) -> dict[str, object]:
    """Which modules of ``root.blocks`` convert, without converting them.

    Rules, from the TP metadata the modules already carry:

    * R1: a row-parallel ``Linear`` that all-reduces (``reduce_output``) becomes a
      reduce-scatter (``"row"``);
    * R2: an ``MLP`` / ``GatedMLP`` whose ``down_proj`` all-reduces converts whole (``"mlp"``);
    * R3: a VisualGen ``Attention``'s projections that read the hidden states all-gather
      first (``"column"``): ``qkv_proj``, ``to_q``, and ``to_k`` / ``to_v`` when
      ``separate_qkv_is_self_attention``.

    ``exceptions`` (``{block-relative module-name pattern: kind, adapter class or "keep"}``)
    override the rules.

    Raises ``ValueError`` when a block has nothing to convert, when an all-reduce would act
    on a token shard (anything but a converted projection's own, a TP-aware RMSNorm's
    per-token head reduction, or a column ``Linear``'s unused one), when an exception names an
    unknown kind or matches no module, or when two blocks would convert differently.

    Returns:
        ``{qualified module name: kind or adapter class}``.
    """
    exceptions = dict(exceptions or {})
    result: dict[str, object] = {}
    first = None
    for index, block in root.blocks.named_children():
        where = f"blocks.{index}"
        kinds = _classify_block(block, exceptions)
        _check_block(block, where, kinds, exceptions)
        if first is None:
            first = (where, kinds)
        elif kinds != first[1]:
            raise ValueError(
                f"token-sharded TP: {where} and {first[0]} convert differently "
                f"({sorted(kinds)} vs {sorted(first[1])}); the blocks share one compiled graph "
                "and one conversion."
            )
        result.update({f"{where}.{name}": kind for name, kind in kinds.items()})
    return result


# =============================================================================
# Conversion
# =============================================================================


def _convert(module: nn.Module, kind: object, tp: "TokenShardedTP", name: str) -> None:
    adapter = _KIND_ADAPTERS.get(kind, kind)
    adapter.prepare(module, tp, name)
    module.__class__ = _adapted_class(adapter, type(module))
    module._token_sharded_tp = tp


def convert_to_token_sharded_tp(
    root: nn.Module,
    tp: "TokenShardedTP",
    *,
    exceptions: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Convert ``root.blocks`` in place (see :func:`classify`); returns what converted."""
    for name, m in root.blocks.named_modules():
        if isinstance(m, TokenShardedAdapter):
            raise ValueError(f"token-sharded TP: blocks.{name} is already token-sharded.")
    plan = classify(root, exceptions=exceptions)
    for name, kind in plan.items():
        _convert(root.get_submodule(name), kind, tp, name)
    counts: dict[str, int] = {}
    for kind in plan.values():
        label = kind if isinstance(kind, str) else kind.__name__
        counts[label] = counts.get(label, 0) + 1
    logger.info(
        f"Token-sharded TP: converted {type(root).__name__} "
        f"({', '.join(f'{n} {k}' for k, n in sorted(counts.items()))})."
    )
    return plan
