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
``exceptions`` override the rules. The adapters hand their projections to the TP helper at
``prepare()`` (``note_consumer`` / ``note_row_consumer``) so its copy-engine modes can size
their pools and validate the FP8 operands once.

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
from ..attention.attention import Attention, QKVMode
from ..modules.rms_norm import RMSNormTPAware
from .token_sharded_shared_input import SharedTokenShardedInput
from .token_sharded_tp import fp8_block_scale_prequant_ok

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

    Takes this rank's rows (``[n, g, K]`` or ``[m, K]``: bf16, a static-scale NVFP4
    ``Fp4QuantizedTensor`` or an FP8 block-scale ``Fp8BlockScaledActivation``) and returns
    ``[B, S, N_local]`` for all tokens, so code after it must take batch and sequence lengths
    from its output, not from the block input (``Attention._attn_impl`` re-derives them from
    ``q`` / ``k``). A bf16 input is quantized before the all-gather when the Linear takes a
    pre-quantized input it would produce itself (static NVFP4, or FP8 block scales on the
    DeepGEMM path). In the TP's copy-engine gather mode an FP8 block-scale input takes the
    fused gather + GEMM instead (``TokenShardedTP.ce_gather_gemm``), whose output is the
    GEMM's on the gathered rows.
    """

    @classmethod
    def prepare(cls, module: nn.Module, tp: "TokenShardedTP", name: str) -> None:
        _check_linear(module, name, TensorParallelMode.COLUMN, tp)
        if module.gather_output:
            raise ValueError(
                f"token-sharded TP: {name} must be a column-parallel Linear without "
                "gather_output to all-gather its input."
            )
        tp.note_consumer(module)

    def forward(self, input, *args, **kwargs):
        tp = self._token_sharded_tp
        # Linear.forward(input, *, lora_params=None) applies its adapters to the dense input
        # when lora_params is truthy; the argument is keyword-only.
        lora_params = kwargs.get("lora_params")
        if isinstance(input, SharedTokenShardedInput):
            if args:
                raise TypeError("shared column inputs require keyword-only options")
            value = input.for_consumer(self, tp, lora_params)
            out = value if input.outputs else super().forward(value, *args, **kwargs)
        elif tp.uses_ce_gather(self, input, lora_params):
            out = tp.ce_gather_gemm(self, input)
        else:
            gathered = tp.gather_input(self, input, prequantize=not lora_params)
            out = super().forward(gathered, *args, **kwargs)
        p = tp.plan
        return out.view(p.batch_size, p.seq_len, -1)


class TokenShardedRow(TokenShardedAdapter):
    """Row-parallel projection writing the token stream: the GEMM, then a reduce-scatter.

    Takes all tokens (``[B, S, K_local]`` or ``[B * S, K_local]``) and returns this rank's
    reduced rows as ``[n, g, N]``. A padded plan zero-pads the GEMM input per sample (cheaper
    than padding the output partial; zero rows leave a dynamic amax unchanged), so the output's
    pad rows hold only rank 0's bias and are dropped at the next gather. In the TP's
    copy-engine reduce-scatter mode an FP8 block-scale Linear on a bf16 input takes the fused
    GEMM + reduce-scatter instead (``TokenShardedTP.ce_gemm_reduce_scatter``), with the same
    padding and bias placement.
    """

    @classmethod
    def prepare(cls, module: nn.Module, tp: "TokenShardedTP", name: str) -> None:
        _check_linear(module, name, TensorParallelMode.ROW, tp)
        _stop_all_reduce(module, name)
        tp.note_row_consumer(module)

    def forward(self, input, *args, **kwargs):
        tp = self._token_sharded_tp
        # Decided on the padded input, the tensor the module's own GEMM takes (its row count
        # and strides are what the stock prologue sees; a free reshape in an unpadded plan).
        # An active LoRA (Linear.forward's lora_params) needs the module's own forward.
        padded = tp.pad_row_input(input)
        if tp.uses_ce_reduce_scatter(self, padded, kwargs.get("lora_params")):
            return tp.local_view(tp.ce_gemm_reduce_scatter(self, input))
        partial = super().forward(padded, *args, **kwargs)
        return tp.local_view(tp.reduce_scatter(partial))


class TokenShardedMLP(TokenShardedAdapter):
    """MLP whose ``down_proj`` writes the token stream: all-gather, the MLP, reduce-scatter.

    Converted whole rather than per projection because its fused GELU paths call the
    up-projection's quant method directly. The MLP runs on the ``B * S`` real rows only (pad
    rows are non-zero after a norm and would perturb a dynamic amax); its output partial is
    padded per sample before the reduce-scatter. The input is quantized before the all-gather
    by the up-projection's rule, as for :class:`TokenShardedColumn`.

    The two copy-engine modes are independent here as everywhere. In the TP's copy-engine
    gather mode an ``MLP`` whose ``up_proj`` takes an FP8 block-scale input runs the fused
    gather + GEMM for ``up_proj``; in the TP's copy-engine reduce-scatter mode an ``MLP``
    whose ``up_proj`` and ``down_proj`` are both FP8 block-scale runs ``up_proj`` on the
    NCCL-gathered input itself. On either path the adapter then runs ``MLP._act_from_up``
    (the activation step ``MLP.forward`` runs between its two projections -- for an FP8
    block-scale ``up_proj`` ``MLP.forward`` is exactly ``down_proj(_act_from_up(up_proj(x)))``,
    its fused-GELU paths needing an NVFP4 or unquantized ``up_proj``) and owns the
    ``down_proj`` call: the fused GEMM + reduce-scatter (``TokenShardedTP.ce_gemm_reduce_scatter``)
    when the TP takes it, else ``down_proj`` and the NCCL reduce-scatter. Every other case
    (the NCCL modes, an active LoRA, other quantizations, a ``GatedMLP`` with its fused
    gate-up paths) is ``MLP.forward`` / ``GatedMLP.forward`` on the gathered input and the NCCL
    reduce-scatter, byte for byte the NCCL-mode code.

    An explicit shared input bypasses the adapter's gather. A plain MLP retains the same
    down-projection dispatch on the shared up-projection result or gathered activation;
    a GatedMLP runs its activation/down-projection tail and the NCCL reduce-scatter.
    """

    @classmethod
    def prepare(cls, module: nn.Module, tp: "TokenShardedTP", name: str) -> None:
        if not isinstance(module, (MLP, GatedMLP)):
            raise ValueError(f"token-sharded TP: {name} must be an MLP or GatedMLP.")
        up = "up_proj" if hasattr(module, "up_proj") else "gate_up_proj"
        _check_linear(getattr(module, up, None), f"{name}.{up}", TensorParallelMode.COLUMN, tp)
        _check_linear(module.down_proj, f"{name}.down_proj", TensorParallelMode.ROW, tp)
        _stop_all_reduce(module.down_proj, f"{name}.down_proj")
        tp.note_consumer(getattr(module, up))
        if isinstance(module, MLP):
            tp.note_row_consumer(module.down_proj)

    def forward(self, x, *args, **kwargs):
        tp = self._token_sharded_tp
        consumer = getattr(self, "up_proj", None) or getattr(self, "gate_up_proj", None)
        # An active LoRA takes the dense input, so the FP8 pre-quantize (and the fused
        # copy-engine ops) is skipped for exactly the calls the wrapped forward routes to
        # its LoRA path.
        lora_active = self._lora_active(args, kwargs)
        if isinstance(x, SharedTokenShardedInput):
            value = x.for_consumer(self, tp, lora_active)
            if isinstance(self, MLP) and x.outputs:
                x_act = self._act_from_up(value)
            elif isinstance(self, MLP) and self._owns_down_proj_for_ce_reduce_scatter(lora_active):
                x_act = self._act_from_up(self.up_proj(value))
            else:
                if x.outputs:
                    final_params = (
                        args[1] if len(args) >= 2 else kwargs.get("final_all_reduce_params")
                    )
                    partial = self._down_from_gate_up(value, final_params)
                else:
                    partial = super().forward(value, *args, **kwargs)
                return tp.local_view(tp.reduce_scatter(partial))
        elif isinstance(self, MLP) and tp.uses_ce_gather(consumer, x, lora_active):
            x_act = self._ce_up_and_activation(x)
        elif isinstance(self, MLP) and self._owns_down_proj_for_ce_reduce_scatter(lora_active):
            x_act = self._act_from_up(self.up_proj(tp.gather_input(consumer, x)))
        else:
            gathered = tp.gather_input(consumer, x, prequantize=not lora_active)
            partial = super().forward(gathered, *args, **kwargs)
            return tp.local_view(tp.reduce_scatter(partial))
        if tp.uses_ce_reduce_scatter(self.down_proj, x_act, lora_active):
            return tp.local_view(tp.ce_gemm_reduce_scatter(self.down_proj, x_act))
        return tp.local_view(tp.reduce_scatter(self.down_proj(x_act)))

    def _lora_active(self, args: tuple, kwargs: dict) -> bool:
        """Whether the wrapped ``forward`` takes its LoRA path for this call.

        ``MLP.forward(x, lora_params=None)`` does when ``lora_params is not None``;
        ``GatedMLP.forward(x, all_rank_num_tokens=None, final_all_reduce_params=None,
        lora_params=None, **kwargs)`` when it is truthy. Either may be passed positionally.
        """
        if isinstance(self, GatedMLP):
            return bool(kwargs.get("lora_params", args[2] if len(args) > 2 else None))
        return kwargs.get("lora_params", args[0] if args else None) is not None

    def _owns_down_proj_for_ce_reduce_scatter(self, lora_params) -> bool:
        """Whether, on the NCCL-gather path, the adapter runs ``up_proj`` and the activation
        itself so that ``down_proj`` can take the fused copy-engine GEMM + reduce-scatter.

        True iff the TP's reduce-scatter mode is effectively ``"copy_engine"``, no LoRA is
        active and both projections are FP8 block-scale on the DeepGEMM path (then
        ``MLP.forward`` is exactly ``down_proj(_act_from_up(up_proj(x)))``, and ``down_proj``
        is a candidate for the fused op; the final call-time decision is
        ``TokenShardedTP.uses_ce_reduce_scatter`` on the activation output). False keeps
        ``MLP.forward`` whole, so the NCCL-mode path is byte for byte the NCCL-mode code.
        """
        tp = self._token_sharded_tp
        return (
            tp.effective_reduce_scatter_mode == "copy_engine"
            and not lora_params
            and fp8_block_scale_prequant_ok(self.up_proj)
            and fp8_block_scale_prequant_ok(self.down_proj)
        )

    def _ce_up_and_activation(self, x):
        """``down_proj``'s input with the up-projection as the fused copy-engine gather + GEMM.

        The fused GEMM's output goes through ``MLP._act_from_up`` (the activation, or the
        fused ReLU2 + NVFP4 quantize), the same step ``MLP.forward`` runs between its
        ``up_proj`` and ``down_proj`` calls. The fused up-projection + GELU paths of
        ``MLP.forward`` need an NVFP4 or unquantized ``up_proj`` and never apply to an FP8
        block-scale one, the only kind the fused gather takes.
        """
        return self._act_from_up(self._token_sharded_tp.ce_gather_gemm(self.up_proj, x))


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
