# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""This target's changes to the DFlash drafter's GEMMs and to the DFlash worker (``drafter_fc_shard``,
``fused_accept``, ``drafter_fixed_gemms``).

The drafter (``llm.draft_model``, a built-in ``DFlashForCausalLM``) and the
worker (``llm.spec_worker``, a built-in ``DFlashWorker``) are built by the
built-in speculative causal LM; the drafter's ``fc``, ``hidden_norm`` and
DFlash 2 convolutions exist only once its checkpoint has loaded. The shell
therefore calls ``install`` from ``load_draft_weights``, after the built-in
load returns and before any forward is captured. Every install changes
methods or re-classes modules in place; the module tree, the parameters and
their names stay the built-in ones, and nothing changes the drafter's
``dflash_forward`` (which another fast path may replace).

Each fast path reproduces a few lines of a built-in method:
``DFlashForCausalLM.project_target_hidden`` (``drafter_fc_shard``),
``DFlashWorker._sample_and_accept_draft_tokens_base`` and
``DFlashWorker._apply_force_accepted_tokens`` (``fused_accept``), and
``DFlashForCausalLM.precompute_context_kv`` (``drafter_fixed_gemms``). Installs check that the
class still runs those built-ins (no subclass overrides them) and that no
other fast path has already replaced the same method on the instance; a fast
path that finds either declines, leaving the built-in in place, and ``install``
raises with the reason: a selected fast path never runs as the built-in under
its name. The shell calls
``install`` from every load hook, so ``install`` logs a fast path only on the
call that changed something.

``drafter_fc_shard`` -- sharded ``fc``. ``project_target_hidden`` (``fc`` then
``hidden_norm``) runs on the rows of the five tapped target layers,
``[rows, 5 * 8192]``, which every rank holds: the ``K + 1`` rows per request of
a decode step, and every token of a prefill chunk. The built-in multiplies
them by the whole replicated ``[8192, 40960]`` BF16 weight on every rank
(671 MB per rank per call). Here each of the TP ranks multiplies by its own
``8192 / TP`` output rows of that weight (a view of the current ``fc.weight``
taken on every call, no copy) and the slices are gathered in rank order;
then the built-in ``hidden_norm``. The gather depends on the TP width:

* TP8 (two nodes of four): the eight ``[rows, 1024]`` slices go over the
  NVLink fabric with the built-in projection-gather kernel
  (``musespark_open_projection_gather.py``) on a workspace of its own. It is
  written for decode steps at batch 1, so calls of more than
  ``FC_SHARD_MAX_ROWS`` rows run the built-in method.
* TP4 (one node of four): the four ``[rows, 2048]`` slices are joined by the
  built-in NCCL all-gather (``tensorrt_llm._torch.distributed.allgather``),
  which takes any row count, so every call is sharded.

The rank's GEMM runs on the fixed low-M tactic listed for its shape
(``ops/gemm/fixed_low_m_gemm.py``) whether or not ``drafter_fixed_gemms`` is
selected, and on ``F.linear`` for the shapes no tactic lists. Each output
element is still one full-K dot product accumulated in fp32 and rounded
once, but possibly in another kernel's K order, so it can differ from the
built-in in the last bf16 place. Only the draft tokens can change: the
target model verifies every draft.

``fused_accept`` -- fused acceptance. The worker's
``_sample_and_accept_draft_tokens_base`` keeps the built-in target-token
sampler and replaces the torch chain behind it (strict acceptance plus the
``TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`` override) with
``ops/spec/dflash_accept.py``: one launch instead of about twenty, bitwise
the same tokens and counts. It composes with another fast path's ``DFlashWorker``
subclass as long as that subclass leaves the acceptance methods alone.

``drafter_fixed_gemms`` -- the drafter's small GEMMs on fixed low-M tactics
(``ops/gemm/fixed_low_m_gemm.py``): the ten replicated convolution
coefficient projections (plain ``nn.Linear``, cuBLAS split-K plus a reduce
kernel), the fused context K/V projection (``F.linear``), and the TP-sharded
``qkv_proj`` / ``o_proj`` / ``gate_up_proj`` / ``down_proj`` (whose per-rank
autotuner choices differ between ranks). Same tactic on every rank; drafter
numerics only. The install reports every candidate call site it left
built-in, and why.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch.models.modeling_dflash import DFlash2BlockConv, DFlashForCausalLM
from tensorrt_llm._torch.modules.linear import Linear, UnquantizedLinearMethod
from tensorrt_llm._torch.speculative import interface as _spec_interface
from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm.logger import logger

from .dispatch import LAUNCH_PDL
from .ops.gemm.fixed_low_m_gemm import fixed_low_m_gemm, splitk_tactics, tactic_for, takes
from .ops.spec.dflash_accept import (
    COUNTER_STRIDE,
    FORCE_FIXED,
    FORCE_FRACTIONAL,
    FORCE_NONE,
    MAX_REQUESTS,
    SLOT_STRIDE,
    dflash_accept,
)

#: The sharded ``fc`` path's row bound at TP8, where the fabric projection
#: gather serves decode steps at batch 1 (K + 1 <= 8 rows); larger calls
#: (prefill chunks, batched decode) run the built-in. TP4 has no row bound.
FC_SHARD_MAX_ROWS = 8

#: The TP widths the sharded ``fc`` is written for: 8 gathers over the fabric
#: projection-gather workspace, 4 with the built-in NCCL all-gather.
FC_SHARD_WORLDS = (4, 8)

#: Output rows of ``fc`` per rank that the TP8 projection gather takes.
FC_SHARD_TP8_ROWS = 1024


def _gemm(
    x: torch.Tensor, weight: torch.Tensor, enable_pdl: bool, tactics: Optional[dict] = None
) -> torch.Tensor:
    """``x @ weight.T`` on the fixed low-M tactic ``tactics`` lists for the shape, else ``F.linear``.

    ``tactics`` is the table of the drafter's TP size (``splitk_tactics``); the
    device's TP8 one when None.
    """
    if takes(x, weight):
        tactic = tactic_for(x.shape[0], x.shape[1], weight.shape[0], tactics)
        if tactic is not None:
            return fixed_low_m_gemm(x, weight, tactic, enable_pdl)
    return F.linear(x, weight)


def _listed(in_features: int, out_features: int, tactics: dict) -> bool:
    """Whether ``tactics`` lists a tactic for a ``[out_features, in_features]`` weight."""
    return any((k, n) == (in_features, out_features) for _, k, n in tactics)


def _bound_to(bound, function: Callable) -> bool:
    """Whether ``bound`` (an instance attribute, or None) is ``function`` bound as a method."""
    return isinstance(bound, types.MethodType) and bound.__func__ is function


def _describe(bound) -> str:
    function = getattr(bound, "__func__", bound)
    module = getattr(function, "__module__", None) or "?"
    return f"{module}.{getattr(function, '__qualname__', type(function).__name__)}"


# --------------------------------------------------------------------------
# drafter_fc_shard: sharded fc
# --------------------------------------------------------------------------
@dataclass
class ShardedFc:
    """What the sharded ``fc`` path holds: this rank's block of ``fc`` output rows and its gather.

    The weight rows themselves are taken from the drafter's ``fc`` on every
    call, as the built-in reads ``self.fc``: the DFlash loader builds a new
    ``fc`` on each load, so a stored view would outlive a reload.
    """

    rank: int
    rows: int
    gather: object
    enable_pdl: bool
    #: TP width the ``fc`` is sharded over.
    world: int = 8
    #: Largest call the gather takes, or None for any row count.
    max_rows: Optional[int] = FC_SHARD_MAX_ROWS


class AllGatherColumns:
    """The TP4 gather of the sharded ``fc``: the built-in NCCL all-gather, slices joined by columns.

    Takes any row count. Every rank of the TP group makes the matching call
    with the same row count, as for every other collective of the step.
    """

    def __init__(self, mapping) -> None:
        self.mapping = mapping

    def gather(self, local: torch.Tensor) -> torch.Tensor:
        from tensorrt_llm._torch.distributed import allgather

        return allgather(local, self.mapping, dim=-1)


def sharded_project_target_hidden(
    self: DFlashForCausalLM, hidden_states: torch.Tensor
) -> torch.Tensor:
    """The built-in ``project_target_hidden`` with ``fc`` column-sharded over the TP group.

    Bound to the drafter as ``project_target_hidden`` by ``install_sharded_fc``.
    Calls of more rows than the gather takes (``ShardedFc.max_rows``), or
    operands it does not take, run the built-in method. Every rank sees the
    same row count and holds the same ``fc`` shapes, so every rank takes the
    same branch and the gather's call order stays identical across ranks.
    """
    state: Optional[ShardedFc] = self.__dict__.get("_fc_shard")
    weight = self.fc.weight
    x = hidden_states.to(weight.dtype)
    if (
        state is None
        or self.fc.bias is not None
        or x.dim() != 2
        or x.shape[0] < 1
        or (state.max_rows is not None and x.shape[0] > state.max_rows)
        or x.dtype != torch.bfloat16
        or not weight.is_contiguous()
        or tuple(weight.shape) != (state.rows * state.world, x.shape[1])
    ):
        return DFlashForCausalLM.project_target_hidden(self, hidden_states)
    begin = state.rank * state.rows
    shard = weight.detach()[begin : begin + state.rows]
    local = _gemm(x.contiguous(), shard, state.enable_pdl)
    return self.hidden_norm(state.gather.gather(local))


def sharded_fc_unsupported_reason(draft_model: nn.Module, mapping) -> Optional[str]:
    """Why the sharded ``fc`` cannot stand in for this drafter's projection, or None."""
    if not isinstance(draft_model, DFlashForCausalLM):
        return f"not a DFlashForCausalLM ({type(draft_model).__name__})"
    if type(draft_model).project_target_hidden is not DFlashForCausalLM.project_target_hidden:
        return f"{type(draft_model).__name__} overrides project_target_hidden"
    fc = getattr(draft_model, "fc", None)
    if (
        not isinstance(fc, nn.Linear)
        or fc.bias is not None
        or not hasattr(draft_model, "hidden_norm")
    ):
        return "the drafter has no bias-free fc + hidden_norm"
    if fc.weight.dtype != torch.bfloat16 or not fc.weight.is_contiguous():
        return f"fc.weight is {fc.weight.dtype}, contiguous={fc.weight.is_contiguous()}"
    world = getattr(mapping, "tp_size", None)
    if mapping is None or world not in FC_SHARD_WORLDS:
        widths = " or ".join(map(str, FC_SHARD_WORLDS))
        return f"TP size is {world}, the gathers are written for {widths}"
    if getattr(mapping, "enable_attention_dp", False) or getattr(mapping, "cp_size", 1) != 1:
        return "attention DP or context parallelism"
    if fc.weight.shape[0] % world:
        return f"fc has {fc.weight.shape[0]} outputs, not a multiple of TP {world}"
    if world == 8 and fc.weight.shape[0] // world != FC_SHARD_TP8_ROWS:
        return (
            f"fc has {fc.weight.shape[0]} outputs; "
            f"the TP8 gather takes {FC_SHARD_TP8_ROWS} per rank"
        )
    return None


def sharded_fc_installed(draft_model: nn.Module) -> bool:
    """Whether ``install_sharded_fc`` has bound the sharded path, with its state, on ``draft_model``."""
    return _bound_to(
        draft_model.__dict__.get("project_target_hidden"), sharded_project_target_hidden
    ) and isinstance(draft_model.__dict__.get("_fc_shard"), ShardedFc)


def install_sharded_fc(
    draft_model: nn.Module, mapping, enable_pdl: bool, gather=None
) -> Optional[str]:
    """Bind ``sharded_project_target_hidden`` on the drafter; return None when installed, else why not.

    ``gather`` is the projection-gather workspace to use; by default one is
    built here (a TP-collective allocation: every rank must call this).
    Idempotent: a repeated install (the shell installs after every drafter
    load) keeps the bound method and its workspace, and the next call reads
    the newly loaded ``fc``. If a reload left a drafter this path does not
    cover, the built-in method is put back. Declines when another feature has
    already replaced ``project_target_hidden`` on the instance.
    """
    reason = sharded_fc_unsupported_reason(draft_model, mapping)
    bound = draft_model.__dict__.get("project_target_hidden")
    ours = _bound_to(bound, sharded_project_target_hidden)
    if reason is not None:
        if ours:
            del draft_model.project_target_hidden
            draft_model.__dict__.pop("_fc_shard", None)
        return reason
    if bound is not None and not ours:
        return f"project_target_hidden is already replaced on this drafter by {_describe(bound)}"
    state: Optional[ShardedFc] = draft_model.__dict__.get("_fc_shard")
    if ours and state is not None:
        state.enable_pdl = enable_pdl
        return None
    world = mapping.tp_size
    if gather is None and world == 8:
        from tensorrt_llm._torch.models.musespark_open_projection_gather import (
            ProjectionGatherWorkspace,
        )

        try:
            gather = ProjectionGatherWorkspace(mapping)
        except ValueError as exc:  # raised alike on every rank (it all-gathers the check)
            return f"projection gather unavailable: {exc}"
    elif gather is None:
        gather = AllGatherColumns(mapping)
    draft_model._fc_shard = ShardedFc(
        rank=mapping.tp_rank,
        rows=draft_model.fc.weight.shape[0] // world,
        gather=gather,
        enable_pdl=enable_pdl,
        world=world,
        max_rows=FC_SHARD_MAX_ROWS if world == 8 else None,
    )
    draft_model.project_target_hidden = types.MethodType(sharded_project_target_hidden, draft_model)
    return None


# --------------------------------------------------------------------------
# fused_accept: fused acceptance
# --------------------------------------------------------------------------
class FusedAcceptDFlashWorker(DFlashWorker):
    """The built-in ``DFlashWorker`` with strict acceptance + forced override in one launch.

    Never constructed: ``install_fused_accept`` re-classes the worker, as this
    class or as a class derived from this one and the worker's own
    ``DFlashWorker`` subclass. Only ``_sample_and_accept_draft_tokens_base``
    changes, and only for uniform verify blocks of at most ``MAX_REQUESTS``
    requests; anything else runs the built-in method.
    """

    def _force_plan(self, runtime_draft_len: int, spec_metadata) -> tuple[int, int, int, float]:
        """The built-in ``_apply_force_accepted_tokens`` branches, as (mode, base, max, frac)."""
        rate = self.force_num_accepted_tokens
        if rate == 0.0:
            return FORCE_NONE, 0, 0, 0.0
        if spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing():
            return FORCE_NONE, 0, 0, 0.0  # eager warmup: no override
        int_part = int(rate)
        frac_part = rate - int_part
        max_total = runtime_draft_len + 1
        base_total = min(int_part + 1, max_total)
        if frac_part > 0.0 and base_total < max_total:
            return FORCE_FRACTIONAL, base_total, max_total, frac_part
        return FORCE_FIXED, base_total, max_total, 0.0

    def _sample_and_accept_draft_tokens_base(
        self, logits, draft_tokens, num_contexts, batch_size, spec_metadata
    ):
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        num_gens = batch_size - num_contexts
        if (
            spec_metadata is None
            or spec_metadata.is_ragged_verify
            or not logits.is_cuda
            or draft_tokens.dim() != 2
            or draft_tokens.shape[0] != num_gens
            or not 1 <= draft_tokens.shape[1] <= 63
            or draft_tokens.stride(-1) != 1
            or draft_tokens.dtype.is_floating_point
            or batch_size > MAX_REQUESTS
            or logits.shape[0] != num_contexts + num_gens * (draft_tokens.shape[1] + 1)
        ):
            return super()._sample_and_accept_draft_tokens_base(
                logits, draft_tokens, num_contexts, batch_size, spec_metadata
            )
        runtime_draft_len = draft_tokens.shape[1]
        target_tokens = self._sample_tokens_for_batch(
            logits, spec_metadata, num_contexts, batch_size
        )
        mode, base_total, max_total, frac = self._force_plan(runtime_draft_len, spec_metadata)
        pool = counter = None
        if mode == FORCE_FRACTIONAL:
            self._ensure_force_accept_rng_state(logits.device)
            pool, counter = self._force_accept_rng_pool, self._force_accept_rng_counter
        return dflash_accept(
            target_tokens.reshape(-1),
            draft_tokens,
            num_contexts,
            mode,
            base_total,
            max_total,
            frac,
            pool,
            counter,
        )


#: Worker methods ``FusedAcceptDFlashWorker`` reproduces or relies on; a worker
#: class that overrides any of them is left alone.
ACCEPT_METHODS = (
    "_sample_and_accept_draft_tokens_base",
    "_apply_force_accepted_tokens",
    "_ensure_force_accept_rng_state",
)

#: ``FusedAcceptDFlashWorker`` combined with another feature's worker class, by that class.
_COMPOSED_WORKERS: dict[type, type] = {}


def fused_accept_unsupported_reason(spec_worker) -> Optional[str]:
    """Why ``FusedAcceptDFlashWorker`` cannot stand in for this worker's acceptance, or None."""
    if not isinstance(spec_worker, DFlashWorker):
        return f"the spec worker is {type(spec_worker).__name__}, not a DFlashWorker"
    cls = type(spec_worker)
    for name in ACCEPT_METHODS:
        if getattr(cls, name) is not getattr(DFlashWorker, name):
            return f"the spec worker class {cls.__name__} overrides {name}"
        if name in vars(spec_worker):
            return (
                f"{name} is already replaced on this worker by {_describe(vars(spec_worker)[name])}"
            )
    builtin = (
        _spec_interface._FORCE_ACCEPT_RNG_COUNTER_STRIDE,
        _spec_interface._FORCE_ACCEPT_RNG_SLOT_STRIDE,
    )
    if builtin != (COUNTER_STRIDE, SLOT_STRIDE):
        return f"the built-in forced-acceptance strides are {builtin}, not {(COUNTER_STRIDE, SLOT_STRIDE)}"
    return None


def install_fused_accept(spec_worker) -> Optional[str]:
    """Re-class the DFlash worker to run ``FusedAcceptDFlashWorker``'s acceptance; None when installed.

    A built-in ``DFlashWorker`` becomes a ``FusedAcceptDFlashWorker``. A
    worker another feature has already re-classed to its own ``DFlashWorker``
    subclass becomes a class derived from both (this one first), provided
    that subclass overrides none of ``ACCEPT_METHODS``. Idempotent.
    """
    if isinstance(spec_worker, FusedAcceptDFlashWorker):
        return None
    reason = fused_accept_unsupported_reason(spec_worker)
    if reason is not None:
        return reason
    cls = type(spec_worker)
    if cls is DFlashWorker:
        spec_worker.__class__ = FusedAcceptDFlashWorker
        return None
    composed = _COMPOSED_WORKERS.get(cls)
    if composed is None:
        composed = type(
            f"FusedAccept{cls.__name__}",
            (FusedAcceptDFlashWorker, cls),
            {
                "__module__": __name__,
                "__doc__": f"``{cls.__qualname__}`` with ``FusedAcceptDFlashWorker``'s acceptance.",
            },
        )
        _COMPOSED_WORKERS[cls] = composed
    spec_worker.__class__ = composed
    return None


# --------------------------------------------------------------------------
# drafter_fixed_gemms: fixed-tactic drafter GEMMs
# --------------------------------------------------------------------------
class FixedTacticProjection(nn.Linear):
    """A plain ``nn.Linear`` (no bias) whose listed shapes run ``fixed_low_m_gemm``.

    Never constructed: ``install_fixed_tactic_gemms`` re-classes the
    convolutions' built-in ``kernel_projection``. ``_fixed_gemm_pdl`` and
    ``_fixed_gemm_tactics`` (the drafter TP size's table) are set then.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.bias is None and input.dim() == 2:
            return _gemm(input, self.weight, self._fixed_gemm_pdl, self._fixed_gemm_tactics)
        return super().forward(input)


class FixedTacticLinearMethod(UnquantizedLinearMethod):
    """The built-in unquantized linear method; bias-free listed shapes run ``fixed_low_m_gemm``.

    Installed per module (a fresh instance each) on the drafter's BF16 TP
    ``Linear`` modules. The fixed GEMM takes the slot of the built-in's low-M
    dispatcher, and like it stays out of the way of a module on the custom
    cuBLAS GEMM (whose output may have to live in an NCCL symmetric-memory
    window). Every other path (weights, bias, unlisted shapes, more than 32
    rows) is the built-in method's.
    """

    def __init__(self, enable_pdl: bool, tactics: Optional[dict] = None) -> None:
        super().__init__()
        self.enable_pdl = enable_pdl
        #: The tactic table of the drafter's TP size (``splitk_tactics``; the device's
        #: TP8 one when None).
        self.tactics = splitk_tactics() if tactics is None else tactics

    def tactic_for(self, m: int, k: int, n: int):
        """The fixed tactic this method runs a ``[m, k] x [n, k]^T`` call on, or None."""
        return tactic_for(m, k, n, self.tactics)

    def apply(self, module, input, bias):
        if (
            bias is None
            and isinstance(input, torch.Tensor)
            and input.dim() >= 2
            and not getattr(module, "use_custom_cublas_mm", False)
        ):
            x = input.reshape(-1, input.shape[-1])
            if takes(x, module.weight):
                tactic = self.tactic_for(x.shape[0], x.shape[1], module.weight.shape[0])
                if tactic is not None:
                    out = fixed_low_m_gemm(x, module.weight, tactic, self.enable_pdl)
                    return out.view(*input.shape[:-1], module.weight.shape[0])
        return super().apply(module, input, bias)


def fixed_tactic_precompute_context_kv(
    self: DFlashForCausalLM, projected_hidden: torch.Tensor, positions: torch.Tensor
):
    """The built-in ``precompute_context_kv`` with its fused K/V GEMM on the fixed tactic.

    Bound to the drafter as ``precompute_context_kv`` by
    ``install_fixed_tactic_gemms``; written out line for line from the
    built-in ``DFlashForCausalLM.precompute_context_kv``, the GEMM call being
    the one change.
    """
    if self._fused_kv_weight is None:
        self._build_fused_kv_buffers()
    if self._fused_kv_bias is not None:
        return DFlashForCausalLM.precompute_context_kv(self, projected_hidden, positions)
    N = projected_hidden.shape[0]
    L = self._num_attn_layers
    nkv = self._num_kv_heads
    hd = self._head_dim
    weight_dtype = self._fused_kv_weight.dtype
    if getattr(self, "_input_ln_eps", None) is not None:
        ph = projected_hidden.float()
        ph = ph * torch.rsqrt(ph.pow(2).mean(-1, keepdim=True) + self._input_ln_eps)
        projected_hidden = ph.to(weight_dtype)
    elif projected_hidden.dtype != weight_dtype:
        projected_hidden = projected_hidden.to(weight_dtype)

    kv_flat = _gemm(
        projected_hidden, self._fused_kv_weight, self._fixed_gemm_pdl, self._fixed_gemm_tactics
    )
    kv = kv_flat.view(N, L, 2, nkv, hd)
    k = kv[:, :, 0].contiguous()
    v = kv[:, :, 1].contiguous()

    if self._k_norm_stacked is not None:
        k = F.rms_norm(k, (hd,), eps=self._k_norm_eps)
        k = k * self._k_norm_stacked.view(1, L, 1, hd)

    self._fused_rope_inplace(k.view(N * L, nkv * hd), positions, N, L)
    return k, v


@dataclass(frozen=True)
class FixedTacticSites:
    """What ``install_fixed_tactic_gemms`` found: call sites on fixed tactics, and why the rest are not.

    ``changed`` counts the sites this call moved onto a fixed tactic (0 on a
    repeated install); it is left out of the comparison, which describes the
    drafter's state.
    """

    installed: int
    declined: tuple[str, ...] = ()
    changed: int = field(default=0, compare=False)


def _projection_reason(proj: nn.Module, tactics: dict) -> Optional[str]:
    if type(proj) not in (nn.Linear, FixedTacticProjection):
        return f"is {type(proj).__name__}, not a plain nn.Linear"
    if proj.bias is not None:
        return "has a bias"
    if proj.weight.dtype != torch.bfloat16:
        return f"weight is {proj.weight.dtype}"
    if not _listed(proj.in_features, proj.out_features, tactics):
        return f"no fixed tactic for a [{proj.out_features}, {proj.in_features}] weight"
    return None


def _linear_reason(module: Linear, tactics: dict) -> Optional[str]:
    method = module.quant_method
    if type(method) is not UnquantizedLinearMethod:
        return f"quant method is {type(method).__name__}"
    if module.weight.dtype != torch.bfloat16:
        return f"weight is {module.weight.dtype}"
    if module.bias is not None:
        return "has a bias"
    if getattr(module, "use_custom_cublas_mm", False):
        return "runs the custom cuBLAS GEMM (output may live in an NCCL window)"
    out_features, in_features = module.weight.shape
    if not _listed(in_features, out_features, tactics):
        return f"no fixed tactic for a [{out_features}, {in_features}] weight"
    return None


def _context_kv_reason(draft_model: DFlashForCausalLM) -> Optional[str]:
    if type(draft_model).precompute_context_kv is not DFlashForCausalLM.precompute_context_kv:
        return f"{type(draft_model).__name__} overrides precompute_context_kv"
    bound = draft_model.__dict__.get("precompute_context_kv")
    if bound is not None and not _bound_to(bound, fixed_tactic_precompute_context_kv):
        return f"precompute_context_kv is already replaced on this drafter by {_describe(bound)}"
    return None


def install_fixed_tactic_gemms(
    draft_model: nn.Module, enable_pdl: bool, tactics: Optional[dict] = None
) -> FixedTacticSites:
    """Put the drafter's small GEMMs on fixed tactics; return which call sites changed and which not.

    ``tactics`` is the table of the drafter's TP size (``splitk_tactics``; the
    device's TP8 one when None). Idempotent. The candidates are the DFlash 2 convolutions'
    ``kernel_projection`` (``nn.Linear``), the decoder layers' ``Linear``
    modules, and the fused context K/V GEMM of ``precompute_context_kv``. A
    candidate changes only if it is still the built-in (class, quant method,
    method binding), bias-free BF16, and its shape has a listed tactic;
    every other candidate is reported in ``declined`` with the reason.
    """
    tactics = splitk_tactics() if tactics is None else tactics
    if not isinstance(draft_model, DFlashForCausalLM):
        return FixedTacticSites(0, (f"not a DFlashForCausalLM ({type(draft_model).__name__})",))
    installed, changed, declined = 0, 0, []
    for kind in ("attention_convs", "mlp_convs"):
        for index, conv in enumerate(getattr(draft_model, kind, None) or ()):
            where = f"{kind}.{index}"
            if not isinstance(conv, DFlash2BlockConv):
                declined.append(f"{where} is {type(conv).__name__}, not a DFlash2BlockConv")
                continue
            proj = conv.kernel_projection
            reason = _projection_reason(proj, tactics)
            if reason is not None:
                declined.append(f"{where}.kernel_projection {reason}")
                continue
            if type(proj) is nn.Linear:
                proj.__class__ = FixedTacticProjection
                changed += 1
            proj._fixed_gemm_pdl = enable_pdl
            proj._fixed_gemm_tactics = tactics
            installed += 1
    for name, module in draft_model.model.layers.named_modules():
        if not isinstance(module, Linear):
            continue
        method = module.quant_method
        if type(method) is FixedTacticLinearMethod:
            method.enable_pdl = enable_pdl
            method.tactics = tactics
            installed += 1
            continue
        reason = _linear_reason(module, tactics)
        if reason is not None:
            declined.append(f"model.layers.{name} {reason}")
            continue
        module.quant_method = FixedTacticLinearMethod(enable_pdl, tactics)
        installed += 1
        changed += 1
    reason = _context_kv_reason(draft_model)
    if reason is not None:
        declined.append(f"context K/V GEMM: {reason}")
    else:
        draft_model._fixed_gemm_pdl = enable_pdl
        draft_model._fixed_gemm_tactics = tactics
        if not _bound_to(
            draft_model.__dict__.get("precompute_context_kv"), fixed_tactic_precompute_context_kv
        ):
            draft_model.precompute_context_kv = types.MethodType(
                fixed_tactic_precompute_context_kv, draft_model
            )
            changed += 1
        installed += 1
    return FixedTacticSites(installed, tuple(declined), changed)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def install(
    draft_model: Optional[nn.Module],
    spec_worker,
    mapping,
    fc_shard: bool,
    fused_accept: bool,
    fixed_gemms: bool,
) -> None:
    """Install the drafter worker fast paths asked for on the drafter and the worker.

    ``fc_shard`` (``drafter_fc_shard``) column-shards the drafter's fc, ``fused_accept``
    fuses the worker's acceptance, ``fixed_gemms`` (``drafter_fixed_gemms``) puts
    the drafter's GEMM call sites on fixed low-M tactics.

    The shell calls this from every load hook, so on all but the first call a
    fast path is usually in place already: a fast path is logged only by the call
    that changed something (a reload that rebuilt the drafter's modules logs
    again).

    Raises:
        ValueError: if ``fc_shard`` or ``fused_accept`` is asked for and cannot be installed
            (no drafter / worker, or the reason its install declined), or
            ``fixed_gemms`` is asked for on a TP size without a tactic table
            (``splitk_tactics``: 4 and 8) or some drafter GEMM call site takes no fixed
            tactic -- running the built-in path under the fast path's name would
            misreport what runs.
    """
    enable_pdl = LAUNCH_PDL
    if fc_shard:
        if draft_model is None:
            raise ValueError("[musespark_open] drafter_fc_shard cannot run here: no drafter")
        was_installed = sharded_fc_installed(draft_model)
        reason = install_sharded_fc(draft_model, mapping, enable_pdl)
        if reason is not None:
            raise ValueError(f"[musespark_open] drafter_fc_shard cannot run here: {reason}")
        if not was_installed:
            gather = (
                "a fabric gather"
                if draft_model._fc_shard.world == 8
                else "an NCCL all-gather, any row count"
            )
            logger.info(
                f"[musespark_open] drafter_fc_shard: DFlash fc column-sharded over "
                f"TP{draft_model._fc_shard.world} with {gather}"
            )
    if fused_accept:
        if spec_worker is None:
            raise ValueError("[musespark_open] fused_accept cannot run here: no spec worker")
        was_installed = isinstance(spec_worker, FusedAcceptDFlashWorker)
        reason = install_fused_accept(spec_worker)
        if reason is not None:
            raise ValueError(f"[musespark_open] fused_accept cannot run here: {reason}")
        if not was_installed:
            logger.info(
                "[musespark_open] fused_accept: DFlash acceptance + forced override in one launch "
                f"(worker class {type(spec_worker).__name__})"
            )
    if fixed_gemms:
        if draft_model is None:
            raise ValueError("[musespark_open] drafter_fixed_gemms cannot run here: no drafter")
        tp_size = getattr(mapping, "tp_size", None)
        try:
            tactics = splitk_tactics(tp_size)
        except ValueError as error:
            raise ValueError(
                f"[musespark_open] drafter_fixed_gemms cannot run here: {error}"
            ) from error
        sites = install_fixed_tactic_gemms(draft_model, enable_pdl, tactics)
        total = sites.installed + len(sites.declined)
        if not sites.installed or sites.declined:
            # The sites follow from the drafter's structure, the same on every rank.
            raise ValueError(
                "[musespark_open] drafter_fixed_gemms cannot run here: "
                f"{len(sites.declined)} of {total} drafter GEMM call sites take no fixed "
                f"tactic ({'; '.join(sites.declined) or 'the drafter has none'})"
            )
        if sites.changed:
            logger.info(
                f"[musespark_open] drafter_fixed_gemms: {sites.installed} of {total} drafter GEMM call sites "
                f"on fixed low-M tactics (the TP{tp_size} table)"
            )
