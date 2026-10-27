# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The strict draft acceptance and the next step's input tokens of a decode-only speculative step, in one launch each.

The in-tree strict path verifies ``D`` draft tokens per request against the target's sampled tokens with about ten
small torch launches (empty / ones / reshape / == / int / cumprod / sum / += / slice copy), and builds the next
step's input tokens with three more (index, unsqueeze, concat). Both are integer copies and compares, so these
kernels are bitwise to that path by construction; the forced-acceptance override stays the caller's (it advances
the in-tree RNG pool and counters).

Compiled on the first call per (draft length, dtypes), which must run outside CUDA-graph capture (``precompile``).
"""

from __future__ import annotations

import importlib
import threading
from typing import Dict, Sequence

import torch

from .._op import Arch, Cell, OpWrapper

_lock = threading.Lock()
_compiled: Dict[tuple, object] = {}
_modules: Dict[str, object] = {}


def _kernel_module():
    """The CuTe DSL kernels beside this file, imported on first use (they pull in cutlass)."""
    mod = _modules.get("kernel")
    if mod is None:
        mod = _modules["kernel"] = importlib.import_module(".strict_accept_kernel", __package__)
    return mod


def _arg(t: torch.Tensor, align: int):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t.detach(), assumed_align=align).mark_layout_dynamic(leading_dim=0)


def _int_align(t: torch.Tensor) -> int:
    return t.element_size()


def max_requests() -> int:
    return _kernel_module().MAX_REQUESTS


def _compiled_fn(name: str, key: tuple, args: tuple, scalars: tuple, consts: tuple, stream):
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"trtllm::{name} must run once per shape outside CUDA-graph capture first"
            )
        import cutlass.cute as cute

        with _lock:
            fn = _compiled.get(key)
            if fn is None:
                fn = _compiled[key] = cute.compile(
                    getattr(_kernel_module(), name), *args, *scalars, *consts, stream
                )
    return fn


@torch.library.custom_op("trtllm::spec_strict_accept", mutates_args=())
def _spec_strict_accept(target: torch.Tensor, draft: torch.Tensor) -> list[torch.Tensor]:
    import cuda.bindings.driver as cuda_driver

    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    n_req, draft_len = draft.shape
    if target.numel() != n_req * (draft_len + 1):
        raise ValueError(
            f"spec_strict_accept: target {tuple(target.shape)} vs draft {tuple(draft.shape)}"
        )
    if n_req > max_requests() or not target.is_contiguous() or not draft.is_contiguous():
        raise ValueError("spec_strict_accept: unsupported operands (see supports())")
    device = target.device
    accepted = torch.empty(n_req, draft_len + 1, dtype=torch.int32, device=device)
    num_accepted = torch.empty(n_req, dtype=torch.int32, device=device)
    args = (
        _arg(target.view(-1), _int_align(target)),
        _arg(draft.view(-1), _int_align(draft)),
        _arg(accepted.view(-1), 4),
        _arg(num_accepted, 4),
    )
    use_pdl = get_env_enable_pdl()
    stream = cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
    key = ("strict_accept", target.dtype, draft.dtype, draft_len, use_pdl)
    fn = _compiled_fn("strict_accept", key, args, (n_req,), (draft_len, use_pdl), stream)
    fn(*args, n_req, stream)
    return [accepted, num_accepted]


@_spec_strict_accept.register_fake
def _(target, draft):
    n_req, draft_len = draft.shape
    return [
        target.new_empty((n_req, draft_len + 1), dtype=torch.int32),
        target.new_empty((n_req,), dtype=torch.int32),
    ]


@torch.library.custom_op("trtllm::spec_next_new_tokens", mutates_args=())
def _spec_next_new_tokens(
    accepted: torch.Tensor,
    num_accepted: torch.Tensor,
    batch_idx: torch.Tensor,
    next_draft: torch.Tensor,
) -> torch.Tensor:
    import cuda.bindings.driver as cuda_driver

    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    n_req, draft_len = next_draft.shape
    if n_req > max_requests() or accepted.stride(1) != 1 or next_draft.stride(1) != 1:
        raise ValueError("spec_next_new_tokens: unsupported operands (see supports())")
    device = accepted.device
    out = torch.empty(
        n_req,
        draft_len + 1,
        dtype=torch.promote_types(accepted.dtype, next_draft.dtype),
        device=device,
    )
    # Flat views over the storage each row index reads (strided rows: the row strides are passed).
    acc_flat = accepted.as_strided(
        ((accepted.shape[0] - 1) * accepted.stride(0) + accepted.shape[1],), (1,)
    )
    nd_flat = next_draft.as_strided(((n_req - 1) * next_draft.stride(0) + draft_len,), (1,))
    args = (
        _arg(acc_flat, _int_align(accepted)),
        _arg(num_accepted, _int_align(num_accepted)),
        _arg(batch_idx, _int_align(batch_idx)),
        _arg(nd_flat, _int_align(next_draft)),
        _arg(out.view(-1), _int_align(out)),
    )
    scalars = (int(accepted.stride(0)), int(next_draft.stride(0)), n_req)
    use_pdl = get_env_enable_pdl()
    stream = cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
    key = (
        "next_new_tokens",
        accepted.dtype,
        num_accepted.dtype,
        batch_idx.dtype,
        next_draft.dtype,
        draft_len,
        use_pdl,
    )
    fn = _compiled_fn("next_new_tokens", key, args, scalars, (draft_len, use_pdl), stream)
    fn(*args, *scalars, stream)
    return out


@_spec_next_new_tokens.register_fake
def _(accepted, num_accepted, batch_idx, next_draft):
    n_req, draft_len = next_draft.shape
    return accepted.new_empty(
        (n_req, draft_len + 1), dtype=torch.promote_types(accepted.dtype, next_draft.dtype)
    )


@torch.library.custom_op("trtllm::spec_kv_rewind", mutates_args=("kv_lens", "rewind"))
def _spec_kv_rewind(
    num_accepted: torch.Tensor, kv_lens: torch.Tensor, rewind: torch.Tensor, nc: int, n: int
) -> None:
    import cuda.bindings.driver as cuda_driver

    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    if not 0 < n <= max_requests() or rewind.numel() < n or rewind.dtype != num_accepted.dtype:
        raise ValueError("spec_kv_rewind: unsupported operands (see supports_kv_rewind())")
    device = num_accepted.device
    args = (
        _arg(num_accepted, _int_align(num_accepted)),
        _arg(kv_lens, _int_align(kv_lens)),
        _arg(rewind, _int_align(rewind)),
    )
    use_pdl = get_env_enable_pdl()
    stream = cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
    key = ("kv_rewind", num_accepted.dtype, kv_lens.dtype, use_pdl)
    fn = _compiled_fn("kv_rewind", key, args, (int(nc), int(n)), (use_pdl,), stream)
    fn(*args, int(nc), int(n), stream)


@_spec_kv_rewind.register_fake
def _(num_accepted, kv_lens, rewind, nc, n):
    return None


_INT_DTYPES = (torch.int32, torch.int64)


def supports_strict_accept(target: torch.Tensor, draft: torch.Tensor) -> bool:
    """Whether the kernel takes these operands; a caller that gets False runs the in-tree path."""
    return (
        target.is_cuda
        and draft.is_cuda
        and target.dtype in _INT_DTYPES
        and draft.dtype in _INT_DTYPES
        and draft.dim() == 2
        and 0 < draft.shape[0] <= max_requests()
        and draft.shape[1] > 0
        and target.numel() == draft.shape[0] * (draft.shape[1] + 1)
        and target.is_contiguous()
        and draft.is_contiguous()
    )


def supports_next_new_tokens(
    accepted: torch.Tensor,
    num_accepted: torch.Tensor,
    batch_idx: torch.Tensor,
    next_draft: torch.Tensor,
) -> bool:
    """Whether the kernel takes these operands; a caller that gets False runs the in-tree path."""
    return (
        all(
            t.is_cuda and t.dtype in _INT_DTYPES
            for t in (accepted, num_accepted, batch_idx, next_draft)
        )
        and accepted.dim() == 2
        and next_draft.dim() == 2
        and accepted.stride(1) == 1
        and next_draft.stride(1) == 1
        and 0 < next_draft.shape[0] <= max_requests()
        and num_accepted.dim() == 1
        and batch_idx.dim() == 1
        and num_accepted.is_contiguous()
        and batch_idx.is_contiguous()
        and num_accepted.numel() >= next_draft.shape[0]
        and batch_idx.numel() >= next_draft.shape[0]
    )


def supports_kv_rewind(
    num_accepted: torch.Tensor, kv_lens: torch.Tensor, rewind: torch.Tensor, nc: int, n: int
) -> bool:
    """Whether the kernel takes these operands; a caller that gets False runs the in-tree lines."""
    return (
        all(t.is_cuda and t.dtype in _INT_DTYPES for t in (num_accepted, kv_lens, rewind))
        and all(t.dim() == 1 and t.is_contiguous() for t in (num_accepted, kv_lens, rewind))
        and rewind.dtype == num_accepted.dtype
        and 0 < n <= max_requests()
        and nc >= 0
        and num_accepted.numel() >= nc + n
        and kv_lens.numel() >= nc + n
        and rewind.numel() >= n
    )


def is_compiled_kv_rewind(num_accepted: torch.Tensor, kv_lens: torch.Tensor) -> bool:
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    return ("kv_rewind", num_accepted.dtype, kv_lens.dtype, get_env_enable_pdl()) in _compiled


def precompile_kv_rewind(device, dtypes: Sequence[torch.dtype] = (torch.int32,)) -> None:
    """Compile the kv_rewind variants (num_accepted dtype x kv_lens dtype) outside capture."""
    for a_dtype in dtypes:
        for k_dtype in dtypes:
            num = torch.ones(1, dtype=a_dtype, device=device)
            torch.ops.trtllm.spec_kv_rewind(
                num, torch.zeros(1, dtype=k_dtype, device=device), torch.empty_like(num), 0, 1
            )
    torch.cuda.synchronize()


def precompile(
    draft_lens: Sequence[int], device, dtypes: Sequence[torch.dtype] = _INT_DTYPES
) -> None:
    """Compile every (draft length, operand dtype) variant a one-model worker passes, outside capture: target
    tokens and drafts of either int dtype; int32 accepted / num_accepted (this op's) and batch indices (the spec
    metadata's); next drafts of either int dtype."""
    for draft_len in draft_lens:
        batch_idx = torch.zeros(1, dtype=torch.int32, device=device)
        for t_dtype in dtypes:
            for d_dtype in dtypes:
                target = torch.zeros(draft_len + 1, dtype=t_dtype, device=device)
                draft = torch.zeros(1, draft_len, dtype=d_dtype, device=device)
                accepted, num = torch.ops.trtllm.spec_strict_accept(target, draft)
        for d_dtype in dtypes:
            draft = torch.zeros(1, draft_len, dtype=d_dtype, device=device)
            torch.ops.trtllm.spec_next_new_tokens(accepted, num, batch_idx, draft)
    torch.cuda.synchronize()


def is_compiled_strict_accept(target: torch.Tensor, draft: torch.Tensor) -> bool:
    """Whether this call's variant is compiled (a captured step may only call compiled variants)."""
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    key = ("strict_accept", target.dtype, draft.dtype, draft.shape[1], get_env_enable_pdl())
    return key in _compiled


def is_compiled_next_new_tokens(accepted, num_accepted, batch_idx, next_draft) -> bool:
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    key = (
        "next_new_tokens",
        accepted.dtype,
        num_accepted.dtype,
        batch_idx.dtype,
        next_draft.dtype,
        next_draft.shape[1],
        get_env_enable_pdl(),
    )
    return key in _compiled


class SpecStrictAccept(OpWrapper):
    """Strict acceptance of a decode-only batch: ``target`` [n_req * (D + 1)] (the target's sampled token per verify
    row, request-major) and ``draft`` [n_req, D] (the drafts being verified), int32 or int64.

    Returns ``(accepted [n_req, D + 1] int32, num_accepted [n_req] int32)``: the target tokens, and 1 + the length of
    the prefix of drafts equal to the target's tokens. No forced-acceptance override (the caller's).
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="K3 c1: one request, int32 tokens (the fused sampler's) vs int32 drafts",
            spec=dict(n_req=1, draft_len=3),
        ),
        Cell(
            why="K3 c8: the 32-row step, mixed prefix lengths",
            spec=dict(n_req=8, draft_len=3),
        ),
        Cell(
            why="K3 at 12 requests: the 48-row bound of the decode targets",
            spec=dict(n_req=12, draft_len=3),
        ),
        Cell(
            why="K7 c1 and c8: 8 verify rows per request",
            spec=dict(n_req=8, draft_len=7),
        ),
        Cell(
            why="int64 target tokens (torch.argmax, the all-greedy path)",
            spec=dict(n_req=8, draft_len=3, target_dtype=torch.int64),
        ),
        Cell(
            why="padded / dropped drafts (-1): never equal to a token",
            spec=dict(n_req=8, draft_len=3, pad_drafts=True),
        ),
        Cell(
            why="every draft accepted, and none accepted",
            spec=dict(n_req=8, draft_len=3, extremes=True),
        ),
        Cell(
            why="the kernel's request bound (one CTA)",
            spec=dict(n_req=64, draft_len=3),
        ),
    )

    note = """
    Integer compares and copies only: the reference is exact, and the op must equal it bit for bit. Decode-only
    batches (no context rows). The ragged verify layout and the forced-acceptance override are not this op's.
    """

    def __call__(self, target: torch.Tensor, draft: torch.Tensor):
        return torch.ops.trtllm.spec_strict_accept(target, draft)

    def reference(self, target: torch.Tensor, draft: torch.Tensor):
        """The in-tree strict path (``SpecWorkerBase._sample_and_accept_draft_tokens_base``, no contexts)."""
        n_req, draft_len = draft.shape
        accepted = torch.empty((n_req, draft_len + 1), dtype=torch.int, device=target.device)
        num_accepted = torch.ones(n_req, dtype=torch.int, device=target.device)
        gen_target = target.reshape(n_req, draft_len + 1)
        num_accepted += torch.cumprod((draft == gen_target[:, :draft_len]).int(), dim=-1).sum(1)
        accepted[:, : draft_len + 1] = gen_target
        return [accepted, num_accepted]


class SpecNextNewTokens(OpWrapper):
    """The next step's input tokens: ``out[b] = [accepted[batch_idx[b], num_accepted[b] - 1], *next_draft[b]]``.

    ``accepted`` [rows, >= D + 1] (row stride free), ``num_accepted`` [>= n_req], ``batch_idx`` [>= n_req],
    ``next_draft`` [n_req, D] (row stride free); int32 or int64 each. The output dtype is torch's promotion of
    ``accepted`` and ``next_draft`` (as ``torch.concat``).
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="K3 c1, int32 throughout",
            spec=dict(n_req=1, draft_len=3),
        ),
        Cell(
            why="K3 c8 with shuffled batch indices",
            spec=dict(n_req=8, draft_len=3, shuffle=True),
        ),
        Cell(
            why="K7 c8",
            spec=dict(n_req=8, draft_len=7),
        ),
        Cell(
            why="int64 next drafts: the promoted int64 output",
            spec=dict(n_req=8, draft_len=3, draft_dtype=torch.int64),
        ),
        Cell(
            why="a wider accepted buffer (max draft length > runtime) and strided drafts",
            spec=dict(n_req=8, draft_len=3, wide=True),
        ),
    )

    note = """
    Index copies only: the reference is exact. The op does not bound batch_idx or num_accepted; is_valid states them.
    """

    def __call__(self, accepted, num_accepted, batch_idx, next_draft):
        return torch.ops.trtllm.spec_next_new_tokens(accepted, num_accepted, batch_idx, next_draft)

    def reference(self, accepted, num_accepted, batch_idx, next_draft):
        """The in-tree ``SpecWorkerBase._prepare_next_new_tokens``."""
        n_req = next_draft.shape[0]
        first = accepted[batch_idx[:n_req], num_accepted[:n_req] - 1].unsqueeze(1)
        return torch.concat([first, next_draft], dim=1)

    def is_valid(self, accepted, num_accepted, batch_idx, next_draft) -> None:
        n_req = next_draft.shape[0]
        assert bool(((batch_idx[:n_req] >= 0) & (batch_idx[:n_req] < accepted.shape[0])).all())
        assert bool(
            ((num_accepted[:n_req] >= 1) & (num_accepted[:n_req] <= accepted.shape[1])).all()
        )


class SpecKvRewind(OpWrapper):
    """DFlash's draft-forward KV bookkeeping for the generation rows ``nc .. nc + n`` after the forced-acceptance
    override: ``rewind[:n] = 1 - num_accepted[nc : nc + n]`` and ``kv_lens[nc : nc + n] += 1`` (in place).

    ``num_accepted`` and ``kv_lens`` int32 or int64 (1-D, contiguous); ``rewind`` [>= n] of num_accepted's dtype.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="K3 c1: one generation row, int32 counts and lengths",
            spec=dict(nc=0, n=1),
        ),
        Cell(
            why="K3 c8: eight generation rows",
            spec=dict(nc=0, n=8),
        ),
        Cell(
            why="context rows first (nc > 0): only the generation rows change",
            spec=dict(nc=3, n=5),
        ),
        Cell(
            why="int64 kv_lens",
            spec=dict(nc=0, n=8, kv_dtype=torch.int64),
        ),
        Cell(
            why="the kernel's request bound (one CTA)",
            spec=dict(nc=0, n=64),
        ),
    )

    note = """
    Integer arithmetic only: the reference is exact. kv_lens is updated in place; rows outside nc .. nc + n are
    untouched. The caller decides the step (it runs where the in-tree lines run, after the forced override).
    """

    def __call__(self, num_accepted, kv_lens, rewind, nc: int, n: int):
        torch.ops.trtllm.spec_kv_rewind(num_accepted, kv_lens, rewind, int(nc), int(n))
        return rewind[:n]

    def reference(self, num_accepted, kv_lens, rewind, nc: int, n: int):
        """DFlashWorker._prepare_kv_for_draft_forward's two lines."""
        out = 1 - num_accepted[nc : nc + n]
        kv_lens[nc : nc + n] += 1
        return out


spec_strict_accept = SpecStrictAccept()
spec_next_new_tokens = SpecNextNewTokens()
spec_kv_rewind = SpecKvRewind()
