# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""B1: M3.1 TP2's exact BF16 gathered draft head plus FP32 Markov consumer.

An explicit target allowlist and its already-opted-in A1 processor scope this adapter.
Other models and S4 small-request steps keep the unmodified head/refinement path.
"""

import importlib
import os
from pathlib import Path

import torch

_DISABLED = os.environ.get("TRTLLM_M31_DISABLE_DRAFT_AG_BF16", "0") == "1"
_TARGET = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2"
)
_REQUESTS = range(13, 29)
_DRAFT_ROWS = tuple(n * 7 for n in _REQUESTS)
_COMPILED = {}
_READY = set()
_LOGGED = set()
_LOADED = False


def _load():
    global _LOADED
    if not _LOADED:
        import tensorrt_llm

        lib = Path(tensorrt_llm.__file__).parent / "libs/libm31_draft_bf16.so"
        if not hasattr(torch.ops.trtllm, "minimax_m31_markov_epilogue_bf16") and lib.is_file():
            torch.ops.load_library(str(lib))
        if hasattr(torch.ops.trtllm, "minimax_m31_markov_epilogue_bf16"):

            @torch.library.register_fake("trtllm::minimax_m31_markov_epilogue_bf16")
            def fake(base, k, bias, out, prev, w1, emb_out, partials, counters, splits):
                return None

        _LOADED = True
    return hasattr(torch.ops.trtllm, "minimax_m31_markov_epilogue_bf16")


def _kernel(shard, out, ws_uc, ws_mc, ws_flags, rank, world):
    import cuda.bindings.driver as driver
    import cutlass.cute as cute

    from ..comm import allgather_cast as ag
    from . import allgather_bf16_kernel as kernel

    grid = ag.pick_grid(shard.shape[0], shard.device, world)
    args = (
        ag._arg(shard.view(-1).view(torch.int32)),
        ag._arg(out.view(-1).view(torch.int32)),
        ag._arg(ws_uc),
        ag._arg(ws_mc),
        ag._arg(ws_flags, align=4),
    )
    stream = driver.CUstream(torch.cuda.current_stream(shard.device).cuda_stream)
    key = (shard.device.index, shard.shape[0], shard.shape[1], world, grid)
    fn = _COMPILED.get(key)
    if fn is None:
        assert not torch.cuda.is_current_stream_capturing(), "B1 must precompile before capture"
        fn = cute.compile(
            kernel.allgather_cast,
            *args,
            rank,
            *ag._consts(shard.shape[0], shard.shape[1], world, grid),
            False,
            stream,
        )
        _COMPILED[key] = fn
    return fn, args, stream


@torch.library.custom_op("trtllm::m31_draft_gather_bf16", mutates_args=())
def gather(
    shard: torch.Tensor,
    ws_uc: torch.Tensor,
    ws_mc: torch.Tensor,
    ws_flags: torch.Tensor,
    rank: int,
    world: int,
) -> torch.Tensor:
    out = shard.new_empty((shard.shape[0], world * shard.shape[1]))
    fn, args, stream = _kernel(shard, out, ws_uc, ws_mc, ws_flags, rank, world)
    fn(*args, rank, stream)
    return out


@gather.register_fake
def _(shard, ws_uc, ws_mc, ws_flags, rank, world):
    return shard.new_empty((shard.shape[0], world * shard.shape[1]))


def prepare(shard, ws):
    """Called inside A1's existing all-rank compile/agreement region; no collective launch."""
    if (
        _DISABLED
        or shard.ndim != 2
        or shard.shape[0] not in _DRAFT_ROWS
        or shard.shape[1] != 100032
        or ws["world"] != 2
        or torch.cuda.get_device_capability(shard.device) != (10, 7)
        or not _load()
    ):
        return
    out = shard.new_empty((shard.shape[0], 200064))
    _kernel(shard, out, ws["ws_uc"], ws["ws_mc"], ws["ws_flags"], ws["rank"], ws["world"])
    epilogue = importlib.import_module(_TARGET + ".ops.spec.m31_markov_epilogue")
    assert epilogue.prebuild(shard.device) is not None
    epilogue._splits(shard.shape[0] // 7, 200064)
    _READY.add((shard.device.index, shard.shape[0]))


def try_head_and_refine(worker, model, hidden, metadata, inputs, spec_metadata, num_gens, k):
    """None means no work was launched and the existing head + refine path must run."""
    processor = model.logits_processor
    if (
        _DISABLED
        or type(worker).__module__ != _TARGET + ".dspark"
        or type(processor).__module__ != _TARGET + ".dspark"
        or type(processor).__name__ != "_DraftA1Logits"
        or num_gens not in _REQUESTS
        or k != 7
        or hidden.shape != (num_gens * 7, 6144)
        or hidden.dtype != torch.bfloat16
        or not hidden.is_cuda
        or not hidden.is_contiguous()
        or (hidden.device.index, num_gens * 7) not in _READY
        or metadata is None
        or metadata.num_contexts != 0
        or getattr(worker, "_s4_step", None) is not None
        or not getattr(model, "has_markov_head", False)
    ):
        return None
    s4 = importlib.import_module(_TARGET + ".dspark_s4")
    if (
        not s4.S4FE
        or model.lm_head is not processor._head[0]
        or num_gens * 7 not in processor._rows
    ):
        return None
    w1, w2 = model.markov_w1, model.markov_w2
    if (
        w1.shape != (200064, 256)
        or w2.shape != (200064, 256)
        or w1.dtype != torch.bfloat16
        or w2.dtype != torch.bfloat16
        or not w1.is_contiguous()
        or not w2.is_contiguous()
        or w1.device != hidden.device
        or w2.device != hidden.device
    ):
        return None
    ws = processor._workspace
    if ws["world"] != 2:
        return None
    shard = model.lm_head.apply_linear(hidden, None)
    base = gather(shard, ws["ws_uc"], ws["ws_mc"], ws["ws_flags"], ws["rank"], ws["world"])
    base = base.view(num_gens, 7, 200064)
    epilogue = importlib.import_module(_TARGET + ".ops.spec.m31_markov_epilogue")
    partials, counters = epilogue.prebuild(base.device)
    splits = epilogue._splits(num_gens, 200064)
    result = torch.empty(base.shape, dtype=torch.float32, device=base.device)
    prev = torch.empty(num_gens, dtype=torch.int64, device=base.device)
    emb = epilogue._prev_embeddings(inputs["first_prev_tokens"], w1)
    for pos in range(7):
        bias = torch.nn.functional.linear(emb, w2)
        nxt = torch.empty_like(emb) if pos < 6 else None
        torch.ops.trtllm.minimax_m31_markov_epilogue_bf16(
            base, pos, bias, result, prev, w1, nxt, partials, counters, splits
        )
        emb = nxt
    log_key = (ws["rank"], num_gens)
    if log_key not in _LOGGED:
        from tensorrt_llm.logger import logger

        logger.info(
            f"[M3.1] DRAFT_AG_BF16 taken rank={ws['rank']} verify_rows={num_gens * 8} draft_rows={num_gens * 7}"
        )
        _LOGGED.add(log_key)
    return result


_ARMED = False


def arm_draft_bf16() -> None:
    """Prepare B1 before A1's all-rank agreement without editing digested comm source."""
    global _ARMED
    if _ARMED or _DISABLED:
        return
    from ..comm import allgather_cast as ag

    stock = ag.precompile

    def precompile(mapping, cols, rows, device, group=None, agree=None):
        eligible_rows = tuple(r for r in rows if r in _DRAFT_ROWS)
        if cols != 100032 or not eligible_rows or agree is None:
            return stock(mapping, cols, rows, device, group=group, agree=agree)

        def agree_with_b1(ok):
            if ok:
                try:
                    ws = ag.workspace_args(mapping, cols, group)
                    for row_count in eligible_rows:
                        shard = torch.zeros(row_count, cols, dtype=torch.bfloat16, device=device)
                        prepare(shard, ws)
                except Exception as exc:  # noqa: BLE001 -- all-rank agreement contains preparation failures
                    from tensorrt_llm.logger import logger

                    logger.warning(f"B1: preparation failed before A1 agreement: {exc}")
                    ok = False
            accepted = agree(ok)
            if not accepted:
                _READY.clear()
            return accepted

        return stock(mapping, cols, rows, device, group=group, agree=agree_with_b1)

    ag.precompile = precompile
    _ARMED = True
