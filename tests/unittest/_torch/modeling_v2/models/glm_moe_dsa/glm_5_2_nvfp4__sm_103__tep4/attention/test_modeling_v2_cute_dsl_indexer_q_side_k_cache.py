# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_indexer_q_side_k_cache entry (MTP3 / MTP5 wide decode)."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip(
        "cute_dsl_indexer_q_side_k_cache runs an sm_103 CuTe DSL kernel", allow_module_level=True
    )

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention import (  # noqa: E402, E501
    cute_dsl_indexer_k_cache_update as k_module,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention import (  # noqa: E402, E501
    cute_dsl_indexer_q_side_fused as q_module,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention.cute_dsl_indexer_q_side_k_cache import (  # noqa: E402, E501
    build,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention.cute_dsl_indexer_q_side_k_cache import (  # noqa: E402, E501
    cute_dsl_indexer_q_side_k_cache as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

HEADS, HEAD_DIM, ROPE_DIM = 32, 128, 64
#: Page size and per-token bytes of the indexer K pool (128 codes + one fp32 scale).
PAGE, TOKEN_BYTES = 32, 132
BLOCKS = 64
#: The layer of the pool written: neither the first nor the last, so a stray write shows.
LAYER = 3
POISON = 0xA7
THETA = 8.0e6  # the checkpoint's `rope_theta`
MAX_POS = 8192
#: The target's `idx_weight_scale`: head_dim**-0.5 * heads**-0.5.
WEIGHT_SCALE = HEAD_DIM**-0.5 * HEADS**-0.5
#: Slot-mapping entries past the step's tokens (the engine's padded batch): never read.
SLOT_TAIL = 7

_COMPILED: dict = {}


def _handles(contiguous: bool):
    """(fused, served q side, served K update), built once per cache addressing."""
    if contiguous not in _COMPILED:
        dev = torch.cuda.current_device()
        _COMPILED[contiguous] = (
            build(dev, is_neox=False, cache_contiguous=contiguous),
            q_module.build(dev, is_neox=False),
            k_module.build(dev, is_neox=False, cache_contiguous=contiguous),
        )
    return _COMPILED[contiguous]


def _rope_table(positions: int) -> torch.Tensor:
    """The target's indexer table `[positions, cos(32) | sin(32)]`."""
    half = ROPE_DIM // 2
    inv = 1.0 / (THETA ** (torch.arange(half, dtype=torch.float64) * 2.0 / ROPE_DIM))
    ang = torch.arange(positions, dtype=torch.float64)[:, None] * inv[None, :]
    return torch.cat([ang.cos(), ang.sin()], dim=1).float().cuda().contiguous()


def _view(pool: torch.Tensor) -> torch.Tensor:
    """The written layer's `[blocks, page, 1, 132]` view of the block-major pool."""
    return pool[:, min(LAYER, pool.shape[1] - 1), :].view(BLOCKS, PAGE, 1, TOKEN_BYTES)


def _inputs(tokens: int, pad_rows: int, pool_layers: int, seed: int) -> dict:
    """One step's operands. Each token gets a distinct (page, slot) and the TRT-LLM slot
    pair (codes at `block * page * 132 + slot * 128`, the scale at `block * page * 132 +
    page * 128 + slot * 4`); the last `pad_rows` get -1, as CUDA-graph padding rows do."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    cpu = torch.Generator().manual_seed(seed)
    flat = torch.randperm(BLOCKS * PAGE, generator=cpu)[:tokens]
    block, slot = flat // PAGE, flat % PAGE
    fp8_slot = block * PAGE * TOKEN_BYTES + slot * HEAD_DIM
    scale_slot = block * PAGE * TOKEN_BYTES + PAGE * HEAD_DIM + slot * 4
    if pad_rows:
        fp8_slot[-pad_rows:] = -1
        scale_slot[-pad_rows:] = -1
    tail = torch.full((SLOT_TAIL,), -1, dtype=torch.int64)
    return dict(
        q=(torch.randn(tokens, HEADS, HEAD_DIM, generator=g, device="cuda") * 2.0).to(torch.bfloat16),
        pos=torch.randint(0, MAX_POS, (tokens,), generator=g, device="cuda").to(torch.int32),
        table=_rope_table(MAX_POS),
        wq=torch.randn(tokens, HEADS, generator=g, device="cuda"),
        keys=(torch.randn(tokens, HEAD_DIM, generator=g, device="cuda") * 3.0 + 0.5).to(torch.bfloat16),
        nw=1.0 + 0.1 * torch.randn(HEAD_DIM, generator=g, device="cuda"),
        nb=0.1 * torch.randn(HEAD_DIM, generator=g, device="cuda"),
        pool=torch.full((BLOCKS, pool_layers, PAGE * TOKEN_BYTES), POISON, dtype=torch.uint8, device="cuda"),
        fp8=torch.cat([fp8_slot, tail]).cuda(),
        scale=torch.cat([scale_slot, tail]).cuda(),
    )  # fmt: skip


def _outputs(tokens: int):
    """`(q_fp8, weights)` filled with a pattern no launch writes."""
    q8 = torch.empty(tokens, HEADS, HEAD_DIM, dtype=torch.float8_e4m3fn, device="cuda")
    q8.view(torch.uint8).fill_(0x5A)
    return q8, torch.full((tokens, HEADS), float("nan"), device="cuda")


def _args(x: dict, q8, w, pool):
    return (x["q"], x["pos"], x["table"], x["wq"], WEIGHT_SCALE, q8, w, x["keys"], x["nw"], x["nb"],
            _view(pool), x["fp8"], x["scale"])  # fmt: skip


def _fused(x: dict, check: bool = False):
    """The fused launch on fresh outputs and a fresh copy of the pool."""
    contiguous = _view(x["pool"]).is_contiguous()
    q8, w = _outputs(x["q"].shape[0])
    pool = x["pool"].clone()
    handle = _handles(contiguous)[0]
    if check:
        with validating(op):
            op(*_args(x, q8, w, pool), is_neox=False, compiled=handle)
    else:
        op(*_args(x, q8, w, pool), is_neox=False, compiled=handle)
    return q8, w, pool


def _served(x: dict):
    """The served path: the q-side launch, then the K-update launch."""
    tokens = x["q"].shape[0]
    _, hq, hk = _handles(_view(x["pool"]).is_contiguous())
    q8, w = _outputs(tokens)
    pool = x["pool"].clone()
    q_module.cute_dsl_indexer_q_side_fused(
        x["q"], x["pos"], x["table"], x["wq"], WEIGHT_SCALE, q8, w, is_neox=False, compiled=hq
    )
    k_module.cute_dsl_indexer_k_cache_update(
        x["keys"], x["nw"], x["nb"], x["pos"], x["table"].view(-1, 2, ROPE_DIM // 2), _view(pool),
        x["fp8"], x["scale"], tokens, is_neox=False, compiled=hk, validate=False,
    )  # fmt: skip
    return q8, w, pool


def _same(a, b) -> bool:
    return all(torch.equal(u.view(torch.uint8) if u.dtype == torch.float8_e4m3fn else u,
                           v.view(torch.uint8) if v.dtype == torch.float8_e4m3fn else v)
               for u, v in zip(a, b))  # fmt: skip


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    """Every byte the two served launches write (q codes, head weights, the whole pool)
    bit for bit; the q side equal to its reference exactly, the cache's scales and every
    byte outside the written slots equal to the K update's reference; non-degenerate."""
    spec = cell.spec
    x = _inputs(spec["tokens"], spec["pad_rows"], spec["pool_layers"],
                zlib.crc32(cell.why.encode()) % 2**31)  # fmt: skip
    got = _fused(x, check=True)
    want = _served(x)
    torch.cuda.synchronize()
    assert _same(got, want), f"{spec}: differs from the served q-side + K-update launches"
    q8, w, pool = got
    ref_q8, ref_w, ref_cache = op.reference(*_args(x, *_outputs(spec["tokens"]), x["pool"]),
                                            is_neox=False, compiled=None)  # fmt: skip
    assert torch.equal(q8.view(torch.uint8), ref_q8.view(torch.uint8)) and torch.equal(w, ref_w)
    tokens, live = spec["tokens"], x["fp8"][: spec["tokens"]] >= 0
    fp8_at, scale_at = x["fp8"][:tokens][live], x["scale"][:tokens][live]
    code_at = (fp8_at.unsqueeze(1) + torch.arange(HEAD_DIM, device="cuda")).flatten()
    scale_at = (scale_at.unsqueeze(1) + torch.arange(4, device="cuda")).flatten()
    cache, ref = _view(pool).contiguous().view(-1), ref_cache.view(-1)
    written = torch.zeros_like(cache, dtype=torch.bool)
    written[code_at] = True
    written[scale_at] = True
    assert torch.equal(cache[~written], ref[~written]), "a byte outside the slots differs"
    assert torch.equal(cache[scale_at], ref[scale_at]), "a row scale differs"
    others = [i for i in range(pool.shape[1]) if i != min(LAYER, pool.shape[1] - 1)]
    assert bool((pool[:, others] == POISON).all()), "a write landed in another layer"
    # Non-degenerate: varied codes and weights, and most written code bytes changed.
    assert int(torch.unique(q8.view(torch.uint8)).numel()) > 64 and float(w.std()) > 0.0
    assert float((cache[code_at] != POISON).float().mean()) > 0.9


@pytest.mark.parametrize("tokens", (128, 150, 192))
def test_replay_is_eager_bitwise_on_new_operands(tokens) -> None:
    """Eager twice is bitwise; in a CUDA graph, replay writes the eager launch's bytes, and
    after new operands are copied into the captured buffers, the eager launch's on them."""
    x = _inputs(tokens, 2, 78, seed=tokens)
    first = _fused(x)
    assert _same(first, _fused(x)), "eager twice differs"
    handle = _handles(False)[0]
    q8, w = _outputs(tokens)
    pool = x["pool"].clone()
    args = _args(x, q8, w, pool)
    op(*args, is_neox=False, compiled=handle)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(*args, is_neox=False, compiled=handle)
    q8.view(torch.uint8).fill_(0)
    w.fill_(0.0)
    pool.copy_(x["pool"])
    graph.replay()
    torch.cuda.synchronize()
    assert _same((q8, w, pool), first)
    y = _inputs(tokens, 3, 78, seed=tokens + 1)
    for key in ("q", "pos", "wq", "keys", "nw", "nb", "fp8", "scale"):
        x[key].copy_(y[key])
    pool.copy_(y["pool"])
    graph.replay()
    torch.cuda.synchronize()
    fresh = _fused(y)
    assert _same((q8, w, pool), fresh) and not _same((q8, w, pool), first)


def test_guards() -> None:
    """Keys the launcher cannot load (misaligned, wrong shape) are refused before launch."""
    x = _inputs(96, 0, 78, seed=5)
    handle = _handles(False)[0]
    storage = torch.empty(96 * HEAD_DIM + 1, dtype=torch.bfloat16, device="cuda")
    x["keys"] = storage[1:].view(96, HEAD_DIM)
    q8, w = _outputs(96)
    with pytest.raises(ValueError, match="q side \\| K update"):
        op(*_args(x, q8, w, x["pool"]), is_neox=False, compiled=handle)
    x["keys"] = torch.zeros(95, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="q side \\| K update"):
        op(*_args(x, q8, w, x["pool"]), is_neox=False, compiled=handle)
