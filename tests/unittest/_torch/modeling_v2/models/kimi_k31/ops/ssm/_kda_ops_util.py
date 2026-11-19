# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the Kimi K3.1 ssm/ entry tests share: pools laid out like the V2 hybrid manager's per-slot coalesced ones
(contiguous within a slot, padded slot stride), K3.1's KDA constants, the specialized targets' stacked
KDA projection, the planned prefill schedule, the one-token decode's arguments, the comparison bands
and the configs of a small KDA + DSA hybrid cache manager.

Not a test file: the entry tests import it by name from their own directory.
"""

from __future__ import annotations

import math
import zlib
from types import SimpleNamespace

import torch

__extra_import_path__ = ["../.."]
from _k31_utils import on_sm

HEAD_DIM = 128  # K == V
CONV_WIDTH = 4
COMMITTED = CONV_WIDTH - 1
LOWER_BOUND = -5.0
SCALE = HEAD_DIM**-0.5
EPS = 1e-5
#: The specialized targets pad the rows of their fused KDA projection to this multiple.
STACK_ROW_MULTIPLE = 16

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}

requires_sm103 = on_sm(103, 107)


def cell_seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


def generator(cell) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(cell_seed(cell))


def padded_slots(
    num_slots: int,
    row_shape: tuple[int, ...],
    dtype: torch.dtype,
    pad_bytes: int = 48,
    device="cuda",
) -> torch.Tensor:
    """`[num_slots, *row_shape]`, contiguous within each slot, with `pad_bytes` of padding between slots
    (a multiple of 16 bytes, as the V2 manager's per-slot coalesced pools keep 16-byte alignment)."""
    row = math.prod(row_shape)
    pad = pad_bytes // torch.tensor([], dtype=dtype).element_size()
    storage = torch.zeros(num_slots, row + pad, dtype=dtype, device=device)
    return storage[:, :row].view(num_slots, *row_shape)


def transposed_padded_slots(
    num_slots: int, rows: int, cols: int, dtype: torch.dtype, pad_bytes: int = 48
) -> torch.Tensor:
    """`[num_slots, cols, rows]` view of `[num_slots, rows, cols]` padded-slot storage: the dim-contiguous
    replay conv caches (`stride(1) == 1`)."""
    return padded_slots(num_slots, (rows, cols), dtype, pad_bytes).transpose(1, 2)


def replay_caches(num_slots: int, heads: int, num_spec: int, gen: torch.Generator) -> dict:
    """One layer's fused-verify replay caches with the target's dtypes and layouts, random finite contents:
    raw conv inputs in `cs_*` (bf16 [slots, D, 3 + num_spec]), normalized keys and SiLU values in k / v
    (bf16), log-decays in [LOWER_BOUND, 0) in g (fp32) and sigmoid gains in beta (bf16 rows padded to 16
    bytes)."""
    dim = heads * HEAD_DIM
    caches = {
        name: transposed_padded_slots(num_slots, COMMITTED + num_spec, dim, torch.bfloat16)
        for name in ("cs_q", "cs_k", "cs_v")
    }
    for name in ("cs_q", "cs_k", "cs_v"):
        caches[name].copy_(torch.randn(caches[name].shape, generator=gen, device="cuda") * 0.5)
    keys = torch.randn(num_slots, num_spec, heads, HEAD_DIM, generator=gen, device="cuda")
    keys = torch.nn.functional.normalize(keys, dim=-1).reshape(num_slots, num_spec, dim)
    caches["k_cache"] = padded_slots(num_slots, (num_spec, dim), torch.bfloat16)
    caches["k_cache"].copy_(keys)
    caches["v_cache"] = padded_slots(num_slots, (num_spec, dim), torch.bfloat16)
    caches["v_cache"].copy_(torch.randn(num_slots, num_spec, dim, generator=gen, device="cuda"))
    caches["g_cache"] = padded_slots(num_slots, (num_spec, dim), torch.float32)
    caches["g_cache"].copy_(
        LOWER_BOUND * torch.rand(num_slots, num_spec, dim, generator=gen, device="cuda")
    )
    padded_heads = (heads + 7) // 8 * 8
    beta = padded_slots(num_slots, (num_spec, padded_heads), torch.bfloat16)[..., :heads]
    beta.copy_(torch.rand(num_slots, num_spec, heads, generator=gen, device="cuda"))
    caches["beta_cache"] = beta
    return caches


def step_schedule(seq_lens, device) -> tuple[torch.Tensor, torch.Tensor]:
    """`(seq_order, cu_chunks)`: the step's shared prefill schedule."""
    order = sorted(range(len(seq_lens)), key=lambda i: -seq_lens[i])
    chunks = [0]
    for length in seq_lens:
        chunks.append(chunks[-1] + (length + 15) // 16)
    as_int32 = dict(dtype=torch.int32, device=device)
    return torch.tensor(order, **as_int32), torch.tensor(chunks, **as_int32)


def stacked_columns(heads: int) -> dict[str, int]:
    """The specialized targets' fused KDA projection `[q | k | v | g | f_a | b | 0]`: one MXFP8 GEMM
    whose weight rows are stacked in that order and zero-padded to a multiple of 16. The first
    column of each section, where the padding starts, and the row pitch `N` of the output (6288 at
    12 heads, 49376 at 96)."""
    dim = heads * HEAD_DIM
    cols = dict(q=0, k=dim, v=2 * dim, g=3 * dim, f_a=4 * dim, b=4 * dim + HEAD_DIM)
    cols["pad"] = cols["b"] + heads
    cols["N"] = -(-cols["pad"] // STACK_ROW_MULTIPLE) * STACK_ROW_MULTIPLE
    return cols


def stacked_projection(tokens: int, heads: int, gen: torch.Generator, device) -> torch.Tensor:
    """A bf16 `[tokens, N]` output of that GEMM: random sections, zero padding (zero weight rows)."""
    cols = stacked_columns(heads)
    out = torch.randn(tokens, cols["N"], generator=gen, device=device)
    out[:, cols["pad"] :] = 0
    return out.to(torch.bfloat16)


def decode_args(
    spec: dict,
    gen: torch.Generator,
    *,
    stacked: bool = False,
    state_dtype: torch.dtype = torch.float32,
) -> tuple[dict, torch.Tensor]:
    """`kda_decode`'s arguments for a cell spec (heads, batch, pool_slots) as the KDA layer passes
    them, and the packed conv pool `[pool_slots, 3D, 3]` its three section views share.

    q / k / v / output gate are column views of the fused [q | k | v | g] GEMM output and beta a
    column view of the merged [f_a | b | pad] projection, or with `stacked` all of them views of
    `stacked_projection`. The forget-gate logits come from their own GEMM, A_log and dt_bias are
    drawn over the engine's ranges, and the conv and `state_dtype` state pools have padded slots.
    """
    heads, batch, slots = spec["heads"], spec["batch"], spec["pool_slots"]
    dim = heads * HEAD_DIM

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen, device="cuda") * scale).to(torch.bfloat16)

    if stacked:
        proj = stacked_projection(batch, heads, gen, "cuda")
        merged, b0 = proj, stacked_columns(heads)["b"]
    else:
        proj = rnd(batch, 4 * dim)
        merged, b0 = rnd(batch, (HEAD_DIM + heads + 7) // 8 * 8), HEAD_DIM
    gate = rnd(batch, dim)
    conv_pool = padded_slots(slots, (3 * dim, 3), torch.bfloat16)
    conv_pool.copy_(torch.randn(conv_pool.shape, generator=gen, device="cuda") * 0.5)
    state = padded_slots(slots, (heads, HEAD_DIM, HEAD_DIM), state_dtype)
    indices = torch.randperm(slots, generator=gen, device="cuda")[:batch].to(torch.int32)
    state.copy_(torch.randn(state.shape, generator=gen, device="cuda") * 0.05)

    def heads_view(t):
        return t.unflatten(-1, (heads, HEAD_DIM)).unsqueeze(0)

    zeros = torch.zeros(dim, dtype=torch.bfloat16, device="cuda")
    args = dict(
        x_q=heads_view(proj[:, :dim]),
        x_k=heads_view(proj[:, dim : 2 * dim]),
        x_v=heads_view(proj[:, 2 * dim : 3 * dim]),
        w_q_t=rnd(4, dim, scale=0.3),
        w_k_t=rnd(4, dim, scale=0.3),
        w_v_t=rnd(4, dim, scale=0.3),
        bias_q=zeros,
        bias_k=zeros,
        bias_v=zeros,
        conv_state_q=conv_pool[:, :dim],
        conv_state_k=conv_pool[:, dim : 2 * dim],
        conv_state_v=conv_pool[:, 2 * dim :],
        a_log=torch.log(torch.empty(heads, device="cuda").uniform_(1, 16, generator=gen)),
        g=heads_view(gate),
        dt_bias=torch.empty(dim, device="cuda").uniform_(
            math.log(1e-3), math.log(1e-1), generator=gen
        ),
        beta=merged[:, b0 : b0 + heads].unsqueeze(0),
        onorm_g=heads_view(proj[:, 3 * dim : 4 * dim]),
        onorm_weight=1 + torch.randn(HEAD_DIM, generator=gen, device="cuda") * 0.1,
        ssm_state_indices=indices,
        state=state,
        lower_bound=LOWER_BOUND,
        scale=SCALE,
        onorm_eps=EPS,
        output=torch.empty(batch, 1, heads, HEAD_DIM, dtype=torch.bfloat16, device="cuda"),
    )
    return args, conv_pool


def clone_strided(t: torch.Tensor) -> torch.Tensor:
    """A copy with the same strides, so the same layout constraints hold for the reference run."""
    out = torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device)
    out.copy_(t)
    return out


def clone_written(args: dict, names) -> dict:
    """A copy of `args` whose `names` tensors are same-stride copies (what the op writes)."""
    return dict(args, **{name: clone_strided(args[name]) for name in names})


def assert_scaled_close(got: torch.Tensor, ref: torch.Tensor, atol: float, what: str) -> None:
    """Within `atol` of `ref`'s largest magnitude (kernel accumulation order and approximate exp / rcp)."""
    scale = ref.float().abs().max().clamp_min(1e-6)
    torch.testing.assert_close(
        got.float() / scale, ref.float() / scale, rtol=0.0, atol=atol, msg=lambda m: f"{what}: {m}"
    )


def assert_rows(pool: torch.Tensor, ref_pool: torch.Tensor, written, atol: float) -> None:
    """The pool rows in `written` within `atol` of the reference's (`assert_scaled_close`), every
    other row bit for bit the reference's."""
    for row in range(pool.shape[0]):
        if row in written:
            assert_scaled_close(pool[row], ref_pool[row], atol, f"state row {row}")
        else:
            assert torch.equal(bits(pool[row]), bits(ref_pool[row])), f"unwritten row {row} changed"


def bits(t: torch.Tensor) -> torch.Tensor:
    """Reinterpret as integers, so that equality means identical bits."""
    return t.view({1: torch.int8, 2: torch.int16, 4: torch.int32}[t.element_size()])


def bare(cls, **attributes):
    """An instance of ``cls`` without its constructor, carrying the given attributes (a cache
    manager without V2 pools, say)."""
    obj = object.__new__(cls)
    for name, value in attributes.items():
        setattr(obj, name, value)
    return obj


def dsa_sparse_config(indexer_k_dtype: str = "fp8"):
    """The hybrid cache manager tests' `DeepSeekSparseAttentionConfig`: 4 index heads of 128, top 64."""
    # Imported here: the CPU reference tests import this module and need only torch.
    from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig

    return DeepSeekSparseAttentionConfig(
        index_n_heads=4,
        index_head_dim=128,
        index_topk=64,
        indexer_k_dtype=indexer_k_dtype,
    )


def dsa_text_config(
    mla_layers: tuple,
    num_hidden_layers: int,
    kv_lora_rank: int = 32,
    qk_rope_head_dim: int = 32,
) -> SimpleNamespace:
    """Text config whose MLA layers (1-indexed) are all DSA layers; every other layer is KDA."""
    return SimpleNamespace(
        architectures=["KimiLinearForCausalLM"],
        model_type="kimi_linear",
        num_hidden_layers=num_hidden_layers,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        linear_attn_config={
            "head_dim": 8,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
            "kda_layers": [
                layer for layer in range(1, num_hidden_layers + 1) if layer not in mla_layers
            ],
            "full_attn_layers": list(mla_layers),
        },
        dsa_config={"dsa_layers": list(mla_layers), "index_topk_freq": 1},
        index_topk_freq=1,
        index_skip_topk_offset=1,
        dtype=torch.bfloat16,
    )
