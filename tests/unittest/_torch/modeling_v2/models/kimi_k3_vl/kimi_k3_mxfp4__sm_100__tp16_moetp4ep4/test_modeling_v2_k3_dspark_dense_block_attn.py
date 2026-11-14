# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the target's attention/k3_dspark_dense_block_attn entry (sm_100): the DSpark drafter's block attention
on the worker's private context arena (``attention_backend="VANILLA"``).

The arguments are built as the built-in DFlash drafter builds them for its VANILLA backend (``DFlashForCausalLM.
dflash_forward``, unwindowed non-causal layer): the arena [slots, layers, columns, KV, 64], the batch's slots, the
block's append slots and columns, the lengths after the block. Each cell checks

* the output and the arena bit for bit against the built-in drafter's own store and flash-attn call in the built-in
  drafter's order (with the built-in drafter's GQA packing): the entry adds nothing;
* the output against ``reference`` within the band flash-attn's bf16 P allows, and the arena against the reference's
  bit for bit (the store is a copy);

and one case replays a CUDA graph captured after an eager call, bit for bit against eager.
"""

import importlib
import zlib

import pytest
import torch

assert torch.cuda.is_available(), "k3_dspark_dense_block_attn requires a CUDA device"

if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
    pytest.skip(
        "k3_dspark_dense_block_attn is certified on sm_100 and sm_103", allow_module_level=True
    )

_attn = importlib.import_module(
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl."
    "kimi_k3_mxfp4__sm_100__tp16_moetp4ep4.ops.attention.k3_dspark_dense_block_attn"
)
k3_dspark_dense_block_attn = _attn.k3_dspark_dense_block_attn

if _attn.unavailability_reason() is not None:
    pytest.skip(
        f"k3_dspark_dense_block_attn: {_attn.unavailability_reason()}", allow_module_level=True
    )

__extra_import_path__ = ["../../.."]
from _validating import validating  # noqa: E402

D = 64
LAYERS = 3
LAYER = 1
# max |err| / max |ref|: flash-attn rounds P and the output to bf16 (~4e-3 each), as for the paged entry.
TOL = 1e-2
# Context lengths before the block, cycled over the requests.
LENGTHS = (0, 64, 60, 127, 1000, 5, 2041, 128)


def _rel_err(a: torch.Tensor, ref: torch.Tensor) -> float:
    return (a.double() - ref.double()).abs().max().item() / max(ref.abs().max().item(), 1e-6)


class _Case:
    """B requests of a T-row block over an arena of ``B + 3`` slots, with the built-in drafter's argument tensors."""

    def __init__(self, seed: int, heads: int, kv: int, batch: int, block: int):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        self.heads, self.kv, self.b, self.t = heads, kv, batch, block
        lengths = [LENGTHS[(seed + 3 * i) % len(LENGTHS)] for i in range(batch)]
        slots = batch + 3
        columns = max(lengths) + block + 5
        shape = (slots, LAYERS, columns, kv, D)
        self.k_cache = (torch.randn(*shape, generator=gen, device="cuda") * 0.5).bfloat16()
        self.v_cache = (torch.randn(*shape, generator=gen, device="cuda") * 0.5).bfloat16()
        self.q = (
            torch.randn(batch, block, heads, D, generator=gen, device="cuda") * 0.5
        ).bfloat16()
        self.k = (torch.randn(batch * block, kv, D, generator=gen, device="cuda") * 0.5).bfloat16()
        self.v = (torch.randn(batch * block, kv, D, generator=gen, device="cuda") * 0.5).bfloat16()
        # The built-in drafter's: cache_seqlens_i32, cache_batch_idx_i32, seq_lens_after and the append indices.
        cache_seqlens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
        self.cache_batch_idx = torch.randperm(slots, generator=gen, device="cuda")[:batch].to(
            torch.int32
        )
        self.seq_lens_after = cache_seqlens + block
        offsets = torch.arange(block, dtype=torch.int32, device="cuda")
        self.append_batch_indices = (
            self.cache_batch_idx.view(-1, 1).expand(-1, block).reshape(-1).long()
        )
        self.append_positions = (cache_seqlens.view(-1, 1) + offsets).reshape(-1).long()

    def args(self, k_cache=None, v_cache=None, pack_gqa=True):
        return (
            self.q,
            self.k,
            self.v,
            self.k_cache.clone() if k_cache is None else k_cache,
            self.v_cache.clone() if v_cache is None else v_cache,
            LAYER,
            self.append_batch_indices,
            self.append_positions,
            self.seq_lens_after,
            self.cache_batch_idx,
            pack_gqa,
        )


def _builtin(case: _Case, k_cache, v_cache, pack_gqa: bool) -> torch.Tensor:
    """The built-in VANILLA store and attention, as ``DFlashForCausalLM.dflash_forward`` runs them per layer."""
    from tensorrt_llm._torch.speculative.dflash_attention import get_dflash_flash_attention

    flash_attention = get_dflash_flash_attention()
    b, t, h, kv = case.b, case.t, case.heads, case.kv
    group = h // kv
    k_cache[case.append_batch_indices, LAYER, case.append_positions] = case.k
    v_cache[case.append_batch_indices, LAYER, case.append_positions] = case.v
    if pack_gqa:
        q_in = (
            case.q.reshape(b, t, kv, group, D).permute(0, 3, 1, 2, 4).reshape(b, group * t, kv, D)
        )
    else:
        q_in = case.q
    out = flash_attention(
        q=q_in,
        k_cache=k_cache[:, LAYER],
        v_cache=v_cache[:, LAYER],
        cache_seqlens=case.seq_lens_after,
        cache_batch_idx=case.cache_batch_idx,
        causal=False,
        window_size=(-1, -1),
    )
    if pack_gqa:
        out = out.view(b, group, t, kv, D).permute(0, 2, 3, 1, 4)
    # The built-in drafter's next step: the rows o_proj reads, [B T, H D].
    return out.reshape(b * t, h * D)


@pytest.mark.parametrize("cell", k3_dspark_dense_block_attn.CELLS, ids=lambda c: c.why)
def test_k3_dspark_dense_block_attn(cell) -> None:
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    case = _Case(seed, spec["heads"], spec["kv"], spec["batch"], spec["block"])
    pack = spec["pack_gqa"]
    args = case.args(pack_gqa=pack)
    with validating(k3_dspark_dense_block_attn):
        out = k3_dspark_dense_block_attn(*args)
    k_builtin, v_builtin = case.k_cache.clone(), case.v_cache.clone()
    want_builtin = _builtin(case, k_builtin, v_builtin, pack)
    ref_args = case.args(pack_gqa=pack)
    ref = k3_dspark_dense_block_attn.reference(*ref_args)
    torch.cuda.synchronize()
    assert out.shape == (case.b, case.t, case.heads, D)
    # The drafter flattens the rows for o_proj the way the built-in drafter does: the same values, bit for bit.
    assert torch.equal(out.reshape(case.b * case.t, case.heads * D), want_builtin), (
        "the entry differs from the built-in store + flash-attn call"
    )
    assert torch.equal(args[3], k_builtin) and torch.equal(args[4], v_builtin), (
        "the arena differs from the built-in drafter's"
    )
    assert torch.equal(args[3], ref_args[3]) and torch.equal(args[4], ref_args[4]), (
        "the store is not a copy"
    )
    err = _rel_err(out, ref)
    assert err <= TOL, f"rel err {err:.3e}"


def test_loader_is_the_speculative_runtimes() -> None:
    """The op loads the same ``flash_attn_with_kvcache`` the DFlash speculative runtime loads."""
    from tensorrt_llm._torch.speculative.dflash_attention import get_dflash_flash_attention

    assert _attn.flash_attention() is get_dflash_flash_attention()


def test_packing_does_not_change_the_result_beyond_rounding() -> None:
    """Packed and unpacked GQA are the same attention; flash-attn may order the reduction differently."""
    case = _Case(7, 4, 1, 2, 7)
    packed = k3_dspark_dense_block_attn(*case.args(pack_gqa=True))
    unpacked = k3_dspark_dense_block_attn(*case.args(pack_gqa=False))
    assert _rel_err(packed, unpacked) <= TOL


def test_capture_replays_the_eager_result() -> None:
    case = _Case(11, 4, 1, 2, 7)
    k_cache, v_cache = case.k_cache.clone(), case.v_cache.clone()
    eager = k3_dspark_dense_block_attn(*case.args(k_cache=k_cache.clone(), v_cache=v_cache.clone()))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = k3_dspark_dense_block_attn(*case.args(k_cache=k_cache, v_cache=v_cache))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, eager)


@pytest.mark.parametrize("bad", ["fp16 q", "int64 lengths", "4-D arena"])
def test_is_valid_refuses(bad) -> None:
    case = _Case(3, 4, 1, 1, 7)
    args = list(case.args())
    if bad == "fp16 q":
        args[0] = args[0].half()
    elif bad == "int64 lengths":
        args[8] = args[8].long()
    else:
        args[3] = args[3][:, 0]
    with pytest.raises(AssertionError):
        k3_dspark_dense_block_attn.is_valid(*args)
